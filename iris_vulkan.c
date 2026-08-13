/*
 * Iris Vulkan Acceleration
 *
 * Compute backend for the Z-Image GPU path. The implementation intentionally
 * exports the existing iris_gpu_* contract so the transformer does not need a
 * second backend-specific execution graph.
 */

#include "iris_metal.h"
#include "iris_platform.h"
#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct iris_gpu_tensor {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t elements;
    size_t bytes;
    int is_f16;
    int persistent;
    int cached;
    int host_visible;
    int mapped;
    struct iris_gpu_tensor *deferred_next;
} iris_gpu_tensor_impl_t;

typedef struct vk_cached_buffer {
    const void *host_ptr;
    size_t bytes;
    int is_f16;
    iris_gpu_tensor_impl_t *tensor;
    struct vk_cached_buffer *next;
} vk_cached_buffer_t;

typedef enum {
    VK_PIPE_LINEAR,
    VK_PIPE_LINEAR_BF16,
    VK_PIPE_LINEAR_BF16_COOP,
    VK_PIPE_LINEAR_FP8,
    VK_PIPE_LINEAR_FP8_COOP,
    VK_PIPE_FP8_TO_BF16,
    VK_PIPE_RMS_NORM,
    VK_PIPE_QK_RMS_NORM,
    VK_PIPE_ROPE_PAIR,
    VK_PIPE_ATTENTION,
    VK_PIPE_ATTENTION_SUBGROUP,
    VK_PIPE_SILU_MUL,
    VK_PIPE_ADD,
    VK_PIPE_GATED_ADD,
    VK_PIPE_ADALN_NORM,
    VK_PIPE_GROUP_NORM,
    VK_PIPE_SWISH,
    VK_PIPE_UPSAMPLE,
    VK_PIPE_CONV2D,
    VK_PIPE_CONV2D_COOP,
    VK_PIPE_F32_TO_BF16,
    VK_PIPE_BF16_TO_F32,
    VK_PIPE_LINEAR_BF16_NATIVE,
    VK_PIPE_LINEAR_BF16_NATIVE_QKV,
    VK_PIPE_LINEAR_BF16_NATIVE_GATE_UP,
    VK_PIPE_RMS_NORM_BF16,
    VK_PIPE_RMS_NORM_BF16_F32_WEIGHT,
    VK_PIPE_HEAD_RMS_NORM_BF16,
    VK_PIPE_QK_RMS_NORM_BF16_F32_WEIGHT,
    VK_PIPE_ELEMENTWISE_BF16,
    VK_PIPE_GATED_ADD_BF16_F32_GATE,
    VK_PIPE_ROPE_PAIR_BF16,
    VK_PIPE_ATTENTION_SUBGROUP_BF16,
    VK_PIPE_ROPE_TEXT_BF16,
    VK_PIPE_CAUSAL_ATTENTION_BF16,
    VK_PIPE_COUNT
} vk_pipeline_id_t;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkDescriptorSetLayout descriptor_layout;
    VkDescriptorPool descriptor_pool;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipelines[VK_PIPE_COUNT];
    VkBuffer dummy_buffer;
    VkDeviceMemory dummy_memory;
    int dummy_host_visible;
    int initialized;
    int device_lost;
    int batch_active;
    iris_gpu_tensor_impl_t *deferred_tensors;
    vk_cached_buffer_t *cache;
    size_t memory_used;
    iris_gpu_tensor_impl_t *stream_weight;
    iris_gpu_tensor_impl_t *stream_staging;
    iris_gpu_tensor_impl_t *decoded_weight;
    size_t stream_weight_bytes;
    size_t decoded_weight_bytes;
    const void *stream_source;
    size_t stream_source_bytes;
    int stream_source_fp8;
    size_t fp8_cache_bytes;
    size_t fp8_cache_limit;
    double last_submit_ms;
    double max_submit_ms;
    double trace_window_start_ms;
    double trace_window_submit_ms;
    double trace_window_max_ms;
    unsigned int trace_window_packets;
    int dedicated_compute_queue;
    int global_low_priority;
    int cooperative_matrix;
    int subgroup_attention;
} vk_context_t;

static vk_context_t vk_ctx;
static int vk_friendly_requested;

static void vk_report(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: %s failed (%d)\n", operation, result);
        /* Stop submitting work after the device has been lost */
        if (result == VK_ERROR_DEVICE_LOST) {
            vk_ctx.device_lost = 1;
            vk_ctx.batch_active = 0;
        }
    }
}

static int vk_ready(void) {
    return vk_ctx.initialized && !vk_ctx.device_lost;
}

static int vk_trace_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        /* Enable dispatch tracing only when explicitly requested */
        enabled = getenv("IRIS_VULKAN_TRACE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int vk_packet_trace_enabled(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        /* Keep high-volume packet logging behind a separate diagnostic switch */
        enabled = getenv("IRIS_VULKAN_TRACE_PACKETS") != NULL;
        initialized = 1;
    }
    return enabled;
}

void iris_gpu_set_friendly_mode(int enable) {
    /* Retain the preference before or after Vulkan context initialization */
    vk_friendly_requested = enable != 0;
}

static uint32_t vk_find_memory_type(uint32_t type_bits,
                                    VkMemoryPropertyFlags required,
                                    VkMemoryPropertyFlags preferred) {
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(vk_ctx.physical_device, &properties);

    for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((properties.memoryTypes[i].propertyFlags & required) == required &&
            (properties.memoryTypes[i].propertyFlags & preferred) == preferred) {
            return i;
        }
    }
    for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int vk_create_buffer(size_t bytes, VkBufferUsageFlags usage,
                            VkBuffer *buffer, VkDeviceMemory *memory,
                            int *host_visible) {
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes ? bytes : 4;
    buffer_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(vk_ctx.device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkCreateBuffer");
        return 0;
    }

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(vk_ctx.device, *buffer, &requirements);
    uint32_t memory_type = vk_find_memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (memory_type == UINT32_MAX) {
        memory_type = vk_find_memory_type(requirements.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        *host_visible = 0;
    } else {
        *host_visible = 1;
    }
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return 0;
    }

    VkMemoryAllocateInfo allocation = {0};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk_ctx.device, &allocation, NULL, memory);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkAllocateMemory");
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return 0;
    }

    result = vkBindBufferMemory(vk_ctx.device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkBindBufferMemory");
        vkFreeMemory(vk_ctx.device, *memory, NULL);
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        *memory = VK_NULL_HANDLE;
        return 0;
    }
    vk_ctx.memory_used += requirements.size;
    return 1;
}

static iris_gpu_tensor_impl_t *vk_tensor_alloc_bytes(size_t bytes, int is_f16,
                                                      int cached) {
    iris_gpu_tensor_impl_t *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    if (!vk_create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          &tensor->buffer, &tensor->memory,
                          &tensor->host_visible)) {
        free(tensor);
        return NULL;
    }
    tensor->bytes = bytes;
    tensor->elements = is_f16 ? bytes / 2 : bytes / sizeof(float);
    tensor->is_f16 = is_f16;
    tensor->cached = cached;
    return tensor;
}

static int vk_create_device_local_buffer(size_t bytes, VkBuffer *buffer,
                                         VkDeviceMemory *memory) {
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes ? bytes : 4;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    /* Create the immutable weight buffer in device-local memory */
    VkResult result = vkCreateBuffer(vk_ctx.device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkCreateBuffer");
        return 0;
    }

    /* Choose device-local memory so immutable weights do not consume host commit */
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(vk_ctx.device, *buffer, &requirements);
    uint32_t memory_type = vk_find_memory_type(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return 0;
    }

    /* Allocate and bind the device-local storage */
    VkMemoryAllocateInfo allocation = {0};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk_ctx.device, &allocation, NULL, memory);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkAllocateMemory");
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return 0;
    }
    result = vkBindBufferMemory(vk_ctx.device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkBindBufferMemory");
        vkFreeMemory(vk_ctx.device, *memory, NULL);
        vkDestroyBuffer(vk_ctx.device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        *memory = VK_NULL_HANDLE;
        return 0;
    }
    vk_ctx.memory_used += requirements.size;
    return 1;
}

static iris_gpu_tensor_impl_t *vk_tensor_alloc_device_local_bytes(size_t bytes,
                                                                    int is_f16,
                                                                    int cached) {
    iris_gpu_tensor_impl_t *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    if (!vk_create_device_local_buffer(bytes, &tensor->buffer, &tensor->memory)) {
        free(tensor);
        return NULL;
    }
    tensor->bytes = bytes;
    tensor->elements = is_f16 ? bytes / 2 : bytes / sizeof(float);
    tensor->is_f16 = is_f16;
    tensor->cached = cached;
    tensor->host_visible = 0;
    return tensor;
}

static void vk_tensor_unmap(iris_gpu_tensor_impl_t *tensor);

static void vk_tensor_destroy(iris_gpu_tensor_impl_t *tensor) {
    if (!tensor || tensor->cached) return;
    /* Release CPU bookkeeping without touching a device that has stopped responding */
    if (vk_ctx.device_lost) {
        free(tensor);
        return;
    }
    vk_tensor_unmap(tensor);
    if (tensor->buffer) vkDestroyBuffer(vk_ctx.device, tensor->buffer, NULL);
    if (tensor->memory) vkFreeMemory(vk_ctx.device, tensor->memory, NULL);
    if (vk_ctx.memory_used >= tensor->bytes) vk_ctx.memory_used -= tensor->bytes;
    free(tensor);
}

static void vk_flush_deferred_tensors(void) {
    /* Release buffers only after all recorded references have finished executing */
    iris_gpu_tensor_impl_t *tensor = vk_ctx.deferred_tensors;
    vk_ctx.deferred_tensors = NULL;
    while (tensor) {
        iris_gpu_tensor_impl_t *next = tensor->deferred_next;
        tensor->deferred_next = NULL;
        vk_tensor_destroy(tensor);
        tensor = next;
    }
}

static void vk_tensor_destroy_deferred(iris_gpu_tensor_impl_t *tensor) {
    if (!tensor || tensor->cached) return;
    /* Keep buffers alive while the active command buffer still references them */
    if (vk_ctx.batch_active && !vk_ctx.device_lost) {
        tensor->deferred_next = vk_ctx.deferred_tensors;
        vk_ctx.deferred_tensors = tensor;
        return;
    }
    vk_tensor_destroy(tensor);
}

static int vk_tensor_map(iris_gpu_tensor_impl_t *tensor, void **data) {
    if (!tensor || !data || !tensor->host_visible || vk_ctx.device_lost) return 0;
    if (tensor->mapped) return 0;
    VkResult result = vkMapMemory(vk_ctx.device, tensor->memory, 0,
                                   tensor->bytes ? tensor->bytes : 4, 0, data);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkMapMemory");
        return 0;
    }
    tensor->mapped = 1;
    return 1;
}

static void vk_tensor_unmap(iris_gpu_tensor_impl_t *tensor) {
    if (tensor && tensor->host_visible && tensor->mapped) {
        /* Forget mappings without issuing Vulkan calls after device loss */
        if (vk_ctx.device_lost) {
            tensor->mapped = 0;
            return;
        }
        vkUnmapMemory(vk_ctx.device, tensor->memory);
        tensor->mapped = 0;
    }
}

static uint16_t vk_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float vk_bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void vk_host_copy(void *dst, const void *src, size_t bytes) {
    /* Keep the normal path at full memcpy throughput */
    if (!vk_friendly_requested) {
        memcpy(dst, src, bytes);
        return;
    }

    /* Fault and copy mapped model pages in scheduler-friendly pieces */
    double start_ms = iris_time_ms();
    const size_t chunk_bytes = 1024u * 1024u;
    size_t offset = 0;
    while (offset < bytes) {
        size_t count = bytes - offset;
        if (count > chunk_bytes) count = chunk_bytes;
        memcpy((uint8_t *)dst + offset, (const uint8_t *)src + offset, count);
        offset += count;
        if (offset < bytes) iris_sleep_ms(1);
    }
    /* Report large host transfers only when Vulkan tracing is requested */
    if (vk_trace_enabled() && bytes >= 1024u * 1024u)
        fprintf(stderr, "Vulkan: friendly host copy %.1f MiB in %.1f ms\n",
                (double)bytes / (1024.0 * 1024.0), iris_time_ms() - start_ms);
}

static void vk_host_convert_bf16(uint16_t *dst, const float *src,
                                  size_t elements) {
    const size_t chunk_elements = 256u * 1024u;
    size_t offset = 0;

    /* Convert mapped model pages in bounded CPU scheduling intervals */
    while (offset < elements) {
        size_t count = elements - offset;
        if (vk_friendly_requested && count > chunk_elements)
            count = chunk_elements;
        for (size_t i = 0; i < count; i++)
            dst[offset + i] = vk_float_to_bf16(src[offset + i]);
        offset += count;
        if (vk_friendly_requested && offset < elements) iris_sleep_ms(1);
    }
}

static int vk_begin_command_buffer(void) {
    /* Do not record commands after a lost device has been detected */
    if (vk_ctx.device_lost) return 0;
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult result = vkBeginCommandBuffer(vk_ctx.command_buffer, &begin);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkBeginCommandBuffer");
        return 0;
    }
    return 1;
}

static void vk_memory_barrier(void) {
    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(vk_ctx.command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
}

static int vk_submit_command_buffer(void) {
    /* Abandon pending work after a lost device has been detected */
    if (vk_ctx.device_lost) {
        vk_ctx.batch_active = 0;
        vk_flush_deferred_tensors();
        return 0;
    }
    VkResult result = vkEndCommandBuffer(vk_ctx.command_buffer);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkEndCommandBuffer");
        return 0;
    }

    /* Measure the complete queue submission and fence latency */
    double submit_start = iris_time_ms();
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk_ctx.command_buffer;
    result = vkQueueSubmit(vk_ctx.queue, 1, &submit, vk_ctx.fence);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkQueueSubmit");
        return 0;
    }
    result = vkWaitForFences(vk_ctx.device, 1, &vk_ctx.fence, VK_TRUE, UINT64_MAX);
    vk_report(result, "vkWaitForFences");
    vk_ctx.last_submit_ms = iris_time_ms() - submit_start;
    if (vk_ctx.last_submit_ms > vk_ctx.max_submit_ms)
        vk_ctx.max_submit_ms = vk_ctx.last_submit_ms;
    /* Aggregate normal tracing so terminal rendering cannot perturb GPU scheduling */
    if (vk_trace_enabled()) {
        double now_ms = iris_time_ms();
        if (vk_ctx.trace_window_start_ms == 0.0)
            vk_ctx.trace_window_start_ms = now_ms;
        vk_ctx.trace_window_packets++;
        vk_ctx.trace_window_submit_ms += vk_ctx.last_submit_ms;
        if (vk_ctx.last_submit_ms > vk_ctx.trace_window_max_ms)
            vk_ctx.trace_window_max_ms = vk_ctx.last_submit_ms;
        if (vk_packet_trace_enabled()) {
            fprintf(stderr, "Vulkan: packet %.1f ms\n", vk_ctx.last_submit_ms);
        } else if (now_ms - vk_ctx.trace_window_start_ms >= 10000.0) {
            /* Report observed queue duty cycle over the completed trace window */
            double window_ms = now_ms - vk_ctx.trace_window_start_ms;
            double duty_percent = 100.0 * vk_ctx.trace_window_submit_ms / window_ms;
            if (duty_percent > 100.0) duty_percent = 100.0;
            fprintf(stderr,
                    "Vulkan trace: %u packets, %.1f ms GPU (%.0f%% duty), "
                    "%.1f ms longest\n",
                    vk_ctx.trace_window_packets, vk_ctx.trace_window_submit_ms,
                    duty_percent, vk_ctx.trace_window_max_ms);
            vk_ctx.trace_window_start_ms = now_ms;
            vk_ctx.trace_window_submit_ms = 0.0;
            vk_ctx.trace_window_max_ms = 0.0;
            vk_ctx.trace_window_packets = 0;
        }
    }
    if (result != VK_SUCCESS && vk_ctx.device_lost) {
        /* Leave lost-device handles untouched because cleanup cannot safely use them */
        vk_ctx.batch_active = 0;
        vk_flush_deferred_tensors();
        return 0;
    }
    vkResetFences(vk_ctx.device, 1, &vk_ctx.fence);
    vkResetCommandPool(vk_ctx.device, vk_ctx.command_pool, 0);
    if (vk_ctx.descriptor_pool)
        vkResetDescriptorPool(vk_ctx.device, vk_ctx.descriptor_pool, 0);

    /* Reclaim transient buffers after the fence proves the batch is complete */
    vk_flush_deferred_tensors();

    /* Leave a brief scheduling window between bounded GPU submissions */
    if (result == VK_SUCCESS && vk_friendly_requested) iris_sleep_ms(1);
    return result == VK_SUCCESS;
}

static int vk_friendly_next_chunk(int current, int maximum, int quantum) {
    /* Grow only while the measured packet remains comfortably interactive */
    if (!vk_friendly_requested) return maximum;
    if (vk_ctx.last_submit_ms < 12.0 && current <= maximum / 2)
        return current * 2;

    /* Back off when a packet exceeded the desktop latency budget */
    if (vk_ctx.last_submit_ms > 30.0 && current > quantum)
        return current / 2;
    return current;
}

static int vk_submit_and_resume_batch(void) {
    /* Leave standalone dispatches alone because they already submitted */
    if (!vk_ctx.batch_active) return 1;

    /* Finish the bounded packet and wait before recording more work */
    vk_ctx.batch_active = 0;
    if (!vk_submit_command_buffer()) return 0;

    /* Resume the caller's logical batch with fresh command resources */
    if (!vk_begin_command_buffer()) return 0;
    vk_ctx.batch_active = 1;
    return 1;
}

static int vk_begin_if_needed(int *temporary_batch) {
    /* Refuse new work once Vulkan has reported device loss */
    if (!vk_ready()) return 0;
    if (vk_ctx.batch_active) {
        *temporary_batch = 0;
        return 1;
    }
    vkResetCommandPool(vk_ctx.device, vk_ctx.command_pool, 0);
    if (!vk_begin_command_buffer()) return 0;
    vk_ctx.batch_active = 1;
    *temporary_batch = 1;
    return 1;
}

static int vk_end_if_temporary(int temporary_batch) {
    if (!temporary_batch) return 1;
    vk_ctx.batch_active = 0;
    return vk_submit_command_buffer();
}

static int vk_descriptor_set(VkDescriptorSet *set,
                             iris_gpu_tensor_impl_t *const *buffers,
                             int count) {
    /* Refuse descriptor allocation once Vulkan has reported device loss */
    if (!vk_ready()) return 0;
    VkDescriptorSetAllocateInfo allocation = {0};
    allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocation.descriptorPool = vk_ctx.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &vk_ctx.descriptor_layout;
    VkResult result = vkAllocateDescriptorSets(vk_ctx.device, &allocation, set);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkAllocateDescriptorSets");
        return 0;
    }

    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 8; i++) {
        iris_gpu_tensor_impl_t *tensor = (i < count && buffers[i])
            ? buffers[i] : (iris_gpu_tensor_impl_t *)NULL;
        infos[i].buffer = tensor ? tensor->buffer : vk_ctx.dummy_buffer;
        infos[i].offset = 0;
        infos[i].range = tensor && tensor->bytes ? tensor->bytes : VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = *set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk_ctx.device, 8, writes, 0, NULL);
    return 1;
}

static int vk_dispatch(vk_pipeline_id_t pipeline_id,
                       iris_gpu_tensor_impl_t *const *buffers, int count,
                       const uint32_t *push, uint32_t push_count,
                       uint32_t groups_x, uint32_t groups_y,
                       uint32_t groups_z) {
    /* Refuse dispatch once Vulkan has reported device loss */
    if (!vk_ready() || !vk_ctx.pipelines[pipeline_id]) return 0;
    if (vk_packet_trace_enabled()) {
        /* Print the kernel and workgroup dimensions before recording it */
        fprintf(stderr, "Vulkan: dispatch %d groups %u,%u,%u batch=%d memory=%zu\n",
                pipeline_id, groups_x, groups_y, groups_z, vk_ctx.batch_active,
                vk_ctx.memory_used);
    }
    int temporary_batch = 0;
    if (!vk_begin_if_needed(&temporary_batch)) return 0;

    VkDescriptorSet descriptor_set;
    if (!vk_descriptor_set(&descriptor_set, buffers, count)) {
        if (temporary_batch) {
            vk_ctx.batch_active = 0;
            vkResetCommandPool(vk_ctx.device, vk_ctx.command_pool, 0);
        }
        return 0;
    }
    vkCmdBindPipeline(vk_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      vk_ctx.pipelines[pipeline_id]);
    vkCmdBindDescriptorSets(vk_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk_ctx.pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    if (push_count) {
        vkCmdPushConstants(vk_ctx.command_buffer, vk_ctx.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           push_count * sizeof(uint32_t), push);
    }
    vkCmdDispatch(vk_ctx.command_buffer, groups_x, groups_y, groups_z);
    vk_memory_barrier();
    return vk_end_if_temporary(temporary_batch);
}

static int vk_copy_region(iris_gpu_tensor_impl_t *dst, size_t dst_offset,
                          iris_gpu_tensor_impl_t *src, size_t src_offset,
                          size_t bytes) {
    /* Refuse buffer copies once Vulkan has reported device loss */
    if (!vk_ready() || !dst || !src) return 0;

    /* Preserve an enclosing batch while splitting standalone transfers */
    int enclosing_batch = vk_ctx.batch_active;
    double start_ms = iris_time_ms();
    const size_t friendly_chunk = 1024u * 1024u;
    size_t offset = 0;
    while (offset < bytes) {
        size_t count = bytes - offset;
        if (vk_friendly_requested && count > friendly_chunk)
            count = friendly_chunk;
        int temporary_batch = 0;
        if (!vk_begin_if_needed(&temporary_batch)) return 0;
        VkBufferCopy copy = {0};
        copy.srcOffset = src_offset + offset;
        copy.dstOffset = dst_offset + offset;
        copy.size = count;
        vkCmdCopyBuffer(vk_ctx.command_buffer, src->buffer, dst->buffer, 1, &copy);
        vk_memory_barrier();
        if (!enclosing_batch) {
            vk_ctx.batch_active = 0;
            if (!vk_submit_command_buffer()) return 0;
        }
        offset += count;
    }
    /* Report large GPU transfers only when Vulkan tracing is requested */
    if (vk_trace_enabled() && vk_friendly_requested && bytes >= 1024u * 1024u)
        fprintf(stderr, "Vulkan: friendly GPU copy %.1f MiB in %.1f ms\n",
                (double)bytes / (1024.0 * 1024.0), iris_time_ms() - start_ms);
    return 1;
}

static iris_gpu_tensor_impl_t *vk_cached_key(const void *cache_key,
                                              const void *host_ptr,
                                              size_t bytes, int is_f16) {
    /* Refuse new cached buffers once Vulkan has reported device loss */
    if (!vk_ready() || !cache_key) return NULL;
    for (vk_cached_buffer_t *entry = vk_ctx.cache; entry; entry = entry->next) {
        if (entry->host_ptr == cache_key && entry->bytes == bytes &&
            entry->is_f16 == is_f16) {
            void *mapped = NULL;
            /* Refresh mutable host arrays before reusing their cached buffer */
            if (!is_f16 && host_ptr && vk_tensor_map(entry->tensor, &mapped)) {
                vk_host_copy(mapped, host_ptr, bytes);
                vk_tensor_unmap(entry->tensor);
            }
            return entry->tensor;
        }
    }
    /* Reject a cache miss when the caller has already released its host copy */
    if (!host_ptr) return NULL;
    iris_gpu_tensor_impl_t *tensor = NULL;
    if (is_f16 || bytes >= 1024u * 1024u)
        tensor = vk_tensor_alloc_device_local_bytes(bytes, is_f16, 1);
    if (!tensor && is_f16 != 2)
        tensor = vk_tensor_alloc_bytes(bytes, is_f16, 1);
    if (!tensor) return NULL;

    /* Upload device-local weights through a temporary host-visible staging buffer */
    if (tensor->host_visible) {
        void *mapped = NULL;
        if (!vk_tensor_map(tensor, &mapped)) {
            tensor->cached = 0;
            vk_tensor_destroy(tensor);
            return NULL;
        }
        vk_host_copy(mapped, host_ptr, bytes);
        vk_tensor_unmap(tensor);
    } else {
        iris_gpu_tensor_impl_t *staging = vk_tensor_alloc_bytes(bytes, is_f16, 0);
        void *mapped = NULL;
        if (!staging || !vk_tensor_map(staging, &mapped)) {
            if (staging) vk_tensor_destroy(staging);
            tensor->cached = 0;
            vk_tensor_destroy(tensor);
            return NULL;
        }
        vk_host_copy(mapped, host_ptr, bytes);
        vk_tensor_unmap(staging);
        if (!vk_copy_region(tensor, 0, staging, 0, bytes)) {
            vk_tensor_destroy(staging);
            tensor->cached = 0;
            vk_tensor_destroy(tensor);
            return NULL;
        }
        vk_tensor_destroy_deferred(staging);
    }

    vk_cached_buffer_t *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        /* Release the allocation when the cache node cannot be created */
        tensor->cached = 0;
        vk_tensor_destroy(tensor);
        return NULL;
    }
    entry->host_ptr = cache_key;
    entry->bytes = bytes;
    entry->is_f16 = is_f16;
    entry->tensor = tensor;
    entry->next = vk_ctx.cache;
    vk_ctx.cache = entry;
    return tensor;
}

static iris_gpu_tensor_impl_t *vk_cached_find(const void *cache_key,
                                               size_t bytes, int storage_kind) {
    /* Locate an immutable cached payload without refreshing its host data */
    for (vk_cached_buffer_t *entry = vk_ctx.cache; entry; entry = entry->next) {
        if (entry->host_ptr == cache_key && entry->bytes == bytes &&
            entry->is_f16 == storage_kind)
            return entry->tensor;
    }
    return NULL;
}

static iris_gpu_tensor_impl_t *vk_cached_fp8(const uint8_t *weights,
                                              size_t bytes) {
    /* Reuse an existing FP8 allocation before applying the cache budget */
    iris_gpu_tensor_impl_t *tensor = vk_cached_find(weights, bytes, 2);
    if (tensor) return tensor;
    if (!weights || !bytes || bytes > vk_ctx.fp8_cache_limit -
        (vk_ctx.fp8_cache_bytes < vk_ctx.fp8_cache_limit
            ? vk_ctx.fp8_cache_bytes : vk_ctx.fp8_cache_limit))
        return NULL;

    /* Upload immutable FP8 data into budgeted device-local storage */
    tensor = vk_cached_key(weights, weights, bytes, 2);
    if (tensor) vk_ctx.fp8_cache_bytes += bytes;
    return tensor;
}

static iris_gpu_tensor_impl_t *vk_cached(const void *host_ptr, size_t bytes,
                                          int is_f16) {
    /* Use the payload address as the default cache key */
    return vk_cached_key(host_ptr, host_ptr, bytes, is_f16);
}

static int vk_stream_weight_prepare(size_t bytes) {
    if (vk_ctx.stream_weight && vk_ctx.stream_staging &&
        vk_ctx.stream_weight_bytes >= bytes)
        return 1;

    /* Replace undersized reusable streaming buffers between submitted kernels */
    vk_tensor_destroy_deferred(vk_ctx.stream_weight);
    vk_tensor_destroy_deferred(vk_ctx.stream_staging);
    vk_ctx.stream_weight = NULL;
    vk_ctx.stream_staging = NULL;
    vk_ctx.stream_weight_bytes = 0;
    vk_ctx.stream_source = NULL;
    vk_ctx.stream_source_bytes = 0;
    vk_ctx.stream_source_fp8 = 0;

    /* Allocate one device-local weight buffer and one host-visible conversion buffer */
    vk_ctx.stream_weight = vk_tensor_alloc_device_local_bytes(bytes, 1, 0);
    vk_ctx.stream_staging = vk_tensor_alloc_bytes(bytes, 1, 0);
    if (!vk_ctx.stream_weight || !vk_ctx.stream_staging ||
        !vk_ctx.stream_staging->host_visible) {
        vk_tensor_destroy(vk_ctx.stream_weight);
        vk_tensor_destroy(vk_ctx.stream_staging);
        vk_ctx.stream_weight = NULL;
        vk_ctx.stream_staging = NULL;
        return 0;
    }
    vk_ctx.stream_weight_bytes = bytes;
    return 1;
}

static int vk_stream_weight_upload(const float *weights, size_t elements) {
    size_t bytes = elements * sizeof(uint16_t);
    if (!weights || !elements || !vk_stream_weight_prepare(bytes)) return 0;

    /* Invalidate the source identity because F32 conversion rewrites the buffer */
    vk_ctx.stream_source = NULL;
    vk_ctx.stream_source_bytes = 0;
    vk_ctx.stream_source_fp8 = 0;

    /* Preserve queued uploads with private staging storage during a logical batch */
    iris_gpu_tensor_impl_t *staging = vk_ctx.batch_active
        ? vk_tensor_alloc_bytes(bytes, 1, 0) : vk_ctx.stream_staging;
    if (!staging) return 0;

    /* Convert mapped F32 weights directly into the selected host-visible buffer */
    void *mapped = NULL;
    if (!vk_tensor_map(staging, &mapped)) {
        if (staging != vk_ctx.stream_staging) vk_tensor_destroy(staging);
        return 0;
    }
    uint16_t *bf16 = mapped;
    vk_host_convert_bf16(bf16, weights, elements);
    vk_tensor_unmap(staging);

    /* Upload the converted matrix before the reusable buffer is overwritten */
    int copied = vk_copy_region(vk_ctx.stream_weight, 0, staging, 0, bytes);
    if (staging != vk_ctx.stream_staging) vk_tensor_destroy_deferred(staging);
    return copied;
}

static int vk_stream_fp8_upload(const uint8_t *weights, size_t elements) {
    size_t bytes = (elements + 3u) & ~(size_t)3u;
    if (!weights || !elements || !vk_stream_weight_prepare(bytes)) return 0;

    /* Reuse a fused payload across its Q, K, and V projections */
    if (vk_ctx.stream_source == weights && vk_ctx.stream_source_bytes == elements &&
        vk_ctx.stream_source_fp8)
        return 1;

    /* Preserve queued uploads with private staging storage during a logical batch */
    iris_gpu_tensor_impl_t *staging = vk_ctx.batch_active
        ? vk_tensor_alloc_bytes(bytes, 1, 0) : vk_ctx.stream_staging;
    if (!staging) return 0;

    /* Copy raw FP8 bytes and clear any alignment padding */
    void *mapped = NULL;
    if (!vk_tensor_map(staging, &mapped)) {
        if (staging != vk_ctx.stream_staging) vk_tensor_destroy(staging);
        return 0;
    }
    vk_host_copy(mapped, weights, elements);
    if (bytes > elements) memset((uint8_t *)mapped + elements, 0, bytes - elements);
    vk_tensor_unmap(staging);
    if (!vk_copy_region(vk_ctx.stream_weight, 0, staging, 0, bytes)) {
        if (staging != vk_ctx.stream_staging) vk_tensor_destroy_deferred(staging);
        return 0;
    }
    if (staging != vk_ctx.stream_staging) vk_tensor_destroy_deferred(staging);
    vk_ctx.stream_source = weights;
    vk_ctx.stream_source_bytes = elements;
    vk_ctx.stream_source_fp8 = 1;
    return 1;
}

static int vk_decoded_weight_prepare(size_t elements) {
    size_t bytes = elements * sizeof(uint16_t);
    if (vk_ctx.decoded_weight && vk_ctx.decoded_weight_bytes >= bytes) return 1;

    /* Grow the reusable BF16 matrix scratch while preserving queued users */
    vk_tensor_destroy_deferred(vk_ctx.decoded_weight);
    vk_ctx.decoded_weight = vk_tensor_alloc_device_local_bytes(bytes, 1, 0);
    if (!vk_ctx.decoded_weight) {
        vk_ctx.decoded_weight_bytes = 0;
        return 0;
    }
    vk_ctx.decoded_weight_bytes = bytes;
    return 1;
}

static int vk_decode_fp8_weight(iris_gpu_tensor_impl_t *source,
                                size_t source_offset, float scale,
                                size_t destination_offset, size_t elements,
                                size_t total_elements) {
    if (!source || !vk_decoded_weight_prepare(total_elements)) return 0;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    iris_gpu_tensor_impl_t *buffers[2] = {source, vk_ctx.decoded_weight};
    uint32_t push[4] = {(uint32_t)source_offset, (uint32_t)elements,
                        scale_bits, (uint32_t)destination_offset};

    /* Decode each packed FP8 pair once before the tensor-core projection */
    return vk_dispatch(VK_PIPE_FP8_TO_BF16, buffers, 2, push, 4,
                       ((uint32_t)((elements + 1u) / 2u) + 255u) / 256u,
                       1, 1);
}

static int vk_linear_bf16_dispatch(iris_gpu_tensor_impl_t *out,
                                    iris_gpu_tensor_impl_t *x,
                                    iris_gpu_tensor_impl_t *weight,
                                    int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight, out};
    vk_pipeline_id_t pipeline = vk_ctx.cooperative_matrix &&
        (seq_len & 15) == 0 && (in_dim & 15) == 0 && (out_dim & 15) == 0
        ? VK_PIPE_LINEAR_BF16_COOP : VK_PIPE_LINEAR_BF16;
    int watchdog_packets = seq_len >= 2048;
    int row_chunk = vk_friendly_requested
                    ? (pipeline == VK_PIPE_LINEAR_BF16_COOP ? 512 : 64) :
                    (watchdog_packets ? 2048 : 1024);

    /* Split large projections into bounded submissions for the Windows watchdog */
    for (int row_offset = 0; row_offset < seq_len;) {
        int row_count = seq_len - row_offset;
        if (row_count > row_chunk) row_count = row_chunk;
        uint32_t push[4] = {(uint32_t)seq_len, (uint32_t)in_dim,
                            (uint32_t)out_dim, (uint32_t)row_offset};
        uint32_t rows_per_group = pipeline == VK_PIPE_LINEAR_BF16_COOP
            ? 64u : 16u;
        uint32_t columns_per_group = 16u;
        if (!vk_dispatch(pipeline, buffers, 3, push, 4,
                         ((uint32_t)out_dim + columns_per_group - 1u) /
                             columns_per_group,
                         ((uint32_t)row_count + rows_per_group - 1u) /
                             rows_per_group, 1))
            return 0;
        /* Fence high-resolution chunks before their combined time triggers TDR */
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        row_offset += row_count;
        if (!watchdog_packets)
            row_chunk = vk_friendly_next_chunk(row_chunk, 1024, 16);
    }
    return 1;
}

int iris_gpu_linear_bf16_into_key(iris_gpu_tensor_t out_tensor,
                                  iris_gpu_tensor_t x_tensor,
                                  const void *cache_key,
                                  const uint16_t *weights,
                                  int seq_len, int in_dim, int out_dim);

static int vk_linear_fp8_dispatch(iris_gpu_tensor_impl_t *out,
                                  iris_gpu_tensor_impl_t *x,
                                  iris_gpu_tensor_impl_t *weight,
                                  size_t weight_offset, float weight_scale,
                                  int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight, out};
    int watchdog_packets = seq_len >= 2048;
    int row_chunk = vk_friendly_requested ? 64 :
                    (watchdog_packets ? 2048 : 1024);
    uint32_t scale_bits;
    memcpy(&scale_bits, &weight_scale, sizeof(scale_bits));
    vk_pipeline_id_t pipeline = VK_PIPE_LINEAR_FP8;

    /* Use range-safe BF16 tensor-core tiles when every matrix edge is complete */
    if (vk_ctx.cooperative_matrix && (seq_len & 15) == 0 &&
        (in_dim & 15) == 0 && (out_dim & 15) == 0)
        pipeline = VK_PIPE_LINEAR_FP8_COOP;

    /* Split large projections into bounded submissions for the Windows watchdog */
    for (int row_offset = 0; row_offset < seq_len;) {
        int row_count = seq_len - row_offset;
        if (row_count > row_chunk) row_count = row_chunk;
        uint32_t push[6] = {(uint32_t)seq_len, (uint32_t)in_dim,
                            (uint32_t)out_dim, (uint32_t)row_offset,
                            (uint32_t)weight_offset, scale_bits};
        if (!vk_dispatch(pipeline, buffers, 3, push, 6,
                         ((uint32_t)out_dim + 15u) / 16u,
                         ((uint32_t)row_count + 15u) / 16u, 1))
            return 0;
        /* Fence high-resolution chunks before their combined time triggers TDR */
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        row_offset += row_count;
        if (!watchdog_packets)
            row_chunk = vk_friendly_next_chunk(row_chunk, 1024, 16);
    }
    return 1;
}

static int vk_load_spirv(const char *path, uint32_t **words, size_t *word_count) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return 0; }
    long size = ftell(file);
    if (size <= 0 || (size % 4) != 0) { fclose(file); return 0; }
    rewind(file);
    uint32_t *data = malloc((size_t)size);
    if (!data) { fclose(file); return 0; }
    size_t read_count = fread(data, 1, (size_t)size, file);
    fclose(file);
    if (read_count != (size_t)size) { free(data); return 0; }
    *words = data;
    *word_count = (size_t)size / 4;
    return 1;
}

static int vk_create_pipeline(vk_pipeline_id_t id, const char *path) {
    uint32_t *words = NULL;
    size_t word_count = 0;
    if (!vk_load_spirv(path, &words, &word_count)) {
        fprintf(stderr, "Vulkan: missing shader %s\n", path);
        return 0;
    }
    VkShaderModuleCreateInfo module_info = {0};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = word_count * sizeof(uint32_t);
    module_info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(vk_ctx.device, &module_info, NULL, &module);
    free(words);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkCreateShaderModule");
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage = {0};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info = {0};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = vk_ctx.pipeline_layout;
    result = vkCreateComputePipelines(vk_ctx.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL,
                                      &vk_ctx.pipelines[id]);
    vkDestroyShaderModule(vk_ctx.device, module, NULL);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkCreateComputePipelines");
        return 0;
    }
    return 1;
}

static int vk_init_context(void) {
    VkApplicationInfo application = {0};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "iris";
    application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    application.pEngineName = "iris";
    application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info = {0};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    VkResult result = vkCreateInstance(&instance_info, NULL, &vk_ctx.instance);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateInstance"); return 0; }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk_ctx.instance, &device_count, NULL);
    if (device_count == 0) return 0;
    VkPhysicalDevice *devices = calloc(device_count, sizeof(*devices));
    if (!devices) return 0;
    vkEnumeratePhysicalDevices(vk_ctx.instance, &device_count, devices);
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            vk_ctx.physical_device = devices[i];
            break;
        }
    }
    if (!vk_ctx.physical_device) vk_ctx.physical_device = devices[0];
    free(devices);

    /* Retain at most three quarters of device-local memory for immutable FP8 weights */
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkDeviceSize largest_device_heap = 0;
    vkGetPhysicalDeviceMemoryProperties(vk_ctx.physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
        if ((memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            memory_properties.memoryHeaps[i].size > largest_device_heap)
            largest_device_heap = memory_properties.memoryHeaps[i].size;
    }
    vk_ctx.fp8_cache_limit = (size_t)(largest_device_heap - largest_device_heap / 4u);

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx.physical_device, &queue_count, NULL);
    VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
    if (!queues) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx.physical_device, &queue_count, queues);
    /* Prefer a compute-only queue so desktop graphics uses a different engine */
    int found_compute = 0;
    for (uint32_t i = 0; i < queue_count; i++) {
        if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            vk_ctx.queue_family = i;
            vk_ctx.dedicated_compute_queue = 1;
            found_compute = 1;
            break;
        }
    }
    if (!found_compute) {
        for (uint32_t i = 0; i < queue_count; i++) {
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                vk_ctx.queue_family = i;
                found_compute = 1;
                break;
            }
        }
    }
    free(queues);

    /* Detect optional scheduling and cooperative-matrix extensions */
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(vk_ctx.physical_device, NULL,
                                         &extension_count, NULL);
    VkExtensionProperties *extensions = calloc(extension_count, sizeof(*extensions));
    int has_global_priority = 0;
    int has_cooperative_matrix = 0;
    int has_shader_bfloat16 = 0;
    if (extensions) {
        vkEnumerateDeviceExtensionProperties(vk_ctx.physical_device, NULL,
                                             &extension_count, extensions);
        for (uint32_t i = 0; i < extension_count; i++) {
            if (strcmp(extensions[i].extensionName,
                       VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0)
                has_global_priority = 1;
            if (strcmp(extensions[i].extensionName,
                       VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0)
                has_cooperative_matrix = 1;
            if (strcmp(extensions[i].extensionName,
                       VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) == 0)
                has_shader_bfloat16 = 1;
        }
        free(extensions);
    }

    /* Verify the subgroup width used by the fused attention shader */
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {0};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2 = {0};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup_properties;
    vkGetPhysicalDeviceProperties2(vk_ctx.physical_device, &properties2);
    vk_ctx.subgroup_attention =
        subgroup_properties.subgroupSize == 32u &&
        (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (subgroup_properties.supportedOperations &
         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);

    /* Query the shader features and exact BF16-to-FP32 matrix tile */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_features = {0};
    cooperative_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceVulkan12Features vulkan12_features = {0};
    vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features = {0};
    bfloat16_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    cooperative_features.pNext = has_shader_bfloat16 ? &bfloat16_features : NULL;
    vulkan12_features.pNext = has_cooperative_matrix ? &cooperative_features : NULL;
    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vulkan12_features;
    vkGetPhysicalDeviceFeatures2(vk_ctx.physical_device, &features2);
    if (has_cooperative_matrix && has_shader_bfloat16 &&
        cooperative_features.cooperativeMatrix &&
        bfloat16_features.shaderBFloat16Type &&
        bfloat16_features.shaderBFloat16CooperativeMatrix &&
        vulkan12_features.vulkanMemoryModel) {
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_matrix_properties =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(vk_ctx.instance,
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t matrix_count = 0;
        if (get_matrix_properties &&
            get_matrix_properties(vk_ctx.physical_device, &matrix_count, NULL) == VK_SUCCESS) {
            VkCooperativeMatrixPropertiesKHR *matrix_properties =
                calloc(matrix_count, sizeof(*matrix_properties));
            if (matrix_properties) {
                for (uint32_t i = 0; i < matrix_count; i++)
                    matrix_properties[i].sType =
                        VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                if (get_matrix_properties(vk_ctx.physical_device, &matrix_count,
                                          matrix_properties) == VK_SUCCESS) {
                    for (uint32_t i = 0; i < matrix_count; i++) {
                        VkCooperativeMatrixPropertiesKHR *entry = &matrix_properties[i];
                        if (entry->MSize == 16u && entry->NSize == 16u &&
                            entry->KSize == 16u &&
                            entry->AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            entry->BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            entry->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            entry->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            entry->scope == VK_SCOPE_SUBGROUP_KHR) {
                            vk_ctx.cooperative_matrix = 1;
                            break;
                        }
                    }
                }
                free(matrix_properties);
            }
        }
    }

    float priority = vk_friendly_requested ? 0.25f : 1.0f;
    VkDeviceQueueGlobalPriorityCreateInfo global_priority = {0};
    global_priority.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO;
    global_priority.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW;
    VkDeviceQueueCreateInfo queue_info = {0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk_ctx.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    if (vk_friendly_requested && has_global_priority)
        queue_info.pNext = &global_priority;
    VkDeviceCreateInfo device_info = {0};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    const char *device_extensions[3];
    uint32_t enabled_extension_count = 0;

    /* Enable only the optional extensions used by the selected fast paths */
    if (vk_ctx.cooperative_matrix)
        device_extensions[enabled_extension_count++] =
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    if (vk_ctx.cooperative_matrix)
        device_extensions[enabled_extension_count++] =
            VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
    if (vk_friendly_requested && has_global_priority)
        device_extensions[enabled_extension_count++] =
            VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
    device_info.enabledExtensionCount = enabled_extension_count;
    device_info.ppEnabledExtensionNames = device_extensions;

    /* Enable the matrix and SPIR-V memory-model features required by the shader */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cooperative = {0};
    VkPhysicalDeviceShaderBfloat16FeaturesKHR enabled_bfloat16 = {0};
    VkPhysicalDeviceVulkan12Features enabled_vulkan12 = {0};
    if (vk_ctx.cooperative_matrix) {
        enabled_cooperative.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        enabled_cooperative.cooperativeMatrix = VK_TRUE;
        enabled_bfloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        enabled_bfloat16.shaderBFloat16Type = VK_TRUE;
        enabled_bfloat16.shaderBFloat16CooperativeMatrix = VK_TRUE;
        enabled_cooperative.pNext = &enabled_bfloat16;
        enabled_vulkan12.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        enabled_vulkan12.pNext = &enabled_cooperative;
        enabled_vulkan12.vulkanMemoryModel = VK_TRUE;
        device_info.pNext = &enabled_vulkan12;
    }
    result = vkCreateDevice(vk_ctx.physical_device, &device_info, NULL, &vk_ctx.device);
    if (result != VK_SUCCESS && queue_info.pNext) {
        /* Retry without global priority when policy or the driver refuses it */
        queue_info.pNext = NULL;
        if (enabled_extension_count &&
            strcmp(device_extensions[enabled_extension_count - 1],
                   VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0)
            enabled_extension_count--;
        device_info.enabledExtensionCount = enabled_extension_count;
        result = vkCreateDevice(vk_ctx.physical_device, &device_info, NULL, &vk_ctx.device);
    } else if (result == VK_SUCCESS && queue_info.pNext) {
        vk_ctx.global_low_priority = 1;
    }
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateDevice"); return 0; }
    if (vk_friendly_requested) {
        fprintf(stderr, "Vulkan friendly mode: %s queue, %s priority\n",
                vk_ctx.dedicated_compute_queue ? "dedicated compute" : "shared graphics",
                vk_ctx.global_low_priority ? "global low" : "local low");
        fprintf(stderr, "Vulkan friendly FP8 cache limit: %.2f GiB\n",
                (double)vk_ctx.fp8_cache_limit / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "Vulkan friendly activations: device-local VRAM\n");
    }
    fprintf(stderr, "Vulkan kernels: %s attention, %s FP8 linear\n",
            vk_ctx.subgroup_attention ? "subgroup-fused" : "portable",
            vk_ctx.cooperative_matrix
                ? "BF16 cooperative-matrix" : "stable FP32 scalar");
    vkGetDeviceQueue(vk_ctx.device, vk_ctx.queue_family, 0, &vk_ctx.queue);

    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk_ctx.queue_family;
    result = vkCreateCommandPool(vk_ctx.device, &pool_info, NULL, &vk_ctx.command_pool);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateCommandPool"); return 0; }

    VkCommandBufferAllocateInfo command_info = {0};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = vk_ctx.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(vk_ctx.device, &command_info, &vk_ctx.command_buffer);
    if (result != VK_SUCCESS) { vk_report(result, "vkAllocateCommandBuffers"); return 0; }

    VkFenceCreateInfo fence_info = {0};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(vk_ctx.device, &fence_info, NULL, &vk_ctx.fence);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateFence"); return 0; }

    VkDescriptorSetLayoutBinding bindings[8];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 8; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 8;
    layout_info.pBindings = bindings;
    result = vkCreateDescriptorSetLayout(vk_ctx.device, &layout_info, NULL,
                                         &vk_ctx.descriptor_layout);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateDescriptorSetLayout"); return 0; }

    VkDescriptorPoolSize pool_size = {0};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 8 * 4096;
    VkDescriptorPoolCreateInfo descriptor_pool_info = {0};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 4096;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    result = vkCreateDescriptorPool(vk_ctx.device, &descriptor_pool_info, NULL,
                                    &vk_ctx.descriptor_pool);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateDescriptorPool"); return 0; }

    VkPushConstantRange push_range = {0};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = 64;
    VkPipelineLayoutCreateInfo pipeline_layout_info = {0};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk_ctx.descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    result = vkCreatePipelineLayout(vk_ctx.device, &pipeline_layout_info, NULL,
                                     &vk_ctx.pipeline_layout);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreatePipelineLayout"); return 0; }

    if (!vk_create_buffer(4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          &vk_ctx.dummy_buffer, &vk_ctx.dummy_memory,
                          &vk_ctx.dummy_host_visible)) return 0;
    const char *shader_paths[VK_PIPE_COUNT] = {
        "shaders/iris_vulkan_linear.comp.spv",
        "shaders/iris_vulkan_linear_bf16.comp.spv",
        "shaders/iris_vulkan_linear_bf16_coop.comp.spv",
        "shaders/iris_vulkan_linear_fp8.comp.spv",
        "shaders/iris_vulkan_linear_fp8_coop.comp.spv",
        "shaders/iris_vulkan_fp8_to_bf16.comp.spv",
        "shaders/iris_vulkan_rms_norm.comp.spv",
        "shaders/iris_vulkan_qk_rms_norm.comp.spv",
        "shaders/iris_vulkan_rope_pair.comp.spv",
        "shaders/iris_vulkan_attention.comp.spv",
        "shaders/iris_vulkan_attention_subgroup.comp.spv",
        "shaders/iris_vulkan_silu_mul.comp.spv",
        "shaders/iris_vulkan_add.comp.spv",
        "shaders/iris_vulkan_gated_add.comp.spv",
        "shaders/iris_vulkan_adaln_norm.comp.spv",
        "shaders/iris_vulkan_group_norm.comp.spv",
        "shaders/iris_vulkan_swish.comp.spv",
        "shaders/iris_vulkan_upsample.comp.spv",
        "shaders/iris_vulkan_conv2d.comp.spv",
        "shaders/iris_vulkan_conv2d_coop.comp.spv",
        "shaders/iris_vulkan_f32_to_bf16.comp.spv",
        "shaders/iris_vulkan_bf16_to_f32.comp.spv",
        "shaders/iris_vulkan_linear_bf16_native.comp.spv",
        "shaders/iris_vulkan_linear_bf16_native_qkv.comp.spv",
        "shaders/iris_vulkan_linear_bf16_native_gate_up.comp.spv",
        "shaders/iris_vulkan_rms_norm_bf16.comp.spv",
        "shaders/iris_vulkan_rms_norm_bf16_f32_weight.comp.spv",
        "shaders/iris_vulkan_head_rms_norm_bf16.comp.spv",
        "shaders/iris_vulkan_qk_rms_norm_bf16_f32_weight.comp.spv",
        "shaders/iris_vulkan_elementwise_bf16.comp.spv",
        "shaders/iris_vulkan_gated_add_bf16_f32_gate.comp.spv",
        "shaders/iris_vulkan_rope_pair_bf16.comp.spv",
        "shaders/iris_vulkan_attention_subgroup_bf16.comp.spv",
        "shaders/iris_vulkan_rope_text_bf16.comp.spv",
        "shaders/iris_vulkan_causal_attention_bf16.comp.spv"
    };
    for (int i = 0; i < VK_PIPE_COUNT; i++) {
        /* Skip optional pipelines that the selected device cannot execute */
        if (i == VK_PIPE_LINEAR_BF16_COOP && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_LINEAR_FP8_COOP && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_LINEAR_BF16_NATIVE && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_LINEAR_BF16_NATIVE_QKV && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_LINEAR_BF16_NATIVE_GATE_UP && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_CONV2D_COOP && !vk_ctx.cooperative_matrix) continue;
        if (i == VK_PIPE_ATTENTION_SUBGROUP && !vk_ctx.subgroup_attention) continue;
        if (i == VK_PIPE_ATTENTION_SUBGROUP_BF16 && !vk_ctx.subgroup_attention) continue;
        if (!vk_create_pipeline((vk_pipeline_id_t)i, shader_paths[i])) return 0;
    }
    vk_ctx.initialized = 1;
    return 1;
}

int iris_metal_init(void) {
    if (vk_ctx.initialized) return 1;
    /* Do not retry a context whose device has already been lost */
    if (vk_ctx.device_lost) return 0;
    memset(&vk_ctx, 0, sizeof(vk_ctx));
    if (!vk_init_context()) {
        iris_metal_cleanup();
        return 0;
    }
    return 1;
}

int iris_metal_available(void) {
    return iris_metal_init();
}

int iris_metal_shaders_available(void) {
    return iris_metal_init();
}

int iris_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!bf16_weights || num_elements == 0 || !vk_ready()) return 0;
    /* Upload immutable BF16 weights before their CPU copies are released */
    return vk_cached(bf16_weights, num_elements * sizeof(uint16_t), 1) != NULL;
}

int iris_metal_warmup_bf16_key(const void *cache_key,
                               const uint16_t *bf16_weights,
                               size_t num_elements) {
    /* Upload a temporary BF16 buffer under a stable caller-owned key */
    if (!cache_key || !bf16_weights || num_elements == 0 || !vk_ready()) return 0;
    return vk_cached_key(cache_key, bf16_weights,
                         num_elements * sizeof(uint16_t), 1) != NULL;
}

int iris_metal_warmup_fp8(const uint8_t *weights, size_t num_elements) {
    /* Retain one immutable device-local copy for all projections using this payload */
    if (!weights || !num_elements || !vk_ready()) return 0;
    return vk_cached_fp8(weights, num_elements) != NULL;
}

size_t iris_metal_fp8_cache_used(void) {
    return vk_ctx.fp8_cache_bytes;
}

static void vk_clear_cache(void) {
    vk_cached_buffer_t *entry = vk_ctx.cache;
    while (entry) {
        vk_cached_buffer_t *next = entry->next;
        if (entry->tensor) {
            entry->tensor->cached = 0;
            vk_tensor_destroy(entry->tensor);
        }
        free(entry);
        entry = next;
    }
    vk_ctx.cache = NULL;
    vk_ctx.fp8_cache_bytes = 0;
}

static void vk_clear_cache_kind(int storage_kind) {
    /* Remove only cached payloads of the requested storage representation */
    vk_cached_buffer_t **link = &vk_ctx.cache;
    while (*link) {
        vk_cached_buffer_t *entry = *link;
        if (entry->is_f16 != storage_kind) {
            link = &entry->next;
            continue;
        }
        *link = entry->next;
        if (entry->tensor) {
            entry->tensor->cached = 0;
            vk_tensor_destroy(entry->tensor);
        }
        free(entry);
    }
}

void iris_metal_cleanup(void) {
    if (!vk_ctx.device && !vk_ctx.instance) return;
    int device_lost = vk_ctx.device_lost;
    /* Wait only while the device is known to be responsive */
    if (vk_ctx.device && !device_lost) vkDeviceWaitIdle(vk_ctx.device);
    vk_flush_deferred_tensors();
    vk_clear_cache();
    /* Release reusable streamed-weight storage while the device is responsive */
    vk_tensor_destroy(vk_ctx.stream_weight);
    vk_tensor_destroy(vk_ctx.stream_staging);
    vk_tensor_destroy(vk_ctx.decoded_weight);
    vk_ctx.stream_weight = NULL;
    vk_ctx.stream_staging = NULL;
    vk_ctx.decoded_weight = NULL;
    vk_ctx.stream_weight_bytes = 0;
    vk_ctx.decoded_weight_bytes = 0;
    vk_ctx.stream_source = NULL;
    vk_ctx.stream_source_bytes = 0;
    vk_ctx.stream_source_fp8 = 0;
    if (vk_ctx.device && !device_lost) {
        for (int i = 0; i < VK_PIPE_COUNT; i++)
            if (vk_ctx.pipelines[i]) vkDestroyPipeline(vk_ctx.device, vk_ctx.pipelines[i], NULL);
        if (vk_ctx.dummy_buffer) vkDestroyBuffer(vk_ctx.device, vk_ctx.dummy_buffer, NULL);
        if (vk_ctx.dummy_memory) vkFreeMemory(vk_ctx.device, vk_ctx.dummy_memory, NULL);
        if (vk_ctx.pipeline_layout) vkDestroyPipelineLayout(vk_ctx.device, vk_ctx.pipeline_layout, NULL);
        if (vk_ctx.descriptor_pool) vkDestroyDescriptorPool(vk_ctx.device, vk_ctx.descriptor_pool, NULL);
        if (vk_ctx.descriptor_layout) vkDestroyDescriptorSetLayout(vk_ctx.device, vk_ctx.descriptor_layout, NULL);
        if (vk_ctx.fence) vkDestroyFence(vk_ctx.device, vk_ctx.fence, NULL);
        if (vk_ctx.command_pool) vkDestroyCommandPool(vk_ctx.device, vk_ctx.command_pool, NULL);
        vkDestroyDevice(vk_ctx.device, NULL);
    }
    /* Avoid teardown calls through a device that the driver has lost */
    if (vk_ctx.instance && !device_lost) vkDestroyInstance(vk_ctx.instance, NULL);
    memset(&vk_ctx, 0, sizeof(vk_ctx));
}

void iris_metal_reset(void) { vk_clear_cache(); }
void iris_metal_rope_cache_begin(void) { }
void iris_metal_reset_transient(void) { }
void iris_metal_clear_weight_cache_only(void) { vk_clear_cache(); }
void iris_metal_clear_bf16_cache_only(void) { vk_clear_cache_kind(1); }
void iris_metal_clear_f16_cache_only(void) { }
void iris_metal_clear_activation_pool_only(void) { }

void iris_gpu_sync(void) {
    if (!vk_ready()) return;
    /* Submit recorded work before waiting for all queue activity */
    if (vk_ctx.batch_active) {
        vk_ctx.batch_active = 0;
        vk_submit_command_buffer();
        return;
    }
    vkDeviceWaitIdle(vk_ctx.device);
}

void iris_gpu_batch_begin(void) {
    if (!vk_ready() || vk_ctx.batch_active) return;
    vkResetCommandPool(vk_ctx.device, vk_ctx.command_pool, 0);
    if (vk_begin_command_buffer()) vk_ctx.batch_active = 1;
}

void iris_gpu_batch_end(void) {
    if (!vk_ctx.batch_active) return;
    vk_ctx.batch_active = 0;
    /* Drop pending work if the device failed before submission */
    if (vk_ctx.device_lost) {
        vk_flush_deferred_tensors();
        return;
    }
    vk_submit_command_buffer();
}

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements) {
    if (!data || !vk_ready()) return NULL;
    size_t bytes = num_elements * sizeof(float);

    /* Create activations in device-local memory to avoid PCIe-backed compute */
    iris_gpu_tensor_impl_t *tensor = vk_tensor_alloc_device_local_bytes(bytes, 0, 0);
    iris_gpu_tensor_impl_t *staging = vk_tensor_alloc_bytes(bytes, 0, 0);
    if (!tensor || !staging) {
        vk_tensor_destroy(tensor);
        vk_tensor_destroy(staging);
        return NULL;
    }
    void *mapped = NULL;
    if (!vk_tensor_map(staging, &mapped)) {
        vk_tensor_destroy(tensor);
        vk_tensor_destroy(staging);
        return NULL;
    }
    vk_host_copy(mapped, data, bytes);
    vk_tensor_unmap(staging);
    if (!vk_copy_region(tensor, 0, staging, 0, bytes)) {
        vk_tensor_destroy(tensor);
        vk_tensor_destroy_deferred(staging);
        return NULL;
    }
    vk_tensor_destroy_deferred(staging);
    return tensor;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements) {
    if (!vk_ready()) return NULL;
    /* Keep uninitialized floating-point activations in device-local memory */
    return vk_tensor_alloc_device_local_bytes(num_elements * sizeof(float), 0, 0);
}

iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements) {
    if (!vk_ready()) return NULL;
    /* Keep uninitialized BF16 activations in device-local memory */
    return vk_tensor_alloc_device_local_bytes(num_elements * sizeof(uint16_t), 1, 0);
}

void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t tensor, int persistent) {
    if (tensor) ((iris_gpu_tensor_impl_t *)tensor)->persistent = persistent;
}

void iris_gpu_tensor_free(iris_gpu_tensor_t tensor) {
    vk_tensor_destroy_deferred((iris_gpu_tensor_impl_t *)tensor);
}

size_t iris_gpu_tensor_size(iris_gpu_tensor_t tensor) {
    return tensor ? ((iris_gpu_tensor_impl_t *)tensor)->elements : 0;
}

int iris_gpu_tensor_is_f16(iris_gpu_tensor_t tensor) {
    return tensor ? ((iris_gpu_tensor_impl_t *)tensor)->is_f16 : 0;
}

void iris_gpu_tensor_write(iris_gpu_tensor_t tensor, const float *data) {
    iris_gpu_tensor_impl_t *dst = (iris_gpu_tensor_impl_t *)tensor;
    if (!dst || !data || !vk_ready()) return;
    size_t bytes = dst->bytes;
    iris_gpu_tensor_impl_t *target = dst;
    iris_gpu_tensor_impl_t *staging = NULL;

    /* Stage CPU writes when the destination lives only in device memory */
    if (!dst->host_visible) {
        staging = vk_tensor_alloc_bytes(bytes, dst->is_f16, 0);
        if (!staging) return;
        target = staging;
    }
    void *mapped = NULL;
    if (!vk_tensor_map(target, &mapped)) {
        vk_tensor_destroy(staging);
        return;
    }
    if (dst->is_f16) {
        uint16_t *out = mapped;
        vk_host_convert_bf16(out, data, dst->elements);
    } else {
        vk_host_copy(mapped, data, dst->elements * sizeof(float));
    }
    vk_tensor_unmap(target);
    if (staging) {
        vk_copy_region(dst, 0, staging, 0, bytes);
        vk_tensor_destroy_deferred(staging);
    }
}

void iris_gpu_tensor_read(iris_gpu_tensor_t tensor, float *out) {
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)tensor;
    if (!src || !out || !vk_ready()) return;
    iris_gpu_sync();
    iris_gpu_tensor_impl_t *readable = src;
    iris_gpu_tensor_impl_t *staging = NULL;

    /* Stage device-local activations before CPU conversion or copying */
    if (!src->host_visible) {
        staging = vk_tensor_alloc_bytes(src->bytes, src->is_f16, 0);
        if (!staging || !vk_copy_region(staging, 0, src, 0, src->bytes)) {
            vk_tensor_destroy(staging);
            return;
        }
        readable = staging;
    }
    void *mapped = NULL;
    if (!vk_tensor_map(readable, &mapped)) {
        vk_tensor_destroy(staging);
        return;
    }
    if (src->is_f16) {
        const uint16_t *input = mapped;
        for (size_t i = 0; i < src->elements; i++) out[i] = vk_bf16_to_float(input[i]);
    } else {
        vk_host_copy(out, mapped, src->elements * sizeof(float));
    }
    vk_tensor_unmap(readable);
    vk_tensor_destroy(staging);
}

float *iris_gpu_tensor_data(iris_gpu_tensor_t tensor) {
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)tensor;
    if (!src || src->is_f16 || !src->host_visible || !vk_ready()) return NULL;
    iris_gpu_sync();
    void *mapped = NULL;
    if (!vk_tensor_map(src, &mapped)) return NULL;
    return mapped;
}

iris_gpu_tensor_t iris_gpu_linear(iris_gpu_tensor_t x_tensor,
                                  const float *weights, const float *bias,
                                  int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    if (!x || !weights || !vk_ready()) return NULL;
    /* Keep projection output on the device for downstream kernels */
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_device_local_bytes(
        (size_t)seq_len * out_dim * sizeof(float), 0, 0);
    iris_gpu_tensor_impl_t *weight = vk_cached(weights,
        (size_t)in_dim * out_dim * sizeof(float), 0);
    iris_gpu_tensor_impl_t *bias_buffer = bias
        ? vk_cached(bias, (size_t)out_dim * sizeof(float), 0) : NULL;
    if (!out || !weight || (bias && !bias_buffer)) {
        vk_tensor_destroy(out);
        return NULL;
    }
    iris_gpu_tensor_impl_t *buffers[4] = {x, weight, bias_buffer, out};
    uint32_t push[4] = {(uint32_t)seq_len, (uint32_t)in_dim,
                        (uint32_t)out_dim, bias ? 1u : 0u};
    if (!vk_dispatch(VK_PIPE_LINEAR, buffers, 4, push, 4,
                     ((uint32_t)out_dim + 15u) / 16u,
                     ((uint32_t)seq_len + 15u) / 16u, 1)) {
        vk_tensor_destroy(out);
        return NULL;
    }
    return out;
}

int iris_gpu_linear_bf16_into(iris_gpu_tensor_t out_tensor,
                              iris_gpu_tensor_t x_tensor,
                              const uint16_t *weights,
                              int seq_len, int in_dim, int out_dim) {
    return iris_gpu_linear_bf16_into_key(out_tensor, x_tensor, weights, weights,
                                         seq_len, in_dim, out_dim);
}

int iris_gpu_linear_bf16_into_key(iris_gpu_tensor_t out_tensor,
                                  iris_gpu_tensor_t x_tensor,
                                  const void *cache_key,
                                  const uint16_t *weights,
                                  int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    if (!out || !x || !cache_key || !vk_ready() || out->is_f16 || x->is_f16) return 0;
    iris_gpu_tensor_impl_t *weight = vk_cached_key(cache_key, weights,
        (size_t)in_dim * out_dim * sizeof(uint16_t), 1);
    if (!weight) return 0;
    return vk_linear_bf16_dispatch(out, x, weight, seq_len, in_dim, out_dim);
}

int iris_gpu_linear_f32_stream_into(iris_gpu_tensor_t out_tensor,
                                    iris_gpu_tensor_t x_tensor,
                                    const float *weights,
                                    int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t elements = (size_t)in_dim * out_dim;
    if (!out || !x || !weights || !vk_ready() || out->is_f16 || x->is_f16)
        return 0;

    /* Convert and upload only the matrix needed by this projection */
    if (!vk_stream_weight_upload(weights, elements)) return 0;
    return vk_linear_bf16_dispatch(out, x, vk_ctx.stream_weight,
                                   seq_len, in_dim, out_dim);
}

int iris_gpu_linear_fp8_stream_into(iris_gpu_tensor_t out_tensor,
                                    iris_gpu_tensor_t x_tensor,
                                    const uint8_t *weights,
                                    size_t weight_elements,
                                    size_t weight_offset,
                                    float weight_scale,
                                    int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t matrix_elements = (size_t)in_dim * out_dim;
    if (!out || !x || !weights || !vk_ready() || out->is_f16 || x->is_f16 ||
        weight_offset > weight_elements || matrix_elements > weight_elements - weight_offset)
        return 0;

    /* Prefer the persistent cache and stream only when the budget rejected this payload */
    iris_gpu_tensor_impl_t *weight = vk_cached_find(weights, weight_elements, 2);
    if (!weight) {
        if (!vk_stream_fp8_upload(weights, weight_elements)) return 0;
        weight = vk_ctx.stream_weight;
    }

    /* Decode once per projection on GPUs whose matrix hardware consumes BF16 */
    if (vk_ctx.cooperative_matrix && (seq_len & 15) == 0 &&
        (in_dim & 15) == 0 && (out_dim & 15) == 0) {
        if (!vk_decode_fp8_weight(weight, weight_offset, weight_scale,
                                  0, matrix_elements, matrix_elements)) return 0;
        return vk_linear_bf16_dispatch(out, x, vk_ctx.decoded_weight,
                                       seq_len, in_dim, out_dim);
    }
    return vk_linear_fp8_dispatch(out, x, weight,
                                  weight_offset, weight_scale,
                                  seq_len, in_dim, out_dim);
}

static iris_gpu_tensor_impl_t *vk_fp8_source(const uint8_t *weights,
                                              size_t weight_elements) {
    iris_gpu_tensor_impl_t *source = vk_cached_find(weights, weight_elements, 2);
    if (source) return source;

    /* Stream an uncached FP8 payload through the reusable upload buffer */
    if (!vk_stream_fp8_upload(weights, weight_elements)) return NULL;
    return vk_ctx.stream_weight;
}

static int vk_linear_bf16_native_dispatch(iris_gpu_tensor_impl_t *out,
                                           iris_gpu_tensor_impl_t *x,
                                           iris_gpu_tensor_impl_t *weight,
                                           int seq_len, int in_dim,
                                           int out_dim) {
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight, out};
    int watchdog_packets = seq_len >= 2048;
    int row_chunk = vk_friendly_requested ? 512 :
                    (watchdog_packets ? 2048 : 1024);

    /* Split native BF16 projections into watchdog-safe row ranges */
    for (int row_offset = 0; row_offset < seq_len;) {
        int row_count = seq_len - row_offset;
        if (row_count > row_chunk) row_count = row_chunk;
        uint32_t push[4] = {(uint32_t)seq_len, (uint32_t)in_dim,
                            (uint32_t)out_dim, (uint32_t)row_offset};
        if (!vk_dispatch(VK_PIPE_LINEAR_BF16_NATIVE, buffers, 3, push, 4,
                         ((uint32_t)out_dim + 15u) / 16u,
                         ((uint32_t)row_count + 63u) / 64u, 1))
            return 0;
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        row_offset += row_count;
        if (!watchdog_packets)
            row_chunk = vk_friendly_next_chunk(row_chunk, 1024, 16);
    }
    return 1;
}

int iris_gpu_linear_fp8_bf16_into(iris_gpu_tensor_t out_tensor,
                                  iris_gpu_tensor_t x_tensor,
                                  const uint8_t *weights,
                                  size_t weight_elements,
                                  size_t weight_offset,
                                  float weight_scale,
                                  int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t matrix_elements = (size_t)in_dim * (size_t)out_dim;
    if (!out || !x || !weights || !out->is_f16 || !x->is_f16 ||
        !vk_ctx.cooperative_matrix || (seq_len & 15) != 0 ||
        (in_dim & 15) != 0 || (out_dim & 15) != 0 ||
        weight_offset > weight_elements ||
        matrix_elements > weight_elements - weight_offset)
        return 0;

    /* Decode the FP8 matrix once and keep activations BF16 throughout */
    iris_gpu_tensor_impl_t *source = vk_fp8_source(weights, weight_elements);
    if (!source || !vk_decode_fp8_weight(source, weight_offset, weight_scale,
                                          0, matrix_elements, matrix_elements))
        return 0;
    return vk_linear_bf16_native_dispatch(out, x, vk_ctx.decoded_weight,
                                           seq_len, in_dim, out_dim);
}

int iris_gpu_linear_fp8_qkv_bf16_into(iris_gpu_tensor_t q_tensor,
                                      iris_gpu_tensor_t k_tensor,
                                      iris_gpu_tensor_t v_tensor,
                                      iris_gpu_tensor_t x_tensor,
                                      const uint8_t *weights,
                                      size_t weight_elements,
                                      size_t weight_offset,
                                      float weight_scale,
                                      int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *v = (iris_gpu_tensor_impl_t *)v_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t matrix_elements = (size_t)in_dim * (size_t)out_dim;
    size_t total_elements = matrix_elements * 3u;
    if (!q || !k || !v || !x || !weights || !q->is_f16 || !k->is_f16 ||
        !v->is_f16 || !x->is_f16 || !vk_ctx.cooperative_matrix ||
        (seq_len & 15) != 0 || (in_dim & 15) != 0 || (out_dim & 15) != 0 ||
        weight_offset > weight_elements ||
        total_elements > weight_elements - weight_offset)
        return 0;

    /* Decode contiguous QKV matrices into one reusable BF16 weight slab */
    iris_gpu_tensor_impl_t *source = vk_fp8_source(weights, weight_elements);
    if (!source) return 0;
    for (size_t projection = 0; projection < 3u; projection++) {
        size_t offset = projection * matrix_elements;
        if (!vk_decode_fp8_weight(source, weight_offset + offset, weight_scale,
                                  offset, matrix_elements, total_elements))
            return 0;
    }

    iris_gpu_tensor_impl_t *buffers[5] = {x, vk_ctx.decoded_weight, q, k, v};
    int watchdog_packets = seq_len >= 2048;
    int row_chunk = vk_friendly_requested ? 512 :
                    (watchdog_packets ? 2048 : 1024);
    /* Fuse three projections while sharing every BF16 activation tile */
    for (int row_offset = 0; row_offset < seq_len;) {
        int row_count = seq_len - row_offset;
        if (row_count > row_chunk) row_count = row_chunk;
        uint32_t push[4] = {(uint32_t)seq_len, (uint32_t)in_dim,
                            (uint32_t)out_dim, (uint32_t)row_offset};
        if (!vk_dispatch(VK_PIPE_LINEAR_BF16_NATIVE_QKV, buffers, 5, push, 4,
                         ((uint32_t)out_dim + 15u) / 16u,
                         ((uint32_t)row_count + 63u) / 64u, 1))
            return 0;
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        row_offset += row_count;
        if (!watchdog_packets)
            row_chunk = vk_friendly_next_chunk(row_chunk, 1024, 64);
    }
    return 1;
}

int iris_gpu_linear_fp8_gate_up_bf16_into(iris_gpu_tensor_t gate_tensor,
                                          iris_gpu_tensor_t up_tensor,
                                          iris_gpu_tensor_t x_tensor,
                                          const uint8_t *gate_weights,
                                          size_t gate_elements,
                                          size_t gate_offset,
                                          float gate_scale,
                                          const uint8_t *up_weights,
                                          size_t up_elements,
                                          size_t up_offset,
                                          float up_scale,
                                          int seq_len, int in_dim,
                                          int out_dim) {
    iris_gpu_tensor_impl_t *gate = (iris_gpu_tensor_impl_t *)gate_tensor;
    iris_gpu_tensor_impl_t *up = (iris_gpu_tensor_impl_t *)up_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t matrix_elements = (size_t)in_dim * (size_t)out_dim;
    size_t total_elements = matrix_elements * 2u;
    if (!gate || !up || !x || !gate_weights || !up_weights ||
        !gate->is_f16 || !up->is_f16 || !x->is_f16 ||
        !vk_ctx.cooperative_matrix || (seq_len & 15) != 0 ||
        (in_dim & 15) != 0 || (out_dim & 15) != 0 ||
        gate_offset > gate_elements || matrix_elements > gate_elements - gate_offset ||
        up_offset > up_elements || matrix_elements > up_elements - up_offset)
        return 0;

    /* Decode gate and up matrices side by side without retaining BF16 copies */
    iris_gpu_tensor_impl_t *gate_source = vk_fp8_source(gate_weights, gate_elements);
    if (!gate_source || !vk_decode_fp8_weight(gate_source, gate_offset, gate_scale,
                                               0, matrix_elements, total_elements))
        return 0;
    iris_gpu_tensor_impl_t *up_source = vk_cached_find(up_weights, up_elements, 2);
    if (!up_source) {
        if (gate_source == vk_ctx.stream_weight && !vk_submit_and_resume_batch())
            return 0;
        up_source = vk_fp8_source(up_weights, up_elements);
    }
    if (!up_source || !vk_decode_fp8_weight(up_source, up_offset, up_scale,
                                             matrix_elements, matrix_elements,
                                             total_elements))
        return 0;

    iris_gpu_tensor_impl_t *buffers[4] = {x, vk_ctx.decoded_weight, gate, up};
    int watchdog_packets = seq_len >= 2048;
    int row_chunk = vk_friendly_requested ? 512 :
                    (watchdog_packets ? 2048 : 1024);
    /* Fuse the SwiGLU input projections while sharing activation tiles */
    for (int row_offset = 0; row_offset < seq_len;) {
        int row_count = seq_len - row_offset;
        if (row_count > row_chunk) row_count = row_chunk;
        uint32_t push[4] = {(uint32_t)seq_len, (uint32_t)in_dim,
                            (uint32_t)out_dim, (uint32_t)row_offset};
        if (!vk_dispatch(VK_PIPE_LINEAR_BF16_NATIVE_GATE_UP, buffers, 4, push, 4,
                         ((uint32_t)out_dim + 15u) / 16u,
                         ((uint32_t)row_count + 63u) / 64u, 1))
            return 0;
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        row_offset += row_count;
        if (!watchdog_packets)
            row_chunk = vk_friendly_next_chunk(row_chunk, 1024, 64);
    }
    return 1;
}

iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x,
                                       const uint16_t *weights,
                                       int seq_len, int in_dim, int out_dim) {
    /* Keep projection output on the device for downstream kernels */
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_device_local_bytes(
        (size_t)seq_len * out_dim * sizeof(float), 0, 0);
    if (!out || !iris_gpu_linear_bf16_into(out, x, weights, seq_len, in_dim, out_dim)) {
        vk_tensor_destroy(out);
        return NULL;
    }
    return out;
}

void iris_gpu_rms_norm_f32(iris_gpu_tensor_t out_tensor, iris_gpu_tensor_t x_tensor,
                           const float *weight, int seq, int hidden, float eps) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *weight_buffer = vk_cached(weight,
        (size_t)hidden * sizeof(float), 0);
    if (!out || !x || !weight_buffer) return;
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight_buffer, out};
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)hidden, eps_bits};
    vk_dispatch(VK_PIPE_RMS_NORM, buffers, 3, push, 3, (uint32_t)seq, 1, 1);
}

void iris_gpu_qk_rms_norm(iris_gpu_tensor_t q_tensor, iris_gpu_tensor_t k_tensor,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *qw = vk_cached(q_weight, (size_t)head_dim * sizeof(float), 0);
    iris_gpu_tensor_impl_t *kw = vk_cached(k_weight, (size_t)head_dim * sizeof(float), 0);
    if (!q || !k || !qw || !kw) return;
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, qw, kw};
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[4] = {(uint32_t)(seq * heads), (uint32_t)head_dim,
                        eps_bits, (uint32_t)heads};
    vk_dispatch(VK_PIPE_QK_RMS_NORM, buffers, 4, push, 4,
                (uint32_t)(seq * heads), 1, 1);
}

void iris_gpu_rope_single_pair_f32(iris_gpu_tensor_t q_tensor,
                                   iris_gpu_tensor_t k_tensor,
                                   const float *cos_freq, const float *sin_freq,
                                   int seq, int heads, int head_dim) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    size_t table_bytes = (size_t)seq * head_dim * sizeof(float);
    iris_gpu_tensor_impl_t *cos_buffer = vk_cached(cos_freq, table_bytes, 0);
    iris_gpu_tensor_impl_t *sin_buffer = vk_cached(sin_freq, table_bytes, 0);
    if (!q || !k || !cos_buffer || !sin_buffer) return;
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, cos_buffer, sin_buffer};
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)heads, (uint32_t)head_dim};
    uint32_t pairs = (uint32_t)(seq * heads * (head_dim / 2));
    vk_dispatch(VK_PIPE_ROPE_PAIR, buffers, 4, push, 3,
                (pairs + 255u) / 256u, 1, 1);
}

int iris_gpu_attention_fused(iris_gpu_tensor_t out_tensor,
                             iris_gpu_tensor_t q_tensor,
                             iris_gpu_tensor_t k_tensor,
                             iris_gpu_tensor_t v_tensor,
                             int seq_q, int seq_k, int heads,
                             int head_dim, float scale) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *v = (iris_gpu_tensor_impl_t *)v_tensor;
    if (!out || !q || !k || !v || !vk_ready() || head_dim > 128) return 0;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, v, out};
    /* Split query tiles so each dispatch stays below the Windows watchdog */
    int watchdog_packets = seq_q >= 2048 || seq_k >= 2048;
    int query_chunk = vk_friendly_requested ? 32 :
                      (watchdog_packets ? 128 : 64);
    for (int query_offset = 0; query_offset < seq_q;) {
        int query_count = seq_q - query_offset;
        if (query_count > query_chunk) query_count = query_chunk;
        uint32_t push[6] = {(uint32_t)seq_q, (uint32_t)seq_k, (uint32_t)heads,
                            (uint32_t)head_dim, scale_bits, (uint32_t)query_offset};
        vk_pipeline_id_t pipeline = vk_ctx.subgroup_attention
            ? VK_PIPE_ATTENTION_SUBGROUP : VK_PIPE_ATTENTION;
        uint32_t query_groups = vk_ctx.subgroup_attention
            ? ((uint32_t)query_count + 7u) / 8u : (uint32_t)query_count;
        if (!vk_dispatch(pipeline, buffers, 4, push, 6,
                         query_groups * (uint32_t)heads, 1, 1)) return 0;
        /* Fence high-resolution query tiles before their combined time triggers TDR */
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        query_offset += query_count;
        if (!watchdog_packets)
            query_chunk = vk_friendly_next_chunk(query_chunk, 64, 4);
    }
    return 1;
}

int iris_gpu_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t q,
                            iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                            int seq_q, int seq_k, int heads, int head_dim,
                            float scale) {
    iris_gpu_tensor_impl_t *out_impl = (iris_gpu_tensor_impl_t *)out;
    iris_gpu_tensor_impl_t *q_impl = (iris_gpu_tensor_impl_t *)q;
    iris_gpu_tensor_impl_t *k_impl = (iris_gpu_tensor_impl_t *)k;
    iris_gpu_tensor_impl_t *v_impl = (iris_gpu_tensor_impl_t *)v;
    if (!out_impl || !q_impl || !k_impl || !v_impl ||
        !out_impl->is_f16 || !q_impl->is_f16 || !k_impl->is_f16 ||
        !v_impl->is_f16 || !vk_ctx.subgroup_attention || head_dim > 128)
        return 0;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    iris_gpu_tensor_impl_t *buffers[4] = {q_impl, k_impl, v_impl, out_impl};
    int watchdog_packets = seq_q >= 2048 || seq_k >= 2048;
    int query_chunk = vk_friendly_requested ? 32 :
                      (watchdog_packets ? 128 : 64);

    /* Execute BF16 attention in bounded query tiles with FP32 softmax state */
    for (int query_offset = 0; query_offset < seq_q;) {
        int query_count = seq_q - query_offset;
        if (query_count > query_chunk) query_count = query_chunk;
        uint32_t push[6] = {(uint32_t)seq_q, (uint32_t)seq_k,
                            (uint32_t)heads, (uint32_t)head_dim,
                            scale_bits, (uint32_t)query_offset};
        uint32_t query_groups = ((uint32_t)query_count + 7u) / 8u;
        if (!vk_dispatch(VK_PIPE_ATTENTION_SUBGROUP_BF16, buffers, 4,
                         push, 6, query_groups * (uint32_t)heads, 1, 1))
            return 0;
        if (watchdog_packets && !vk_submit_and_resume_batch()) return 0;
        query_offset += query_count;
        if (!watchdog_packets)
            query_chunk = vk_friendly_next_chunk(query_chunk, 64, 8);
    }
    return 1;
}

int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t q,
                                  iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                                  int seq_q, int seq_k, int heads, int head_dim,
                                  float scale) {
    return iris_gpu_attention_bf16(out, q, k, v, seq_q, seq_k,
                                   heads, head_dim, scale);
}

int iris_gpu_head_rms_norm_bf16(iris_gpu_tensor_t x_tensor,
                                iris_gpu_tensor_t weight_tensor,
                                int seq, int heads, int head_dim, float eps) {
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *weight = (iris_gpu_tensor_impl_t *)weight_tensor;
    if (!x || !weight || !x->is_f16 || !weight->is_f16 ||
        !vk_ctx.subgroup_attention) return 0;
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[4] = {(uint32_t)seq, (uint32_t)heads,
                        (uint32_t)head_dim, eps_bits};
    iris_gpu_tensor_impl_t *buffers[2] = {x, weight};
    /* Normalize each sequence-head pair entirely within one subgroup */
    return vk_dispatch(VK_PIPE_HEAD_RMS_NORM_BF16, buffers, 2, push, 4,
                       (uint32_t)(seq * heads), 1, 1);
}

void iris_gpu_rms_norm_bf16(iris_gpu_tensor_t out_tensor,
                            iris_gpu_tensor_t x_tensor,
                            iris_gpu_tensor_t weight_tensor,
                            int seq, int hidden, float eps) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *weight = (iris_gpu_tensor_impl_t *)weight_tensor;
    if (!out || !x || !weight || !out->is_f16 || !x->is_f16 ||
        !weight->is_f16 || !vk_ctx.subgroup_attention) return;
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)hidden, eps_bits};
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight, out};
    /* Normalize each hidden-state row with FP32 accumulation */
    vk_dispatch(VK_PIPE_RMS_NORM_BF16, buffers, 3, push, 3,
                (uint32_t)seq, 1, 1);
}

void iris_gpu_rms_norm_bf16_f32_weight(iris_gpu_tensor_t out_tensor,
                                       iris_gpu_tensor_t x_tensor,
                                       const float *weight,
                                       int seq, int hidden, float eps) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *weight_buffer = vk_cached(
        weight, (size_t)hidden * sizeof(float), 0);
    if (!out || !x || !weight_buffer || !out->is_f16 || !x->is_f16 ||
        !vk_ctx.subgroup_attention) return;
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)hidden, eps_bits};
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight_buffer, out};
    /* Retain small affine weights in FP32 while activations remain BF16 */
    vk_dispatch(VK_PIPE_RMS_NORM_BF16_F32_WEIGHT, buffers, 3, push, 3,
                (uint32_t)seq, 1, 1);
}

void iris_gpu_qk_rms_norm_bf16_f32_weight(iris_gpu_tensor_t q_tensor,
                                          iris_gpu_tensor_t k_tensor,
                                          const float *q_weight,
                                          const float *k_weight,
                                          int seq, int heads, int head_dim,
                                          float eps) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *qw = vk_cached(
        q_weight, (size_t)head_dim * sizeof(float), 0);
    iris_gpu_tensor_impl_t *kw = vk_cached(
        k_weight, (size_t)head_dim * sizeof(float), 0);
    if (!q || !k || !qw || !kw || !q->is_f16 || !k->is_f16 ||
        !vk_ctx.subgroup_attention) return;
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[4] = {(uint32_t)seq, (uint32_t)heads,
                        (uint32_t)head_dim, eps_bits};
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, qw, kw};
    /* Normalize Q and K concurrently with FP32 per-head weights */
    vk_dispatch(VK_PIPE_QK_RMS_NORM_BF16_F32_WEIGHT, buffers, 4,
                push, 4, (uint32_t)(seq * heads), 1, 1);
}

void iris_gpu_rope_pair_bf16(iris_gpu_tensor_t q_tensor,
                             iris_gpu_tensor_t k_tensor,
                             const float *cos_freq,
                             const float *sin_freq,
                             int seq, int heads, int head_dim) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    size_t table_bytes = (size_t)seq * (size_t)head_dim * sizeof(float);
    iris_gpu_tensor_impl_t *cos_buffer = vk_cached(cos_freq, table_bytes, 0);
    iris_gpu_tensor_impl_t *sin_buffer = vk_cached(sin_freq, table_bytes, 0);
    if (!q || !k || !q->is_f16 || !k->is_f16 ||
        !cos_buffer || !sin_buffer) return;
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)heads,
                        (uint32_t)head_dim};
    uint32_t pairs = (uint32_t)(seq * heads * (head_dim / 2));
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, cos_buffer, sin_buffer};
    /* Apply the preassembled three-axis RoPE table to packed BF16 Q and K */
    vk_dispatch(VK_PIPE_ROPE_PAIR_BF16, buffers, 4, push, 3,
                (pairs + 255u) / 256u, 1, 1);
}

void iris_gpu_gated_add_bf16_f32_gate(iris_gpu_tensor_t out_tensor,
                                      const float *gate,
                                      iris_gpu_tensor_t projection_tensor,
                                      int seq, int hidden) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *projection =
        (iris_gpu_tensor_impl_t *)projection_tensor;
    iris_gpu_tensor_impl_t *gate_buffer = vk_cached(
        gate, (size_t)hidden * sizeof(float), 0);
    if (!out || !projection || !gate_buffer || !out->is_f16 ||
        !projection->is_f16) return;
    uint32_t push[2] = {(uint32_t)seq, (uint32_t)hidden};
    uint32_t pairs = ((uint32_t)(seq * hidden) + 1u) / 2u;
    iris_gpu_tensor_impl_t *buffers[3] = {out, gate_buffer, projection};
    /* Apply FP32 modulation gates directly to packed BF16 residuals */
    vk_dispatch(VK_PIPE_GATED_ADD_BF16_F32_GATE, buffers, 3, push, 2,
                (pairs + 255u) / 256u, 1, 1);
}

static void vk_elementwise_bf16(iris_gpu_tensor_impl_t *out,
                                iris_gpu_tensor_impl_t *a,
                                iris_gpu_tensor_impl_t *b,
                                int n, uint32_t operation) {
    if (!out || !a || !b || !out->is_f16 || !a->is_f16 || !b->is_f16)
        return;
    iris_gpu_tensor_impl_t *buffers[3] = {a, b, out};
    uint32_t push[2] = {(uint32_t)n, operation};
    /* Process packed BF16 pairs without read-modify-write races */
    vk_dispatch(VK_PIPE_ELEMENTWISE_BF16, buffers, 3, push, 2,
                (((uint32_t)n + 1u) / 2u + 255u) / 256u, 1, 1);
}

void iris_gpu_add_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t a,
                       iris_gpu_tensor_t b, int n) {
    vk_elementwise_bf16((iris_gpu_tensor_impl_t *)out,
                        (iris_gpu_tensor_impl_t *)a,
                        (iris_gpu_tensor_impl_t *)b, n, 0u);
}

void iris_gpu_silu_mul_bf16(iris_gpu_tensor_t gate, iris_gpu_tensor_t up,
                            int n) {
    vk_elementwise_bf16((iris_gpu_tensor_impl_t *)gate,
                        (iris_gpu_tensor_impl_t *)gate,
                        (iris_gpu_tensor_impl_t *)up, n, 1u);
}

void iris_gpu_copy_bf16(iris_gpu_tensor_t dst_tensor,
                        iris_gpu_tensor_t src_tensor, size_t n) {
    iris_gpu_tensor_impl_t *dst = (iris_gpu_tensor_impl_t *)dst_tensor;
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)src_tensor;
    if (dst && src && dst->is_f16 && src->is_f16)
        vk_copy_region(dst, 0, src, 0, n * sizeof(uint16_t));
}

void iris_gpu_rope_text_bf16(iris_gpu_tensor_t q_tensor,
                             iris_gpu_tensor_t k_tensor,
                             const float *cos_cache, const float *sin_cache,
                             int seq, int num_q_heads, int num_kv_heads,
                             int head_dim) {
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    size_t frequency_bytes = (size_t)seq * (size_t)(head_dim / 2) * sizeof(float);
    iris_gpu_tensor_impl_t *cos_buffer = vk_cached(cos_cache, frequency_bytes, 0);
    iris_gpu_tensor_impl_t *sin_buffer = vk_cached(sin_cache, frequency_bytes, 0);
    if (!q || !k || !q->is_f16 || !k->is_f16 ||
        !cos_buffer || !sin_buffer) return;
    iris_gpu_tensor_impl_t *buffers[4] = {q, k, cos_buffer, sin_buffer};
    uint32_t push[4] = {(uint32_t)seq, (uint32_t)num_q_heads,
                        (uint32_t)num_kv_heads, (uint32_t)head_dim};
    uint32_t max_heads = (uint32_t)(num_q_heads > num_kv_heads
        ? num_q_heads : num_kv_heads);
    uint32_t pairs = (uint32_t)seq * max_heads *
                     (((uint32_t)head_dim / 2u + 1u) / 2u);
    /* Apply split-half Qwen RoPE directly to packed BF16 tensors */
    vk_dispatch(VK_PIPE_ROPE_TEXT_BF16, buffers, 4, push, 4,
                (pairs + 63u) / 64u, 1, 1);
}

int iris_gpu_causal_attention_bf16(iris_gpu_tensor_t out_tensor,
                                   iris_gpu_tensor_t q_tensor,
                                   iris_gpu_tensor_t k_tensor,
                                   iris_gpu_tensor_t v_tensor,
                                   const int *attention_mask,
                                   int seq, int num_q_heads,
                                   int num_kv_heads, int head_dim,
                                   float scale) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *v = (iris_gpu_tensor_impl_t *)v_tensor;
    iris_gpu_tensor_impl_t *mask = attention_mask
        ? vk_cached(attention_mask, (size_t)seq * sizeof(int), 0) : NULL;
    if (!out || !q || !k || !v || !out->is_f16 || !q->is_f16 ||
        !k->is_f16 || !v->is_f16 || !vk_ctx.subgroup_attention ||
        num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 ||
        head_dim > 128 || (attention_mask && !mask)) return 0;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    uint32_t push[6] = {(uint32_t)seq, (uint32_t)num_q_heads,
                        (uint32_t)num_kv_heads, (uint32_t)head_dim,
                        scale_bits, attention_mask ? 1u : 0u};
    iris_gpu_tensor_impl_t *buffers[5] = {q, k, v, out, mask};
    /* Dispatch fused causal GQA attention for every query-head pair */
    return vk_dispatch(VK_PIPE_CAUSAL_ATTENTION_BF16, buffers, 5, push, 6,
                       (uint32_t)(seq * num_q_heads), 1, 1);
}

void iris_gpu_silu_mul(iris_gpu_tensor_t gate_tensor, iris_gpu_tensor_t up_tensor,
                       int n) {
    iris_gpu_tensor_impl_t *gate = (iris_gpu_tensor_impl_t *)gate_tensor;
    iris_gpu_tensor_impl_t *up = (iris_gpu_tensor_impl_t *)up_tensor;
    if (!gate || !up) return;
    iris_gpu_tensor_impl_t *buffers[2] = {gate, up};
    uint32_t push[1] = {(uint32_t)n};
    vk_dispatch(VK_PIPE_SILU_MUL, buffers, 2, push, 1,
                ((uint32_t)n + 255u) / 256u, 1, 1);
}

void iris_gpu_add_f32(iris_gpu_tensor_t out_tensor, iris_gpu_tensor_t a_tensor,
                      iris_gpu_tensor_t b_tensor, int n) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *a = (iris_gpu_tensor_impl_t *)a_tensor;
    iris_gpu_tensor_impl_t *b = (iris_gpu_tensor_impl_t *)b_tensor;
    if (!out || !a || !b) return;
    if (out != a) vk_copy_region(out, 0, a, 0, (size_t)n * sizeof(float));
    iris_gpu_tensor_impl_t *buffers[2] = {out, b};
    uint32_t push[1] = {(uint32_t)n};
    vk_dispatch(VK_PIPE_ADD, buffers, 2, push, 1,
                ((uint32_t)n + 255u) / 256u, 1, 1);
}

void iris_gpu_gated_add(iris_gpu_tensor_t out_tensor, const float *gate,
                        iris_gpu_tensor_t proj_tensor, int seq, int hidden) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *proj = (iris_gpu_tensor_impl_t *)proj_tensor;
    iris_gpu_tensor_impl_t *gate_buffer = vk_cached(gate,
        (size_t)hidden * sizeof(float), 0);
    if (!out || !proj || !gate_buffer) return;
    iris_gpu_tensor_impl_t *buffers[3] = {out, proj, gate_buffer};
    uint32_t push[2] = {(uint32_t)seq, (uint32_t)hidden};
    vk_dispatch(VK_PIPE_GATED_ADD, buffers, 3, push, 2,
                ((uint32_t)(seq * hidden) + 255u) / 256u, 1, 1);
}

void iris_gpu_adaln_norm(iris_gpu_tensor_t out_tensor, iris_gpu_tensor_t x_tensor,
                         const float *shift, const float *scale,
                         int seq, int hidden, float eps) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *shift_buffer = vk_cached(shift,
        (size_t)hidden * sizeof(float), 0);
    iris_gpu_tensor_impl_t *scale_buffer = vk_cached(scale,
        (size_t)hidden * sizeof(float), 0);
    if (!out || !x || !shift_buffer || !scale_buffer) return;
    iris_gpu_tensor_impl_t *buffers[4] = {x, shift_buffer, scale_buffer, out};
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[3] = {(uint32_t)seq, (uint32_t)hidden, eps_bits};
    vk_dispatch(VK_PIPE_ADALN_NORM, buffers, 4, push, 3, (uint32_t)seq, 1, 1);
}

void iris_gpu_group_norm_f32(iris_gpu_tensor_t out_tensor,
                             iris_gpu_tensor_t x_tensor,
                             const float *gamma, const float *beta,
                             int batch, int channels, int spatial,
                             int num_groups, float eps) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *gamma_buffer = vk_cached(
        gamma, (size_t)channels * sizeof(float), 0);
    iris_gpu_tensor_impl_t *beta_buffer = vk_cached(
        beta, (size_t)channels * sizeof(float), 0);
    if (!out || !x || !gamma_buffer || !beta_buffer ||
        channels % num_groups != 0) return;
    iris_gpu_tensor_impl_t *buffers[4] = {
        x, gamma_buffer, beta_buffer, out
    };
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(eps_bits));
    uint32_t push[5] = {(uint32_t)batch, (uint32_t)channels,
                        (uint32_t)spatial, (uint32_t)num_groups, eps_bits};
    /* Dispatch one reduction workgroup for each batch normalization group */
    vk_dispatch(VK_PIPE_GROUP_NORM, buffers, 4, push, 5,
                (uint32_t)(batch * num_groups), 1, 1);
}

void iris_gpu_swish_f32(iris_gpu_tensor_t out_tensor,
                        iris_gpu_tensor_t x_tensor, int n) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    if (!out || !x || n <= 0) return;
    iris_gpu_tensor_impl_t *buffers[2] = {x, out};
    uint32_t push[1] = {(uint32_t)n};
    /* Apply the activation across the complete contiguous tensor */
    vk_dispatch(VK_PIPE_SWISH, buffers, 2, push, 1,
                ((uint32_t)n + 255u) / 256u, 1, 1);
}

iris_gpu_tensor_t iris_gpu_upsample_nearest_2x_f32(iris_gpu_tensor_t x_tensor,
                                                    int channels, int H, int W) {
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    size_t elements = (size_t)channels * (size_t)H * (size_t)W * 4u;
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_device_local_bytes(
        elements * sizeof(float), 0, 0);
    if (!x || !out) {
        vk_tensor_destroy(out);
        return NULL;
    }
    iris_gpu_tensor_impl_t *buffers[2] = {x, out};
    uint32_t push[3] = {(uint32_t)channels, (uint32_t)H, (uint32_t)W};
    /* Expand each input pixel to a two-by-two output block */
    if (!vk_dispatch(VK_PIPE_UPSAMPLE, buffers, 2, push, 3,
                     ((uint32_t)elements + 255u) / 256u, 1, 1)) {
        vk_tensor_destroy_deferred(out);
        return NULL;
    }
    return out;
}

iris_gpu_tensor_t iris_gpu_conv2d_f32(iris_gpu_tensor_t x_tensor,
                                      const float *weight, const float *bias,
                                      int batch, int in_ch, int out_ch,
                                      int H, int W, int kH, int kW,
                                      int stride, int padding) {
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    int out_h = (H + 2 * padding - kH) / stride + 1;
    int out_w = (W + 2 * padding - kW) / stride + 1;
    size_t output_elements = (size_t)batch * (size_t)out_ch *
                             (size_t)out_h * (size_t)out_w;
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_device_local_bytes(
        output_elements * sizeof(float), 0, 0);
    iris_gpu_tensor_impl_t *weight_buffer = vk_cached(
        weight, (size_t)out_ch * (size_t)in_ch * (size_t)kH *
                (size_t)kW * sizeof(float), 0);
    iris_gpu_tensor_impl_t *bias_buffer = bias
        ? vk_cached(bias, (size_t)out_ch * sizeof(float), 0) : NULL;
    if (!x || !out || !weight_buffer || (bias && !bias_buffer)) {
        vk_tensor_destroy(out);
        return NULL;
    }
    iris_gpu_tensor_impl_t *buffers[4] = {x, weight_buffer, bias_buffer, out};

    /* Use implicit-im2col BF16 matrix tiles when cooperative matrices are available */
    if (vk_ctx.cooperative_matrix) {
        uint32_t pixels = (uint32_t)(out_h * out_w);
        uint32_t pixel_chunk = pixels >= 16384u ? 16384u : pixels;

        /* Split large spatial grids into independently fenced matrix packets */
        for (uint32_t pixel_offset = 0; pixel_offset < pixels;
             pixel_offset += pixel_chunk) {
            uint32_t pixel_count = pixels - pixel_offset;
            if (pixel_count > pixel_chunk) pixel_count = pixel_chunk;
            uint32_t push[13] = {
                (uint32_t)batch, (uint32_t)in_ch, (uint32_t)out_ch,
                (uint32_t)H, (uint32_t)W, (uint32_t)kH, (uint32_t)kW,
                (uint32_t)stride, (uint32_t)padding, (uint32_t)out_h,
                (uint32_t)out_w, bias ? 1u : 0u, pixel_offset
            };
            if (!vk_dispatch(VK_PIPE_CONV2D_COOP, buffers, 4, push, 13,
                             (pixel_count + 15u) / 16u,
                             ((uint32_t)out_ch + 15u) / 16u,
                             (uint32_t)batch)) {
                vk_tensor_destroy_deferred(out);
                return NULL;
            }
            /* Fence high-resolution convolution tiles before Windows triggers TDR */
            if (pixels >= 16384u && !vk_submit_and_resume_batch()) {
                vk_tensor_destroy_deferred(out);
                return NULL;
            }
        }
        return out;
    }
    const uint32_t max_elements = vk_friendly_requested ? 16384u : 65536u;

    /* Record bounded output slices so the scheduler can preempt between dispatches */
    for (uint32_t offset = 0; offset < output_elements; offset += max_elements) {
        uint32_t count = (uint32_t)(output_elements - offset);
        if (count > max_elements) count = max_elements;
        uint32_t push[13] = {
            (uint32_t)batch, (uint32_t)in_ch, (uint32_t)out_ch,
            (uint32_t)H, (uint32_t)W, (uint32_t)kH, (uint32_t)kW,
            (uint32_t)stride, (uint32_t)padding, (uint32_t)out_h,
            (uint32_t)out_w, offset, bias ? 1u : 0u
        };
        if (!vk_dispatch(VK_PIPE_CONV2D, buffers, 4, push, 13,
                         (count + 255u) / 256u, 1, 1)) {
            vk_tensor_destroy_deferred(out);
            return NULL;
        }
    }
    return out;
}

void iris_gpu_copy_f32(iris_gpu_tensor_t dst_tensor, iris_gpu_tensor_t src_tensor,
                       size_t n) {
    iris_gpu_tensor_impl_t *dst = (iris_gpu_tensor_impl_t *)dst_tensor;
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)src_tensor;
    if (dst && src) vk_copy_region(dst, 0, src, 0, n * sizeof(float));
}

void iris_gpu_copy_region_f32(iris_gpu_tensor_t dst_tensor, size_t dst_offset,
                              iris_gpu_tensor_t src_tensor, size_t src_offset,
                              size_t n) {
    iris_gpu_tensor_impl_t *dst = (iris_gpu_tensor_impl_t *)dst_tensor;
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)src_tensor;
    if (dst && src) vk_copy_region(dst, dst_offset * sizeof(float), src,
                                   src_offset * sizeof(float), n * sizeof(float));
}

void iris_gpu_copy_region_bf16(iris_gpu_tensor_t dst_tensor, size_t dst_offset,
                               iris_gpu_tensor_t src_tensor, size_t src_offset,
                               size_t n) {
    iris_gpu_tensor_impl_t *dst = (iris_gpu_tensor_impl_t *)dst_tensor;
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)src_tensor;
    if (dst && src && dst->is_f16 && src->is_f16)
        vk_copy_region(dst, dst_offset * sizeof(uint16_t), src,
                       src_offset * sizeof(uint16_t), n * sizeof(uint16_t));
}

void iris_gpu_split_qkv_mlp(iris_gpu_tensor_t fused_tensor,
                            iris_gpu_tensor_t q_tensor, iris_gpu_tensor_t k_tensor,
                            iris_gpu_tensor_t v_tensor, iris_gpu_tensor_t gate_tensor,
                            iris_gpu_tensor_t up_tensor, int seq, int hidden,
                            int mlp_hidden) {
    iris_gpu_tensor_impl_t *fused = (iris_gpu_tensor_impl_t *)fused_tensor;
    iris_gpu_tensor_impl_t *q = (iris_gpu_tensor_impl_t *)q_tensor;
    iris_gpu_tensor_impl_t *k = (iris_gpu_tensor_impl_t *)k_tensor;
    iris_gpu_tensor_impl_t *v = (iris_gpu_tensor_impl_t *)v_tensor;
    iris_gpu_tensor_impl_t *gate = (iris_gpu_tensor_impl_t *)gate_tensor;
    iris_gpu_tensor_impl_t *up = (iris_gpu_tensor_impl_t *)up_tensor;
    if (!fused) return;
    if (hidden == 0) {
        if (gate) vk_copy_region(gate, 0, fused, 0,
                                 (size_t)seq * mlp_hidden * sizeof(float));
        if (up) vk_copy_region(up, 0, fused,
                               (size_t)seq * mlp_hidden * sizeof(float),
                               (size_t)seq * mlp_hidden * sizeof(float));
    } else {
        size_t bytes = (size_t)seq * hidden * sizeof(float);
        if (q) vk_copy_region(q, 0, fused, 0, bytes);
        if (k) vk_copy_region(k, 0, fused, bytes, bytes);
        if (v) vk_copy_region(v, 0, fused, bytes * 2, bytes);
    }
}

int iris_gpu_convert_f32_to_bf16_into(iris_gpu_tensor_t out_tensor,
                                      iris_gpu_tensor_t in_tensor) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *in = (iris_gpu_tensor_impl_t *)in_tensor;
    if (!out || !in || !out->is_f16 || in->is_f16) return 0;
    size_t count = out->elements < in->elements ? out->elements : in->elements;
    iris_gpu_tensor_impl_t *buffers[2] = {in, out};
    uint32_t push[1] = {(uint32_t)count};
    /* Convert device-local FP32 activations directly into packed BF16 */
    return vk_dispatch(VK_PIPE_F32_TO_BF16, buffers, 2, push, 1,
                       (((uint32_t)count + 1u) / 2u + 255u) / 256u, 1, 1);
}

int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t out_tensor,
                                      iris_gpu_tensor_t in_tensor) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *in = (iris_gpu_tensor_impl_t *)in_tensor;
    if (!out || !in || out->is_f16 || !in->is_f16) return 0;
    size_t count = out->elements < in->elements ? out->elements : in->elements;
    iris_gpu_tensor_impl_t *buffers[2] = {in, out};
    uint32_t push[1] = {(uint32_t)count};
    /* Convert packed BF16 activations directly into device-local FP32 */
    return vk_dispatch(VK_PIPE_BF16_TO_F32, buffers, 2, push, 1,
                       (((uint32_t)count + 1u) / 2u + 255u) / 256u, 1, 1);
}

iris_gpu_tensor_t iris_gpu_tensor_f32_to_bf16(iris_gpu_tensor_t input) {
    iris_gpu_tensor_impl_t *source = (iris_gpu_tensor_impl_t *)input;
    if (!source || source->is_f16) return NULL;
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc_f16(source->elements);
    /* Convert into a separate BF16 activation tensor */
    if (!out || !iris_gpu_convert_f32_to_bf16_into(out, input)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

iris_gpu_tensor_t iris_gpu_tensor_bf16_to_f32(iris_gpu_tensor_t input) {
    iris_gpu_tensor_impl_t *source = (iris_gpu_tensor_impl_t *)input;
    if (!source || !source->is_f16) return NULL;
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc(source->elements);
    /* Convert into a separate FP32 activation tensor */
    if (!out || !iris_gpu_convert_bf16_to_f32_into(out, input)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

int iris_gpu_linear_bf16_native_into(iris_gpu_tensor_t out_tensor,
                                     iris_gpu_tensor_t x_tensor,
                                     const uint16_t *weights,
                                     int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    iris_gpu_tensor_impl_t *weight = vk_cached(
        weights, (size_t)in_dim * (size_t)out_dim * sizeof(uint16_t), 1);
    if (!out || !x || !weight || !out->is_f16 || !x->is_f16 ||
        !vk_ctx.cooperative_matrix || (out_dim & 1) != 0) return 0;
    /* Multiply BF16 activations and weights on cooperative matrix hardware */
    return vk_linear_bf16_native_dispatch(out, x, weight,
                                           seq_len, in_dim, out_dim);
}

iris_gpu_tensor_t iris_gpu_linear_bf16_native(iris_gpu_tensor_t x,
                                              const uint16_t *weights,
                                              int seq_len, int in_dim,
                                              int out_dim) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc_f16(
        (size_t)seq_len * (size_t)out_dim);
    /* Allocate and fill a native BF16 projection result */
    if (!out || !iris_gpu_linear_bf16_native_into(
            out, x, weights, seq_len, in_dim, out_dim)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

int iris_metal_causal_attention(float *out, const float *q, const float *k,
                                const float *v, const int *attention_mask,
                                int seq, int num_q_heads, int num_kv_heads,
                                int head_dim, float scale) {
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)attention_mask;
    (void)seq;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)scale;
    /* Keep the legacy host-pointer path on CPU; Vulkan uses GPU tensors */
    return 0;
}

void iris_gpu_rope_2d(iris_gpu_tensor_t x, const float *cos_freq,
                      const float *sin_freq, int seq, int heads,
                      int head_dim, int axis_dim) {
    (void)x; (void)cos_freq; (void)sin_freq;
    (void)seq; (void)heads; (void)head_dim; (void)axis_dim;
}
