/*
 * Iris Vulkan Acceleration
 *
 * Compute backend for the Z-Image GPU path. The implementation intentionally
 * exports the existing iris_gpu_* contract so the transformer does not need a
 * second backend-specific execution graph.
 */

#include "iris_metal.h"
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
    VK_PIPE_RMS_NORM,
    VK_PIPE_QK_RMS_NORM,
    VK_PIPE_ROPE_PAIR,
    VK_PIPE_ATTENTION,
    VK_PIPE_SILU_MUL,
    VK_PIPE_ADD,
    VK_PIPE_GATED_ADD,
    VK_PIPE_ADALN_NORM,
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
    vk_cached_buffer_t *cache;
    size_t memory_used;
} vk_context_t;

static vk_context_t vk_ctx;

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
    free(tensor);
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
        return 0;
    }
    VkResult result = vkEndCommandBuffer(vk_ctx.command_buffer);
    if (result != VK_SUCCESS) {
        vk_report(result, "vkEndCommandBuffer");
        return 0;
    }

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
    if (result != VK_SUCCESS && vk_ctx.device_lost) {
        /* Leave lost-device handles untouched because cleanup cannot safely use them */
        vk_ctx.batch_active = 0;
        return 0;
    }
    vkResetFences(vk_ctx.device, 1, &vk_ctx.fence);
    vkResetCommandPool(vk_ctx.device, vk_ctx.command_pool, 0);
    if (vk_ctx.descriptor_pool)
        vkResetDescriptorPool(vk_ctx.device, vk_ctx.descriptor_pool, 0);
    return result == VK_SUCCESS;
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
    if (vk_trace_enabled()) {
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
    if ((pipeline_id == VK_PIPE_LINEAR || pipeline_id == VK_PIPE_LINEAR_BF16 ||
         pipeline_id == VK_PIPE_ATTENTION) && !temporary_batch) {
        /* Submit long-running kernels separately so one dispatch cannot trip TDR */
        vk_ctx.batch_active = 0;
        return vk_submit_command_buffer();
    }
    return vk_end_if_temporary(temporary_batch);
}

static int vk_copy_region(iris_gpu_tensor_impl_t *dst, size_t dst_offset,
                          iris_gpu_tensor_impl_t *src, size_t src_offset,
                          size_t bytes) {
    /* Refuse buffer copies once Vulkan has reported device loss */
    if (!vk_ready() || !dst || !src) return 0;
    int temporary_batch = 0;
    if (!vk_begin_if_needed(&temporary_batch)) return 0;
    VkBufferCopy copy = {0};
    copy.srcOffset = src_offset;
    copy.dstOffset = dst_offset;
    copy.size = bytes;
    vkCmdCopyBuffer(vk_ctx.command_buffer, src->buffer, dst->buffer, 1, &copy);
    vk_memory_barrier();
    if (!temporary_batch) {
        /* Submit standalone copies before recording the next large kernel */
        vk_ctx.batch_active = 0;
        return vk_submit_command_buffer();
    }
    return vk_end_if_temporary(temporary_batch);
}

static iris_gpu_tensor_impl_t *vk_cached(const void *host_ptr, size_t bytes,
                                          int is_f16) {
    /* Refuse new cached buffers once Vulkan has reported device loss */
    if (!vk_ready() || !host_ptr) return NULL;
    for (vk_cached_buffer_t *entry = vk_ctx.cache; entry; entry = entry->next) {
        if (entry->host_ptr == host_ptr && entry->bytes == bytes &&
            entry->is_f16 == is_f16) {
            void *mapped = NULL;
            /* Refresh mutable host arrays before reusing their cached buffer */
            if (!is_f16 && vk_tensor_map(entry->tensor, &mapped)) {
                memcpy(mapped, host_ptr, bytes);
                vk_tensor_unmap(entry->tensor);
            }
            return entry->tensor;
        }
    }
    iris_gpu_tensor_impl_t *tensor = NULL;
    if (is_f16 && !vk_ctx.batch_active)
        tensor = vk_tensor_alloc_device_local_bytes(bytes, is_f16, 1);
    if (!tensor)
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
        memcpy(mapped, host_ptr, bytes);
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
        memcpy(mapped, host_ptr, bytes);
        vk_tensor_unmap(staging);
        if (!vk_copy_region(tensor, 0, staging, 0, bytes)) {
            vk_tensor_destroy(staging);
            tensor->cached = 0;
            vk_tensor_destroy(tensor);
            return NULL;
        }
        vk_tensor_destroy(staging);
    }

    vk_cached_buffer_t *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        /* Release the allocation when the cache node cannot be created */
        tensor->cached = 0;
        vk_tensor_destroy(tensor);
        return NULL;
    }
    entry->host_ptr = host_ptr;
    entry->bytes = bytes;
    entry->is_f16 = is_f16;
    entry->tensor = tensor;
    entry->next = vk_ctx.cache;
    vk_ctx.cache = entry;
    return tensor;
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

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx.physical_device, &queue_count, NULL);
    VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
    if (!queues) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx.physical_device, &queue_count, queues);
    for (uint32_t i = 0; i < queue_count; i++) {
        if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vk_ctx.queue_family = i;
            break;
        }
    }
    free(queues);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk_ctx.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {0};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(vk_ctx.physical_device, &device_info, NULL, &vk_ctx.device);
    if (result != VK_SUCCESS) { vk_report(result, "vkCreateDevice"); return 0; }
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
        "shaders/iris_vulkan_rms_norm.comp.spv",
        "shaders/iris_vulkan_qk_rms_norm.comp.spv",
        "shaders/iris_vulkan_rope_pair.comp.spv",
        "shaders/iris_vulkan_attention.comp.spv",
        "shaders/iris_vulkan_silu_mul.comp.spv",
        "shaders/iris_vulkan_add.comp.spv",
        "shaders/iris_vulkan_gated_add.comp.spv",
        "shaders/iris_vulkan_adaln_norm.comp.spv"
    };
    for (int i = 0; i < VK_PIPE_COUNT; i++) {
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
}

void iris_metal_cleanup(void) {
    if (!vk_ctx.device && !vk_ctx.instance) return;
    int device_lost = vk_ctx.device_lost;
    /* Wait only while the device is known to be responsive */
    if (vk_ctx.device && !device_lost) vkDeviceWaitIdle(vk_ctx.device);
    vk_clear_cache();
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
void iris_metal_clear_bf16_cache_only(void) { }
void iris_metal_clear_f16_cache_only(void) { }
void iris_metal_clear_activation_pool_only(void) { }

void iris_gpu_sync(void) {
    /* Synchronize only while the device is known to be responsive */
    if (vk_ready()) vkDeviceWaitIdle(vk_ctx.device);
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
    if (vk_ctx.device_lost) return;
    vk_submit_command_buffer();
}

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements) {
    if (!data || !vk_ready()) return NULL;
    iris_gpu_tensor_impl_t *tensor = vk_tensor_alloc_bytes(num_elements * sizeof(float), 0, 0);
    if (!tensor) return NULL;
    void *mapped = NULL;
    if (!vk_tensor_map(tensor, &mapped)) { vk_tensor_destroy(tensor); return NULL; }
    memcpy(mapped, data, num_elements * sizeof(float));
    vk_tensor_unmap(tensor);
    return tensor;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements) {
    if (!vk_ready()) return NULL;
    return vk_tensor_alloc_bytes(num_elements * sizeof(float), 0, 0);
}

iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements) {
    if (!vk_ready()) return NULL;
    return vk_tensor_alloc_bytes(num_elements * sizeof(uint16_t), 1, 0);
}

void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t tensor, int persistent) {
    if (tensor) ((iris_gpu_tensor_impl_t *)tensor)->persistent = persistent;
}

void iris_gpu_tensor_free(iris_gpu_tensor_t tensor) {
    vk_tensor_destroy((iris_gpu_tensor_impl_t *)tensor);
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
    void *mapped = NULL;
    if (!vk_tensor_map(dst, &mapped)) return;
    if (dst->is_f16) {
        uint16_t *out = mapped;
        for (size_t i = 0; i < dst->elements; i++) out[i] = vk_float_to_bf16(data[i]);
    } else {
        memcpy(mapped, data, dst->elements * sizeof(float));
    }
    vk_tensor_unmap(dst);
}

void iris_gpu_tensor_read(iris_gpu_tensor_t tensor, float *out) {
    iris_gpu_tensor_impl_t *src = (iris_gpu_tensor_impl_t *)tensor;
    if (!src || !out || !vk_ready()) return;
    iris_gpu_sync();
    void *mapped = NULL;
    if (!vk_tensor_map(src, &mapped)) return;
    if (src->is_f16) {
        const uint16_t *input = mapped;
        for (size_t i = 0; i < src->elements; i++) out[i] = vk_bf16_to_float(input[i]);
    } else {
        memcpy(out, mapped, src->elements * sizeof(float));
    }
    vk_tensor_unmap(src);
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
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_bytes(
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
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *x = (iris_gpu_tensor_impl_t *)x_tensor;
    if (!out || !x || !weights || !vk_ready() || out->is_f16 || x->is_f16) return 0;
    iris_gpu_tensor_impl_t *weight = vk_cached(weights,
        (size_t)in_dim * out_dim * sizeof(uint16_t), 1);
    if (!weight) return 0;
    iris_gpu_tensor_impl_t *buffers[3] = {x, weight, out};
    uint32_t push[3] = {(uint32_t)seq_len, (uint32_t)in_dim, (uint32_t)out_dim};
    return vk_dispatch(VK_PIPE_LINEAR_BF16, buffers, 3, push, 3,
                       ((uint32_t)out_dim + 15u) / 16u,
                       ((uint32_t)seq_len + 15u) / 16u, 1);
}

iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x,
                                       const uint16_t *weights,
                                       int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_impl_t *out = vk_tensor_alloc_bytes(
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
    /* Split query rows so each attention dispatch stays below the Windows watchdog */
    const int query_chunk = 16;
    for (int query_offset = 0; query_offset < seq_q; query_offset += query_chunk) {
        int query_count = seq_q - query_offset;
        if (query_count > query_chunk) query_count = query_chunk;
        uint32_t push[6] = {(uint32_t)seq_q, (uint32_t)seq_k, (uint32_t)heads,
                            (uint32_t)head_dim, scale_bits, (uint32_t)query_offset};
        if (!vk_dispatch(VK_PIPE_ATTENTION, buffers, 4, push, 6,
                         (uint32_t)query_count * (uint32_t)heads, 1, 1)) return 0;
    }
    return 1;
}

int iris_gpu_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t q,
                            iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                            int seq_q, int seq_k, int heads, int head_dim,
                            float scale) {
    (void)out; (void)q; (void)k; (void)v;
    (void)seq_q; (void)seq_k; (void)heads; (void)head_dim; (void)scale;
    return 0;
}

int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t q,
                                  iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                                  int seq_q, int seq_k, int heads, int head_dim,
                                  float scale) {
    return iris_gpu_attention_bf16(out, q, k, v, seq_q, seq_k,
                                   heads, head_dim, scale);
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
    void *out_data = NULL;
    void *in_data = NULL;
    if (!vk_tensor_map(out, &out_data) || !vk_tensor_map(in, &in_data)) {
        if (out_data) vk_tensor_unmap(out);
        if (in_data) vk_tensor_unmap(in);
        return 0;
    }
    uint16_t *dst = out_data;
    const float *src = in_data;
    size_t count = out->elements < in->elements ? out->elements : in->elements;
    for (size_t i = 0; i < count; i++) dst[i] = vk_float_to_bf16(src[i]);
    vk_tensor_unmap(in);
    vk_tensor_unmap(out);
    return 1;
}

int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t out_tensor,
                                      iris_gpu_tensor_t in_tensor) {
    iris_gpu_tensor_impl_t *out = (iris_gpu_tensor_impl_t *)out_tensor;
    iris_gpu_tensor_impl_t *in = (iris_gpu_tensor_impl_t *)in_tensor;
    if (!out || !in || out->is_f16 || !in->is_f16) return 0;
    void *out_data = NULL;
    void *in_data = NULL;
    if (!vk_tensor_map(out, &out_data) || !vk_tensor_map(in, &in_data)) {
        if (out_data) vk_tensor_unmap(out);
        if (in_data) vk_tensor_unmap(in);
        return 0;
    }
    float *dst = out_data;
    const uint16_t *src = in_data;
    size_t count = out->elements < in->elements ? out->elements : in->elements;
    for (size_t i = 0; i < count; i++) dst[i] = vk_bf16_to_float(src[i]);
    vk_tensor_unmap(in);
    vk_tensor_unmap(out);
    return 1;
}

void iris_gpu_rope_2d(iris_gpu_tensor_t x, const float *cos_freq,
                      const float *sin_freq, int seq, int heads,
                      int head_dim, int axis_dim) {
    (void)x; (void)cos_freq; (void)sin_freq;
    (void)seq; (void)heads; (void)head_dim; (void)axis_dim;
}
