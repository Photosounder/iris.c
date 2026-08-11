/*
 * iris_platform.h - Small platform portability helpers
 */

#ifndef IRIS_PLATFORM_H
#define IRIS_PLATFORM_H

#include <stddef.h>

/* Return a monotonic elapsed-time value in milliseconds. */
double iris_time_ms(void);

/* Return the number of logical processors available to the process. */
int iris_cpu_count(void);

/* Create a unique temporary directory and write its path to the caller's buffer. */
int iris_create_temp_dir(char *path, size_t path_size);

/* Create a unique temporary file, optionally preserving the requested suffix. */
int iris_create_temp_file(char *path, size_t path_size, const char *suffix);

/* Remove a temporary file using the platform's native API. */
int iris_unlink(const char *path);

/* Duplicate a string using the allocator used by the C runtime. */
char *iris_strdup(const char *value);

#endif /* IRIS_PLATFORM_H */
