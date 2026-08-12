/*
 * iris_platform.c - Small platform portability helpers
 */

#include "iris_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <io.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#endif

double iris_time_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int initialized;
    LARGE_INTEGER counter;

    /* Cache the high-resolution timer frequency once. */
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timeval tv;

    /* Use the portable wall-clock fallback on Unix-like systems. */
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

int iris_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO info;

    /* Ask Windows for the number of logical processors in the system. */
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);

    /* Keep callers safe when the operating system cannot report a count. */
    return count > 0 ? (int)count : 1;
#endif
}

void iris_sleep_ms(unsigned int milliseconds) {
#ifdef _WIN32
    /* Yield the processor through the native Windows scheduler */
    Sleep(milliseconds);
#else
    /* Build and apply a nanosecond-resolution duration on Unix-like systems */
    struct timespec duration;
    duration.tv_sec = milliseconds / 1000u;
    duration.tv_nsec = (long)(milliseconds % 1000u) * 1000000L;
    nanosleep(&duration, NULL);
#endif
}

void iris_lower_process_priority(void) {
#ifdef _WIN32
    /* Lower CPU, memory, and disk scheduling priority for the whole process */
    if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN))
        SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#else
    /* Leave process priority unchanged where no portable lowering API is used */
    return;
#endif
}

int iris_create_temp_dir(char *path, size_t path_size) {
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char candidate[MAX_PATH];

    /* Ask Windows for a unique temporary filename, then turn it into a directory. */
    if (GetTempPathA((DWORD)sizeof(temp_path), temp_path) == 0 ||
        GetTempFileNameA(temp_path, "iri", 0, candidate) == 0) {
        return -1;
    }
    DeleteFileA(candidate);
    if (CreateDirectoryA(candidate, NULL) == 0 ||
        strlen(candidate) + 1 > path_size) {
        RemoveDirectoryA(candidate);
        return -1;
    }
    strcpy(path, candidate);
    return 0;
#else
    char template_path[512];

    /* Use the standard Unix mkdtemp implementation for temporary output. */
    snprintf(template_path, sizeof(template_path), "%s/iris-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (strlen(template_path) + 1 > path_size || !mkdtemp(template_path)) {
        return -1;
    }
    strcpy(path, template_path);
    return 0;
#endif
}

int iris_create_temp_file(char *path, size_t path_size, const char *suffix) {
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char base_path[MAX_PATH];
    char suffixed_path[MAX_PATH];

    /* Create a file in the Windows temporary directory and append its suffix. */
    if (GetTempPathA((DWORD)sizeof(temp_path), temp_path) == 0 ||
        GetTempFileNameA(temp_path, "iri", 0, base_path) == 0) {
        return -1;
    }
    if (!suffix || !suffix[0]) {
        if (strlen(base_path) + 1 > path_size) {
            DeleteFileA(base_path);
            return -1;
        }
        strcpy(path, base_path);
        return 0;
    }
    if (snprintf(suffixed_path, sizeof(suffixed_path), "%s%s", base_path, suffix) < 0 ||
        strlen(suffixed_path) + 1 > path_size ||
        !MoveFileA(base_path, suffixed_path)) {
        DeleteFileA(base_path);
        return -1;
    }
    strcpy(path, suffixed_path);
    return 0;
#else
    char template_path[512];
    int fd;

    /* Use mkstemps so the generated file keeps the extension required by image I/O. */
    snprintf(template_path, sizeof(template_path), "%s/iris_XXXXXX%s",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", suffix ? suffix : "");
    fd = mkstemps(template_path, suffix ? (int)strlen(suffix) : 0);
    if (fd < 0 || strlen(template_path) + 1 > path_size) {
        if (fd >= 0) close(fd);
        return -1;
    }
    close(fd);
    strcpy(path, template_path);
    return 0;
#endif
}

int iris_unlink(const char *path) {
#ifdef _WIN32
    /* Use the CRT unlink spelling available in MinGW and MSVC-compatible builds. */
    return _unlink(path);
#else
    /* Use the POSIX unlink operation on Unix-like systems. */
    return unlink(path);
#endif
}

char *iris_strdup(const char *value) {
    size_t length;
    char *copy;

    /* Allocate and copy the complete NUL-terminated string. */
    if (!value) return NULL;
    length = strlen(value) + 1;
    copy = (char *)malloc(length);
    if (copy) memcpy(copy, value, length);
    return copy;
}
