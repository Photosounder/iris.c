/* CIT Allocator Windows integration used by the Windows build */
#ifndef IRIS_CITA_H
#define IRIS_CITA_H

#if defined(_WIN32) && defined(IRIS_USE_CITA)
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(IRIS_CITA_IMPLEMENTATION)
#define CITA_WIN_IMPLEMENTATION
#endif
#include <cita_windows.h>
#endif

#endif /* IRIS_CITA_H */
