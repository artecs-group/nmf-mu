#ifndef _COMMON_
#define _COMMON_

#include <sys/time.h>

#ifdef REAL_S
#define C_REAL float
#else
#define C_REAL double
#endif

double gettime();

#endif