
#ifndef _DEF_H
#define _DEF_H

#include "platform.h"
#include "aging.h"
#include "parallel.h"


/// set all variables according to flag settings
#if _GPU_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "helper_cuda.h"

#define DEVICE_VECTOR	device_vector
#define	__HOST__	__host__
#define	__DEVICE__	__device__
#define __FORCEINLINE__	__forceinline__
#define __SHARED__	__shared__

#define THREADS_PER_BLOCK	256
#define MAX_THREADS_PER_BLOCK	1024
static const char PLATFORM[4] = "GPU";

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
#else
/// treat device vector as host vector in CPU version
#define DEVICE_VECTOR	host_vector
#define	__HOST__
#define	__DEVICE__
#define __FORCEINLINE
#define __SHARED__

static const char PLATFORM[4] = "CPU";
#endif

#if ANTS_AGING
static const char AGING[7] = "anging";
#else
static const char AGING[9] = "noanging";
#endif

/// host vector will remain as-is in both CPU and GPU versions
#define HOST_VECTOR	host_vector

#define TRUE	1
#define FALSE	0

#define DEBUG( x )	x

#define OUTPUT_PRECISION	8
#define MAX_DISTANCE	0x7fffffff

#define IN
#define OUT
#define IN_OUT


#define CALCULATE_DISTANCE_ERROR(c,o)	(((c-o) / o) * 100)


//static const char BASE_DIR[MAX_INPUT_LEN] = "aco/release/";
#define BASE_DIR		"aco/release/"

typedef unsigned long long int		QWORD;
typedef unsigned int				DWORD;
typedef unsigned short				WORD;
typedef unsigned char				BYTE;

#endif // _DEF_H

