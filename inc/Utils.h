#pragma once

#include <string>
#include "def.h"

//Linear Congruential Random Number Generator Values
#define IA 16807
#define IM 2147483647
#define AM (1.0f/(float)IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876


class Utils {
public:
	Utils(void);
	~Utils(void);

	static std::string int2String(int t);
	static float random( long *idum );

		 
	static void sort2(int v[], int v2[], int left, int right);

	static void compute_distance( const float* IN xaxis, const float* IN yaxis, const int IN nodes, int* OUT distance);


private:
	 
	static void swap2(int v[], int v2[], int i, int j);
	static int geo_distance (float xi, float xj, float yi, float yj);
	static float exact_distance (float xi, float xj, float yi, float yj);
	static int round_distance (float xi, float xj, float yi, float yj);
	static int ceil_distance (float xi, float xj, float yi, float yj);
	static int floor_distance (float xi, float xj, float yi, float yj);
};

#if _GPU_
__DEVICE__ __FORCEINLINE__
float dev_random(long* idum, int idx ) {
	long k, id = idum[idx];
	float ans;

	k =(id)/IQ;
	id = IA * (id - k * IQ) - IR * k;
	if (id < 0 ) id += IM;
	ans = AM * id;
	idum[idx] = id;
	return ans;
}

inline static unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline static unsigned int round2BlockSize(unsigned int threads, unsigned int blockSize) {
	return ((threads + blockSize - 1)/blockSize)*blockSize;
}

#endif
#if 0
template<typename T>
__DEVICE__ void swap(T* parray, const int i, const int j) {
	T temp = parray[i];
	parray[i] = parray[j];
	parray[j] = temp;
}

__global__ void sort_step(float* dev_values, int* r_values, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
 
	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i) {
		if ((i&k)==0) {
			/* Sort ascending */
			if (dev_values[i]>dev_values[ixj]) {
				/* exchange(i,ixj); */
				swap<float>(dev_values, i, ixj);
				swap<int>(r_values, i, ixj);

			}
		}
		if ((i&k)!=0) {
			/* Sort descending */
			if (dev_values[i]<dev_values[ixj]) {
				/* exchange(i,ixj); */
				swap<float>(dev_values, i, ixj);
				swap<int>(r_values, i, ixj);
			}
		}
	}
} 
 
/**
* Inplace bitonic sort using CUDA.
*/
void bitonic_sort(float* dev_values, int* r_values, const int size, int threadsPerBlock, int blocksPerGrid)
{
	
	int j, k;
	/* Major step */
	for (k = 2; k <= size; k <<= 1) {
		/* Minor step */
		for (j=k>>1; j>0; j=j>>1) {
			sort_step<<<blocksPerGrid, threadsPerBlock>>>(dev_values, j, k);
		}
	}
	
} 
#endif
