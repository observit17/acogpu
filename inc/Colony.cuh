
#ifndef COLONY_CUH
#define COLONY_CUH

#include "Colony.h"
#include "default.h"


#if ANTS_AGING
__DEVICE__ unsigned int numValidAnts;
int isconverged;
#endif

__DEVICE__  __HOST__
float compute_fittness(const float pheromen, const float alpha, const float hurestic
#if ANTS_AGING
	, const float pheromone_mean, const int age, const int maxage, const int isConverged
#endif
	) {

	float p = pheromen;

#if ANTS_AGING
	if(age < maxage && isConverged)
		//p = (p*age)/maxage;
		p = ((p-pheromone_mean) * ((float)age/(float)maxage)) + pheromone_mean;
#endif

	//return __powf(p, alpha) * hurestic;
	return p * hurestic;
}

__DEVICE__  __HOST__
int select_local_best(const int* nnlist, const unsigned char* visited, const int numCities) {
	int s = -1;
	for(int i=numCities-1; i>=0 ; i--) {
		if(visited[nnlist[i]] == FALSE) {
			s = i;
			break;
		}
	}
	return s;
}

#if _GPU_

__device__ float dev_atomic_add_f1(float* address, float val) { 
	unsigned int* addr = (unsigned int*)address; 
	unsigned int old = *addr, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(addr, assumed, __float_as_int(val + __int_as_float(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __int_as_float(old); 
}

__global__ void dev_init_seeds(curandState* OUT state, const int numElements, const long seed) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;	

    if (k < numElements) {
       unsigned long s = seed << k;
       curand_init ( s*s, k+1, 0, &state[k] );
    }
}

__device__ float dev_random_uniform(curandState* state) {
	curandState localState = *state;
	float rnd = curand_uniform( &localState );
	(*state) = localState;
	return rnd;
}

__global__ void dev_init_pheromone(float* OUT pheromones, const int numElements, const float pher) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;	

    if (k < numElements) {
       pheromones[k] = pher;    		
    }
}

__global__  void dev_compute_hurestics(int* distances, float* OUT hurestics, const int numElements, const float beta) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;	

    if (k < numElements) {
       hurestics[k] = HURESTIC(distances[k], beta);		
    }
}


__global__  void dev_compute_selection(const float* pheromones, const float* hurestics, int* OUT probabilities, 
							const float alpha, const int numElements) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;	

    if (k < numElements*numElements)    {
		probabilities[k] = hurestics[k] == 0 ? 0 : //powf(pheromones[k], alpha)*hurestics[k]; 
			__float_as_int(pheromones[k]*hurestics[k]); 
    }
}

__global__ void dev_compute_nnlist(const int* distances, int* OUT nnlist, const int n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx < n*n) {  		

		int cidx = idx;
		int iv = idx % n;		

		const int start = idx - iv; //((int)(idx / n)) * n;
		const int end = start + n;
		
		for(int i = start; i< idx; i++) {
			if(distances[i] > distances[idx])
				cidx--;
		}
		for(int i = idx+1; i< end; i++) {
			if(distances[i] < distances[idx])
				cidx++;
		}
		nnlist[cidx] = iv;		
	}
}


__global__ void dev_clear_visits(unsigned char* OUT visited, long* OUT tour, const int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
    if (i < n) {  
		visited[i] = FALSE;	
		tour[i] = 0;
    }
}

__global__ void dev_place_ants(curandState* IN states, long* OUT tour, unsigned char* OUT visited, long* OUT tourlength,
#if ANTS_AGING
							   unsigned char* IN validants, 
#endif
							   const int na, const int nc) {

	int k = blockDim.x * blockIdx.x + threadIdx.x;
	
	
    if (k < na 
#if ANTS_AGING
		&& validants[k]
#endif
	) {  
		//curandState localState = states[k];
		//float frnd = curand_uniform( &localState );
		//states[k] = localState;

		float frnd = dev_random_uniform(&(states[k]));
		
		long rnd = (long)(frnd * (float) nc);
		tour[k*nc] = rnd; 
		visited[k*nc+rnd] = TRUE;	
		tourlength[k] = 0;
		
    }
}

__global__ void dev_compute_pheromone_mean(const float* IN pheromone, float* OUT pheromone_mean, const int cities, const int n) {

	float *sdata = SharedMemory<float>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.x*cities + threadIdx.x;

    sdata[tid] = (tid < cities && i < n) ? pheromone[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) pheromone_mean[blockIdx.x] = sdata[0]/cities;
}


__global__ void dev_compute_probabilities(int* OUT prob_ptr, const long* IN tour, const unsigned char* IN visited,										  
										  const int* IN probabilities, const int* IN nnlist, 
										  const float* IN pheromones, const float* IN hurestics,
#if ANTS_AGING
										  const float* IN pheromone_mean, const unsigned char* IN validants, const int* IN antage, 
										  const int minRetirementAge, const int isConverged,
#endif
										  const int phase, const int ants, const int cities,
										  const float alpha) {

		
	int i = blockDim.x * blockIdx.x + threadIdx.x; // current thread
	
    if (i < ants * cities) {

		int k = i / cities;
#if ANTS_AGING
		if(validants[k]) {
#endif
			int n = i % cities;
			int current_city = tour[k*cities+phase-1]; 

			int xy = current_city*cities+nnlist[current_city*cities+n];

			prob_ptr[i] = visited[k*cities+nnlist[current_city*cities+n]] ? 0 :
#if ANTS_AGING				
				compute_fittness(pheromones[xy], alpha, hurestics[xy]
						, pheromone_mean[current_city], antage[k], minRetirementAge, isConverged
				);
#else
				probabilities[xy];
#endif
#if ANTS_AGING
		}
#endif
    }
}

template <typename T> 
__global__ 	void dev_init_data(T* data, const T f, unsigned int n) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<n) {
		data[i] = f;
	}
}

__global__ 	void dev_reduce_probabilities(float* IN prob, float* OUT sprob, 
#if ANTS_AGING
										  const unsigned char* IN validants, 
#endif
										  unsigned int blocks, unsigned int cities, unsigned int n) {
#if ANTS_AGING
	if(validants[blockIdx.x] == FALSE)
		return;
#endif

    //float *sdata = SharedMemory<float>();
	extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    //unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int i = blockIdx.x*cities + threadIdx.x;
	unsigned int bid = blockIdx.x % blocks;
	unsigned int c = bid * blockDim.x + tid;
	unsigned int b = blockIdx.x / blocks;
	unsigned int i = b * cities + c;

    //sdata[tid] = (tid < cities && i < n) ? prob[i] : 0;
	sdata[tid] = (c < cities) ? prob[i] : 0;
	//if( s < n )	sprob[s] = (c < cities) ? prob[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
		//sprob[blockIdx.x] = sdata[0];
		dev_atomic_add_f1(&(sprob[b]), sdata[0]);
	}
}

__device__ void update_local_pheromone(long* tourlength, int* distance, const int c1, const int c2, const int k, const int cities) {

	//tourlength[k] += distance[c1->c2]
}

__global__ void dev_move_ants(const int phase, const int ants, const int cities, int* IN prob_ptr, curandState* IN states,
							  long* OUT tour, unsigned char* OUT visited, const QWORD* IN sum_prob, int* IN nnlist,
#if ANTS_AGING
					const unsigned char* IN validants,
#endif
					int* IN distances, long* dtourlength, int* tourmap, QWORD* info) {
	
	
	int k = blockDim.x * blockIdx.x + threadIdx.x; // current thread

    if (k < ants 
#if ANTS_AGING
		&& validants[k]
#endif
	) {

		//QWORD* kinfo = &(info[k*5]);
		//kinfo[0] = 0;

		float rnd = dev_random_uniform(&(states[k]));
				
		int current_city = tour[k*cities+phase-1]; 
		int* pb_ptr = &(prob_ptr[k*cities]);
		int* nnl = &(nnlist[current_city*cities]);
		
		float irnd = (rnd * __int_as_float((int)(sum_prob[k])));
		int i = 0;
		float partial_sum = __int_as_float(pb_ptr[i]);
		while ( partial_sum <= irnd ) {
			i++;
			if(i >= cities) {
				i = select_local_best(nnl, &(visited[k*cities]), cities);	
				//kinfo[0] = 1;
				break;
			}
			partial_sum += __int_as_float(pb_ptr[i]); 
		}		
		int target_city = nnl[i];	

		//kinfo[1] = current_city;
		//kinfo[2] = target_city;
		//kinfo[3] = sum_prob[k];
		//kinfo[4] = irnd;

		tour[k*cities+phase] = target_city; 
		visited[k*cities+target_city] = TRUE;

		dtourlength[k] += distances[current_city*cities+target_city];
		if(phase == cities-1) {
			dtourlength[k] += distances[tour[k*cities]*cities+target_city];
		}
		if(current_city < target_city)
			tourmap[current_city*ants+k] = target_city;
		else
			tourmap[target_city*ants+k] = current_city;
	
    }
}


__global__ void dev_calculate_tour_lengths_(int* IN distances, long* IN tour, long* OUT tourlength, 
#if ANTS_AGING
										   int* OUT antperformance, const unsigned char* IN validants, 
#endif
										   unsigned int cities, unsigned int n) {

#if ANTS_AGING
    if(validants[blockIdx.x] == FALSE)
		return;
#endif

	long* sdata = SharedMemory<long>();


    // load shared mem
    unsigned int tid = threadIdx.x;
    //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = blockIdx.x*cities;
	unsigned int i = k + tid;

    sdata[tid] = (tid < cities-1) ? //g_idata[i] : 0;
		distances[tour[i]*cities+tour[i+1]] : ((tid == cities-1)? distances[tour[i]*cities+tour[k]] :  0);

	 //distances[tour[k*numCities+i]*numCities+tour[k*numCities+i+1]];
	 //distances[tour[k*numCities+numCities-1]*numCities+tour[k*numCities]];

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
#if ANTS_AGING
		antperformance[blockIdx.x] = tourlength[blockIdx.x] == sdata[0] ? antperformance[blockIdx.x]+1 : 0;
#endif
		tourlength[blockIdx.x] = sdata[0];
	}
}

__global__ void dev_calculate_tour_lengths(int* IN distances, long* IN tour, long* OUT tourlength, 
#if ANTS_AGING
										   int* OUT antperformance, const unsigned char* IN validants, 
#endif
										   unsigned int blocks, unsigned int cities, unsigned int n) {

#if ANTS_AGING
    if(validants[blockIdx.x] == FALSE)
		return;
#endif

	long* sdata = SharedMemory<long>();

	 // load shared mem
    unsigned int tid = threadIdx.x;
    //unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int i = blockIdx.x*cities + threadIdx.x;
	unsigned int bid = blockIdx.x % blocks;
	unsigned int c = bid * blockDim.x + tid;
	unsigned int b = blockIdx.x / blocks;
	unsigned int i = b * cities + c;

    //sdata[tid] = (tid < cities && i < n) ? prob[i] : 0;
	sdata[tid] = (c < cities-1) ? 
		distances[tour[i]*cities+tour[i+1]] : ((c == cities-1)? distances[tour[i]*cities+tour[b * cities]] :  0);
	


    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
#if ANTS_AGING
		antperformance[blockIdx.x] = tourlength[blockIdx.x] == sdata[0] ? antperformance[blockIdx.x]+1 : 0;
#endif
		//tourlength[blockIdx.x] = sdata[0];
		atomicAdd((unsigned long long int*)(&(tourlength[b])), (unsigned long long int)sdata[0]);
	}
}

__global__ void dev_compute_best_tour(const long* IN tour, 
#if ANTS_AGING
									  const unsigned char* IN validants, 
#endif
									  long* OUT btour, int* OUT bindex,  unsigned int ants, int threadsPerBlock2) {

	long* sdata = SharedMemory<long>();
	//int* sindex = SharedMemory<int>();
	int* sindex = (int*)(&sdata[threadsPerBlock2]);

    // load shared mem
    unsigned int tid = threadIdx.x;
    //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(tid < ants 
#if ANTS_AGING
		&& validants[tid]
#endif
	) {
		sdata[tid] = tour[tid];	
		sindex[tid] = tid;
	}
	else {
		sdata[tid] = MAX_DISTANCE;
		sindex[tid] = -1;
	}


    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            if(sdata[tid] > sdata[tid + s]) {
				sdata[tid] = sdata[tid + s];
				sindex[tid] = sindex[tid + s];
			}
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {		
		btour[blockIdx.x] = sdata[0];
		bindex[blockIdx.x] = sindex[0];
		//bindex[0] = 0; 
	}
}

#if ANTS_AGING
__global__ void dev_init_ant_ages(int* OUT antage, unsigned char* OUT validants, int* OUT antperformance, 
								  const int initialpopulation, const int ants) {

	int k = blockDim.x * blockIdx.x + threadIdx.x; // current thread

	if(k < ants) {
		antage[k] = 0;
		validants[k] = k < initialpopulation ? TRUE : FALSE;
		antperformance[k] = 0;

		if(k==0)
			numValidAnts = initialpopulation;
	}

	
}

__global__ void dev_update_ant_ages(int* IN_OUT antage, const int ants) {

	int k = blockDim.x * blockIdx.x + threadIdx.x; // current thread

	if(k < ants) {
		antage[k] = antage[k]+1;
	}
}

__global__ void dev_retire_ants(int* IN antage, unsigned char* IN validants, int* IN antperformance, const int ants, 
								const int minRetirementAge, const int maxRetirementAge, const int localPeformance) {

	__shared__ unsigned int rant;
	int i = blockDim.x * blockIdx.x + threadIdx.x; // current thread
	if(i < ants) {

		if (threadIdx.x == 0) {
			rant = MAX_INT;						
		}
		__syncthreads();

		if(validants[i] && (antage[i] >= maxRetirementAge 
#if 0
			|| (antage[i] >= minRetirementAge && antperformance[i] >= localPeformance)
#endif
			)) {
			atomicMin(&rant, (unsigned int)i);
		}

		__syncthreads();
		if (threadIdx.x == 0) {			
			if(rant != MAX_INT) {
				validants[rant] = FALSE;
				atomicDec(&numValidAnts, MAX_DISTANCE);	
			}
		}
	}

#if 0
	int i = blockDim.x * blockIdx.x + threadIdx.x; // current thread

	if(i < ants) {
		if(validants[i] && (antage[i] >= maxRetirementAge || 
			(antage[i] >= minRetirementAge && antperformance[i] >= localPeformance))) {
				validants[i] = FALSE;
				atomicDec(&numValidAnts, MAX_DISTANCE);	
				//numValidAnts--;
				//isconverged = TRUE;
		}
	}
#endif
}

__global__ void dev_recruite_ants(int* IN antage, unsigned char* IN validants, const int ants, 
								const int plimit) {

	int i = blockDim.x * blockIdx.x + threadIdx.x; // current thread

	if(i < ants) {
		if(validants[i] == FALSE) {
			validants[i] = TRUE;
			antage[i] = 0;
			atomicInc(&numValidAnts, MAX_DISTANCE);
			//numValidAnts++;			
		}
	}
#if 0
	int i = blockDim.x * blockIdx.x + threadIdx.x; // current thread

	if(i < ants) {
		if(validants[i] == FALSE && numValidAnts < plimit) {
			validants[i] = TRUE;
			antage[i] = 0;
			atomicInc(&numValidAnts, MAX_DISTANCE);
			//numValidAnts++;			
		}
	}
#endif
}

__global__ void dev_copy_num_valid_ants(unsigned int* OUT dnva) {
	*dnva = numValidAnts;			
}

#endif // ANTS_AGING

#endif // __GPU__

#endif // COLONY_CUH