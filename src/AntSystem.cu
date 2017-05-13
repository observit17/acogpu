

#include "AntSystem.h"
#include "default.h"

#include "dump.h"
const static char ANT_SYSTEM_NAME[] = "Ant System";

__DEVICE__ __HOST__ float adj_pheromone(float pheromone) {
	return pheromone > MIN_PHEROMONE ? pheromone : MIN_PHEROMONE;
}
#if _GPU_


__device__ float dev_atomic_add_f2(float* address, float val) { 
	unsigned int* addr = (unsigned int*)address; 
	unsigned int old = *addr, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(addr, assumed, __float_as_int(val + __int_as_float(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __int_as_float(old); 
}

__global__ void dev_evaporate_pheromones(float *pheromones, float rho, unsigned int n) {
    
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < n)
		pheromones[i] = adj_pheromone((1 - rho) * pheromones[i]);    
}


#if 0
// at cpu
//__global__ 
void dev_update_pheromones(float *dpheromones, long* dtour, 
#if ANTS_AGING
							const unsigned char* dvalidants,
#endif
						   unsigned int ants, unsigned int cities, long* tourlength)
{
	float* pheromones = new float[cities*cities];
	long* tour = new long[ants*cities];
	

	checkCudaErrors(cudaMemcpy(pheromones, dpheromones, cities*cities*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(tour, dtour, ants*cities*sizeof(long), cudaMemcpyDeviceToHost));
#if ANTS_AGING
	unsigned char* validants = new unsigned char[ants];
	checkCudaErrors(cudaMemcpy(validants, dvalidants, ants*sizeof(unsigned char), cudaMemcpyDeviceToHost));
#endif

    for (int k = 0 ; k < ants; k++ ) {
#if ANTS_AGING
		if(validants[k]) {
#endif
		float d_tau = 1.0 / (float) tourlength[k];
		for (int i = 0 ; i < cities-1; i++ ) {
			int j = tour[k*cities+i];
			int h = tour[k*cities+i+1];
			pheromones[j*cities+h] += d_tau;
			pheromones[h*cities+j] = pheromones[j*cities+h];
		}
		int j = tour[k*cities+cities-1];
		int h = tour[k*cities];
		pheromones[j*cities+h] += d_tau;
		pheromones[h*cities+j] = pheromones[j*cities+h];

#if ANTS_AGING
		}
#endif
	}

	checkCudaErrors(cudaMemcpy(dpheromones, pheromones, cities*cities*sizeof(float), cudaMemcpyHostToDevice));

#if ANTS_AGING
	delete[] validants;
#endif
	delete[] pheromones;
	delete[] tour;
}

#endif

#if 1
__global__ void dev_compute_tau(const long* tourlength, 
#if ANTS_AGING
							const unsigned char* validants,
#endif
								float* tau, unsigned int ants) {

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < ants 
#if ANTS_AGING
		&& validants[i]
#endif
	)
		tau[i] = 1.0f / (float) tourlength[i];  
}

// atomic operation
__global__ 
void dev_update_pheromones(const long* tour, const float* tau, float *pheromones, 
#if ANTS_AGING
							const unsigned char* validants,
#endif
							unsigned int ants, unsigned int cities, const int numElements)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < numElements) {
		int k = int(i/cities);
#if ANTS_AGING		
		if(validants[k]) {
#endif
		int j = tour[i];
		int h = ((i+1) % cities) == 0 ? tour[i+1-cities] : tour[i+1];
		
		dev_atomic_add_f2(&(pheromones[j*cities+h]), tau[k]);
		//pheromones[j*cities+h] += tau[i/cities];
#if ANTS_AGING
		}
#endif
	}	
}

#endif

#if 0
// with tourmap
__global__ 
void dev_update_pheromones(int* tourmap, const long* tourlength, float *pheromones,

							unsigned int ants, unsigned int cities)
{
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(j < cities) {
		int* tm = &(tourmap[j*ants]);
		for(int a = 0; a < ants; a++) {
			if(tm[a] > -1) {
				int h = tm[a]; 
				pheromones[j*cities+h] = pheromones[h*cities+j] = 
					 pheromones[j*cities+h]+ (1.0f / (float) tourlength[a]);
					//pheromones[j*cities+h]+tau[a];
			}
		}
		
	}	
}

#endif

#endif

//constructor: Allocates memory and sets defaults.
AntSystem::AntSystem(int* newDistances, int newNumCities, int newNumAnts, float alpha, float beta, float rho)
  : Colony(newDistances, newNumCities, newNumAnts) {
	setAlpha(alpha);
	setBeta(beta);
	setRho(rho);
}

#if ANTS_AGING
AntSystem::AntSystem(int* newDistances, int newNumCities, int newNumAnts, float alpha, float beta, float rho,
	  int minRetirementAge, int maxRetirementAge, int localPeformance, int globalPerformance, int minPopulationLimit, int maxPopulationLimit) 
	  : Colony(newDistances, newNumCities, newNumAnts) {
		setAlpha(alpha);
		setBeta(beta);
		setRho(rho);

		setMinRetirementAge(minRetirementAge);
		setMaxRetirementAge(maxRetirementAge);
		setLocalPeformance(localPeformance);
		setGlobalPerformance(globalPerformance);
		setMinPopulationLimit(minPopulationLimit);
		setMaxPopulationLimit(maxPopulationLimit);
}
#endif

AntSystem::~AntSystem() {

}

const char* AntSystem::antSystemName() {
	return ANT_SYSTEM_NAME;
}


//computeParameters: Simply computes neccesary parameters.
void AntSystem::computeParameters()
{
  computeInitialPheromone();
}

//computeInitialPheromone: Computes the initial pheromone level with the formula described by Marco Dorigo.
void AntSystem::computeInitialPheromone()
{
  initialPheromone = DEFAULT_PHEROMONE;
}


void AntSystem::evaporatePheromones() {

#if _GPU_
	int threadsPerBlock1 = THREADS_PER_BLOCK;
    int blocksPerGrid1 = ((numCities*numCities) + threadsPerBlock1 - 1) / threadsPerBlock1;

	dev_evaporate_pheromones<<<blocksPerGrid1, threadsPerBlock1>>>(pheromones, rho, numCities*numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

#else

	for (int i = 0 ; i < numCities; i++ ) {
		for (int j = 0 ; j <= i ; j++ ) {
			pheromones[i*numCities+j] = adj_pheromone((1 - rho) * pheromones[i*numCities+j]);
			pheromones[j*numCities+i] = pheromones[i*numCities+j];
		}
    }
#endif
}

template <typename T> 
__global__ 	void dev_init_data_f(T* data, const T f, unsigned int n) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<n) {
		data[i] = f;
	}
}
//updataPheromones: Evaporates, then the ants lay pheromone, judged by the distances of their tours.
void AntSystem::updatePheromones() {	

#if _GPU_

#if 0	
	// pre computed cities map for pheromone
	int threadsPerBlock1 = THREADS_PER_BLOCK;
	int blocksPerGrid1 = (numCities + threadsPerBlock1 - 1) / threadsPerBlock1;	


	dev_update_pheromones<<<blocksPerGrid1, threadsPerBlock1>>>(tourmap, dtourlength, pheromones, 
							numAnts, numCities);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

#endif

#if 1	
	// atomic phromone update
	float* tau = 0;
	checkCudaErrors(cudaMalloc((void **)&tau, numAnts*sizeof(float)));

	int threadsPerBlock1 = THREADS_PER_BLOCK;
	int blocksPerGrid1 = (numAnts + threadsPerBlock1 - 1) / threadsPerBlock1;	

	
	dev_compute_tau<<<blocksPerGrid1, threadsPerBlock1>>>(dtourlength, 
#if ANTS_AGING
							validants,
#endif
							tau, numAnts);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());


	blocksPerGrid1 = ((numAnts*numCities) + threadsPerBlock1 - 1) / threadsPerBlock1;	


	dev_update_pheromones<<<blocksPerGrid1, threadsPerBlock1>>>(tour, tau, pheromones, 
#if ANTS_AGING
							validants,
#endif
							numAnts, numCities, numAnts*numCities);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

	cudaFree(tau);
#endif
#if 0
	// at cpu
	dev_update_pheromones(pheromones, tour, 
#if ANTS_AGING
								validants,
#endif
								numAnts, numCities, tourlength
								);
#endif
#else
	for (int k = 0 ; k < numAnts; k++ ) {
#if ANTS_AGING
		if(validants[k]) {
#endif
			float d_tau = 1.0 / (float) tourlength[k];
	
			for (int i = 0 ; i < numCities-1; i++ ) {
				int j = tour[k*numCities+i];
				int h = tour[k*numCities+i+1];
				pheromones[j*numCities+h] += d_tau;
				pheromones[h*numCities+j] = pheromones[j*numCities+h];
			}
			int j = tour[k*numCities+numCities-1];
			int h = tour[k*numCities];
			pheromones[j*numCities+h] += d_tau;
			pheromones[h*numCities+j] = pheromones[j*numCities+h];
#if ANTS_AGING
		}
#endif

	}
#endif
}
