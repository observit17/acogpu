


#include <assert.h>
#include <iostream>
#include <limits>
#include "Colony.cuh"
#include "default.h"
#include "Timer.h"
#include "Statistics.h"
#include "Utils.h"

#include "dump.h"




/**
 *	Constructor: Sets defaults and allocates memory.
 *
 *	distances - the distance vector among cities in the TSP problem
 *	cities - the number of cities in the TSP problem
 *	ants - the number of ants to solve the TSP problems
 */
Colony::Colony(int* distances, int cities, int ants) {

	DEBUG( assert ( distances != 0 ); );
	DEBUG( assert ( cities > 0 ); );
	DEBUG( assert ( ants > 0 ); );

	/// set default values
	alpha = DEFAULT_ALPHA;	//< alpha set to 1.0
	beta = DEFAULT_BETA;	//< beta set to 2.0 
	rho = DEFAULT_RHO;	//< evaporation factor set to 0.5
	reps = 0;	//< number of of repetitions
	initialPheromone = DEFAULT_PHEROMONE;

	iterBestDist = MAX_DISTANCE -1;// std::numeric_limits<float>::max() - 1;
	globBestDist = MAX_DISTANCE;// std::numeric_limits<float>::max();  

	/// assign distances from host vector to device vector
	this->distances = distances;

	/// set number of cities from the provided number
	this->numCities = cities;

	/// set number of ants
	this->numAnts = ants;

	


#if _GPU_
	checkCudaErrors(cudaMalloc((void **)&probabilities, numCities*numCities*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&pheromones, numCities*numCities*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&tour, numAnts*numCities*sizeof(long)));
	//checkCudaErrors(cudaMalloc((void **)&gbtour, numCities*sizeof(long)));
	checkCudaErrors(cudaMalloc((void **)&visited, numAnts*numCities*sizeof(unsigned char)));
	//checkCudaErrors(cudaMalloc((void **)&tourlength, numAnts*sizeof(long)));
	checkCudaErrors(cudaMalloc((void **)&nnlist, numCities*numCities*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&dtourlength, numAnts*sizeof(long)));

	checkCudaErrors(cudaMalloc ((void**) &devStates, numAnts*sizeof( curandState ) ));
	
	checkCudaErrors(cudaMalloc((void **)&hurestics, numCities*numCities*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&tourmap, numAnts*numCities*sizeof(int)));
	

#if ANTS_AGING
	checkCudaErrors(cudaMalloc((void **)&antage, numAnts*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&antperformance, numAnts*sizeof(int)));

	checkCudaErrors(cudaMalloc((void **)&validants, numAnts*sizeof(unsigned char)));

	checkCudaErrors(cudaMalloc((void **)&pheromone_mean, numCities*sizeof(float)));
	
#endif

#else

	/// the ant can visit any connected city, idealy, all cities are connected
	/// so an ant can move to n-1 cities from each current city.
	/// as a whole there are n^2 or n*n probabilities for all number of verticies
	this->probabilities = new float[numCities*numCities];

	/// each edge must have a phoremone value
	this->pheromones = new float[numCities*numCities];

	this->tour = new long[numAnts*numCities];

	//this->gbtour = new long[numCities];

	this->visited = new unsigned char[numAnts*numCities];

	//this->tourlength = new long[numAnts];
	
	this->nnlist = new int[numCities*numCities];

	this->hurestics = new float[numCities*numCities];

#if ANTS_AGING
	this->antage = new int[numAnts];

	this->antperformance = new int[numAnts];
	
	this->validants = new unsigned char[numAnts];

	this->pheromone_mean = new float[numCities];
#endif

#endif
	
	this->tourlength = new long[numAnts];
	this->gbtour = new long[numCities];

	//srand(time(NULL));
	

#if ANTS_AGING
	/*
	/// ants aging and retirements
	minretage = DEFAULT_MIN_RITIREMENT_AGE;
	maxretage = DEFAULT_MAX_RITIREMENT_AGE;
	localnoconvergence = DEFAULT_LOCAL_NO_CONVERGENCE;
	globalnoconvergence = DEFAULT_GLOBAL_NO_CONVERGENCE;
	minpopulation = (int)((float)numAnts*0.3f);//DEFAULT_MIN_POPULATION_LIMIT;
	maxpopulation = numAnts;//DEFAULT_MAX_POPULATION_LIMIT;
	*/

	//this->intialpopulation = this->numAnts/2;
	this->intialpopulation = this->numAnts;
	//numValidAnts = this->intialpopulation;
#endif

	
}

Colony::~Colony() {

#if _GPU_
	cudaFree(probabilities);
	cudaFree(pheromones);
	cudaFree(tour);
	//cudaFree(gbtour);
	cudaFree(visited);
	//cudaFree(tourlength);
	cudaFree(nnlist);
	cudaFree(dtourlength);
	cudaFree(devStates);
	cudaFree(hurestics);
	cudaFree(tourmap);
#if ANTS_AGING
	cudaFree(antage);
	cudaFree(antperformance);
	cudaFree(validants);
	cudaFree(pheromone_mean);
#endif

#else
	delete[] this->probabilities;
	delete[] this->pheromones;
	delete[] this->tour;
	//delete[] this->gbtour;
	delete[] this->visited;
	//delete[] this->tourlength;
	delete[] this->nnlist;
	delete[] this->hurestics;
#if ANTS_AGING
	delete[] this->antage;
	delete[] this->antperformance;
	delete[] this->validants;
	delete[] this->pheromone_mean;
#endif

#endif
	
	delete[] this->tourlength;
	delete[] this->gbtour;
}



//initialize: Initializes data, creates maps and keys, performs standard ACO initialization steps etc. 
void Colony::initialize() {
 
  computeParameters();
  
  initPheromone(initialPheromone);

  computeHurestics();

  //computeProbabilities();

  computeNNlist();

#if ANTS_AGING
  initAntAges();
	isconverged = FALSE;
#endif

	initTourLengths();

  
#if _GPU_
	seed = (long) time(NULL);
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = (numAnts + threadsPerBlock - 1) / threadsPerBlock;
	dev_init_seeds<<<blocksPerGrid, threadsPerBlock>>>(devStates, numAnts, seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#endif
	
}

void Colony::initPheromone(const float pher) {

#if _GPU_
	int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ((numCities*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	dev_init_pheromone<<<blocksPerGrid, threadsPerBlock>>>(pheromones, numCities*numCities, pher);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#else
	for(int i =0; i < numCities*numCities; i++)
		pheromones[i] = pher;
#endif
}



void Colony::computeHurestics() {

#if _GPU_
	int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ((numCities*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	dev_compute_hurestics<<<blocksPerGrid, threadsPerBlock>>>(distances, hurestics, numCities*numCities, beta);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#else

	for(int i =0; i < numCities*numCities; i++)
		hurestics[i] = HURESTIC(distances[i], beta);

#endif
}




//computeProbabilities: Computes the probabilities from the distances and pheromones.
void Colony::computeProbabilities() {

#if _GPU_
	int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ((numCities*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	dev_compute_selection<<<blocksPerGrid, threadsPerBlock>>>(pheromones, hurestics, probabilities, alpha, numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#else
	for (int i = 0 ; i < numCities ; i++ ) {
		for (int j = 0 ; j < i ; j++ ) {
			
			probabilities[i*numCities + j] = pow(pheromones[i*numCities+j], alpha) * hurestics[i*numCities+j];
				//powf((1.0 / ((float) distances[i*numCities+j])), beta);
			probabilities[j*numCities+i] = probabilities[i*numCities+j];
		}
		probabilities[i*numCities + i] = 0;
    }
	probabilities[numCities*numCities -1] = 0;
#endif
	
}


void Colony::computeNNlist() {

#if _GPU_
	
	int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ((numCities*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	
	dev_compute_nnlist<<<blocksPerGrid, threadsPerBlock>>>(distances, nnlist, numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
	
#else 
	int* distance_vector;
    int* help_vector;

	distance_vector = new int[numCities];
    help_vector = new int[numCities];

    for (int node = 0 ; node < numCities ; node++ ) { 
		
		for (int i = 0; i < numCities; i++ ) {  /* Copy distances from nodes to the others */
			distance_vector[i] = distances[node*numCities+i];
			help_vector[i] = i;
		}
		
		Utils::sort2(distance_vector, help_vector, 0, numCities-1);

	
		for (int i = 0 ; i < numCities; i++ ) {
			nnlist[node*numCities+i] = help_vector[i];
		}
    }
    delete[] distance_vector;
    delete[] help_vector;
#endif	
}



void Colony::initTourLengths() {
	for (int i = 0 ; i < numAnts ; i++ ) {
		tourlength[i] = 0;		
	}
}


template<typename T>
T copyFromDevice(T* dv) {
	T v;
	checkCudaErrors(cudaMemcpy(&v, dv, 1*sizeof(T), cudaMemcpyDeviceToHost));
	return v;
}

#if ANTS_AGING
void Colony::initAntAges() {	

#if _GPU_
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = (numAnts + threadsPerBlock - 1) / threadsPerBlock;
	dev_init_ant_ages<<<blocksPerGrid, threadsPerBlock>>>(antage, validants, antperformance, intialpopulation, numAnts);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#else
	numValidAnts = this->intialpopulation;	
	for (int i = 0 ; i < intialpopulation ; i++ ) {
		antage[i] = 0;
		validants[i] = TRUE;
		antperformance[i] = 0;
	}
	for (int i = intialpopulation ; i < numAnts ; i++ ) {
		antage[i] = 0;
		validants[i] = FALSE;
		antperformance[i] = 0;
	}
#endif
}

void Colony::updateAntAges() {
#if _GPU_
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = (numAnts + threadsPerBlock - 1) / threadsPerBlock;
	dev_update_ant_ages<<<blocksPerGrid, threadsPerBlock>>>(antage, numAnts);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#else
	for (int i = 0 ; i < numAnts ; i++ )
		antage[i] = antage[i]+1;
#endif
}

void Colony::retireAnts() {

#if _GPU_
	unsigned int va = getNumValidAnts();
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = ((numAnts/2) + threadsPerBlock - 1) / threadsPerBlock;
	dev_retire_ants<<<blocksPerGrid, threadsPerBlock>>>(antage, validants, antperformance, numAnts/2, minRetirementAge, maxRetirementAge, localPeformance);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

	if(va != getNumValidAnts())
		isconverged = TRUE;
#else
	for (int i = 0 ; i < numAnts/2 ; i++ ) {
		//if(numValidAnts <= minpopulation)
		//	break;

		if(validants[i] && (antage[i] >= maxRetirementAge 
#if 0
			|| (antage[i] >= minRetirementAge && antperformance[i] >= localPeformance)
#endif
			)) {
				validants[i] = FALSE;
				numValidAnts--;
				isconverged = TRUE;
				break;
		}
	}
#endif
}



void Colony::recruiteAnts() {

	unsigned int va = getNumValidAnts();
	if(globalperformance >= globalPerformance || va < minPopulationLimit) {
		int plimit = minPopulationLimit + ((maxPopulationLimit - minPopulationLimit)/2);
#if _GPU_	
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = (numAnts + threadsPerBlock - 1) / threadsPerBlock;
	dev_recruite_ants<<<blocksPerGrid, threadsPerBlock>>>(antage, validants, numAnts, plimit);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());	
#else
		for (int i = 0 ; i < numAnts; i++ ) {
			if(validants[i] == FALSE) {
				validants[i] = TRUE;
				antage[i] = 0;
				numValidAnts++;
				//if(numValidAnts >= plimit)
					break;
			}
		}
#endif	
	}	

}



void Colony::computePheromoneMean() {
#if _GPU_
	const int threadsPerBlock2 = nextPow2(numCities);
    int blocksPerGrid2 = numCities;
	int smemSize2 = (threadsPerBlock2 <= 32) ? 2 * threadsPerBlock2 * sizeof(float) : threadsPerBlock2 * sizeof(float);
	dev_compute_pheromone_mean<<<blocksPerGrid2, threadsPerBlock2, smemSize2>>>(pheromones, pheromone_mean, numCities, numCities*numCities);
#else

	for (int i = 0 ; i < numCities; i++ ) {	
		float sum = 0;
		for (int j = 0 ; j < numCities; j++ ) {		
			sum += pheromones[i*numCities+j];		
		}
		pheromone_mean[i] = sum/numCities;
	}
#endif	
}

#endif // ANTS_AGING

unsigned int Colony::getNumValidAnts() {
#if ANTS_AGING
	
#if _GPU_
	unsigned int nva = 0;	
	unsigned int* dnva;
	checkCudaErrors(cudaMalloc((void **)&dnva, 1*sizeof(unsigned int)));
	dev_copy_num_valid_ants<<<1,1>>> (dnva);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
	checkCudaErrors(cudaMemcpy(&nva, dnva, 1*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaFree(dnva);
	return nva;
#else
	return numValidAnts;
#endif

#else // !ANTS_AGING
	return numAnts;
#endif // ANTS_AGING
}

void Colony::initTour() {

	seed = (long) time(NULL);

#if ANTS_AGING
	computePheromoneMean();
#endif
	
#if _GPU_

	int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ((numAnts*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	
	dev_clear_visits<<<blocksPerGrid, threadsPerBlock>>>(visited, tour, numAnts*numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

	blocksPerGrid = (numAnts + threadsPerBlock - 1) / threadsPerBlock;
	dev_place_ants<<<blocksPerGrid, threadsPerBlock>>>(devStates, tour, visited, dtourlength,
#if ANTS_AGING
														validants, 
#endif
														numAnts, numCities);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

 
    blocksPerGrid = ((numAnts*numCities) + threadsPerBlock - 1) / threadsPerBlock;
	
	dev_init_data<int><<<blocksPerGrid, threadsPerBlock>>>(tourmap, -1, numAnts*numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

	
#else

	/// clear ant visits	
	for (int i = 0 ; i < numAnts*numCities ; i++ ) {
		visited[i] = FALSE;
		tour[i] = 0;
	}
	

	/// place ants at random initial city
	 for (int k = 0 ; k < numAnts ; k++ ) {
#if ANTS_AGING
		 if(validants[k]) {
#endif
			long rnd = (long) (Utils::random( &seed ) * (float) numCities); /* random number between 0 .. n-1 */
			tour[k*numCities] = rnd; 
			visited[(k*numCities)+rnd] = TRUE;	
#if ANTS_AGING
		 }
#endif
	 }

#endif
}





/**
 * constructSolution: Main ACO loop. Performs the solution constructruction step, then updates distances, pheromones, probabilities.
 * 
 */
void Colony::constructSolution(float timers[]) {
	// 1- init tour
	// 2- tour
	// 3- compute ant distances
	// 4- evaporate pheromones
	// 5- update pheromones
	// 6- compute probabilities

	Timer timer;
	timer.create();

	
	timer.start();
	initTour();
#if ANTS_AGING
	updateAntAges();
#endif
	timer.stop();
	timers[TIME_INIT_TOUR] = timer.elapsed();

#if !ANTS_AGING
	timer.start();
	computeProbabilities();
	timer.stop();
	timers[TIME_COMPUTE_PROBABILITIES] = timer.elapsed();
#endif
	
	timer.start();
	constructPath();
	timer.stop();
	timers[TIME_TOUR] = timer.elapsed();

	timer.start();
	computeAntDistances();
	timer.stop();
	timers[TIME_COMPUTE_ANT_DISTANCE] = timer.elapsed();

	timer.start();
	evaporatePheromones();
	timer.stop();
	timers[TIME_EVAPORATE_PHEROMONE] = timer.elapsed();

	timer.start();
	updatePheromones();
	timer.stop();
	timers[TIME_UPDATE_PHEROMONE] = timer.elapsed();

#if ANTS_AGING
	timer.start();	
	retireAnts();
	recruiteAnts();
	timer.stop();
	timers[TIME_COMPUTE_PROBABILITIES] = timer.elapsed();
#endif	

	timer.erase();

}

void verifySUM(int* prob_ptr, QWORD* sumprob_ptr, const int numAnts, const int numCities) {

	int* hprob_ptr = new int[numAnts*numCities];
	QWORD* hsumprob_ptr = new QWORD[numAnts];
	checkCudaErrors(cudaMemcpy(hprob_ptr, prob_ptr, numAnts*numCities*sizeof(int), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(hsumprob_ptr, sumprob_ptr, numAnts*sizeof(QWORD), cudaMemcpyDeviceToHost));
	

	for(int k =0; k< numAnts; k++) {
		QWORD spbr = 0;
		for(int c=0; c< numCities; c++) {
			spbr += hprob_ptr[k*numCities+c];
		}
		hsumprob_ptr[k] = spbr;
		//if(spbr != hsumprob_ptr[k]) {
		//	printf("ant: %d\tsumprob_ptr: %.32f\tspbr: %.32f\n", k, hsumprob_ptr[k], spbr);
		//}
	}

	dump2DArray<QWORD>("output/prob_ptr.txt", "sumprob_ptr", sumprob_ptr, 1, numAnts, false);
	dump2DArray<QWORD>("output/prob_ptr.txt", "hsumprob_ptr", hsumprob_ptr, 1, numAnts, true);

	delete[] hprob_ptr;
	delete[] hsumprob_ptr;
}

__global__ 	void dev_reduce_probabilities_xs(float* IN prob, float* OUT sprob, 
#if ANTS_AGING
										  const unsigned char* IN validants, 
#endif
										  unsigned int blocks, unsigned int ants, unsigned int cities, unsigned int n) {

	unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
	if( k < ants ) {
#if ANTS_AGING
	if(validants[k] == FALSE)
		return;
#endif
	
		float* kprob = &(prob[k*cities]);
		sprob[k] = 0;
		for(int c=0; c< cities; c++) {
			sprob[k] += kprob[c];
		}
	}
}

__global__ 	void dev_reduce_probabilities_s(int* IN prob, QWORD* OUT sprob, 
#if ANTS_AGING
										  const unsigned char* IN validants, 
#endif
										  unsigned int blocks, unsigned int ants, unsigned int cities, unsigned int n) {

	QWORD* sdata = SharedMemory<QWORD>();
	//unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int k = blockIdx.x;

	if( k < ants ) {
#if ANTS_AGING
		if(validants[k] == FALSE)
			return;
#endif
		int* kprob = &(prob[k*cities]);
		if(tid < cities) {
			sdata[tid] = kprob[tid];
			for(int b=tid+blockDim.x; b< blocks*blockDim.x; b+=blockDim.x) {
				sdata[tid] += (b < cities ? kprob[b] : 0);
			}
		}
		else {
			sdata[tid] = 0;
		}
		
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
			sprob[k] = sdata[0];			
		}
	}
}



void Colony::constructPath() {

#if _GPU_
	int* prob_ptr=0;
	
	QWORD* sumprob_ptr=0;

	checkCudaErrors(cudaMalloc((void **)&prob_ptr, numAnts*numCities*sizeof(int)));	
	checkCudaErrors(cudaMalloc((void **)&sumprob_ptr, numAnts*sizeof(QWORD)));


	int threadsPerBlock1 = THREADS_PER_BLOCK;
    int blocksPerGrid1 = ((numAnts*numCities) + threadsPerBlock1 - 1) / threadsPerBlock1;
		
	int threadsPerBlock2 = THREADS_PER_BLOCK;	
	int blocks = (numCities + threadsPerBlock2 - 1) / threadsPerBlock2;	
	int smemSize = (threadsPerBlock2 <= 32) ? 2 * threadsPerBlock2 * sizeof(QWORD) : threadsPerBlock2 * sizeof(QWORD);

	int threadsPerBlock3 = THREADS_PER_BLOCK;
    int blocksPerGrid3 = (numAnts + threadsPerBlock3 - 1) / threadsPerBlock3;
	
	 QWORD* info = 0;
	 checkCudaErrors(cudaMalloc((void **)&info, 5*numAnts*sizeof(QWORD)));
	//int f = 1;
	for(int phase = 1; phase < numCities; phase++) {

		/// - compute probablities of selecting cities for each ant
		dev_compute_probabilities<<<blocksPerGrid1, threadsPerBlock1>>>(prob_ptr, tour, visited, probabilities, nnlist, 
										pheromones, hurestics, 
#if ANTS_AGING
										pheromone_mean, validants, antage, minRetirementAge, isconverged,
#endif
										phase, numAnts, numCities, alpha);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaSynchronize());


		dev_reduce_probabilities_s<<<numAnts, threadsPerBlock3, smemSize>>>(prob_ptr, sumprob_ptr,
#if ANTS_AGING
										validants, 
#endif
										blocks, numAnts, numCities, numAnts*numCities);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaSynchronize());
		

		/// move ants to new cities
		dev_move_ants<<<blocksPerGrid3, threadsPerBlock3>>>(phase, numAnts, numCities, prob_ptr, devStates, tour, visited, sumprob_ptr, nnlist,
#if ANTS_AGING
															validants,
#endif
															distances, dtourlength, tourmap, info);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaSynchronize());

		//dump2DArray<int>("output/prob_ptr.txt", "prob_ptr", prob_ptr, numAnts, numCities, false);
		//dump2DArray<QWORD>("output/prob_ptr.txt", "info", info, numAnts, 5, false);

#if 0	
		
		if(f) {	
			verifySUM(prob_ptr, sumprob_ptr, numAnts, numCities);
			f = 0;
		}

#endif
	}

	//dump2DArray<int>("output/prob_ptr.txt", "prob_ptr", prob_ptr, numAnts, numCities, false);
	//dump2DArray<QWORD>("output/prob_ptr.txt", "sumprob_ptr", sumprob_ptr, 1, numAnts, false);
	//dump2DArray<long>("output/prob_ptr.txt", "tour", tour, numAnts, numCities, false);
	checkCudaErrors(cudaFree(info));	
	checkCudaErrors(cudaFree(prob_ptr));	
	checkCudaErrors(cudaFree(sumprob_ptr));


#else
	float* prob_ptr = new float[numCities];
	for(int x = 1; x < numCities; x++)    {		
		for(int k = 0; k < numAnts; k++) {
#if ANTS_AGING
			if(validants[k]) {
#endif
				updateAntVisits(x);

				float sum_prob = calculateProbabilityOfSelection(prob_ptr, k, x);

				if (sum_prob <= 0.0) {		
					chooseBestNext(k, x);
				}     
				else {
					selectCities(prob_ptr, k, x, sum_prob);
				}
#if ANTS_AGING
			}
#endif
		}

	}
	delete[] prob_ptr;
#endif
}

void Colony::updateAntVisits(int x) {
	
}
float Colony::calculateProbabilityOfSelection(float* prob_ptr, int k, int phase) {
	float sum_prob = 0;
	

	int current_city = tour[k*numCities+phase-1]; /* current_city city of ant k */
    DEBUG( assert ( current_city >= 0 && current_city < numCities ); )
	for (int i = 0 ; i < numCities; i++ ) {
		
		if ( visited[k*numCities+nnlist[current_city*numCities+i]] ) 
			prob_ptr[i] = 0.0;   /* city already visited */
		else {
			DEBUG( assert ( nnlist[current_city*numCities+i] >= 0 && nnlist[current_city*numCities+i] < numCities ); )
			int xy = current_city*numCities+nnlist[current_city*numCities+i];
			//prob_ptr[i] = probabilities[current_city*numCities+nnlist[current_city*numCities+i]];
			prob_ptr[i] = compute_fittness(pheromones[xy], alpha, hurestics[xy]
#if ANTS_AGING
			, pheromone_mean[current_city], antage[k], minRetirementAge, isconverged
#endif
				);
			sum_prob += prob_ptr[i];
		} 
    }
	return sum_prob;
}


void Colony::selectCities(float* prob_ptr, int k, int phase, float sum_prob) {
	/* at least one neighbor is eligible, chose one according to the
		   selection probabilities */
	int current_city = tour[k*numCities+phase-1]; /* current_city city of ant k */

	float rnd = (float)Utils::random( &seed );
	
	
	rnd *= sum_prob;
	int i = 0;
	float partial_sum = prob_ptr[i];
	while ( partial_sum <= rnd) {
		i++;
		if(i >= numCities) {
			printf("probability issue %d\n", k);
			// select local best
			i = select_local_best(&(nnlist[current_city*numCities]), &(visited[k*numCities]), numCities);
			break;			
		}
		partial_sum += prob_ptr[i]; 
	}
		
	DEBUG( assert ( 0 <= i && i < numCities); )
	DEBUG( assert ( prob_ptr[i] >= 0.0); )
	int help = nnlist[current_city*numCities+i];			
	DEBUG( assert ( help >= 0 && help < numCities ); )
	DEBUG( assert ( visited[k*numCities+help] == FALSE ); )
	tour[k*numCities+phase] = help; 
	visited[k*numCities+help] = TRUE;		

}

void Colony::chooseBestNext(int k, int phase ) { 

	int current_city = tour[k*numCities+phase-1];
 
	int n = select_local_best(&(nnlist[current_city*numCities]), &(visited[k*numCities]), numCities);
	int next_city = nnlist[current_city*numCities+n];

	DEBUG( assert ( 0 <= next_city && next_city < numCities); )
	DEBUG( assert ( visited[k*numCities+next_city] == FALSE ); )
	tour[k*numCities+phase] = next_city; 
	visited[k*numCities+next_city] = TRUE;
}



//computeAntDistances: Computes the distances of each ant's tour, then updates records.
void Colony::computeAntDistances() {

	int iterBestAnt = -1;
	iterBestDist = MAX_DISTANCE-1;// std::numeric_limits<float>::max() - 1;

#if _GPU_
#if 0
	int threadsPerBlock1 = nextPow2(numCities);
    int blocksPerGrid1 = numAnts;
	int smemSize1 = (threadsPerBlock1 <= 32) ? 2 * threadsPerBlock1 * sizeof(long) : threadsPerBlock1 * sizeof(long);

	/// reduce tour lengths for each ant
	dev_calculate_tour_lengths_<<<blocksPerGrid1, threadsPerBlock1, smemSize1>>>(distances, tour, dtourlength, 
#if ANTS_AGING
													antperformance, validants, 
#endif
													numCities, numAnts*numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());

	
	//dump2DArray<long>("output/tourlength.txt", "dev-tour-length-1", dtourlength, 1, numAnts);
#endif
	//dump2DArray<long>("output/tourlength.txt", "dev-tour-length-1", dtourlength, 1, numAnts);
#if 0
	// compute ants tour lengths
	int threadsPerBlock1 = THREADS_PER_BLOCK;
    int blocksPerGrid1 = (numAnts + threadsPerBlock1 - 1) / threadsPerBlock1;
	dev_init_data<long><<<blocksPerGrid1, threadsPerBlock1>>>(dtourlength, 0, numAnts);

	//int np2 = nextPow2(numCities);
	//int threadsPerBlock = min(np2, threadsPerBlock1);
	int np2 = round2BlockSize(numCities, THREADS_PER_BLOCK);
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks = (numCities + threadsPerBlock - 1) / threadsPerBlock;
	int blocksPerGrid = (numAnts*np2 + threadsPerBlock - 1) / threadsPerBlock;
	int smemSize = (threadsPerBlock <= 32) ? 2 * threadsPerBlock * sizeof(long) : threadsPerBlock * sizeof(long);

	/// reduce tour lengths for each ant
	dev_calculate_tour_lengths<<<blocksPerGrid, threadsPerBlock, smemSize>>>(distances, tour, dtourlength, 
#if ANTS_AGING
													antperformance, validants, 
#endif
													blocks, numCities, numAnts*numCities);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
#endif
	//dump2DArray<long>("output/tourlength.txt", "dev-tour-length-2", dtourlength, 1, numAnts);
#if 1
    checkCudaErrors(cudaMemcpy(tourlength, dtourlength, numAnts*sizeof(long), cudaMemcpyDeviceToHost));

#if ANTS_AGING
	unsigned char* hvalidants = new unsigned char[numAnts];
	checkCudaErrors(cudaMemcpy(hvalidants, validants, numAnts*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	int* hantage = new int[numAnts];
	checkCudaErrors(cudaMemcpy(hantage, antage, numAnts*sizeof(int), cudaMemcpyDeviceToHost));
#endif

	//dumpAntsPerformance(tourlength, numAnts, numCities, optimum
//#if ANTS_AGING
//		, hantage
//#endif
//		);

	for (int k = 0 ; k < numAnts ; k++ ) {

		if(tourlength[k] < iterBestDist
#if ANTS_AGING
			&& hvalidants[k]
#endif
			) {
			iterBestAnt = k;
			iterBestDist = tourlength[k];
		}
	}
#if ANTS_AGING
	delete[] hvalidants;
	delete[] hantage;
#endif
#endif
#if 0
	
	//long iterBestDist1 = MAX_DISTANCE-1;
	//int iterBestAnt1 = -1;

	long* btour;
	int* bindex;
	checkCudaErrors(cudaMalloc((void **)&btour, sizeof(long)));	
	checkCudaErrors(cudaMalloc((void **)&bindex, sizeof(int)));

	int threadsPerBlock2 = nextPow2(numAnts);
    int blocksPerGrid2 = 1;
	int smemSize2 = threadsPerBlock2 * (sizeof(long)+sizeof(int));
	
	dev_compute_best_tour<<<blocksPerGrid2, threadsPerBlock2, smemSize2>>>(dtourlength, 
#if ANTS_AGING
													validants, 
#endif
													btour, bindex,  numAnts, threadsPerBlock2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaSynchronize());
	checkCudaErrors(cudaMemcpy(&iterBestDist, btour, sizeof(long), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&iterBestAnt, bindex, sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(btour);
	cudaFree(bindex);

	//printf("\niterBestDist:%ld\titerBestDist1:%ld\titerBestAnt:%d\titerBestAnt1:%d\n",
	//	iterBestDist,iterBestDist1,iterBestAnt,iterBestAnt1);
#endif


	if(iterBestDist < globBestDist && iterBestAnt > -1) {
		globBestDist = iterBestDist;
#if ANTS_AGING
		globalperformance = 0;
#endif
		checkCudaErrors(cudaMemcpy(gbtour, tour + (iterBestAnt*numCities), numCities*sizeof(long), cudaMemcpyDeviceToHost));		
	}
#if ANTS_AGING
	else {
		globalperformance++;
	}
#endif
   

#else

	for (int k = 0 ; k < numAnts ; k++ ) {
#if ANTS_AGING
		if(validants[k]) {
#endif
			long tour_length = 0;
  	
			for (int i = 0 ; i < numCities-1 ; i++ ) {
					
				tour_length += distances[tour[k*numCities+i]*numCities+tour[k*numCities+i+1]];
			
			}
			tour_length += distances[tour[k*numCities+numCities-1]*numCities+tour[k*numCities]];
#if ANTS_AGING
			if(tour_length == tourlength[k])
				antperformance[k]++;
			else
				antperformance[k] = 0;
#endif	
			tourlength[k] = tour_length;
			if(tour_length < iterBestDist) {
				iterBestAnt = k;
				iterBestDist = tour_length;
			}
#if ANTS_AGING
		}
#endif
    }

	//dumpAntsPerformance(tourlength, numAnts, numCities, optimum
//#if ANTS_AGING
//		, antage
//#endif
//		);
	if(iterBestDist < globBestDist && iterBestAnt > -1) {
		globBestDist = iterBestDist;

		for(int i=0; i < numCities; i++) {
			gbtour[i] = tour[iterBestAnt*numCities+i];
		}
#if ANTS_AGING
		globalperformance = 0;
#endif
	}
#if ANTS_AGING
	else {
		globalperformance++;
	}
#endif

#endif
   
}




//greedyDistance: Returns the value of a simple greedy solution starting at city 0.
float Colony::greedyDistance() {
	return 0.0f;
}

void Colony::setAlpha(float newAlpha)
{
  alpha = newAlpha;
}

void Colony::setBeta(float newBeta)
{
  beta = newBeta;
}

void Colony::setRho(float newRho)
{
  rho = newRho;
}

float Colony::getAlpha()
{
  return alpha;
}

float Colony::getBeta()
{
  return beta;
}

float Colony::getRho()
{
  return rho;
}

int Colony::getNumAnts()
{
  return numAnts;
}

int Colony::getNumCities() {
	return numCities;
}

long Colony::getIterBestDist()
{
  return iterBestDist;
}

long Colony::getGlobBestDist()
{
  return globBestDist;
}

int Colony::getReps()
{
  return reps;
}

#if ANTS_AGING
int Colony::getMinRetirementAge() {
	return minRetirementAge;
}


void Colony::setMinRetirementAge(int minRetirementAge) {
	this->minRetirementAge = minRetirementAge;
}


int Colony::getMaxRetirementAge() {
	return maxRetirementAge;
}


void Colony::setMaxRetirementAge(int maxRetirementAge) {
	this->maxRetirementAge = maxRetirementAge;
}


int Colony::getLocalPeformance() {
	return localPeformance;
}


void Colony::setLocalPeformance(int localPeformance) {
	this->localPeformance = localPeformance;
}


int Colony::getGlobalPerformance() {
	return globalPerformance;
}


void Colony::setGlobalPerformance(int globalPerformance) {
	this->globalPerformance = globalPerformance;
}


int Colony::getMinPopulationLimit() {
	return minPopulationLimit;
}


void Colony::setMinPopulationLimit(int minPopulationLimit) {
	this->minPopulationLimit = minPopulationLimit;
}


int Colony::getMaxPopulationLimit() {
	return maxPopulationLimit;
}


void Colony::setMaxPopulationLimit(int maxPopulationLimit) {
	this->maxPopulationLimit = maxPopulationLimit;
}

#endif

std::string Colony::getTour()
{
	std::string result;
	for(int i = 0; i < numCities; i++){
		result += Utils::int2String(gbtour[i]+1) + ",";
	}
	return result;
}
