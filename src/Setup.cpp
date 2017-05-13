/****************************************
 * Setup.cpp                            *
 * Sets up the ACO                      *
 ****************************************/

/**
 * 
 * ACO setup for
 * 1- huriestics computation
 * 2- probabilities computation
 * 3- inline tour length computation
 * 4- tour map for pehromone update
 * 5- integral probibilities
 *
**/

#include "Colony.h"
#include "AntSystem.h"
#include "TSPReader.h"
#include "Commandline.h"
#include "TSP.h"
#include "Utils.h"
#include <iostream>
#include <string>
#include <cctype>
#include <cstdlib>
#include "Timer.h"
#include "Writer.h"
#include "dump.h"
#include <iomanip>


using namespace std;


void initCuda() {
#if _GPU_	
	int devID = 0;
    //cudaDeviceProp deviceProp;

	checkCudaErrors(cudaSetDevice(devID));
    //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
#endif   
}

void resetCuda() {
#if _GPU_	
	cudaDeviceReset();
#endif   
}


int* calculateDistances(const float* IN xaxis, const float* IN yaxis, const int IN nodes) {

	int* distance = new int[nodes*nodes];
	Utils::compute_distance(xaxis, yaxis, nodes, distance);
	

#if _GPU_
	int* d_distance = 0;
	size_t size = nodes * nodes * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&d_distance, size));
	checkCudaErrors(cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice));
	delete[] distance;
	return d_distance;
#else
	return distance;
#endif
}


int main(int argc, char* argv[]) {	
	
	initCuda();
	int oldprecision = std::cout.precision();
	std::cout << setprecision(OUTPUT_PRECISION);
	Commandline commandline;
	if(!commandline.parseCommandline(argc, argv)) {
		commandline.printHelp();
		return -1;
	}
	TSPReader tSPReader;
	if(!tSPReader.read(commandline.getTsplibfile())) {
		cout << "\nUnable to read tsplib file [" << commandline.getTsplibfile() << "]\n";
		commandline.printHelp();
		return -1;
	}
	Timer timer;
	timer.create();

	timer.start();
	int* distance = calculateDistances(tSPReader.getXcoords(), tSPReader.getYcoords(), tSPReader.getNumNodes());
	timer.stop();
	float calcdisttime = timer.elapsed();
	//dump2DArray<int>("output/distnace.txt", "", distance , tSPReader.getNumNodes(), tSPReader.getNumNodes(), false);
		
	
	//DEBUG(Writer::dump2DArray<float>("output/dump.txt", commandline.getTsplibfile(), tSPReader.getDistances(), tSPReader.getNumNodes(), tSPReader.getNumNodes());)

	AntSystem antSystem(distance, tSPReader.getNumNodes(), commandline.getNumAnts(),
		commandline.getAlpha(), commandline.getBeta(), commandline.getRho()
#if ANTS_AGING
		, commandline.getMinRetirementAge(), commandline.getMaxRetirementAge(), commandline.getLocalPeformance(), 
		commandline.getGlobalPerformance(), commandline.getMinPopulationLimit(), commandline.getMaxPopulationLimit()
#endif
		);
	Writer::VERBOSE = commandline.getVerbose();
	
	Writer writer(commandline.getOutfile());

	writer.writeHeader(PLATFORM, commandline.getAlpha(), commandline.getBeta(), commandline.getRho(), commandline.getNumAnts(), antSystem.antSystemName(), tSPReader.getName().c_str(), calcdisttime);
	writer.close();
	TSP tsp(commandline.getMaxTime(), commandline.getMaxIter(), commandline.getMaxReps(), commandline.getOptimum());
	tsp.execute(tSPReader.getName().c_str(), antSystem, commandline.getOutfile(), calcdisttime);
	timer.erase();

	std::cout << setprecision(oldprecision);

#if _GPU_
		cudaFree(distance);
#else
		delete[] distance;
#endif

	resetCuda();

	return 0;
}

