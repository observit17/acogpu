#ifndef _DUMP_H
#define _DUMP_H


#include <fstream>

//#include <iomanip>

#include "def.h"

template<typename T>
bool dump2DArray(const char* filen, const char* msg, T* darrptr, const int rc, const int cc, bool ishost=false) {
	std::ofstream f; 
	f.open(filen, std::ios_base::app);
	if (!f)	return false;
	f.precision(OUTPUT_PRECISION*4);
	f.setf(std::ios::fixed, std::ios::floatfield);

	if(msg)
		f << "\n\n" << msg;
	//f << std::setprecision(OUTPUT_PRECISION);
	
	

#if _GPU_
	T* harrptr;
	if(ishost) {
		harrptr = darrptr;
	}
	else {
		harrptr = new T[rc*cc];
		checkCudaErrors(cudaMemcpy(harrptr, darrptr, rc*cc*sizeof(T), cudaMemcpyDeviceToHost));
	}
#else
	T* harrptr = darrptr;
#endif

	for (int i = 0 ; i < rc; i++ ) {
		f << "\n\n";
		for (int j = 0 ; j < cc; j++ ) {
			f << harrptr[i*cc+j] << "\t";
		} 
	}

#if _GPU_
	if(!ishost)
		delete[] harrptr;
#endif
	f << "\n\n";

	f.close();
	return true;
}
#if 0
int iteration = 0;

float minf(float a, float b) {
	return a < b ? a : b;
}
void dumpAntsPerformance(const long* tourlength, const int numAnts, const int numCities, const int optimum
#if ANTS_AGING
						 , const int* antage
#endif
						 ) {

#define UPPER_LIMIT		100

	std::ofstream f;
	char fn[1024];
	sprintf(fn, BASE_DIR "output/performaces_%s_%s_%d_%d.csv", PLATFORM, AGING, numCities, numAnts);
	f.open(fn, std::ios_base::app);
	if(!f) return;

	f << iteration;
#if ANTS_AGING
	for (int k = 0 ; k < numAnts ; k++ ) {
		f << "," << antage[k];
	}
#endif
	for (int k = 0 ; k < numAnts ; k++ ) {
		f << "," << tourlength[k];
	}
	for (int k = 0 ; k < numAnts ; k++ ) {
		float d_calulated = (float)tourlength[k];
		float d_optimal = (float)optimum;
		f << "," << UPPER_LIMIT, CALCULATE_DISTANCE_ERROR(d_calulated , d_optimal);
	}
	for (int k = 0 ; k < numAnts ; k++ ) {
		float d_calulated = (float)tourlength[k];
		float d_optimal = (float)optimum;
		f << "," << minf(UPPER_LIMIT, CALCULATE_DISTANCE_ERROR(d_calulated , d_optimal));
	}
	f << "\n";
	iteration++;
	f.close();
}
#endif
#if 0
template<typename T>
bool dump2DArray(const char* filen, const char* msg, T* darrptr, const int rc, const int cc) {
	FILE* f = 0;

	f = fopen(filen, "a");

	if(!f) return false;

	if(msg)
		fprintf(f, "\n\n%s", msg);
	
#if _GPU_
	T* harrptr = new T[rc*cc];
	checkCudaErrors(cudaMemcpy(harrptr, darrptr, rc*cc*sizeof(T), cudaMemcpyDeviceToHost));
#else
	T* harrptr = darrptr;
#endif
	for (int i = 0 ; i < rc; i++ ) {
		fprintf(f, "\n\n");
		for (int j = 0 ; j < cc; j++ ) {
			fprintf(f, "%.20g\t", harrptr[i*cc+j]);
		} 
	}
#if _GPU_
	delete[] harrptr;
#endif
	fprintf(f, "\n\n");

	fclose(f);
	return true;
}


bool dump2DArray(const char* filen, const char* msg, int* darrptr, const int rc, const int cc) {
	FILE* f = 0;

	f = fopen(filen, "a");

	if(!f) return false;

	if(msg)
		fprintf(f, "\n\n%s", msg);
	
#if _GPU_
	int* harrptr = new int[rc*cc];
	checkCudaErrors(cudaMemcpy(harrptr, darrptr, rc*cc*sizeof(int), cudaMemcpyDeviceToHost));
#else
	int* harrptr = darrptr;
#endif

	for (int i = 0 ; i < rc; i++ ) {
		fprintf(f, "\n\n");
		for (int j = 0 ; j < cc; j++ ) {
			fprintf(f, "%d\t", harrptr[i*cc+j]);
		} 
	}
#if _GPU_
	delete[] harrptr;
#endif
	fprintf(f, "\n\n");

	fclose(f);
	return true;
}
#endif
#endif