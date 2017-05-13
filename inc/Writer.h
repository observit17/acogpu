

#ifndef WRITER_H
#define WRITER_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include "def.h"
#include "Timer.h"
#include "Statistics.h"
using namespace std;
/*
#define TIME_INIT_TOUR					0
#define TIME_TOUR						TIME_INIT_TOUR+1
#define TIME_COMPUTE_ANT_DISTANCE		TIME_TOUR+1	
#define TIME_EVAPORATE_PHEROMONE		TIME_COMPUTE_ANT_DISTANCE+1
#define TIME_UPDATE_PHEROMONE			TIME_EVAPORATE_PHEROMONE+1
#define TIME_COMPUTE_PROBABILITIES		TIME_UPDATE_PHEROMONE+1
*/

const char FILE_EXT_LOG[5] = ".log";
const char FILE_EXT_CSV[5] = ".csv";

class Writer { 
public:
	Writer(); // Sets defaults.
	Writer(const char* filen); // Sets defaults and opens the given file.
	Writer(const char* fname, const char* ext);
	~Writer();
	void close();
	bool setFile(const char* filen); // Tries to open given file. If it does, it is changed to writing mode.
	void writeHeader(const char* platform, float alpha, float beta, float rho, int numAnts,string ACO, string TSPName, float calcdisttime); // Writes a header to the file and (if in writing mode) to the file.
	void write(int iter, float iterBest, float globBest, float iterTime, float totalTime); // Writes a standard line of output to stdout and (if in writing mode) to the file.

	void write(const Statistics& statistics);
	void writelog(const SummaryStatistics& summary);
	void writecsv(const SummaryStatistics& summary);

	void writelogheader();
	void writecsvheader();

	void write(const char* dump);
#if 0
	template<typename T>
	static bool dumpArray(const char* filen, const char* msg, T* varrptr, const int size);

	template<typename T>
	static bool dump2DArray(const char* filen, const char* msg, T* arrptr, const int rc, const int cc);
#endif
public:
	static bool VERBOSE;
private:  
	ofstream f; // this is the file
	bool writing;
	string filename;

};

#if 0
// includes CUDA Runtime
#include <cuda_runtime.h>
#include "helper_cuda.h"


template<typename T>
bool Writer::dump2DArray(const char* filen, const char* msg, T* darrptr, const int rc, const int cc) {
	ofstream f; 
	f.open(filen, ios_base::app);
	if (!f)	return false;

	if(msg)
		f << "\n\n" << msg;
	//f << setprecision(OUTPUT_PRECISION);
	f.precision(OUTPUT_PRECISION);
	T* harrptr = new T[rc*cc];
	checkCudaErrors(cudaMemcpy(harrptr, darrptr, rc*cc*sizeof(T), cudaMemcpyDeviceToHost));

	for (int i = 0 ; i < rc; i++ ) {
		f << "\n\n";
		for (int j = 0 ; j < cc; j++ ) {
			f << harrptr[i*cc+j] << "\t";
		} 
	}

	delete[] harrptr;
	f << "\n\n";

	f.close();
	return true;
}

template<typename T>
bool Writer::dumpArray(const char* filen, const char* msg, T* arrptr, const int size) {
	ofstream f; 
	f.open(filen, ios_base::app);
	if (!f)	return false;

	if(msg)
		f << "\n\n" << msg << "\n\n";
	//f << setprecision(OUTPUT_PRECISION);

	for(int i=0; i<size; i++)
		f << arrptr[i] << "\t";
	f << "\n\n";

	f.close();
	return true;
}
#endif

#endif

