/****************************************
 * TSPReader.h                          *
 * Reads .tsp files                     *
 ****************************************/

#ifndef TSPREADER_H
#define TSPREADER_H
#include <iostream>	// Used for command line I/O
#include <fstream>	// Used for file Input
#include <string>
#include <math.h>
#include <cctype>
#include <float.h> // Used to find maximum float
#include "def.h"
using namespace std;


//TSPReader: Used to read .tsp files.
class TSPReader
{	

	
 public:
  //Constructors/Destructors
  TSPReader();
  ~TSPReader();

  bool read(char* filen);	// Reads a given tsp file and extracts data.
 
  string getName();
  float* getXcoords();
  float* getYcoords();  
  int getNumNodes(); 
 

private:
	string name;			// TSP name - max length 16
	int nodes;		// Number of cities (dimension)
	float* xcoords;		// X coords
	float* ycoords;		// Y coords	
	string* cityNames;	
	
};

#endif

