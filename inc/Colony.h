

#ifndef COLONY_H
#define COLONY_H

#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include "def.h"
#include "Timer.h"



#define RANDOM_SEED	12345678

#if _GPU_
#define HURESTIC(d,b)	(d==MAX_DISTANCE ? 0 : __powf((1.0 / ((float)d)), b))
#else
#define HURESTIC(d,b)	(d==MAX_DISTANCE ? 0 : powf((1.0 / ((float)d)), b))
#endif

//Colony: The main ACO functions and data.
class Colony {
protected:
	Colony(){}

 public:
  Colony(int* newDistances, int newNumCities, int newNumAnts);
  virtual ~Colony();
  void initialize(); // Initializes data, creates maps and keys, performs standard ACO initialization steps etc.
  void constructSolution(float timers[]); // Main ACO loop. Performs the solution constructruction step, then updates distances, pheromones, probabilities.
  void computeAntDistances(); // Computes the distances of each ant's tour, then updates records.
  void computeProbabilities(); // Computes the probabilities from the distances and pheromones.
  void setAlpha(float newAlpha);
  void setRho(float newRho);
  void setBeta(float newBeta);
  float getAlpha();
  float getRho();
  float getBeta();
  int getNumAnts();
  int getNumCities();
  long getIterBestDist();
  long getGlobBestDist();
  int getReps();
  virtual void computeParameters() = 0; //Implemented differently in each ACO.
  virtual const char* antSystemName() = 0;
  std::string getTour();
  void setOptimum(const int optimum){ this->optimum = optimum; }
 protected:
  float greedyDistance(); // Returns the value of a simple greedy solution starting at city 0.
  virtual void computeInitialPheromone() = 0; //Implemented differently in each ACO.
  virtual void evaporatePheromones() = 0; //Implemented differently in each ACO.
  virtual void updatePheromones() = 0; //Implemented differently in each ACO.


  void initTourLengths();
  void initTour();
  void initPheromone(const float pher);
  void updateAntVisits(int x);
  float calculateProbabilityOfSelection(float* prob_ptr, int k, int phase);
  void selectCities(float* prob_ptr, int k, int phase, float sum_prob);
  void computeNNlist();
  void computeHurestics();
  

  void chooseBestNext(int k, int phase );

  void constructPath();
 
  int numCities;
  int reps;
  float alpha;
  float beta;
  float rho;
  float initialPheromone;

  float* pheromones;
  int* distances;
  int* probabilities;
  long* tour;
  long* gbtour;
  unsigned char* visited;
  long* tourlength;
  int optimum;
#if _GPU_
  long* dtourlength;
  curandState* devStates;
  int* tourmap;
#endif
  int* nnlist;
  float* hurestics;
  
  int numAnts;
  long iterBestDist;
  long globBestDist;

  long seed;

#if ANTS_AGING
  /// ants aging and retirements
  void initAntAges();
  void updateAntAges();
  void retireAnts();
  void recruiteAnts();
  void computePheromoneMean();

  float* pheromone_mean;

  int* antage;
  int* antperformance;
  unsigned char* validants;
  
  int globalperformance;
  int intialpopulation;
  

  // ants aging
	int minRetirementAge;
	int maxRetirementAge;
	int localPeformance;
	int globalPerformance;
	int minPopulationLimit;
	int maxPopulationLimit;


public:
	int getMinRetirementAge();
	void setMinRetirementAge(int minRetirementAge);
	int getMaxRetirementAge();
	void setMaxRetirementAge(int maxRetirementAge) ;
	int getLocalPeformance();
	void setLocalPeformance(int localPeformance);
	int getGlobalPerformance();
	void setGlobalPerformance(int globalPerformance);
	int getMinPopulationLimit();		
	void setMinPopulationLimit(int minPopulationLimit) ;
	int getMaxPopulationLimit();
	void setMaxPopulationLimit(int maxPopulationLimit);
#endif

#if PARALLEL_ACO_WITH_BROKER && _GPU_

protected:
	

public:

#endif // PARALLEL_ACO_WITH_BROKER

public:
	unsigned int getNumValidAnts();
  

};
#endif
