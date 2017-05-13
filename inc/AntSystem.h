
#ifndef ANTSYSTEM_H
#define ANTSYSTEM_H
#include "Colony.h"

//AntSystem: Provides the neccesary extensions to Colony to create a Rank-Based Ant System
class AntSystem : public Colony {
 public:
  AntSystem(int* newDistances, int newNumCities, int newNumAnts, float alpha, float beta, float rho); // Allocates memory and sets defaults.
#if ANTS_AGING
  AntSystem(int* newDistances, int newNumCities, int newNumAnts, float alpha, float beta, float rho,
	  int minRetirementAge, int maxRetirementAge, int localPeformance, int globalPerformance, int minPopulationLimit, int maxPopulationLimit); // Allocates memory and sets defaults.
#endif
  virtual ~AntSystem();
  const char* antSystemName();
  void computeParameters(); // Simply computes neccesary parameters.

 private:
  void computeInitialPheromone(); // Computes the initial pheromone level
  void evaporatePheromones();
  void updatePheromones(); // Evaporates, then the ants lay pheromone.  
	
};

#endif
