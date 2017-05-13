/****************************************
 * RankBasedAntSystem.h                 *
 * Peter Ahrens                         *
 * Performs specific RBAS procedures    *
 ****************************************/

#ifndef RANKBASEDANTSYSTEM_H
#define RANKBASEDANTSYSTEM_H
#include "Colony.h"

//RankBasedAntSystem: Provides the neccesary extensions to Colony to create a Rank-Based Ant System
class RankBasedAntSystem : public Colony {
 public:
  RankBasedAntSystem(int* distances, int cities, int ants); // Allocates memory and sets defaults.
  virtual ~RankBasedAntSystem();

  void initialize(); // Runs the Colony initialize, then creates additional maps and keys.
  void computeParameters(); // Simply computes neccesary parameters.
  void setW(int w);
  int getW();
  const char* antSystemName();
 private:
  void computeInitialPheromone(); // Computes the initial pheromone level with the formula described by Marco Dorigo.
  void evaporatePheromones();
  void updatePheromones(); // Evaporates, then the ants lay pheromone at levels corresponding to their rank, judged by the distances of their tours.
  int w;
  
};

#endif
