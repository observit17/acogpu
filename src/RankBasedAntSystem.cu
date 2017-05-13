/****************************************
 * RankBasedAntSystem.cu                *
 * Peter Ahrens                         *
 * Performs specific RBAS procedures    *
 ****************************************/

#include "RankBasedAntSystem.h"
#include "default.h"


const static char ANT_SYSTEM_NAME[] = "Rank-based Ant System";

//constructor: Allocates memory and sets defaults.
RankBasedAntSystem::RankBasedAntSystem(int* distances, int cities, int ants)
  : Colony(distances, cities, ants) {
  w = DEFAULT_NUM_RANK_ANTS;//default
  
}

RankBasedAntSystem::~RankBasedAntSystem() {

}

const char* RankBasedAntSystem::antSystemName() {
	return ANT_SYSTEM_NAME;
}

//initialize: Runs the Colony initialize, then creates additional maps and keys.
void RankBasedAntSystem::initialize() {
  Colony::initialize();
  
}

//computeParameters: Simply computes neccesary parameters.
void RankBasedAntSystem::computeParameters() {
  if (numCities < w){
    w = numCities;
  }
  computeInitialPheromone();
}

//computeInitialPheromone: Computes the initial pheromone level with the formula described by Marco Dorigo.
void RankBasedAntSystem::computeInitialPheromone() {
  initialPheromone = DEFAULT_PHEROMONE; /*0.5*w*(w-1)/(rho * Colony::greedyDistance());*/
}


void RankBasedAntSystem::evaporatePheromones() {
	
}

//updataPheromones: Evaporates, then the ants lay pheromone at levels corresponding to their rank, judged by the distances of their tours.
void RankBasedAntSystem::updatePheromones() {
  
  
}


void RankBasedAntSystem::setW(int w)
{
  this->w = w;
}

int RankBasedAntSystem::getW()
{
  return w;
}

