#include <string.h>
#include "TSP.h"
#include "Colony.h"
#include "Writer.h"

static const char SUMMARY_FILE[MAX_INPUT_LEN] = BASE_DIR "output/summary";

TSP::TSP(int maxTime, int maxIter, int maxReps, int optimum)
{
	setMaxTime(maxTime);
	setMaxIter(maxIter);
	setMaxReps(maxReps);
	setOptimum(optimum);
}


TSP::~TSP(void)
{
}



void TSP::execute(const char* tspname, Colony& colony, const char* outstats, float calcdisttime) {

	// 1- init tour
	// 2- tour
	// 3- compute ant distances
	// 4- evaporate pheromones
	// 5- update pheromones
	// 6- compute probabilities
	float timers[TOTAL_TIMERS];
	Writer writer(outstats);
	Statistics statistics;
	SummaryStatistics summary;
	

	initStatistics(colony.getAlpha(), colony.getBeta(), colony.getRho(), statistics, summary);
	
	Timer timer;
	timer.create();
	timer.start();

	colony.setOptimum(optimum);
	colony.initialize();

	float inittime = timer.elapsed();
	timer.erase();

	int i = 0;

#if _GPU_ && 0
	/// warm-up
	colony.constructSolution(timers);
	colony.constructSolution(timers);
	colony.constructSolution(timers);
	colony.constructSolution(timers);
	colony.constructSolution(timers);
#endif

	for(; !terminationCondition(summary.t_totaltime, i, colony.getReps(), colony.getIterBestDist()); i++){

		colony.constructSolution(timers);

		computeStatistics(summary, statistics, i, colony.getNumValidAnts(), colony.getIterBestDist(), colony.getGlobBestDist(), timers);
		
		writer.write(statistics);

		//if(i%1000 == 0)
		//	printf("iter: %d\n", i);

    }

	if(colony.getGlobBestDist() <= optimum)
		writer.write(colony.getTour().c_str());

	writeSummaryStatistics(summary, colony, i, inittime, calcdisttime, optimum);


}

void TSP::writeSummaryStatistics(SummaryStatistics& summary, Colony& colony, int iterations, float inittime, float calcdisttime, int optimum) {
	summary.numants = colony.getNumAnts();
	summary.numcities = colony.getNumCities();
	summary.numiter = iterations;
	summary.t_inittime = inittime;
	//summary.t_evappheromnes = summary.t_evappheromnes;///((float)(summary.numiter));
	//summary.t_updatephoremone = summary.t_updatephoremone;///((float)(summary.numiter));
	summary.t_distcalc = calcdisttime;
	summary.d_optimal = optimum;
	summary.d_calulated = colony.getGlobBestDist();
	summary.d_error = CALCULATE_DISTANCE_ERROR(summary.d_calulated , summary.d_optimal);

	Writer writerlog(SUMMARY_FILE, FILE_EXT_LOG);
	writerlog.writelogheader();
	writerlog.writelog(summary);

	Writer writercsv(SUMMARY_FILE, FILE_EXT_CSV);
	writercsv.writecsvheader();
	writercsv.writecsv(summary);
}

void TSP::initStatistics(float alpha, float beta, float rho, Statistics& statistics, SummaryStatistics& summary) {
	memset(summary.tsplib, '\0', MAX_INPUT_LEN);
	strcpy(summary.platform, PLATFORM);
	summary.p_alpha = alpha;
	summary.p_beta = beta;
	summary.p_rho = rho;
	summary.t_inittime = 0;
	summary.t_distcalc = 0;
	summary.t_inittour = 0;
	summary.t_computeprob = 0;
	summary.t_tour = 0;
	summary.t_antdist = 0;
	summary.t_updatephoremone = 0;
	summary.t_evappheromnes = 0;
	summary.t_totaltime = 0;
}

void TSP::computeStatistics(SummaryStatistics& summary, Statistics& statistics, int iter, int numAnts, long iterbest, long globalbest, float timers[]) {
	statistics.iter = iter;
	statistics.ants = numAnts;
	statistics.iterbest = iterbest;
	statistics.globalbest = globalbest;
	statistics.t_inittour = timers[TIME_INIT_TOUR];
	statistics.t_tour = timers[TIME_TOUR];
	statistics.t_antdist = timers[TIME_COMPUTE_ANT_DISTANCE];
	statistics.t_evapphoremone = timers[TIME_EVAPORATE_PHEROMONE];
	statistics.t_updatephoremone = timers[TIME_UPDATE_PHEROMONE];
	statistics.t_computeprob = timers[TIME_COMPUTE_PROBABILITIES];

	float itertime = 0.0f;
	for(int i=0; i<TOTAL_TIMERS; i++) {
		itertime += timers[i];
	}

	summary.t_totaltime += itertime;
	summary.t_inittour += statistics.t_inittour;
	summary.t_computeprob += statistics.t_computeprob;
	summary.t_tour += statistics.t_tour;
	summary.t_antdist += statistics.t_antdist;
	summary.t_evappheromnes += statistics.t_evapphoremone;
	summary.t_updatephoremone += statistics.t_updatephoremone;

	statistics.t_itertime = itertime;
	statistics.t_globaltime = summary.t_totaltime;
}

void TSP::setMaxTime(int maxTime) {
	this->maxTime = maxTime;
}


void TSP::setMaxIter(int maxIter) {
	this->maxIter = maxIter;
}


void TSP::setMaxReps(int maxReps) {
	this->maxReps = maxReps;
}


void TSP::setOptimum(int optimum) {
	this->optimum = optimum;
}


bool TSP::terminationCondition(const float elapsed, const int iter, const int reps, const float tourlen)
{
	return elapsed >= maxTime || iter >= maxIter || reps >= maxReps || tourlen <= optimum;		
}
