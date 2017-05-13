
#ifndef _STATISTICS_
#define _STATISTICS_

#define MAX_INPUT_LEN	1024

enum StatsTimer {	
	TIME_INIT_TOUR = 0,
	TIME_TOUR,
	TIME_COMPUTE_ANT_DISTANCE,
	TIME_EVAPORATE_PHEROMONE,
	TIME_UPDATE_PHEROMONE,
	TIME_COMPUTE_PROBABILITIES,
	TOTAL_TIMERS
};

struct Statistics {
	int ants;
	int iter;
	long iterbest;
	long globalbest;
	float t_inittour;
	float t_computeprob;
	float t_tour;
	float t_antdist;
	float t_evapphoremone;
	float t_updatephoremone;	
	float t_itertime;
	float t_globaltime;
};

struct SummaryStatistics {
	char platform[MAX_INPUT_LEN];
	char tsplib[MAX_INPUT_LEN];
	int numcities;
	int numants;
	int numiter;
	float t_inittime;
	float t_inittour;
	float t_computeprob;
	float t_tour;
	float t_antdist;
	float t_distcalc;
	float t_updatephoremone;
	float t_evappheromnes;
	float t_totaltime;
	float d_optimal;
	float d_calulated;
	float d_error;
	float p_alpha;
	float p_beta;
	float p_rho;
};

#endif // _STATISTICS_
