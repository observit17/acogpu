#include "Commandline.h"
#include "default.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>

using namespace std;




#define STR_HELP_ALPHA \
	"  -a, --alpha          # alpha (influence of pheromone trails)\n"

#define STR_HELP_BETA \
	"  -b, --beta           # beta (influence of heuristic information)\n"

#define STR_HELP_RHO \
	"  -r, --rho            # rho: pheromone trail evaporation\n"

#define STR_HELP_TOURS \
	"  -s, --tours          # number of steps in each trial\n"

#define STR_HELP_TIME \
	"  -t, --time           # maximum time for each trial\n"

#define STR_HELP_TSPLIBFILE \
	"  -i, --tsplibfile     f inputfile (TSPLIB format necessary)\n"

#define STR_HELP_OUTPATH \
	"  -p, --outfile        f output file where results will be stored\n"

#define STR_HELP_OPTIMUM \
	"  -o, --optimum        # stop if tour better or equal optimum is found\n"

#define STR_HELP_ANTS \
	"  -m, --ants           # number of ants\n"

#define STR_HELP_REPETITIONS \
	"  -c, --repetitions    # number of repetitions at which solution exits\n"

#define STR_HELP_MIN_RETIREMENT_AGE \
	"  -j, --minage           min age for tetirement of an ant\n"

#define STR_HELP_MAX_RETIREMENT_AGE \
	"  -k, --maxage           max age for tetirement of an ant\n"

#define STR_HELP_LOCAL_PERFORMANCE \
	"  -l, --localperformance           local performance of an ant in deciding retirement\n"

#define STR_HELP_GLOBAL_PERFORMANCE \
	"  -g, --globalperformance           global performance of all ants in deciding retirement\n"

#define STR_HELP_MIN_POPULATION_LIMIT \
	"  -x, --minpopulation           min population on ants in deciding recruitment\n"

#define STR_HELP_MAX_POPULATION_LIMIT \
	"  -y, --maxpopulation           max population of ants in deciding retirement\n"

#define STR_HELP_VERBOSE \
	"  -v, --verbose        display output to standard console\n"

#define STR_HELP_HELP \
	"  -h, --help           display this help text and exit\n"




#define STR_HELP \
	STR_HELP_ALPHA \
	STR_HELP_BETA \
	STR_HELP_RHO \
	STR_HELP_TOURS \
	STR_HELP_TIME \
	STR_HELP_TSPLIBFILE \
	STR_HELP_OUTPATH \
	STR_HELP_OPTIMUM \
	STR_HELP_ANTS \
	STR_HELP_REPETITIONS \
	STR_HELP_MIN_RETIREMENT_AGE \
	STR_HELP_MAX_RETIREMENT_AGE \
	STR_HELP_LOCAL_PERFORMANCE \
	STR_HELP_GLOBAL_PERFORMANCE \
	STR_HELP_MIN_POPULATION_LIMIT \
	STR_HELP_MAX_POPULATION_LIMIT \
	STR_HELP_VERBOSE \
	STR_HELP_HELP


Commandline::Commandline(void) {

	setDefaults();
}


void Commandline::setDefaults() {

	tsplibfile[0] = '\0';
	outfile[0] = '\0';
	numAnts = DEFAULT_NUM_ANTS;
	alpha = DEFAULT_ALPHA;
	beta = DEFAULT_BETA;
	rho = DEFAULT_RHO;
	maxTime = DEFAULT_MAX_TIME;
	maxIter = DEFAULT_MAX_ITERATIONS;
	maxReps = DEFAULT_MAX_REPITITIONS;
	optimum = DEFAULT_OPTIMUM;
	verbose = false;

	/// ants aging
	minRetirementAge = DEFAULT_MIN_RITIREMENT_AGE;
	maxRetirementAge = DEFAULT_MAX_RITIREMENT_AGE;
	localPeformance = DEFAULT_LOCAL_NO_CONVERGENCE;
	globalPerformance = DEFAULT_GLOBAL_NO_CONVERGENCE;
	minPopulationLimit = DEFAULT_MIN_POPULATION_LIMIT;
	maxPopulationLimit = DEFAULT_MAX_POPULATION_LIMIT;
}

Commandline::~Commandline(void) {
}

bool Commandline::parseCommandline(int argc, char* argv[]) {
	
	// d,e,f,n,q,u,w,z
	static const char *const optstr_a = "-a";
	static const char *const optstr_b = "-b";
	static const char *const optstr_r = "-r";
	static const char *const optstr_s = "-s";
	static const char *const optstr_t = "-t";
	static const char *const optstr_i = "-i";
	static const char *const optstr_p = "-p";
	static const char *const optstr_o = "-o";
	static const char *const optstr_m = "-m";
	static const char *const optstr_c = "-c";
	static const char *const optstr_v = "-v";
	static const char *const optstr_h = "-h";

	static const char *const optstr_j = "-j";
	static const char *const optstr_k = "-k";
	static const char *const optstr_l = "-l";
	static const char *const optstr_g = "-g";
	static const char *const optstr_x = "-x";
	static const char *const optstr_y = "-y";
	
	
	static const char *const optstr__alpha = "--alpha";	
	static const char *const optstr__beta = "--beta";	
	static const char *const optstr__rho = "--rho";
	static const char *const optstr__tours = "--tours";
	static const char *const optstr__time = "--time";
	static const char *const optstr__tsplibfile = "--tsplibfile";
	static const char *const optstr__outfile = "--outfile";
	static const char *const optstr__optimum = "--optimum";
	static const char *const optstr__ants = "--ants";
	static const char *const optstr__repetitions = "--repetitions";
	static const char *const optstr__verbose = "--verbose";
	static const char *const optstr__help = "--help";

	static const char *const optstr__minage = "--minage";
	static const char *const optstr__maxage = "--maxage";
	static const char *const optstr__localperformance = "--localperformance";
	static const char *const optstr__globalperformance = "--globalperformance";
	static const char *const optstr__minpopulation = "--minpopulation";
	static const char *const optstr__maxpopulation = "--maxpopulation";
	
	

	for(int i = 0; i < argc;i++){
		string option(argv[i]);

		if (option == optstr_a || option == optstr__alpha) {
			alpha = atof(argv[++i]);
		}
		else if (option == optstr_b || option == optstr__beta) {
			beta = atof(argv[++i]);
		}
		else if (option == optstr_r || option == optstr__rho) {
			rho = atof(argv[++i]);
		}
		else if (option == optstr_i || option == optstr__tsplibfile) {
			strcpy(tsplibfile, argv[++i]);
		}	
		else if (option == optstr_p || option == optstr__outfile) {
			strcpy(outfile, argv[++i]);
		}
		else if (option == optstr_m || option == optstr__ants) {
			numAnts = atoi(argv[++i]);
		}		
		else if (option == optstr_t || option == optstr__time) {
			maxTime = atoi(argv[++i]);
		}
		else if (option == optstr_s || option == optstr__tours) {
			maxIter = atoi(argv[++i]);
		}
		else if (option == optstr_o || option == optstr__optimum) {
			optimum = atoi(argv[++i]);
		}
		else if (option == optstr_c || option == optstr__repetitions) {
			maxReps = atoi(argv[++i]);
		}
		else if (option == optstr_v || option == optstr__verbose) {
			verbose = true;
		}
		else if (option == optstr_h || option == optstr__help) {			
			return false;
		}
		else if (option == optstr_j || option == optstr__minage) {
			minRetirementAge = atoi(argv[++i]);
		}
		else if (option == optstr_k || option == optstr__maxage) {
			maxRetirementAge = atoi(argv[++i]);
		}
		else if (option == optstr_l || option == optstr__localperformance) {
			localPeformance = atoi(argv[++i]);
		}
		else if (option == optstr_g || option == optstr__globalperformance) {
			globalPerformance = atoi(argv[++i]);
		}
		else if (option == optstr_x || option == optstr__minpopulation) {
			minPopulationLimit = atoi(argv[++i]);
		}
		else if (option == optstr_y || option == optstr__maxpopulation) {
			maxPopulationLimit = atoi(argv[++i]);
		}
		
	}

	return true;
}

void Commandline::printHelp(void) {
	std::cout << "Usage: [OPTION]... [ARGUMENT]...\n\n Options:\n" << STR_HELP;
}

char* Commandline::getTsplibfile() {
	return tsplibfile;
}

char* Commandline::getOutfile() {
	return outfile;
}

int Commandline::getNumAnts() {
	return numAnts;
}


float Commandline::getAlpha() {
	return alpha;
}


float Commandline::getBeta() {
	return beta;
}


float Commandline::getRho() {
	return rho;
}


int Commandline::getMaxTime() {
	return maxTime;
}


int Commandline::getMaxIter() {
	return maxIter;
}


int Commandline::getMaxReps() {
	return maxReps;
}


int Commandline::getOptimum() {
	return optimum;
}

bool Commandline::getVerbose() {
	return verbose;
}

int Commandline::getMinRetirementAge() {
	return minRetirementAge;
}
int Commandline::getMaxRetirementAge() {
	return maxRetirementAge;
}
int Commandline::getLocalPeformance() {
	return localPeformance;
}
int Commandline::getGlobalPerformance() {
	return globalPerformance;
}
int Commandline::getMinPopulationLimit() {
	return minPopulationLimit;
}
int Commandline::getMaxPopulationLimit() {
	return maxPopulationLimit;
}




