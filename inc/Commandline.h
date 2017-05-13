#pragma once

#define MAX_FILE_PATH	2048

class Commandline
{
public:
	Commandline(void);
	~Commandline(void);
	void setDefaults();

	bool parseCommandline(int argc, char* argv[]);

	void printHelp(void);

	char* getTsplibfile();
	char* getOutfile();
	int getNumAnts();
	float getAlpha();
	float getBeta();
	float getRho();
	int getMaxTime();
	int getMaxIter();
	int getMaxReps();
	int getOptimum();
	bool getVerbose();

	// ants aging
	int getMinRetirementAge();
	int getMaxRetirementAge();
	int getLocalPeformance();
	int getGlobalPerformance();
	int getMinPopulationLimit();
	int getMaxPopulationLimit();

private:
	char tsplibfile[MAX_FILE_PATH];
	char outfile[MAX_FILE_PATH];
	int numAnts;
	float alpha;
	float beta;
	float rho;
	int maxTime;
	int maxIter;
	int maxReps;
	int optimum;
	bool verbose;

	/// ants aging
	int minRetirementAge;
	int maxRetirementAge;
	int localPeformance;
	int globalPerformance;
	int minPopulationLimit;
	int maxPopulationLimit;
	
};

