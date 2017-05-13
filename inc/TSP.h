#pragma once

#include "Statistics.h"

class Colony;

class TSP
{
public:
	TSP(int maxTime, int maxIter, int maxReps, int optimum);
	~TSP(void);

	void execute(const char* tspname, Colony& colony, const char* outstats, float calcdisttime);

	void setMaxTime(int maxTime);
	void setMaxIter(int maxIter);
	void setMaxReps(int maxReps);
	void setOptimum(int optimum);

private:
	int maxTime;
	int maxIter;
	int maxReps;
	int optimum;
protected:
	bool terminationCondition(const float elapsed, const int iter, const int reps, const float tourlen);
	void initStatistics(float alpha, float beta, float rho, Statistics& statistics, SummaryStatistics& summary);
	void computeStatistics(SummaryStatistics& summary, Statistics& statistics, int iter, int numAnts, long iterbest, long globalbest, float timers[]);
	void writeSummaryStatistics(SummaryStatistics& summary, Colony& colony, int iterations, float inittime, float calcdisttime, int optimum);
};

