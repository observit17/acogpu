#include <sstream>
#include "Utils.h"

#define IM1 2147483563
#define IM2 2147483399
#define AM1 (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.e-14
#define RNMX (1.0-EPS)

double ran2(int *idum)
{
    int j;
    int k;
    static int idum2 = 123456789;
    static int iy = 0;
    static int iv[NTAB];
    double temp;

    if (*idum <= 0) {		// *idum < 0 ==> initialize
		if (-(*idum) < 1)
			*idum = 1;
		else
			*idum = -(*idum);
		idum2 = (*idum);

		for (j = NTAB+7; j >= 0; j--) {
			k = (*idum)/IQ1;
			*idum = IA1*(*idum-k*IQ1) - k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy = iv[0];
    }
    k = (*idum)/IQ1;
    *idum = IA1*(*idum-k*IQ1) - k*IR1;
    if (*idum < 0) *idum += IM1;

    k = idum2/IQ2;
    idum2 = IA2*(idum2-k*IQ2)-k*IR2;
    if (idum2 < 0) idum2 += IM2;

    j = iy/NDIV;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if (iy < 1) iy += IMM1;

    if ((temp = AM1*iy) > RNMX)
		return RNMX;
    else
		return temp;
}

#undef IM1
#undef IM2
#undef AM1
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

static int randx = 0;		/* copy of random seed (internal use only) */

#include <time.h>

int srandinter(int seed)	/* initialize the random number generator */
{
    if (seed == 0) seed = (int) time(NULL);	/* initialize from the system
						   clock if seed = 0 */
    randx = -abs(seed);
    return seed;		/* return seed in case we need to repeat */
}

float randinter(float a, float b)	/* return a random number uniformly
					   distributed between a and b */
{
    if (randx == 0) srandinter(0);
    return a + (b-a)*((float)ran2(&randx));
}

#undef M_PI
#define M_PI 3.14159265358979323846264


Utils::Utils(void)
{
}


Utils::~Utils(void)
{
}

#define MIN_RANDOM	0.00000009f
#define MAX_RANDOM	0.999999999999999999f

float Utils::random( long *idum ) {
	
	return randinter(MIN_RANDOM, MAX_RANDOM);
#if 0
	long k;
	float ans;

	k =(long)((*idum)/IQ);
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0 ) *idum += IM;
	ans = AM * (*idum);
	return ans;
#endif
}

std::string Utils::int2String(int t)
{
    std::ostringstream oss;
    std::string s;
    oss << t;
    s = oss.str();
    return s;
}

void Utils::sort2(int v[], int v2[], int left, int right) {
	int k, last;

	if (left >= right) 
	return;
	swap2(v, v2, left, (left + right)/2);
	last = left;
	for (k=left+1; k <= right; k++)
	if (v[k] < v[left])
		swap2(v, v2, ++last, k);
	swap2(v, v2, left, last);
	sort2(v, v2, left, last);
	sort2(v, v2, last+1, right);
}

 
void Utils::swap2(int v[], int v2[], int i, int j) {
	int tmp1 = v[i];
	v[i] = v[j];
	v[j] = tmp1;
	int tmp2 = v2[i];
	v2[i] = v2[j];
	v2[j] = tmp2;
}


void Utils::compute_distance( const float* IN xaxis, const float* IN yaxis, const int IN nodes, int* OUT distance) {	

	int d;
	for(int i = 0; i < nodes-1; i++){
		distance[i * nodes + i] = MAX_DISTANCE;
		for(int j = i+1; j < nodes; j++){
			d = round_distance (xaxis[i], xaxis[j], yaxis[i], yaxis[j]);
				//geo_distance (xaxis[i], xaxis[j], yaxis[i], yaxis[j]);
				//exact_distance(xaxis[i], xaxis[j], yaxis[i], yaxis[j]);
				//ceil_distance(xaxis[i], xaxis[j], yaxis[i], yaxis[j]);
				//floor_distance(xaxis[i], xaxis[j], yaxis[i], yaxis[j]);

			distance[i * nodes + j] = d;
			distance[j * nodes + i] = d;
		}

	}

	distance[nodes * nodes -1] = MAX_DISTANCE;
	
}



int Utils::geo_distance (float xi, float xj, float yi, float yj) {
     double lati, latj, longi, longj;
     double q1, q2, q3, q4, q5;

     lati = M_PI * xi / 180.0;
     latj = M_PI * xj / 180.0;

     longi = M_PI * yi / 180.0;
     longj = M_PI * yj / 180.0;

     q1 = cos (latj) * sin(longi - longj);
     q3 = sin((longi - longj)/2.0);
     q4 = cos((longi - longj)/2.0);
     q2 = sin(lati + latj) * q3 * q3 - sin(lati - latj) * q4 * q4;
     q5 = cos(lati - latj) * q4 * q4 - cos(lati + latj) * q3 * q3;
     return (int) (6378388.0 * atan2(sqrt(q1*q1 + q2*q2), q5) + 1.0);
}

float Utils::exact_distance (float xi, float xj, float yi, float yj) {
    double xd = xi - xj;
    double yd = yi - yj;
    double r  = sqrt(xd*xd + yd*yd);

    return (float) r;
}

int Utils::round_distance (float xi, float xj, float yi, float yj) {
    double xd = xi - xj;
    double yd = yi - yj;
    double r  = sqrt(xd*xd + yd*yd) + 0.5;

    return (int) r;
}

int Utils::ceil_distance (float xi, float xj, float yi, float yj) {
    double xd = xi - xj;
    double yd = yi - yj;
    double r  = sqrt(xd*xd + yd*yd) + 0.000000001;

    return (int)r;
}

int Utils::floor_distance (float xi, float xj, float yi, float yj) {
    double xd = xi - xj;
    double yd = yi - yj;
    double r  = sqrt(xd*xd + yd*yd) + 1.0;

    return (int)r;
}


