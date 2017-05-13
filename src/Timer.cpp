


#include "helper_timer.h"
#include "Timer.h"


Timer::Timer(void) : timer(0), elapsed_time(0) {
	
}


Timer::~Timer(void) {
	if(timer != 0) {
		sdkDeleteTimer(&timer);
	}
}

void Timer::create(void) {
	if(timer != 0)
		erase();

	sdkCreateTimer(&timer);
}

void Timer::erase(void) {
	if(timer != 0) {
		sdkDeleteTimer(&timer);
		timer = 0;
	}
}


void Timer::start(void) {
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	elapsed_time = 0.0f;
}

void Timer::stop(void) {
	
	sdkStopTimer(&timer);

	elapsed_time = sdkGetTimerValue(&timer);
}


float Timer::elapsed(void) {	
	return elapsed_time;
}
