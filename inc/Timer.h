#ifndef TIMER_H
#define TIMER_H

class Timer {
public:
	Timer(void);
	virtual ~Timer(void);

	void create(void);
	void erase(void);
	
	void start(void);
	void stop(void);
	float elapsed(void);
	
private:
	class StopWatchInterface *timer;
	float elapsed_time;
	
};

#endif
