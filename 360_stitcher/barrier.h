#pragma once
#include <mutex>

class Barrier {
public:
	Barrier(unsigned int n);
	void wait();

private:
	std::mutex mtx;
	std::condition_variable cond_var;

	int wait_count;
	int inter_wait_count;
	int const target_wait_count;
};