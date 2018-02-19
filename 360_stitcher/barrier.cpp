#include "barrier.h"

Barrier::Barrier(unsigned int n) 
	: wait_count(0), target_wait_count(n)
{}

void Barrier::wait() {
	std::unique_lock<std::mutex> lk(mtx);
	unsigned int const current_wait_cycle = inter_wait_count;
	++wait_count;
	if (wait_count != target_wait_count) {
		// wait condition must not depend on wait_count
		cond_var.wait(lk,
			[this, current_wait_cycle]() {
			return inter_wait_count != current_wait_cycle;
		});
	}
	else {
		// increasing the second counter allows waiting threads to exit
		++inter_wait_count;
		wait_count = 0;
		cond_var.notify_all();
	}
}