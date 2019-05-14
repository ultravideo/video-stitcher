#pragma once
#include <vector>

template<class T>
class LockableVector : public std::vector<T> {
public:
    LockableVector() : std::vector<T>(), vec_mutex(){
    }

    LockableVector(int num_images) : std::vector<T>(num_images) {
    }

    void lock() {
        vec_mutex.lock();
    }

    void unlock() {
        vec_mutex.unlock();
    }

    std::mutex vec_mutex;
};

