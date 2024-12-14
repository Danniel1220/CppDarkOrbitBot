#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <functional>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> taskQueue;
        std::mutex queueMutex;
        std::condition_variable condition;
        bool stop = false;
        int activeThreads = 0; // Count of threads currently processing tasks

    public:
        explicit ThreadPool(size_t threads);
        ~ThreadPool();

        void enqueue(std::function<void()> task);
        void waitForCompletion();
};

#endif
