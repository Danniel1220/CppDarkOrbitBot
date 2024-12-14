#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t threads) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock, [this]() { return stop || !taskQueue.empty(); });

                    if (stop && taskQueue.empty()) return;

                    task = std::move(taskQueue.front());
                    taskQueue.pop();
                    ++activeThreads;
                }

                task();

                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    --activeThreads;
                    condition.notify_all();
                }
            }
            });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        taskQueue.push(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::waitForCompletion() {
    std::unique_lock<std::mutex> lock(queueMutex);
    condition.wait(lock, [this]() { return taskQueue.empty() && activeThreads == 0; });
}
