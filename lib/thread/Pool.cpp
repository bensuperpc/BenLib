#if __cplusplus >= 201703L

#    include "Pool.hpp"

namespace thread
{
Pool::Pool(std::size_t numberOfThreads) : stopped_(false)
{
    threads_.reserve(numberOfThreads);
    for (std::size_t i = 0; i < numberOfThreads; ++i) {
        threads_.emplace_back([this] {
            while (true) {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> uniqueLock(mutex_);
                    condition_.wait(uniqueLock, [this] { return tasks_.empty() == false || stopped_ == true; });
                    if (tasks_.empty() == false) {
                        task = std::move(tasks_.front());
                        // attention! tasks_.front() moved
                        tasks_.pop();
                    } else { // stopped_ == true (necessarily)
                        return;
                    }
                }
                task();
            }
        });
    }
}

Pool::~Pool()
{
    {
        std::lock_guard<std::mutex> lockGuard(mutex_);
        stopped_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : threads_) {
        worker.join();
    }
}
} // namespace thread
#endif