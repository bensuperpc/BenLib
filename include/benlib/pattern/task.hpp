#ifndef BENLIB_TASK_HPP_
#define BENLIB_TASK_HPP_

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "BS_thread_pool.hpp"
#include "scheduler.hpp"

namespace benlib {
namespace pattern {

enum class TaskStatus : int { NONE = 0, COMPLETED, CANCELLED, FAILED };

struct ITask {
    virtual ~ITask() = default;
    virtual void start() = 0;
    void cancel() { setStatus(TaskStatus::CANCELLED); }
    void fail() { setStatus(TaskStatus::FAILED); }

    virtual bool isDone() = 0;

    TaskStatus status() const { return _status.load(); }
    void setStatus(TaskStatus s) {
        _status.store(s); 
    }

   protected:
    std::atomic<TaskStatus> _status{TaskStatus::NONE};
    mutable std::shared_mutex _mutex;
};

template <typename T>
struct Task : ITask {
    using ValueType = T;

    bool isDone() override {
        auto st = _status.load();
        return st != TaskStatus::NONE;
    }

    template <typename U>
    void setValue(U&& v) {
        std::unique_lock<std::shared_mutex> lk(_mutex);
        _value = std::forward<U>(v);
    }

    T getValue() const {
        std::shared_lock<std::shared_mutex> lk(_mutex);
        return _value;  // copy or move
    }

   private:
    T _value{};
};

template <typename T>
struct ITasker {
    virtual ~ITasker() = default;
    virtual void taskCompleted(std::shared_ptr<Task<T>>) = 0;
};

class TaskHandler : public App {
   public:
    explicit TaskHandler(BS::thread_pool<BS::tp::priority>& pool) : _pool(pool) {}

    template <typename Req>
    void startTask(std::shared_ptr<Req> req, ITasker<typename Req::ValueType>& tasker, BS::pr priority = BS::pr::high) {
        {
            std::unique_lock<std::shared_mutex> lk(_mutex);
            _entries.emplace_back(req);
        }

        std::future<void> _ = _pool.submit_task(
            [req, &tasker]() {
                try {
                    req->start();
                    if (req->status() == TaskStatus::NONE) {
                        req->setStatus(TaskStatus::COMPLETED);
                    }
                } catch (...) {
                    req->setStatus(TaskStatus::FAILED);
                }
                try {
                    tasker.taskCompleted(req);
                } catch (...) {
                }
            },
            priority);
    }

    void update() override {
        for (auto it = _entries.begin(); it != _entries.end();) {
            if ((*it)->isDone()) {
                std::unique_lock<std::shared_mutex> lk(_mutex);
                it = _entries.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool allTaskCompleted() const {
        std::shared_lock<std::shared_mutex> lk(_mutex);
        return _entries.empty();
    }

   private:
    BS::thread_pool<BS::tp::priority>& _pool;
    std::deque<std::shared_ptr<ITask>> _entries;
    mutable std::shared_mutex _mutex;
};

}  // namespace pattern
}  // namespace benlib

#endif  // BENLIB_TASK_HPP_
