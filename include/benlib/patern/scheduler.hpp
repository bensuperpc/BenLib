/**
 * @file scheduler.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-06
 *
 * MIT License
 *
 */

#ifndef BENLIB_PATERN_SCHEDULER_HPP_
#define BENLIB_PATERN_SCHEDULER_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <shared_mutex>

namespace benlib {
namespace patern {

class App {
public:
    App() = default;
    virtual ~App() = default;
    virtual void update() = 0;
};

class Scheduler {
   public:
    explicit Scheduler() : _running(false), _updateFrequency(std::chrono::milliseconds(10)) {
        start();
    }
    virtual ~Scheduler() {
        stop();
    }

    void addApp(std::shared_ptr<App> app) {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _apps.push_back(app);
    }

    void removeApp(std::shared_ptr<App> app) {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _apps.erase(std::remove(_apps.begin(), _apps.end(), app), _apps.end());
    }

    void setUpdateFrequency(std::chrono::milliseconds updateFrequency) {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _updateFrequency = updateFrequency;
    }

    bool isRunning() const {
        return _running.load();
    }

    void stop() {
        _running.store(false);
        if (_thread.joinable()) {
            _thread.join();
        }
    }

    void start() {
        if (!_running.load() && !_thread.joinable()) {
            _running.store(true);
            _thread = std::thread(&Scheduler::runner, this);
            return;
        }
    }

    std::shared_mutex& getMutex() {
        return _mutex;
    }

    private:
    void runner() {
        while (_running.load()) {
            std::this_thread::sleep_for(_updateFrequency);
            {
                std::unique_lock<std::shared_mutex> lock(_mutex);
                for (auto& app : _apps) {
                    app->update();
                }
            }
        }
    }

    std::vector<std::shared_ptr<App>> _apps;
    std::atomic<bool> _running = false;
    std::thread _thread;
    std::shared_mutex _mutex;
    std::chrono::milliseconds _updateFrequency = std::chrono::milliseconds(10);
};

}  // namespace patern
}  // namespace benlib
#endif
