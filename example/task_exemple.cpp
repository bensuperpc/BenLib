
#include "benlib/pattern/task.hpp"

class IntTask : public benlib::pattern::Task<int> {
   public:
    void start() override {
        std::cout << "[IntTask] start\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        setValue(42);
        std::cout << "[IntTask] completed\n";
    }
};
class StringTask : public benlib::pattern::Task<std::string> {
   public:
    void start() override {
        std::cout << "[StringTask] start\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        fail();
        setValue("Hello, world!");
        std::cout << "[StringTask] completed\n";
    }
};
class CharTask : public benlib::pattern::Task<char> {
   public:
    void start() override {
        std::cout << "[CharTask] start\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        cancel();
        setValue('A');
        std::cout << "[CharTask] completed\n";
    }
};
class MyApp : public benlib::pattern::ITasker<int>, public benlib::pattern::ITasker<std::string>, public benlib::pattern::ITasker<char> {
   public:
    MyApp(benlib::pattern::TaskHandler& handler) : _handler(handler) {
        _handler.startTask(std::make_shared<IntTask>(), *this);
        _handler.startTask(std::make_shared<CharTask>(), *this);
        _handler.startTask(std::make_shared<StringTask>(), *this);
    }
    void taskCompleted(std::shared_ptr<benlib::pattern::Task<int>> req) override {
        switch (req->status()) {
            case benlib::pattern::TaskStatus::COMPLETED:
                std::cout << "[MyApp] int completed, status=" << int(req->status()) << ", value=" << req->getValue() << "\n";
                break;
            case benlib::pattern::TaskStatus::CANCELLED:
                std::cout << "[MyApp] int cancelled, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::FAILED:
                std::cout << "[MyApp] int failed, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::INPROGRESS:
                break;
            case benlib::pattern::TaskStatus::NONE:
                std::cout << "[MyApp] int none, status=" << int(req->status()) << "\n";
                break;
        }
    }
    void taskCompleted(std::shared_ptr<benlib::pattern::Task<std::string>> req) override {
        switch (req->status()) {
            case benlib::pattern::TaskStatus::COMPLETED:
                std::cout << "[MyApp] string completed, status=" << int(req->status()) << ", value=" << req->getValue() << "\n";
                break;
            case benlib::pattern::TaskStatus::CANCELLED:
                std::cout << "[MyApp] string cancelled, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::FAILED:
                std::cout << "[MyApp] string failed, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::INPROGRESS:
                break;
            case benlib::pattern::TaskStatus::NONE:
                std::cout << "[MyApp] string none, status=" << int(req->status()) << "\n";
                break;
        }
    }
    void taskCompleted(std::shared_ptr<benlib::pattern::Task<char>> req) override {
        switch (req->status()) {
            case benlib::pattern::TaskStatus::COMPLETED:
                std::cout << "[MyApp] char completed, status=" << int(req->status()) << ", value=" << req->getValue() << "\n";
                break;
            case benlib::pattern::TaskStatus::CANCELLED:
                std::cout << "[MyApp] char cancelled, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::FAILED:
                std::cout << "[MyApp] char failed, status=" << int(req->status()) << "\n";
                break;
            case benlib::pattern::TaskStatus::INPROGRESS:
                break;
            case benlib::pattern::TaskStatus::NONE:
                std::cout << "[MyApp] char none, status=" << int(req->status()) << "\n";
                break;
        }
    }

   private:
    benlib::pattern::TaskHandler& _handler;
};

int main() {
    benlib::pattern::Scheduler scheduler;
    BS::thread_pool<BS::tp::priority> threadPool(4);
    scheduler.setDelayFrequency(std::chrono::milliseconds(50));
    auto handler = std::make_shared<benlib::pattern::TaskHandler>(threadPool);
    scheduler.addApp(handler);
    MyApp app(*handler);
    while (!handler->allTaskCompleted()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    scheduler.stop();
    return 0;
}