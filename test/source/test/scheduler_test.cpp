/**
 * @file test_sphere.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#include "benlib/pattern/scheduler.hpp"

#include "gtest/gtest.h"

class BasicApp : public benlib::pattern::App {
public:
    BasicApp() = default;
    ~BasicApp() = default;

    void update() override {
        counter++;
    }

    uint64_t getCounter() const {
        return counter;
    }

    private:
     uint64_t counter = 0;
};

TEST(Scheduler, basic_1) {
    benlib::pattern::Scheduler scheduler;

    std::shared_ptr<BasicApp> app = std::make_shared<BasicApp>();
    
    scheduler.addApp(app);

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    EXPECT_GT(app->getCounter(), 0);

    scheduler.removeApp(app);

    scheduler.stop();

    EXPECT_EQ(scheduler.isRunning(), false);
}

TEST(Scheduler, basic_2) {
    benlib::pattern::Scheduler scheduler;

    std::vector<std::shared_ptr<BasicApp>> apps;
    for (int i = 0; i < 10; i++) {
        apps.push_back(std::make_shared<BasicApp>());
        scheduler.addApp(apps[i]);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    for (int i = 0; i < 10; i++) {
        EXPECT_GT(apps[i]->getCounter(), 0);
    }

    for (int i = 0; i < 10; i++) {
        scheduler.removeApp(apps[i]);
    }
    
    scheduler.stop();

    EXPECT_EQ(scheduler.isRunning(), false);
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
