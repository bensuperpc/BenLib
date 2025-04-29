#include <benlib/pattern/repository.hpp>
#include <iostream>
#include <memory>

class Test : public benlib::pattern::Repository<Test> {
   private:
    size_t _id = 0;

   public:
    Test() : benlib::pattern::Repository<Test>() { _id = benlib::pattern::Repository<Test>::data().size(); }
    ~Test() {}

    void print() { std::cout << "Hello, i'm Test N°" << _id << std::endl; }
};

auto main() -> int {
    std::unique_ptr<Test> test1 = std::make_unique<Test>();
    std::unique_ptr<Test> test2 = std::make_unique<Test>();
    std::unique_ptr<Test> test3 = std::make_unique<Test>();
    std::unique_ptr<Test> test4 = std::make_unique<Test>();

    std::cout << "There are " << benlib::pattern::Repository<Test>::data().size() << " Test object(s) in the repository." << std::endl;
    std::cout << "Remove test4" << std::endl;
    test4.reset();

    std::cout << "There are " << benlib::pattern::Repository<Test>::data().size() << " Test object(s) in the repository." << std::endl;

    for (auto& test : benlib::pattern::Repository<Test>::data()) {
        test->print();
    }

    return 0;
}