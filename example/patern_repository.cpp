#include <benlib/patern/repository.hpp>
#include <iostream>
#include <memory>

class Test : public benlib::patern::Repository<Test> {
   private:
    size_t _id = 0;

   public:
    Test() : benlib::patern::Repository<Test>() { _id = benlib::patern::Repository<Test>::getData().size(); }
    ~Test() {}

    void print() { std::cout << "Hello, i'm Test N°" << _id << std::endl; }
};

auto main() -> int {
    std::unique_ptr<Test> test1 = std::make_unique<Test>();
    std::unique_ptr<Test> test2 = std::make_unique<Test>();
    std::unique_ptr<Test> test3 = std::make_unique<Test>();
    std::unique_ptr<Test> test4 = std::make_unique<Test>();

    std::cout << "There are " << benlib::patern::Repository<Test>::getData().size() << " Test object(s) in the repository." << std::endl;
    std::cout << "Remove test4" << std::endl;
    test4.reset();

    std::cout << "There are " << benlib::patern::Repository<Test>::getData().size() << " Test object(s) in the repository." << std::endl;

    for (auto& test : benlib::patern::Repository<Test>::getData()) {
        test->print();
    }

    return 0;
}