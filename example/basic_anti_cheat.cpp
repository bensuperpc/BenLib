#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <thread>

class SecureHealth {
   private:
    std::unique_ptr<int> _health1 = nullptr;
    std::unique_ptr<int> _health2 = nullptr;
    std::unique_ptr<int> _health3 = nullptr;

    std::unique_ptr<int> _healthChecksum1 = nullptr;
    std::unique_ptr<int> _healthChecksum2 = nullptr;
    std::unique_ptr<int> _healthChecksum3 = nullptr;

    std::unique_ptr<int> _hashKey1 = nullptr;
    std::unique_ptr<int> _hashKey2 = nullptr;
    std::unique_ptr<int> _hashKey3 = nullptr;

    std::unique_ptr<int> _encryptionKey = nullptr;

    int computeChecksum(int h, int hv) const {
        return h * hv;
    }

    int encrypt(int h) const { return h ^ *_encryptionKey; }
    int decrypt(int h) const { return h ^ *_encryptionKey; }

    int getRandomValue(int min = -10000, int max = 10000) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

   public:
    SecureHealth(int h) {
        setHealth(h);
    }

    void setHealth(int h) {
        _hashKey1 = std::make_unique<int>(getRandomValue());
        _hashKey2 = std::make_unique<int>(getRandomValue());
        _hashKey3 = std::make_unique<int>(getRandomValue());

        _encryptionKey = std::make_unique<int>(getRandomValue());

        _health1 = std::make_unique<int>(encrypt(h));
        _health2 = std::make_unique<int>(encrypt(h));
        _health3 = std::make_unique<int>(encrypt(h));

        _healthChecksum1 = std::make_unique<int>(computeChecksum(decrypt(*_health1), *_hashKey1));
        _healthChecksum2 = std::make_unique<int>(computeChecksum(decrypt(*_health2), *_hashKey2));
        _healthChecksum3 = std::make_unique<int>(computeChecksum(decrypt(*_health3), *_hashKey3));
    }

    bool isTampered() const {  
        if (*_healthChecksum1 != computeChecksum(decrypt(*_health1), *_hashKey1)) {
            return true;
        }
        if (*_healthChecksum2 != computeChecksum(decrypt(*_health2), *_hashKey2)) {
            return true;
        }
        if (*_healthChecksum3 != computeChecksum(decrypt(*_health3), *_hashKey3)) {
            return true;
        }
        if (decrypt(*_health1) != decrypt(*_health2) || decrypt(*_health2) != decrypt(*_health3)) {
            return true;
        }
        return false;
    }

    void randomizeAddress() {
        int move = getRandomValue(0, 30);
        if (isTampered()) {
            std::terminate();
        }

        if (move >= 0 && move <= 10) {
            setHealth(decrypt(*_health1));
        } else if (move > 10 && move <= 20) {
            setHealth(decrypt(*_health2));
        } else {
            setHealth(decrypt(*_health3));
        }
    }

    int getHealth() const { return decrypt(*_health1); }

    // For testing purposes
    int getCopy1Health() const { return decrypt(*_health2); }
    int getCopy2Health() const { return decrypt(*_health3); }
    void externalHealth1Change(int h) {
        *_health1 = h;
    }
    void externalHealth2Change(int h) {
        *_health2 = h;
    }
    void externalHealth3Change(int h) {
        *_health3 = h;
    }
};

int main() {
    SecureHealth playerHealth(100);
    std::cout << "Health: " << playerHealth.getHealth() << "\n";
    playerHealth.randomizeAddress();

    playerHealth.externalHealth1Change(9999);

    if (playerHealth.isTampered()) {
        std::cout << "Memory Tampering Detected!: " << playerHealth.getHealth() << " instead of " << playerHealth.getCopy1Health() << "\n";
    }
}
