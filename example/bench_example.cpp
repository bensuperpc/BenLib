/*

template <typename T>
void arraySubABC(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] - B[i];
    }
}

template <typename T>
void arrayMulABC(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] * B[i];
    }
}

template <typename T>
void arrayDivABC(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] / B[i];
    }
}

template <typename T>
void arrayModABC(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] % B[i];
    }
}
*/


#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

template <typename T>
void arrayAddABC(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr std::size_t attempts = std::pow(2, 12);
    constexpr std::size_t SIZE = std::pow(2, 12);
    volatile int8_t vectorAvalue = 1;
    volatile int8_t vectorBvalue = 2;
    volatile int8_t vectorCvalue = 0;

    std::vector<int8_t> A(SIZE);
    std::vector<int8_t> B(SIZE);
    std::vector<int8_t> C(SIZE);

    std::vector<double> times(attempts);
    std::vector<double> throughputs(attempts);
    std::vector<double> operations(attempts);

    for (std::size_t i = 0; i < attempts; ++i) {
        //std::cout << "Attempt: " << i << std::endl;
        std::iota(A.begin(), A.end(), vectorAvalue);
        std::iota(B.begin(), B.end(), vectorBvalue);
        std::fill(C.begin(), C.end(), vectorCvalue);

        auto start = std::chrono::high_resolution_clock::now();
        arrayAddABC(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
        times[i] = elapsed.count();

        double totalBytes = 3 * SIZE * sizeof(int8_t);
        double throughput = (totalBytes / elapsed.count()) / (1024 * 1024 * 1024);
        //std::cout << "Throughput: " << throughput << " GB/s" << std::endl;
        throughputs[i] = throughput;
        
        double operation = (SIZE / elapsed.count()) / (1024 * 1024 * 1024);
        //std::cout << "Operation: " << operation << " GB/s" << std::endl;
        operations[i] = operation;

        //std::cout << std::endl;
    }

    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(attempts);
    double avgThroughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / static_cast<double>(attempts);
    double avgOperation = std::accumulate(operations.begin(), operations.end(), 0.0) / static_cast<double>(attempts);
    
    double minTime = *std::min_element(times.begin(), times.end());
    double maxTime = *std::max_element(times.begin(), times.end());
    double minOperation = *std::min_element(operations.begin(), operations.end());

    double minThroughput = *std::min_element(throughputs.begin(), throughputs.end());
    double maxThroughput = *std::max_element(throughputs.begin(), throughputs.end());
    double maxOperation = *std::max_element(operations.begin(), operations.end());

    std::cout << "Average time: " << avgTime << " s per attempt with " << SIZE << " elements (" << sizeof(int8_t) << " bytes)" << std::endl;
    std::cout << "Min time: " << minTime << " s per attempt with " << SIZE << " elements (" << sizeof(int8_t) << " bytes)" << std::endl;
    std::cout << "Max time: " << maxTime << " s per attempt with " << SIZE << " elements (" << sizeof(int8_t) << " bytes)" << std::endl;

    std::cout << "Average throughput: " << avgThroughput << " GB/s" << std::endl;
    std::cout << "Min throughput: " << minThroughput << " GB/s" << std::endl;
    std::cout << "Max throughput: " << maxThroughput << " GB/s" << std::endl;

    std::cout << "Average operation: " << avgOperation << " GOps/s (" << sizeof(int8_t) << " bytes)" << std::endl;
    std::cout << "Min operation: " << minOperation << " GOps/s (" << sizeof(int8_t) << " bytes)" << std::endl;
    std::cout << "Max operation: " << maxOperation << " GOps/s (" << sizeof(int8_t) << " bytes)" << std::endl;


    return 0;
}
