#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>
#include "generator/generator.hpp"

static constexpr int64_t multiplier = 4;
static constexpr int64_t minRange = 16;
static constexpr int64_t maxRange = 64;
static constexpr int64_t minThreadRange = 1;
static constexpr int64_t maxThreadRange = 1;
static constexpr int64_t repetitions = 1;

static void DoSetup([[maybe_unused]] const benchmark::State& state) {}

static void DoTeardown([[maybe_unused]] const benchmark::State& state) {}

template <typename Type>
static void benlib_bench(benchmark::State& state) {
    auto range = state.range(0);

    benchmark::DoNotOptimize(range);

    for (auto _ : state) {
        if (get_benlib() != "benlib") {
            state.SkipWithError("get_benlib() != \"benlib\"");
        }

        state.PauseTiming();
        state.ResumeTiming();
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(Type));
}

BENCHMARK(benlib_bench<uint32_t>)
    ->Name("benlib_bench<uint32_t>")
    ->RangeMultiplier(multiplier)
    ->Range(minRange, maxRange)
    ->ThreadRange(minThreadRange, maxThreadRange)
    ->Unit(benchmark::kNanosecond)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Repetitions(repetitions);

// Run the benchmark
// BENCHMARK_MAIN();

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
