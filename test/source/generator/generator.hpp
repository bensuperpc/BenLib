#ifndef WORLD_OF_CUBE_GENERATOR_HPP
#define WORLD_OF_CUBE_GENERATOR_HPP

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// #include <omp.h>

#ifndef FASTNOISE_HPP_INCLUDED
#define FASTNOISE_HPP_INCLUDED
#ifdef __GNUC__
#pragma GCC system_header
#endif
#ifdef __clang__
#pragma clang system_header
#endif
#include "FastNoise/FastNoise.h"
#endif

class Generator {
   public:
    explicit Generator(int32_t _seed);

    explicit Generator();

    ~Generator();

    void reseed(int32_t _seed);

    int32_t randomizeSeed();

    uint32_t get_seed() const;

    void setOctaves(uint32_t _octaves);

    uint32_t getOctaves() const;

    void setLacunarity(float _lacunarity);

    float getLacunarity() const;

    void setGain(float _gain);

    float getGain() const;

    void setFrequency(float _frequency);
    float getFrequency() const;

    void setWeightedStrength(float _weighted_strength);
    float getWeightedStrength() const;

    void setMultiplier(uint32_t _multiplier);

    uint32_t getMultiplier() const;

    std::vector<uint32_t> generate2dMeightmap(const int32_t begin_x,
                                              [[maybe_unused]] const int32_t begin_y,
                                              const int32_t begin_z,
                                              const uint32_t size_x,
                                              [[maybe_unused]] const uint32_t size_y,
                                              const uint32_t size_z);

    std::vector<uint32_t> generate3dHeightmap(const int32_t begin_x,
                                              const int32_t begin_y,
                                              const int32_t begin_z,
                                              const uint32_t size_x,
                                              const uint32_t size_y,
                                              const uint32_t size_z);
    /*
    std::unique_ptr<Chunk> generateChunk(const int32_t chunk_x, const int32_t
    chunk_y, const int32_t chunk_z, const bool generate_3d_terrain);

    [[nodiscard]] std::vector<std::unique_ptr<Chunk>> generateChunks(const int32_t
    begin_chunk_x, const int32_t begin_chunk_y, const int32_t begin_chunk_z, const
    uint32_t size_x, const uint32_t size_y, const uint32_t size_z, const bool
    generate_3d_terrain);

    std::vector<Block> generate2d(const int32_t begin_x,
                                  const int32_t begin_y,
                                  const int32_t begin_z,
                                  const uint32_t size_x,
                                  const uint32_t size_y,
                                  const uint32_t size_z);

    std::vector<Block> generate3d(const int32_t begin_x,
                                  const int32_t begin_y,
                                  const int32_t begin_z,
                                  const uint32_t size_x,
                                  const uint32_t size_y,
                                  const uint32_t size_z);
    */
   private:
    // default seed
    int32_t seed = 404;
    FastNoise::SmartNode<FastNoise::Perlin> fnSimplex;
    FastNoise::SmartNode<FastNoise::FractalFBm> fnFractal;

    int32_t octaves = 6;
    float lacunarity = 0.5f;
    float gain = 3.5f;
    float frequency = 0.4f;
    float weighted_strength = 0.0f;
    uint32_t multiplier = 128;
};

#endif  // WORLD_OF_CUBE_GENERATOR_HPP