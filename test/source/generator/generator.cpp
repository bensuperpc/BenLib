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

#include "generator.hpp"

Generator::Generator(int32_t _seed) : seed(_seed) {
    fnSimplex = FastNoise::New<FastNoise::Perlin>();
    fnFractal = FastNoise::New<FastNoise::FractalFBm>();

    fnFractal->SetSource(fnSimplex);
    fnFractal->SetOctaveCount(octaves);
    fnFractal->SetGain(gain);
    fnFractal->SetLacunarity(lacunarity);
    fnFractal->SetWeightedStrength(weighted_strength);
}

Generator::Generator() {
    fnSimplex = FastNoise::New<FastNoise::Perlin>();
    fnFractal = FastNoise::New<FastNoise::FractalFBm>();

    fnFractal->SetSource(fnSimplex);
    fnFractal->SetOctaveCount(octaves);
    fnFractal->SetGain(gain);
    fnFractal->SetLacunarity(lacunarity);
    fnFractal->SetWeightedStrength(weighted_strength);

    randomizeSeed();
}

Generator::~Generator() {}

void Generator::reseed(int32_t _seed) {
    this->seed = _seed;
}

int32_t Generator::randomizeSeed() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
    this->seed = dis(gen);

    return seed;
}

uint32_t Generator::get_seed() const {
    return seed;
}

void Generator::setOctaves(uint32_t _octaves) {
    this->octaves = _octaves;
    fnFractal->SetOctaveCount(octaves);
}

uint32_t Generator::getOctaves() const {
    return octaves;
}

void Generator::setLacunarity(float _lacunarity) {
    this->lacunarity = _lacunarity;
    fnFractal->SetLacunarity(lacunarity);
}

float Generator::getLacunarity() const {
    return lacunarity;
}

void Generator::setGain(float _gain) {
    this->gain = _gain;
    fnFractal->SetGain(gain);
}

float Generator::getGain() const {
    return gain;
}

void Generator::setFrequency(float _frequency) {
    this->frequency = _frequency;
}

float Generator::getFrequency() const {
    return frequency;
}

void Generator::setWeightedStrength(float _weighted_strength) {
    this->weighted_strength = _weighted_strength;
    fnFractal->SetWeightedStrength(weighted_strength);
}

float Generator::getWeightedStrength() const {
    return weighted_strength;
}

void Generator::setMultiplier(uint32_t _multiplier) {
    this->multiplier = _multiplier;
}

uint32_t Generator::getMultiplier() const {
    return multiplier;
}

std::vector<uint32_t> Generator::generate2dMeightmap(const int32_t begin_x,
                                                     [[maybe_unused]] const int32_t begin_y,
                                                     const int32_t begin_z,
                                                     const uint32_t size_x,
                                                     [[maybe_unused]] const uint32_t size_y,
                                                     const uint32_t size_z) {
    constexpr bool debug = false;

    std::vector<uint32_t> heightmap(size_x * size_z);

    std::vector<float> noise_output(size_x * size_z);

    if (fnFractal.get() == nullptr) {
        std::cout << "fnFractal is nullptr" << std::endl;
        return heightmap;
    }

    fnFractal->GenUniformGrid2D(noise_output.data(), begin_x, begin_z, size_x, size_z, frequency, seed);

    // Convert noise_output to heightmap
    for (uint32_t i = 0; i < size_x * size_z; i++) {
        heightmap[i] = static_cast<uint32_t>((noise_output[i] + 1.0) * multiplier);
        if constexpr (debug) {
            std::cout << "i: " << i << ", value: " << noise_output[i] << ", heightmap: " << heightmap[i] << std::endl;
        }
    }

    if constexpr (debug) {
        // cout max and min
        auto minmax = std::minmax_element(heightmap.begin(), heightmap.end());
        std::cout << "min: " << static_cast<int32_t>(*minmax.first) << std::endl;
        std::cout << "max: " << static_cast<int32_t>(*minmax.second) << std::endl;
    }
    return heightmap;
}

std::vector<uint32_t> Generator::generate3dHeightmap(const int32_t begin_x,
                                                     const int32_t begin_y,
                                                     const int32_t begin_z,
                                                     const uint32_t size_x,
                                                     const uint32_t size_y,
                                                     const uint32_t size_z) {
    constexpr bool debug = false;

    std::vector<uint32_t> heightmap(size_x * size_y * size_z);

    std::vector<float> noise_output(size_x * size_y * size_z);

    if (fnFractal.get() == nullptr) {
        std::cout << "fnFractal is nullptr" << std::endl;
        return heightmap;
    }

    fnFractal->GenUniformGrid3D(noise_output.data(), begin_x, begin_y, begin_z, size_x, size_y, size_z, frequency, seed);

    // Convert noise_output to heightmap
    for (uint32_t i = 0; i < size_x * size_y * size_z; i++) {
        heightmap[i] = static_cast<uint32_t>((noise_output[i] + 1.0) * multiplier);
        if constexpr (debug) {
            std::cout << "i: " << i << ", noise_output: " << noise_output[i] << ", heightmap: " << heightmap[i] << std::endl;
        }
    }

    if constexpr (debug) {
        // cout max and min
        auto minmax = std::minmax_element(heightmap.begin(), heightmap.end());
        std::cout << "min: " << static_cast<int32_t>(*minmax.first) << std::endl;
        std::cout << "max: " << static_cast<int32_t>(*minmax.second) << std::endl;
    }

    return heightmap;
}

/*
std::unique_ptr<Chunk> Generator::generateChunk(const int32_t chunk_x,
                                                const int32_t chunk_y,
                                                const int32_t chunk_z,
                                                const bool generate_3d_terrain)
{ const int32_t real_x = chunk_x * Chunk::chunk_size_x; const int32_t real_y =
chunk_y * Chunk::chunk_size_y; const int32_t real_z = chunk_z *
Chunk::chunk_size_z;

    std::vector<Block> blocks;

    std::unique_ptr<Chunk> _chunk = std::make_unique<Chunk>();

    if (generate_3d_terrain) {
        blocks = std::move(generate3d(real_x, real_y, real_z,
Chunk::chunk_size_x, Chunk::chunk_size_y, Chunk::chunk_size_z)); } else { blocks
= std::move(generate2d(real_x, real_y, real_z, Chunk::chunk_size_x,
Chunk::chunk_size_y, Chunk::chunk_size_z));
    }

    _chunk->set_blocks(blocks);
    _chunk->set_chuck_pos(chunk_x, chunk_y, chunk_z);

    return _chunk;
}

[[nodiscard]] std::vector<std::unique_ptr<Chunk>>
Generator::generateChunks(const int32_t begin_chunk_x, const int32_t
begin_chunk_y, const int32_t begin_chunk_z, const uint32_t size_x, const
uint32_t size_y, const uint32_t size_z, const bool generate_3d_terrain) {
    constexpr bool debug = false;

    std::vector<std::unique_ptr<Chunk>> chunks;
    chunks.reserve(size_x * size_y * size_z);

#pragma omp parallel for collapse(3) schedule(auto)
    for (int32_t x = begin_chunk_x; x < begin_chunk_x + size_x; x++) {
        for (int32_t z = begin_chunk_y; z < begin_chunk_y + size_z; z++) {
            for (int32_t y = begin_chunk_z; y < begin_chunk_z + size_y; y++) {
                auto gen_chunk = generateChunk(x, y, z, generate_3d_terrain);
#pragma omp critical
                chunks.emplace_back(std::move(gen_chunk));
            }
        }
    }

    return chunks;
}
std::vector<Block> Generator::generate2d(const int32_t begin_x,
                                         const int32_t begin_y,
                                         const int32_t begin_z,
                                         const uint32_t size_x,
                                         const uint32_t size_y,
                                         const uint32_t size_z) {
    constexpr bool debug = false;

    std::vector<uint32_t> heightmap;
    std::vector<Block> blocks = std::vector<Block>(size_x * size_y * size_z,
Block());

    heightmap = std::move(generate2dMeightmap(begin_x, begin_y, begin_z, size_x,
size_y, size_z));

    // Generate blocks
    for (uint32_t x = 0; x < size_x; x++) {
        for (uint32_t z = 0; z < size_z; z++) {
            // Noise value is divided by 4 to make it smaller and it is used as
the height of the Block (z) std::vector<Block>::size_type vec_index =
math::convert_to_1d(x, z, size_x, size_z);

            uint32_t noise_value = heightmap[vec_index] / 4;

            for (uint32_t y = 0; y < size_y; y++) {
                // Calculate real y from begin_y
                vec_index = math::convert_to_1d(x, y, z, size_x, size_y,
size_z);

                if constexpr (debug) {
                    std::cout << "x: " << x << ", z: " << z << ", y: " << y << "
index: " << vec_index
                              << ", noise: " <<
static_cast<int32_t>(noise_value) << std::endl;
                }

                Block& current_block = blocks[vec_index];

                // If the noise value is greater than the current Block, make it
air if (noise_value > 120) { current_block.block_type = block_type::stone;
                    continue;
                }
            }
        }
    }
    return blocks;
}

std::vector<Block> Generator::generate3d(const int32_t begin_x,
                                         const int32_t begin_y,
                                         const int32_t begin_z,
                                         const uint32_t size_x,
                                         const uint32_t size_y,
                                         const uint32_t size_z) {
    constexpr bool debug = false;

    std::vector<Block> blocks = std::vector<Block>(size_x * size_y * size_z,
Block());

    std::vector<uint32_t> heightmap = generate3dHeightmap(begin_x, begin_y,
begin_z, size_x, size_y, size_z);

    // Generate blocks
    for (uint32_t x = 0; x < size_x; x++) {
        for (uint32_t z = 0; z < size_z; z++) {
            for (uint32_t y = 0; y < size_y; y++) {
                size_t vec_index = math::convert_to_1d(x, y, z, size_x, size_y,
size_z); const uint32_t noise_value = heightmap[vec_index]; auto& current_block
= blocks[vec_index];

                if constexpr (debug) {
                    std::cout << "x: " << x << ", z: " << z << ", y: " << y << "
index: " << vec_index
                              << ", noise: " <<
static_cast<int32_t>(noise_value) << std::endl;
                }

                if (noise_value > 120) {
                    current_block.block_type = block_type::stone;
                    continue;
                }
            }
        }
    }
    return blocks;
}
*/
