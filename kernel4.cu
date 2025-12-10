
#include "common.h"
#include "matrix.h"
#include "timer.h"
#include <iostream>
#include <chrono>

template <int TILE_SIZE>
__global__ void kernel4(CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                        COOMatrix *cooMatrix_d)
{
    const unsigned int row = blockIdx.x;

    if (row >= csrMatrix1_d->numRows)
        return;

    __shared__ float acc[TILE_SIZE];
    __shared__ unsigned int tile_count;
    __shared__ unsigned int tile_global_start;
    __shared__ unsigned int tile_write_offset;
    unsigned int tile_diff;

    for (int tileStart = 0; tileStart < csrMatrix2_d->numCols; tileStart += TILE_SIZE)
    {
        const int tileEnd = min(tileStart + TILE_SIZE, csrMatrix2_d->numCols);
        tile_diff = tileEnd - tileStart;
        for (unsigned int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x)
            acc[i] = 0.0f;

        __syncthreads();

        int local_count = 0;
        // for each non-zero at A[row (per-thread), colA]
        const auto row1_start = csrMatrix1_d->rowPtrs[row];
        const auto row1_end = csrMatrix1_d->rowPtrs[row + 1];
        for (unsigned int i = row1_start + threadIdx.x; i < row1_end; i += blockDim.x)
        {
            // get colA and the value A[row,colA]
            unsigned int colA = csrMatrix1_d->colIdxs[i];
            float valA = csrMatrix1_d->values[i];

            // for each non-zero at B[colA, colB]
            // hoisting from kernel 1
            const auto rowB_start = csrMatrix2_d->rowPtrs[colA];
            const auto rowB_end = csrMatrix2_d->rowPtrs[colA + 1];
            for (unsigned int j = rowB_start; j < rowB_end; ++j)
            {
                unsigned int colB = csrMatrix2_d->colIdxs[j];
                // only accumulate if colB 256is in current tile
                if (colB >= tileStart && colB < tileEnd)
                {
                    auto prev = atomicAdd(&acc[colB - tileStart], valA * csrMatrix2_d->values[j]);
                    if (prev == 0.0f)
                    {
                        local_count++;
                    }
                }
            }
        }

        // raising this above the previous __syncthreads() allows us to get dual use out of it
        if (threadIdx.x == 0)
        {
            tile_count = 0;
            tile_write_offset = 0;
        }

        __syncthreads();

        // reduction with warp-level shuffls
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2)
        {
            local_count += __shfl_down_sync(mask, local_count, offset);
        }

        // only the first thread uses atomics
        if ((threadIdx.x % 32) == 0)
        {
            atomicAdd(&tile_count, local_count);
        }

        __syncthreads();

        if (threadIdx.x == 0 && tile_count > 0)
        {
            tile_global_start = atomicAdd(&cooMatrix_d->numNonzeros, tile_count);
        }

        __syncthreads();

        if (tile_count > 0)
        {
            for (int i = threadIdx.x; i < tile_diff; i += blockDim.x)
            {
                if (acc[i] != 0.0f)
                {
                    // attomic adds to shmem local index are much faster
                    // however, most of the benefit comes from cooMatrix_d writes being faster
                    // even though the atomicAdd to the num non-zero was only ~6% of execution time, it
                    // sped up the cooMatrix_d writes
                    auto local_index = atomicAdd(&tile_write_offset, 1);

                    // turn back into a global pos
                    auto global_idx = tile_global_start + local_index;

                    cooMatrix_d->rowIdxs[global_idx] = row;
                    cooMatrix_d->colIdxs[global_idx] = tileStart + i;
                    cooMatrix_d->values[global_idx] = acc[i];
                }
            }
        }

        __syncthreads();
    }
}

template <size_t block_size = 256, int TILE_SIZE = 64 * 32>
void tmpl_spmspm_gpu4(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                      CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                      COOMatrix *cooMatrix_d)
{
    cudaError_t err;

    // launch kernels
    kernel4<TILE_SIZE><<<csrMatrix1->numRows, block_size>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void reset_coo_counter(COOMatrix *coo)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        coo->numNonzeros = 0;
    }
}

// Helper to run a specific configuration
template <size_t BLOCK_SIZE, int TILE_SIZE>
void run_benchmark_config(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                          CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                          COOMatrix *cooMatrix_d,
                          size_t skip, size_t retries)
{
    // Warm-up
    for (size_t i = 0; i < skip; ++i)
    {
        reset_coo_counter<<<1, 1>>>(cooMatrix_d);
        tmpl_spmspm_gpu4<BLOCK_SIZE, TILE_SIZE>(csrMatrix1, csrMatrix2, csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
    }
    cudaDeviceSynchronize();

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < retries; ++i)
    {
        reset_coo_counter<<<1, 1>>>(cooMatrix_d);
        tmpl_spmspm_gpu4<BLOCK_SIZE, TILE_SIZE>(csrMatrix1, csrMatrix2, csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_sec = elapsed.count() / retries;

    std::cerr << BLOCK_SIZE
              << "," << TILE_SIZE
              << "," << avg_time_sec << std::endl;
}

// Recursive template to iterate over Tile Sizes
template <size_t BLOCK_SIZE, int CURRENT_TILE, int MAX_TILE>
struct TileLoop
{
    static void run(CSRMatrix *h1, CSRMatrix *h2, CSRMatrix *d1, CSRMatrix *d2, COOMatrix *d_coo, size_t skip, size_t retries)
    {
        run_benchmark_config<BLOCK_SIZE, CURRENT_TILE>(h1, h2, d1, d2, d_coo, skip, retries);
        
        if constexpr (CURRENT_TILE * 2 <= MAX_TILE)
        {
            TileLoop<BLOCK_SIZE, CURRENT_TILE * 2, MAX_TILE>::run(h1, h2, d1, d2, d_coo, skip, retries);
        }
    }
};

// Recursive template to iterate over Block Sizes
template <size_t CURRENT_BLOCK, size_t MAX_BLOCK, int MIN_TILE, int MAX_TILE>
struct BlockLoop
{
    static void run(CSRMatrix *h1, CSRMatrix *h2, CSRMatrix *d1, CSRMatrix *d2, COOMatrix *d_coo, size_t skip, size_t retries)
    {
        // Run all tile variations for this block size
        TileLoop<CURRENT_BLOCK, MIN_TILE, MAX_TILE>::run(h1, h2, d1, d2, d_coo, skip, retries);

        if constexpr (CURRENT_BLOCK * 2 <= MAX_BLOCK)
        {
            BlockLoop<CURRENT_BLOCK * 2, MAX_BLOCK, MIN_TILE, MAX_TILE>::run(h1, h2, d1, d2, d_coo, skip, retries);
        }
    }
};

void test_spspm_gpu4(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                   CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                   COOMatrix *cooMatrix_d)
{
    constexpr size_t min_block_size = 64;
    constexpr size_t max_block_size = 1024;

    constexpr size_t tile_size_min = 64;
    constexpr size_t tile_size_max = 64 * 64;

    constexpr size_t skip = 3;       // number of launches to ignore for warm-up
    constexpr size_t num_retries = 10; // number of timed launches

    std::cerr << "Running Benchmark for Kernel 4..." << std::endl;
    std::cerr << "--------------------------------------------------" << std::endl;
    std::cerr << "Block Size,Tile Size,Time" << std::endl;

    // Start the compile-time loop recursion
    BlockLoop<min_block_size, max_block_size, tile_size_min, tile_size_max>::run(
        csrMatrix1, csrMatrix2, 
        csrMatrix1_d, csrMatrix2_d, 
        cooMatrix_d, 
        skip, num_retries
    );
    
    std::cerr << "--------------------------------------------------" << std::endl;
}

void spmspm_gpu4(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                 CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                 COOMatrix *cooMatrix_d)
{
    tmpl_spmspm_gpu4<128, 2048>(csrMatrix1, csrMatrix2,
                     csrMatrix1_d, csrMatrix2_d,
                     cooMatrix_d);
}
