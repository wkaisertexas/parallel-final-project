
#include "common.h"
#include "matrix.h"
#include "timer.h"

constexpr int TILE_SIZE = 64 * 32;

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
                    atomicAdd(&acc[colB - tileStart], valA * csrMatrix2_d->values[j]);
                }
            }
        }


        // raising this above the previous __syncthreads() allows us to get dual use out of it
        if (threadIdx.x == 0) {
            tile_count = 0;
            tile_write_offset = 0;
        }

        __syncthreads();

        // local counting
        int local_count = 0;
        for (int i = threadIdx.x; i < tile_diff; i += blockDim.x)
        {
            if (acc[i] != 0.0f)
            {
                local_count++;
            }
        }

        // reduction with warp-level shuffls
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2) {
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

                    if (global_idx < cooMatrix_d->capacity)
                    {
                        cooMatrix_d->rowIdxs[global_idx] = row;
                        cooMatrix_d->colIdxs[global_idx] = tileStart + i;
                        cooMatrix_d->values[global_idx] = acc[i];
                    }
                }
            }
        }

        __syncthreads();
    }
}

void spmspm_gpu4(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                 CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                 COOMatrix *cooMatrix_d)
{
    cudaError_t err;
    // launch kernels
    int block_size = 256;
    //   int grid_size = (csrMatrix1->numRows + block_size - 1) / block_size;

    kernel4<<<csrMatrix1->numRows, block_size>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
