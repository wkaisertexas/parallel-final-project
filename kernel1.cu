
#include "common.h"
#include "timer.h"

constexpr int TILE_SIZE = 64;

__global__ void kernel1(CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                        COOMatrix *cooMatrix_d) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= csrMatrix1_d->numRows) return;

    float acc[TILE_SIZE];
    
    for (int tileStart = 0; tileStart < csrMatrix2_d->numCols; tileStart += TILE_SIZE) {
        int tileEnd = min(tileStart + TILE_SIZE, csrMatrix2_d->numCols);
        for(unsigned int i = 0; i < TILE_SIZE; ++i) acc[i] = 0.0f;
        for(unsigned int i = csrMatrix1_d->rowPtrs[row]; i < csrMatrix1_d->rowPtrs[row+1]; ++i){
            unsigned int colA = csrMatrix1_d->colIdxs[i];
            float valA = csrMatrix1_d->values[i];
            for(unsigned int j = csrMatrix2_d->rowPtrs[colA]; j < csrMatrix2_d->rowPtrs[colA+1]; ++j){
                unsigned int colB = csrMatrix2_d->colIdxs[j];
                if (colB >= tileStart && colB < tileEnd) {
                    acc[colB - tileStart] += valA * csrMatrix2_d->values[j];
                }
            }
        }
        int local_nnz = 0;
        for (int i = 0; i < tileEnd - tileStart; i++) {
            if (acc[i] != 0.0f) local_nnz++;
        }
        if (local_nnz > 0) {
            // one atomic add per thread per tile (instead of one per non-zero)
            int start_idx = atomicAdd(&cooMatrix_d->numNonzeros, local_nnz);
            
            int current_idx = start_idx;
            for (int i = 0; i < tileEnd - tileStart; i++) {
                if (acc[i] != 0.0f) {
                    if (current_idx < cooMatrix_d->capacity) {
                        cooMatrix_d->rowIdxs[current_idx] = row;
                        cooMatrix_d->colIdxs[current_idx] = tileStart + i;
                        cooMatrix_d->values[current_idx] = acc[i];
                    }
                    current_idx++;
                }
            }
        }
    }
} 

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    cudaError_t err;
    // launch kernels
    int block_size = 256;
    int grid_size = (csrMatrix1->numRows + block_size - 1) / block_size;
  
    kernel1<<<grid_size, block_size>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
  
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

