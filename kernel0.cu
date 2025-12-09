
#include "common.h"
#include "matrix.h"
#include "timer.h"


constexpr int TILE_SIZE = 64 * 8;


__global__ void kernel0(CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                        COOMatrix *cooMatrix_d) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= csrMatrix1_d->numRows) return;

  float acc[TILE_SIZE];

  for (int tileStart = 0; tileStart < csrMatrix2_d->numCols; tileStart += TILE_SIZE) {
    int tileEnd = min(tileStart + TILE_SIZE, csrMatrix2_d->numCols);
    for(unsigned int i = 0; i < tileEnd - tileStart; ++i) acc[i] = 0.0f;

    // for each non-zero at A[row (per-thread), colA]
    for(unsigned int i = csrMatrix1_d->rowPtrs[row]; i < csrMatrix1_d->rowPtrs[row+1]; ++i){
      // get colA and the value A[row,colA]
      unsigned int colA = csrMatrix1_d->colIdxs[i];
      float valA = csrMatrix1_d->values[i];

      // for each non-zero at B[colA, colB]
      for(unsigned int j = csrMatrix2_d->rowPtrs[colA]; j < csrMatrix2_d->rowPtrs[colA+1]; ++j){
        unsigned int colB = csrMatrix2_d->colIdxs[j];
        // only accumulate if colB is in current tile
        if (colB >= tileStart && colB < tileEnd) {
          acc[colB - tileStart] += valA * csrMatrix2_d->values[j];
        }
      }
  }

  // write non-zeros to coo matrix
  for (int i = 0; i < tileEnd - tileStart; i++) {
    if (acc[i] != 0.0f) {
        int idx = atomicAdd(&cooMatrix_d->numNonzeros, 1);
        if (idx < cooMatrix_d->capacity) {
            cooMatrix_d->rowIdxs[idx] = row;
            cooMatrix_d->colIdxs[idx] = tileStart + i;
            cooMatrix_d->values[idx] = acc[i];
        }
      }
    }
  }
}


void spmspm_gpu0(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
            CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
            COOMatrix *cooMatrix_d) {
  cudaError_t err;
  // launch kernels
  int block_size = 256;
  int grid_size = (csrMatrix1->numRows + block_size - 1) / block_size;

  kernel0<<<grid_size, block_size>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}
