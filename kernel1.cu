
#include "common.h"
#include "timer.h"

constexpr int TILE_SIZE = 64 * 8;

__global__ void kernel1(CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                        COOMatrix *cooMatrix_d) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= csrMatrix1_d->numRows)
    return;

  float acc[TILE_SIZE];

  // hoist/cache invariants
  unsigned int startA = csrMatrix1_d->rowPtrs[row];
  unsigned int endA = csrMatrix1_d->rowPtrs[row + 1];

  const unsigned int *__restrict__ Acols = csrMatrix1_d->colIdxs;
  const float *__restrict__ Avals = csrMatrix1_d->values;

  const unsigned *__restrict__ Brows = csrMatrix2_d->rowPtrs;
  const unsigned int *__restrict__ Bcols = csrMatrix2_d->colIdxs;
  const float *__restrict__ Bvals = csrMatrix2_d->values;
  const unsigned int BnumCols = csrMatrix2_d->numCols;

  for (int tileStart = 0; tileStart < BnumCols; tileStart += TILE_SIZE) {
    int tileEnd = min(tileStart + TILE_SIZE, BnumCols);
    for (unsigned int i = 0; i < tileEnd - tileStart; ++i)
      acc[i] = 0.0f;

    // for each non-zero at A[row (per-thread), colA]
    for (unsigned int i = startA; i < endA; ++i) {
      // get colA and the value A[row,colA]
      unsigned int colA = Acols[i];
      float valA = Avals[i];

      unsigned start = Brows[colA];
      unsigned end = Brows[colA + 1];

      // for each non-zero at B[colA, colB]
      for (unsigned int j = start; j < end; ++j) {
        unsigned int colB = Bcols[j];
        // only accumulate if colB is in current tile
        if (colB >= tileStart && colB < tileEnd) {
          acc[colB - tileStart] += valA * Bvals[j];
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

void spmspm_gpu1(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                 CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                 COOMatrix *cooMatrix_d) {
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
