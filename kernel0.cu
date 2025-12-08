
#include "common.h"
#include "matrix.h"
#include "timer.h"

constexpr size_t max_num_cols = 64; //< Maximum number of columns

// TODO: define some kernel here
__global__ void kernel0(CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                        COOMatrix *cooMatrix_d) {
  // TOOD: check if I need to convert this to compressed spare column in doing a
  // transpose first
  int rowA = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: introduce and atomic for the number of output calls
  if (rowA >= csrMatrix1_d->numRows) {
    return;
  }
  float row_values[max_num_cols]; // this is the accumulator per row mentioned
                                  // in the readme

  const auto a_row_ptrs = csrMatrix1_d->rowPtrs;
  const int start_a = a_row_ptrs[rowA];
  const int end_a = a_row_ptrs[rowA + 1];

  const auto b_row_ptrs = csrMatrix2_d->rowPtrs;

  const auto n_cols_2 = csrMatrix2_d->numCols;
  for (int starting_col_range = 0; starting_col_range < n_cols_2;
       starting_col_range += max_num_cols) {
    // we define an exclusive range [starting_col_range, ending_col_range)
    int ending_col_range = starting_col_range + max_num_cols;
    ending_col_range = min(n_cols_2, ending_col_range);
    
    // clearing the accumulator values
    // saves a bit at the end by not going to 64 if it does not has to
    // probably has zero impact on performance, but it caused some bugs previously
    for (int i = 0; i < ending_col_range - starting_col_range; i++) {
        row_values[i] = 0.0F;
    }

    for (int i = start_a; i < end_a; i++) {
      const auto col_a = csrMatrix1_d->colIdxs[i];
      const auto val_a = csrMatrix1_d->values[i];
      const auto row_b = col_a;

      // inner for loop
      const auto startb = b_row_ptrs[row_b];
      const auto endb = b_row_ptrs[row_b + 1];

      // TODO accumulate the sum
      for(auto elem_b =  startb; elem_b < endb; elem_b++){
        // unpack the value
        auto col_b = csrMatrix2_d->colIdxs[elem_b];
        auto val_b = csrMatrix2_d->values[elem_b];

        // only add to the row values if within the column values
        bool in_range = col_b < ending_col_range && col_b >= starting_col_range;
        // you could accumulate a histogram of non-zero columns and then use that
        // this way this boolean check can not fail if you have a matrix which is
        // [ 1, 2, ..., 100,000]
        // [ 1, 0.0, ..., 100,000 ]
        // ...
        // this would not be a time complexity faster, but I think it would be faster
        if (in_range){
            row_values[col_b - starting_col_range] += val_a * val_b;
        }

      }
    }
    for (auto i = 0; i < ending_col_range - starting_col_range; ++i) {
      auto row_value = row_values[i];
      auto global_col = starting_col_range + i; // can I use iterators in kernels?

      if (row_value != 0.0F) {
        auto outIdx = atomicAdd(&cooMatrix_d->numNonzeros, 1);

        if (outIdx < cooMatrix_d->capacity) {
          cooMatrix_d->rowIdxs[outIdx] = rowA;
          cooMatrix_d->colIdxs[outIdx] = global_col;
          cooMatrix_d->values[outIdx] = row_value;
        }
      }
    }
  }
}

void spmspm_gpu0(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                 CSRMatrix *csrMatrix1_d, CSRMatrix *csrMatrix2_d,
                 COOMatrix *cooMatrix_d) {
  // cooMatrix_d -> the final matrix on the host
  // csrMatrix1 -> the host matrix A
  // csrMatrix2 -> the host matrix B
  // csrMatrix1_d -> the device matrix A
  // csrMatrix2_d -> the device matrix B

  // basic implementation plan
  // allocate *bufffers* for the coo matrix -> the size here is not trivially
  // known according to matrix.h and main.cu this is allready allocated to some
  // large size

  // allocating the number of non-zeros
  // unsigned int* d_nnz_counter;
  cudaError_t err;
  // // err = cudaMalloc((void**)&d_nnz_counter, sizeof(unsigned int));
  // // if(err != cudaSuccess) {
  // //     printf("Cuda malloc failed\n");
  // // }
  // err = cudaMemset(d_nnz_counter, 0, sizeof(unsigned int));
  // if(err != cudaSuccess) {
  //     printf("Cuda memset failed\n");
  // }

  // assert that the destination matrix is of zero size
  // assert(csrMatrix1->numRows == csrMatrix1->numCols);
  // assert(csrMatrix2->numRows == csrMatrix2->numCols);
  // assert(csrMatrix1->numRows == csrMatrix2->numRows);

  // launch kernels
  int block_size = 256;
  int grid_size = (csrMatrix1->numRows + block_size - 1) / block_size;

  kernel0<<<grid_size, block_size>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  // no copy step since it is a device matrix

  // cudaMemcpy(&cooMatrix_d->numNonzeros, d_nnz_counter, sizeof(unsigned int),
  // cudaMemcpyDeviceToHost); cudaFree(d_nnz_counter);
}
