# Kernel 0 

Kernel 0 provides the base implementation for sparse matrix-matrix multiplication. For a multiplication of matrices `A` and `B` into output `C`, traditional matrix multiplication kernels (not sparse) multiply row `i` of `A` by a column `j` of `B` to get `C[i,j]`. 

Due to the CSR representation of `A` and `B`, however, accessing columns of `B` is inefficient because it would require searching for particular column indices. A better implementation leverages the structure of the CSR, which makes traversing rows inexpensive. Firstly, we assign each row of `A` a thread. This thread will update all and only the corresponding row of `C`. The kernel avoids searching for column indices of `B` by performing partial sums. For a row `i` in output matrix `C`, a non-zero element `A[i,j]` will contribute to `C[i,k]` for every non-zero `B[j,k]` in row `j` of `B`. By iterating over row `j` of `B` (efficient in CSR), the kernel accumulates these contributions into the appropriate output columns without ever needing to search for specific column indices.

There is an issue, however, with naively writing the partial sum to the output matrix. Since we will be writing to this position multiple times for different rows of `B`, how can we tell if anything has already been written to that particular position in the COO output? Do we create a new entry or not? Checking a value at a particular position is extremely expensive with COO (requires search), so we address this by writing partial sums to an accumulator. Only once this accumulator is filled with all partial sums do we write to the COO matrix. Since no other threads will write to the same location (because each thread is a row), we avoid the checking problem across threads.

This introduces another issue: we don't have enough memory to store all the partial sums. Therefore, we take a tiled approach. The accumulator is only responsible for accumulating results for `TILE_SIZE` columns at a time. After processing all contributions to the current tile, non-zero values are written to the COO output, and the accumulator is reused for the next tile of columns.

# Kernel 1

Kernel 1 makes minor optimizations that should persist throughout the other optimizations. 

For one, we hoisted certain invariants outside of loops where they don't need to be updated. When iterating over tiles, we will always be using the same row start and end points. Similarly, `csrMatrix2_d->numCols` is constant throughout the entire kernel execution, yet kernel0 accessed it through the struct pointer on every tile iteration. By loading it once into a local variable (`BnumCols`), we avoid repeated pointer dereferencing.

Second, we cache certain struct members that are re-used. Instead of requiring the GPU to first access the struct, then its member, then index, our kernel stores these pointers (e.g. `colIdxs`) into a variable.

Lastly, since we know there isn't memory overlap between pointers like `rowPtrs` and `colIdxs`, we can use the `__restrict__` keyword to cache more aggressively.

All in all, these optimizations resulted in about a 10% speedup. This kernel optimization was about refinement and setting a healthy baseline for future optimizations.

# Kernel 2

Kernel 2 provides a basic implementation of shared memory in the way most conducive to our original implementation and, interestingly, breaks a common pattern of optimization: thread coarsing. In this implementation, the accumulator is shared between threads in a block with each block working on a single row. This avoids global memory for accumulators and coalesses reads to `csrMatrix1_d->colIdxs` and `csrMatrix1_d->values`. However, only completing a row per block is doing the opposite of thread coarsening.

However, this adjustment approximately halves execution time from 22.5 ms to 10.2 ms, so the tradeoff is valid.

When profiling this kernel in nsight compute, it can be seen that the writeback stage to the COO matrix takes 53 % of total execution time (5 % + 14.3 % + 14.7 % + 18.3 %). This insight motivated kernel 3's modifications to use shared memory to perform the accumulation.

![Kernel 2 Nsight Compute "Kernel 2 Nsight Compute Profile"](./imgs/kernel2-ncu-report.png)

## Kernel 3

Kernel 3 introduces 3 new shared variables: `tile_count`, `tile_global_start` and `tile_write_offset`. Before writing to the cooMatrix, the total number of non-zero elements is tallied. Following this, a global offset from the number of non-zeros is computed. Then atomic additions are made to the `tile_write_offset` before writing to global memory.

The magnitude of impact of this optimization was suprising because avoiding the global memory atomic add to the number of non-zeros per non-zero was only 6% of kernel execution time. However, there was more time-savings when it came to writes made to cooMatrix_d.

The most likely explanation for this fact is how writes to COO are more coalessed because of the faster atomic adds to `tile_write_offset`. This is important because this shows how profilers reporting of time-taken per line can be a misleading metric for cache optimization.


## Alternative Things Attempting

When attempting to reduce the use of atomics, a prefix sum for non-zero elements was used. Despite avoided atomic operations, using a prefix-sum marginally increased total execution time.

However, because prefix-sum is a constant-time operation the benefit to it's use would increase depending on the sparsity of the input matrix.