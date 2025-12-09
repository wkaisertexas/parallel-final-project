# Kernel 0 

Kernel 0 provides the base implementation for sparse matrix-matrix multiplication. For a multiplication of matrices `A` and `B` into output `C`, traditional matrix multiplication kernels (not sparse) multiply row `i` of `A` by a column `j` of `B` to get `C[i,j]`. 

Due to the CSR representation of `A` and `B`, however, accessing columns of `B` is inefficient because it would require searching for particular column indices. A better implementation leverages the structure of the CSR, which makes traversing rows inexpensive. Firstly, we assign each row of `A` a thread. This thread will update all and only the corresponding row of `C`. The kernel avoids searching for column indices of `B` by performing partial sums. For a row `i` in output matrix `C`, a non-zero element `A[i,j]` will contribute to `C[i,k]` for every non-zero `B[j,k]` in row `j` of `B`. By iterating over row `j` of `B` (efficient in CSR), the kernel accumulates these contributions into the appropriate output columns without ever needing to search for specific column indices.

There is an issue, however, with naively writing the partial sum to the output matrix. Since we will be writing to this position multiple times for different rows of `B`, how can we tell if anything has already been written to that particular position in the COO output? Do we create a new entry or not? Checking a value at a particular position is extremely expensive with COO (requires search), so we address this by writing partial sums to an accumulator. Only once this accumulator is filled with all partial sums do we write to the COO matrix. Since no other threads will write to the same location (because each thread is a row), we avoid the checking problem across threads.

This introduces another issue: we don't have enough memory to store all the partial sums. Therefore, we take a tiled approach. The accumulator is only responsible for accumulating results for `TILE_SIZE` columns at a time. After processing all contributions to the current tile, non-zero values are written to the COO output, and the accumulator is reused for the next tile of columns.

# Kernel 1

