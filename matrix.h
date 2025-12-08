
#ifndef __MATRIX_H_
#define __MATRIX_H_

#include <stdexcept>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(status)); \
        throw std::runtime_error("Testing"); \
    } \
}

#pragma pack(push, 1)
struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int capacity;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};
#pragma pack(pop)

COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrix(COOMatrix* cooMatrix);

void sortCOOMatrix(COOMatrix* cooMatrix);

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrixOnGPU(COOMatrix* cooMatrix);

void clearCOOMatrixOnGPU(COOMatrix* cooMatrix);
void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h);


#pragma pack(push, 1)

struct CSRMatrix {
    unsigned int numRows = 0;
    unsigned int numCols = 0;
    unsigned int numNonzeros = 0;
    unsigned int* rowPtrs = nullptr;
    unsigned int* colIdxs = nullptr;
    float* values = nullptr;
};
#pragma pack(pop)

CSRMatrix* createCSRMatrixFromFile(const char* fileName);
void freeCSRMatrix(CSRMatrix* csrMatrix);

CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int numNonzeros);
void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix);

void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d);

#endif

