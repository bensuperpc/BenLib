 
#ifndef MY_CUDA
#define MY_CUDA

void vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n);
void vecSub(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n);
void vecMult(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n);
void vecDiv(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n);

#endif