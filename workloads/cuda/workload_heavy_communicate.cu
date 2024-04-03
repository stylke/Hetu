#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <nccl.h>

#define ROUND 100000

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  int gpus[8];
  ncclComm_t comms[8];
  int nDev = argc - 1;
  std::cout << "Use " << nDev << " gpus" << std::endl;
  for (int i = 0; i < nDev; ++i) {
    gpus[i] = std::atoi(argv[i + 1]);
    assert(gpus[i] >= 0 && gpus[i] <= 7); 
  }

  // allocating and initializing device buffers
  int size = 1024 * 1024;
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, gpus));

  size_t cnt = 0;
  while(1) {
    std::cout << "Round " << cnt << std::endl;
    // multiple devices per process
    for (int r = 0; r < ROUND; ++r) {
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());
    }
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(gpus[i]));
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    cnt++;
  }

  // free
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }

  // finalizing NCCL
  for(int i = 0; i < nDev; ++i) {
    ncclCommDestroy(comms[i]);
  }
  std::cout << "Simulation complete";
  return 0;
}