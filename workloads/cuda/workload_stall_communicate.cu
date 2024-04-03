#include <stdio.h>
#include <iostream>
#include <nccl.h>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
      __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \ 
}

#define CHECK_NCCL(call) { \
  ncclResult_t err = call; \
  if(err != ncclSuccess) { \
    fprintf(stderr, "NCCL error in file '%s' in line %i: %s.\n", \
      __FILE__, __LINE__, ncclGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \
}

int main(int argc, char* argv[]) {

  assert(argc == 2);
  int gpu_id = std::atoi(argv[1]);
  assert(gpu_id >= 0 && gpu_id <= 7);

  ncclComm_t comms[8];
  cudaStream_t streams[8];

  // Assuming 8 GPUs are available and are to be used.
  const int nGPUs = 8;
  const size_t dataSize = 1024 * 1024; // Example data size.
  float* data;

  // Initialize NCCL, creating a unique communicator for each GPU.
  CHECK_NCCL(ncclCommInitAll(comms, nGPUs, NULL));

  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaStreamCreate(&streams[gpu_id]));
  CHECK_CUDA(cudaMalloc(&data, dataSize * sizeof(float)));
  std::cout << "Comm-stall straggler " << gpu_id << " start..." << std::endl;
  CHECK_NCCL(ncclAllReduce((const void*)data, (void*)data, dataSize, ncclFloat, ncclSum, comms[gpu_id], streams[gpu_id]));
  CHECK_CUDA(cudaStreamSynchronize(streams[gpu_id]));

  // Finalizing.
  for(int i = 0; i < nGPUs; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    ncclCommDestroy(comms[i]);
  }

  std::cout << "Simulation complete";
  return 0;
}
