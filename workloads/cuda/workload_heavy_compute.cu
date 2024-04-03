#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define ROUND 100000

__global__ void heavyKernel() {
  double sum = 0.0;
  for (size_t i = 0; i < ROUND; i++) {
    sum += sin(i) * cos(i);
  }
}

int main(int argc, char *argv[]) {

  assert(argc == 3);
  int gpu_id = std::atoi(argv[1]);
  cudaStream_t stream;
  int block_num = std::atoi(argv[2]);
  assert(gpu_id >= 0 && gpu_id <= 7);
  size_t cnt = 0;
  // 设置使用的GPU
  cudaSetDevice(gpu_id);
  cudaStreamCreate(&stream);
  // 启动kernel，此处配置很关键，需要足够多的block和thread以充分利用GPU资源
  while(1) {
    heavyKernel<<<block_num, 1024, 0, stream>>>();
    // 等待CUDA kernel完成
    // cudaStreamSynchronize(stream);
    std::cout << "Completed heavy computation on GPU " << gpu_id 
      << ", the total kernel round is " << cnt << std::endl;
    cnt++;
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  std::cout << "Simulation complete";
  return 0;
}
