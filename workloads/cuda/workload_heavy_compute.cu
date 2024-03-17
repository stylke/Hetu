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

  assert(argc == 2);
  int gpu_id = std::atoi(argv[1]);
  assert(gpu_id >= 0 && gpu_id <= 7);
  size_t cnt = 0;
  // 设置使用的GPU
  cudaSetDevice(gpu_id);
  // 启动kernel，此处配置很关键，需要足够多的block和thread以充分利用GPU资源
  while(1) {
    heavyKernel<<<1024, 1024>>>();
    // 等待CUDA kernel完成
    cudaDeviceSynchronize();
    std::cout << "Completed heavy computation on GPU " << gpu_id 
      << ", the total kernel round is " << cnt << std::endl;
    cnt++;
  }
  return 0;
}
