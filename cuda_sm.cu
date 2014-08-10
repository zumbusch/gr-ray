// ----------------------------------------------------------------------
//
// Copyright (c) 2011, 2014, Gerhard Zumbusch.
// All rights reserved.
//
// ----------------------------------------------------------------------

// find CUDA compute capabilities
// compile with nvcc

#include <stdio.h>

int main(int argc, char **argv) {
  int dev = 0;
  if (cudaSetDevice(dev)) return 1;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev)) return 1;
  printf("-arch=sm_%d%d", prop.major, prop.minor);
  cudaDeviceReset();
  return 0;
}
