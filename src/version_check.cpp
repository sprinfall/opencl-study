// version_check.cpp
//
// Check if there's any GPU with OpenCL 1.2 and above.

#include <iostream>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/opencl.hpp"  // or cl2.hpp

static int GetOpenCLVersion(const cl::Device& device) {
  std::string version = device.getInfo<CL_DEVICE_VERSION>();

  if (version.size() < 10) {  // 10: The size of "OpenCL M.m"
    return 0;
  }

  // TODO: More check
  int value = (version[7] - '0') * 10;
  value += version[9] - '0';

  return value;
}

static bool IsOpenCL12GpuAvailable() {
  std::vector<cl::Platform> platforms;
  if (cl::Platform::get(&platforms) != CL_SUCCESS) {
    return false;
  }

  const int kMinVersion = 12;

  for (auto& platform : platforms) {
    std::string version = platform.getInfo<CL_PLATFORM_VERSION>();

    std::vector<cl::Device> devices;
    if (platform.getDevices(CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS) {
      continue;
    }

    for (auto& device : devices) {
      if (GetOpenCLVersion(device) >= kMinVersion) {
        return true;
      }
    }
  }

  return false;
}

int main() {
  std::cout << std::boolalpha << IsOpenCL12GpuAvailable() << std::endl;

  return 0;
}
