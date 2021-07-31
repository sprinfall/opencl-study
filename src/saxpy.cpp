// saxpy.cpp
//
// SAXPY stands for "Single-Precision A * X Plus Y".
// Based on OpenCL 1.2 API for using Nvidia driver.

// Device is explicitly selected and used.
// Neither cl::Platform::setDefault() nor cl::Device::setDefault() is used.

#include <iostream>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"  // or cl2.hpp

#include "common.inc"  // for DeviceTypeName()

// Use C++11 raw string literals for kernel source code.
static const std::string kSaxpyKernelCL = R"CLC(
  __kernel void saxpy(float alpha, __global const float* X, __global float* Y)
  {
    int i = get_global_id(0);
    Y[i] = alpha * X[i];
  }
)CLC";

// Enumerate and select device.
static cl::Device SelectDevice(cl_device_type device_type) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    return cl::Device{};
  }

  cl::Device selected_device;

  for (auto& platform : platforms) {
    std::string version = platform.getInfo<CL_PLATFORM_VERSION>();
    std::cout << "Platform " << version << std::endl;

    std::vector<cl::Device> devices;
    if (platform.getDevices(device_type, &devices) != CL_SUCCESS) {
      std::cout << " No devices found." << std::endl;
      continue;
    }

    for (auto& device : devices) {
      std::cout << " - [" << DeviceTypeName(device_type) << "] "
                << device.getInfo<CL_DEVICE_VENDOR>() << ": "
                << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      std::cout << "   (Max compute units: "
                << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                << ", max work group size: "
                << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << ")\n";

      // Just select the first matched device.
      // Continue to enumerate other devices for study purpose.
      if (selected_device() == nullptr) {
        selected_device = device;
      }
    }

    std::cout << std::endl;
  }

  return selected_device;
}

static const std::size_t N = (1 << 10);

static const float XVAL = static_cast<float>(rand() % 1000000);
static const float YVAL = static_cast<float>(rand() % 1000000);
static const float AVAL = static_cast<float>(rand() % 1000000);

#define USE_NEW_API 0
#define WAIT_FOR_KERNEL_EVENT 0

int main() {
  cl::Device device = SelectDevice(CL_DEVICE_TYPE_GPU);
  if (device() == nullptr) {
    return 1;
  }

  std::cout << "Using " << device.getInfo<CL_DEVICE_VENDOR>() << " "
            << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cout << std::endl;

  cl::Context context{ device };
  cl::Program program{ context, kSaxpyKernelCL };

  try {
    program.build();

  } catch (const cl::BuildError& error) {
    auto build_log = error.getBuildLog();
    for (auto& pair : build_log) {
      std::cerr << "Device: " << pair.first.getInfo<CL_DEVICE_NAME>()
                << std::endl
                << pair.second << std::endl;
    }
    return 1;
  }

  float* host_x = new float[N];
  float* host_y = new float[N];
  for (std::size_t i = 0; i < N; ++i) {
    host_x[i] = XVAL;
    host_y[i] = YVAL;
  }

  std::cout << "Y[0]: " << host_y[0] << std::endl;

  try {
    cl::CommandQueue queue{ context, device };

#if USE_NEW_API
    // cl::copy() is used internally.
    cl::Buffer device_x{ queue, host_x, host_x + N, true };
    cl::Buffer device_y{ queue, host_y, host_y + N, true };
#else
    cl::Buffer device_x{ context, CL_MEM_READ_ONLY, N * sizeof(float) };
    cl::Buffer device_y{ context, CL_MEM_READ_WRITE, N * sizeof(float) };
    queue.enqueueWriteBuffer(device_x, CL_TRUE, 0, N * sizeof(float), host_x);
    queue.enqueueWriteBuffer(device_y, CL_TRUE, 0, N * sizeof(float), host_y);
#endif  // USE_NEW_API

#if USE_NEW_API
    auto saxpy_kernel =
        cl::KernelFunctor<float, cl::Buffer&, cl::Buffer&>{ program, "saxpy" };

#if WAIT_FOR_KERNEL_EVENT
    cl::Event event = saxpy_kernel(cl::EnqueueArgs{ queue, cl::NDRange(N) },
                                   AVAL, device_x, device_y);
    // Wait for the event to complete.
    event.wait();
#else
    saxpy_kernel(cl::EnqueueArgs{ queue, cl::NDRange(N) }, AVAL, device_x,
                 device_y);
#endif  // WAIT_FOR_KERNEL_EVENT

#else
    cl::Kernel saxpy_kernel{ program, "saxpy" };

    saxpy_kernel.setArg(0, AVAL);
    saxpy_kernel.setArg(1, device_x);
    saxpy_kernel.setArg(2, device_y);

#if WAIT_FOR_KERNEL_EVENT
    cl::Event event;
    queue.enqueueNDRangeKernel(saxpy_kernel, cl::NullRange, cl::NDRange(N),
                               cl::NullRange, nullptr, &event);
    // Wait for the event to complete.
    event.wait();
#else
    queue.enqueueNDRangeKernel(saxpy_kernel, cl::NullRange, cl::NDRange(N));
#endif  // WAIT_FOR_KERNEL_EVENT

    queue.finish();
#endif  // USE_NEW_API

#if USE_NEW_API
    // Use memory mapping and copy commands.
    cl::copy(queue, device_y, host_y, host_y + N);
#else
    queue.enqueueReadBuffer(device_y, CL_TRUE, 0, N * sizeof(float), host_y);
#endif  // USE_NEW_API

    std::cout << "Y[0]: " << host_y[0] << std::endl;

  } catch (const cl::Error& error) {
    std::cout << "Error: " << error.what() << std::endl;
  }

  delete[] host_x;
  delete[] host_y;

  return 0;
}
