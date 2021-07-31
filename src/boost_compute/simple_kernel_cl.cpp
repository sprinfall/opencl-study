// Rewrite boost_compute/simple_kernel.cpp using OpenCL C++ bindings.

#include <iostream>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"  // or cl2.hpp

int main() {
  cl::Device device = cl::Device::getDefault();

  cl::Context context{ device };

  constexpr std::size_t N = 4;

  // setup input arrays
  float a[] = { 1, 2, 3, 4 };
  float b[] = { 5, 6, 7, 8 };

  // make space for the output
  float c[] = { 0, 0, 0, 0 };

  // create memory buffers for the input and output
  cl::Buffer buffer_a{ context, a, a + N, true };
  cl::Buffer buffer_b{ context, b, b + N, true };
  cl::Buffer buffer_c{ context, c, c + N, false };

  // source code for the add kernel
  const char source[] =
      "__kernel void add(__global const float *a,"
      "                  __global const float *b,"
      "                  __global float *c)"
      "{"
      "    const uint i = get_global_id(0);"
      "    c[i] = a[i] + b[i];"
      "}";

  // create the program with the source
  cl::Program program{ context, source };

  // compile the program
  program.build();

  // create the kernel
  cl::Kernel kernel{ program, "add" };

  // set the kernel arguments
  kernel.setArg(0, buffer_a);
  kernel.setArg(1, buffer_b);
  kernel.setArg(2, buffer_c);

  // create a command queue
  cl::CommandQueue queue{ context, device };

  constexpr std::size_t SIZE = N * sizeof(float);

  // write the data from 'a' and 'b' to the device
  queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, SIZE, a);
  queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, SIZE, b);

  // run the add kernel
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N),
                             cl::NullRange);

  //queue.finish();

  // transfer results back to the host array 'c'
  queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, SIZE, c);

  // print out results in 'c'
  std::cout << "c: [" << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3]
            << "]" << std::endl;

  return 0;
}
