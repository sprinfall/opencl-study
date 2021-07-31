#ifndef COMMON_INC_
#define COMMON_INC_

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#error Please include this file after define CL_HPP_TARGET_OPENCL_VERSION !
#endif

#include <string>

#include "CL/cl.h"

static std::string DeviceTypeName(cl_device_type device_type) {
  std::string ret;

#define DEVICE_TYPE_TEST(type, name) \
  if ((device_type & type) != 0) {   \
    if (!ret.empty()) ret += ",";    \
    ret += name;                     \
  }

  DEVICE_TYPE_TEST(CL_DEVICE_TYPE_CPU, "CPU");
  DEVICE_TYPE_TEST(CL_DEVICE_TYPE_GPU, "GPU");
  DEVICE_TYPE_TEST(CL_DEVICE_TYPE_ACCELERATOR, "ACCEL");
  DEVICE_TYPE_TEST(CL_DEVICE_TYPE_CUSTOM, "CUSTOM");

  return ret;
}

#endif  // COMMON_INC_
