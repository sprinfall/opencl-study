#include <iostream>

#include <boost/compute/core.hpp>

namespace compute = boost::compute;

int main() {
  compute::device device = compute::system::default_device();

  std::cout << "hello from " << device.name();
  std::cout << " (platform: " << device.platform().name() << ")" << std::endl;

  return 0;
}
