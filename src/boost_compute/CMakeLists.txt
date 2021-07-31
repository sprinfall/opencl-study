
# Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS chrono)
if(Boost_FOUND)
    message(STATUS "Boost version: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
endif()

set(TARGETS
    hello_world
    simple_kernel
    simple_kernel_cl
    )

foreach(target ${TARGETS})
    add_executable(${target} ${target}.cpp)

    target_link_libraries(${target} Boost::chrono)

    target_link_libraries(${target}
        debug ${THIRD_PARTY_DIR}/OpenCL-ICD-Loader/Debug/OpenCL.lib
        optimized ${THIRD_PARTY_DIR}/OpenCL-ICD-Loader/Release/OpenCL.lib
        )

    set_target_properties(${target} PROPERTIES FOLDER "Boost.Compute")
endforeach()
