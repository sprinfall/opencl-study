set(TARGETS
	version_check
	saxpy
	)

foreach(target ${TARGETS})
	add_executable(${target} ${target}.cpp)

	target_link_libraries(${target}
		debug ${THIRD_PARTY_DIR}/OpenCL-ICD-Loader/Debug/OpenCL.lib
		optimized ${THIRD_PARTY_DIR}/OpenCL-ICD-Loader/Release/OpenCL.lib
		)
endforeach()
