add_library(tegra-infer STATIC
        memory.cpp
        memory.h
        macro.h
        tensorrt.cpp
        tensorrt.h
)

target_link_libraries(tegra-infer PUBLIC
        CUDA::cudart
        glog::glog
)

target_include_directories(tegra-infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
