find_package(glog REQUIRED)

add_library(tegra-infer STATIC
        memory.cpp
        memory.h
        macro.h
        tensorrt.cpp
        tensorrt.h
        fs.cpp
        fs.h
        stage.cpp
        stage.h
        pipline.cpp
        pipline.h
        typedef.h
        kernel.cu
        kernel.h
        source.cpp
        source.h
)

target_link_libraries(tegra-infer PUBLIC
        CUDA::cudart
        glog::glog
        jetson-utils
        nvinfer
)

target_include_directories(tegra-infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

add_executable(main
    main.cpp
)

target_link_libraries(main
    tegra-infer
)

add_executable(perf_kernel
    perf_kernel.cpp
)

target_link_libraries(perf_kernel
    tegra-infer
)
