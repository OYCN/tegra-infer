# for jetson-utils
set(UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils")
file(GLOB jetsonUtilitySources
        "${UTILS_DIR}/*.cpp"
        "${UTILS_DIR}/camera/*.cpp"
        "${UTILS_DIR}/codec/*.cpp"
        "${UTILS_DIR}/cuda/*.cu"
        "${UTILS_DIR}/cuda/*.cpp"
        "${UTILS_DIR}/display/*.cpp"
        "${UTILS_DIR}/image/*.cpp"
        "${UTILS_DIR}/input/*.cpp"
        "${UTILS_DIR}/network/*.cpp"
        "${UTILS_DIR}/threads/*.cpp"
        "${UTILS_DIR}/video/*.cpp"
)
add_library(jetson-utils ${jetsonUtilitySources})
target_compile_definitions(jetson-utils PUBLIC -DENABLE_NVMM)
target_link_libraries(jetson-utils PUBLIC
        CUDA::cudart
        CUDA::nppicc
        GL
        GLU
        GLEW
        gstreamer-1.0
        gstapp-1.0
        gstpbutils-1.0
        gstwebrtc-1.0
        gstsdp-1.0
        gstrtspserver-1.0
        json-glib-1.0
        soup-2.4
        gobject-2.0
        glib-2.0
        X11
        nvbufsurface
        nvbufsurftransform
)
target_link_directories(jetson-utils PUBLIC
        /usr/lib/aarch64-linux-gnu/tegra
)
target_include_directories(jetson-utils PUBLIC
        "${UTILS_DIR}/"
        "${UTILS_DIR}/camera"
        "${UTILS_DIR}/codec"
        "${UTILS_DIR}/cuda"
        "${UTILS_DIR}/display"
        "${UTILS_DIR}/image"
        "${UTILS_DIR}/input"
        "${UTILS_DIR}/network"
        "${UTILS_DIR}/threads"
        "${UTILS_DIR}/video"
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/include/libxml2
        /usr/include/json-glib-1.0
        /usr/include/libsoup-2.4
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/gstreamer-1.0/include
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/glib-2.0/include/
        /usr/src/jetson_multimedia_api/include
)

add_executable(camera-viewer
        "${UTILS_DIR}/camera/camera-viewer/camera-viewer.cpp"
)
target_link_libraries(camera-viewer PRIVATE jetson-utils)
