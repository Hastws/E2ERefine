cmake_minimum_required(VERSION 3.10)
project(AutoAlgorama)

set(CMAKE_CXX_STANDARD 14)

#Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# 找SDL2库
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

# ImGui源文件
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-1.91.9b)

set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

include_directories(./include)

add_executable(${PROJECT_NAME}
        exec/auto_algorama.cpp
        src/windows/main_window.cpp
        ${IMGUI_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${SDL2_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${SDL2_LIBRARIES}
        OpenGL::GL
        dl
)
