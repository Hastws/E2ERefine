set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-1.91.9b)

set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl2.cpp
)

include_directories(
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
)

include_directories(external/FADBAD++)