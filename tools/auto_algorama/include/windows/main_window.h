#ifndef INCLUDE_AUTOALG_MAIN_WINDOW_H
#define INCLUDE_AUTOALG_MAIN_WINDOW_H

#include <SDL.h>
#include <SDL_opengl.h>

#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_opengl3.h"

namespace autoalg {
    class MainWindow {
    public:
        MainWindow(int width, int height);

        int Process() const;

        ~MainWindow();

    private:
        int width_ = 0;
        int height_ = 0;

        SDL_Window *window_;
        SDL_GLContext gl_context_;
    };
}

#endif // INCLUDE_AUTOALG_MAIN_WINDOW_H
