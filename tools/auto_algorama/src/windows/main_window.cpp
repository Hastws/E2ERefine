#include "windows/main_window.h"

#include "common/log.h"
#include "functions/functions.h"

namespace autoalg {
    static float amplitude = 1.0f;
    static float frequency = 1.0f;
    static int func_type = 0;

    // 坐标轴范围和刻度
    static float x_min = -10.0f;
    static float x_max = 10.0f;
    static float y_min = -10.0f;
    static float y_max = 10.0f;
    static int x_ticks = 20;
    static int y_ticks = 20;

    REGISTER_FUNCTION(Sigmoid, ::autoalg::Sigmoid);
    REGISTER_FUNCTION(Probit, ::autoalg::Probit);
    REGISTER_FUNCTION(Tanh, ::autoalg::Tanh);
    REGISTER_FUNCTION(VariantSigmoid, ::autoalg::VariantSigmoid);
    REGISTER_FUNCTION(ShiftedScaled, ::autoalg::ShiftedScaledSigmoid);
    REGISTER_FUNCTION(ScaledTanh, ::autoalg::ScaledTanh);
    REGISTER_FUNCTION(BimodalSigmoid, ::autoalg::BimodalSigmoid);

    double EvaluateFunction(int func, float x, float amp, float freq) {
        auto name = FunctionsManager::Instance().GetFunctionName(func);
        return amp * FunctionsManager::Instance().Call(name, freq * x);
    }

    MainWindow::MainWindow(const int width, const int height)
        : width_(width), height_(height), window_(nullptr) {
        DEBUG(MainWindow) << "initialized.";
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_EVENTS) != 0) {
            ERROR(MainWindow) << "SDL_Init failed: " << SDL_GetError();
        } else {
            const char *glsl_version = "#version 150";
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                                SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

            window_ = SDL_CreateWindow(
                "AutoAlgorama", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width_,
                height_,
                SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
            gl_context_ = SDL_GL_CreateContext(window_);
            SDL_GL_MakeCurrent(window_, gl_context_);
            SDL_GL_SetSwapInterval(1);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            ImGui_ImplSDL2_InitForOpenGL(window_, gl_context_);
            ImGui_ImplOpenGL3_Init(glsl_version);
        }
    }

    int MainWindow::Process() const {
        DEBUG(MainWindow) << "processing.";
        if (!window_) {
            return 1;
        }
        constexpr ImVec4 clear_color = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
        bool done = false;
        while (!done) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
                if (event.type == SDL_QUIT) done = true;
                if (event.type == SDL_WINDOWEVENT &&
                    event.window.event == SDL_WINDOWEVENT_CLOSE &&
                    event.window.windowID == SDL_GetWindowID(window_))
                    done = true;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            ImDrawList *bg = ImGui::GetBackgroundDrawList();
            ImGuiViewport *vp = ImGui::GetMainViewport();
            ImVec2 p0 = vp->Pos;
            ImVec2 p1 = ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
            ImVec2 dispSize = vp->Size;

            float data_x_mid = 0.0f;
            float data_y_mid = 0.0f;
            float x_range = x_max - x_min;
            float y_range = y_max - y_min;

            // 背景填充
            bg->AddRectFilled(p0, p1, IM_COL32(20, 20, 20, 255));
            // 网格线
            for (int i = 0; i <= x_ticks; ++i) {
                float t = (float) i / x_ticks;
                float px = p0.x + t * dispSize.x;
                bg->AddLine(ImVec2(px, p0.y), ImVec2(px, p1.y),
                            IM_COL32(50, 50, 50, 100));
            }
            for (int i = 0; i <= y_ticks; ++i) {
                float t = (float) i / y_ticks;
                float py = p0.y + t * dispSize.y;
                bg->AddLine(ImVec2(p0.x, py), ImVec2(p1.x, py),
                            IM_COL32(50, 50, 50, 100));
            }
            // 屏幕中心坐标
            float x0 = p0.x + dispSize.x * (-x_min / x_range);
            float y0 = p0.y + dispSize.y * (1.0f - (-y_min / y_range));
            // 强制中心为坐标原点
            x0 = p0.x + dispSize.x * 0.5f;
            y0 = p0.y + dispSize.y * 0.5f;
            // 绘制坐标轴
            bg->AddLine(ImVec2(p0.x, y0), ImVec2(p1.x, y0),
                        IM_COL32(200, 200, 200, 150));
            bg->AddLine(ImVec2(x0, p0.y), ImVec2(x0, p1.y),
                        IM_COL32(200, 200, 200, 150));
            // 坐标刻度与标签
            for (int i = 0; i <= x_ticks; ++i) {
                float t = (float) i / x_ticks;
                float px = p0.x + t * dispSize.x;
                float data_x = x_min + t * x_range;
                bg->AddLine(ImVec2(px, y0 - 5), ImVec2(px, y0 + 5),
                            IM_COL32(200, 200, 200, 150));
                char buf[32];
                snprintf(buf, 32, "%d", static_cast<int>(data_x));
                bg->AddText(ImVec2(px + 3, y0 + 3), IM_COL32(255, 255, 255, 200), buf);
            }
            for (int i = 0; i <= y_ticks; ++i) {
                float t = 1.0f - (float) i / y_ticks;
                float py = p0.y + t * dispSize.y;
                float data_y = y_min + (1.0f - t) * y_range;
                bg->AddLine(ImVec2(x0 - 5, py), ImVec2(x0 + 5, py),
                            IM_COL32(200, 200, 200, 150));
                int int_y = static_cast<int>(data_y);
                if (int_y == 0) {
                    continue;
                }
                char buf[32];
                snprintf(buf, 32, "%d", int_y);
                bg->AddText(ImVec2(x0 + 6, py - 6), IM_COL32(255, 255, 255, 200), buf);
            }
            // 函数曲线（中心原点映射）
            int samples = std::max(100, x_ticks * 40);
            ImVec2 prev;
            bool firstPt = true;
            for (int i = 0; i <= samples; ++i) {
                float t = (float) i / samples;
                float x = x_min + t * x_range;
                float y = EvaluateFunction(func_type, x, amplitude, frequency);
                float px = x0 + (x - data_x_mid) / x_range * dispSize.x;
                float py = y0 - (y - data_y_mid) / y_range * dispSize.y;
                if (!firstPt)
                    bg->AddLine(prev, ImVec2(px, py), IM_COL32(255, 120, 120, 200), 2.0f);
                prev = ImVec2(px, py);
                firstPt = false;
            }

            // 前台 UI 窗口：函数控件
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("Function Controls", nullptr, ImGuiWindowFlags_None);
            ImGui::Text("Function Settings");
            ImGui::Separator();

            static char search_buf[64] = "";
            ImGui::InputText("Search Function", search_buf, IM_ARRAYSIZE(search_buf));

            ImGui::BeginChild("Function List", ImVec2(0, 76), true);
            for (int i = 0; i < FunctionsManager::Instance().GetNumberOfFunctions(); ++i) {
                const char *name = FunctionsManager::Instance().GetFunctionName(i).c_str();
                if (strstr(name, search_buf)) {
                    if (ImGui::Selectable(name, func_type == i)) {
                        func_type = i;
                    }
                }
            }
            ImGui::EndChild();

            ImGui::SliderFloat("Amplitude", &amplitude, 0.1f, 5.0f);
            ImGui::SliderFloat("Frequency", &frequency, 0.1f, 10.0f);
            ImGui::End();

            // 获取鼠标位置（屏幕坐标）
            ImVec2 mouse_pos = ImGui::GetIO().MousePos;

            // 判断鼠标是否在绘图区域内
            bool mouse_in_rect = (mouse_pos.x >= p0.x && mouse_pos.x <= p1.x &&
                                  mouse_pos.y >= p0.y && mouse_pos.y <= p1.y);
            bool control_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
            if (mouse_in_rect && !control_hovered) {
                // 计算鼠标对应的坐标系中的 x, y 值
                float mouse_x_data = x_min + (mouse_pos.x - p0.x) / dispSize.x * x_range;
                float mouse_y_data = y_max - (mouse_pos.y - p0.y) / dispSize.y * y_range;

                // 绘制横向参考线（贯穿绘图区）
                bg->AddLine(ImVec2(p0.x, mouse_pos.y), ImVec2(p1.x, mouse_pos.y),
                            IM_COL32(200, 200, 100, 150), 1.5f);
                // 绘制纵向参考线（贯穿绘图区）
                bg->AddLine(ImVec2(mouse_pos.x, p0.y), ImVec2(mouse_pos.x, p1.y),
                            IM_COL32(200, 200, 100, 150), 1.5f);

                // 显示坐标文本
                char buf[64];
                snprintf(buf, 64, "x=%.2f, y=%.2f", mouse_x_data, mouse_y_data);

                // 文字位置，鼠标右上方，避免遮挡指针
                ImVec2 text_pos = ImVec2(mouse_pos.x + 10, mouse_pos.y - 20);
                bg->AddText(text_pos, IM_COL32(255, 255, 100, 255), buf);
            }

            // 渲染
            ImGui::Render();
            SDL_GL_MakeCurrent(window_, gl_context_);
            glViewport((int) vp->Pos.x, (int) vp->Pos.y, (int) vp->Size.x,
                       (int) vp->Size.y);
            glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(window_);
        }
        return 0;
    }

    MainWindow::~MainWindow() {
        DEBUG(MainWindow) << "destroyed.";
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        SDL_GL_DeleteContext(gl_context_);
        SDL_DestroyWindow(window_);
        SDL_Quit();
    }
} // namespace autoalg
