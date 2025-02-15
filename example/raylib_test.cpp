#include "raylib-cpp.hpp"

#include <cmath>  // NOLINT

#include "benlib/patern/scheduler.hpp"

#include "gtest/gtest.h"

class BasicApp : public benlib::patern::App {
public:
    BasicApp() : _window(raylib::Window()) {
    }

    ~BasicApp() {
        _window.Close();
    }

    void init() {
        SetTraceLogLevel(LOG_ALL);
        _window.Init(screenWidth, screenHeight, "raylib [shapes] example - collision area");
        _window.SetTargetFPS(60);

        boxA = raylib::Rectangle(10, _window.GetHeight() / 2 - 50, 200, 100);
        boxB = raylib::Rectangle(GetScreenWidth() / 2 - 30, _window.GetHeight() / 2 - 30, 60, 60);
    }

    void update() override {
        if (!isInitialized) [[unlikely]] {
            init();
            isInitialized = true;
        }

        if (_window.ShouldClose()) {
            exit(0);
        }

        updateData();
        updateVisual();
    }

    void updateData() {
        if (!pause)
            boxA.x += boxASpeedX;

        // Bounce box on x screen limits
        if (((boxA.x + boxA.width) >= GetScreenWidth()) || (boxA.x <= 0))
            boxASpeedX *= -1;

        // Update player-controlled-box (box02)
        boxB.x = GetMouseX() - boxB.width / 2;
        boxB.y = GetMouseY() - boxB.height / 2;

        // Make sure Box B does not go out of move area limits
        if ((boxB.x + boxB.width) >= _window.GetWidth())
            boxB.x = _window.GetWidth() - boxB.width;
        else if (boxB.x <= 0)
            boxB.x = 0;

        if ((boxB.y + boxB.height) >= _window.GetHeight())
            boxB.y = _window.GetHeight() - boxB.height;
        else if (boxB.y <= screenUpperLimit)
            boxB.y = screenUpperLimit;

        // Check boxes collision
        collision = boxA.CheckCollision(boxB);

        // Get collision rectangle (only on collision)
        if (collision)
            boxCollision = boxA.GetCollision(boxB);

        // Pause Box A movement
        if (IsKeyPressed(KEY_SPACE))
            pause = !pause;
    }

    void updateVisual() {
        _window.BeginDrawing();

        _window.ClearBackground(RAYWHITE);

        DrawRectangle(0, 0, screenWidth, screenUpperLimit, collision ? RED : BLACK);

        boxA.Draw(GOLD);
        boxB.Draw(BLUE);

        if (collision) {
            // Draw collision area
            boxCollision.Draw(LIME);

            // Draw collision message
            raylib::DrawText("COLLISION!", GetScreenWidth() / 2 - MeasureText("COLLISION!", 20) / 2, screenUpperLimit / 2 - 10, 20, BLACK);

            // Draw collision area
            raylib::DrawText(TextFormat("Collision Area: %i", (int)boxCollision.width * (int)boxCollision.height), GetScreenWidth() / 2 - 100,
                             screenUpperLimit + 10, 20, BLACK);
        }

        _window.DrawFPS(10, 10);

        _window.EndDrawing();
    }

    private:
    const int screenWidth = 800;
    const int screenHeight = 450;
    raylib::Window _window;

    raylib::Rectangle boxA;
    int boxASpeedX = 4;
    raylib::Rectangle boxB;
    raylib::Rectangle boxCollision;

    int screenUpperLimit = 40;

    bool pause = false;
    bool collision = false;

    bool isInitialized = false;
};

int main(void) {

    benlib::patern::Scheduler scheduler;

    auto app = std::make_shared<BasicApp>();

    scheduler.addApp(app);

    while (scheduler.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}