#include "raylib-cpp.hpp"

#include <stdlib.h>
#include "benlib/pattern/scheduler.hpp"

#define MAX_ENEMIES 50000
#define MAX_BATCH_ELEMENTS 16384

typedef struct Enemies {
    raylib::Vector2 position;
    Color color;
} Enemies;

class BasicApp : public benlib::pattern::App {
   public:
    BasicApp() : _window(raylib::Window()), _camera(raylib::Camera2D({0, 0}, {0, 0}, 0.0f, 1.0f)) {}

    ~BasicApp() {
        _window.Close();
        free(enemies);
    }

    void init() {
        SetTraceLogLevel(LOG_ALL);
        _window.Init(screenWidth, screenHeight, "raylib test");
        // raylib::SetConfigFlags(FLAG_MSAA_4X_HINT);
        _window.SetTargetFPS(60);
    }

    void update() override {
        if (!isInitialized) [[unlikely]] {
            init();
            isInitialized = true;
        }

        if (_window.ShouldClose()) {
            exit(0);
        }
        updateKey();
        updateData();
        updateVisual();
    }

    void updateData() {
        _camera.target = Vector2Add(playerPosition, (Vector2){20, 20});
        _camera.offset = (Vector2){screenWidth / 2, screenHeight / 2};
        for (int i = 0; i < enemiesCount; i++) {
            raylib::Vector2 delta = Vector2Subtract(playerPosition, enemies[i].position);
            float distance = Vector2Length(delta);
            if (distance > 20) {
                delta = Vector2Scale(delta, (1.0f / distance) * 2.5f);
                enemies[i].position = Vector2Add(enemies[i].position, delta);
            } else {
                enemies[i].position = Vector2Add(enemies[i].position, (Vector2){GetRandomValue(-1, 1), GetRandomValue(-1, 1)});
            }
        }
    }

    void updateKey() {
        float wheel = GetMouseWheelMove();
        raylib::Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), _camera);
        if (wheel != 0) {
            _camera.offset = GetMousePosition();
            _camera.target = mouseWorldPos;
            _camera.zoom = Clamp(expf(logf(_camera.zoom) + 0.2f * wheel), 0.125f, 64.0f);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            for (int i = 0; i < 10; i++) {
                if (enemiesCount < MAX_ENEMIES) {
                    enemies[enemiesCount].position = mouseWorldPos;
                    enemies[enemiesCount].color = (Color){GetRandomValue(50, 240), GetRandomValue(80, 240), GetRandomValue(100, 240), 255};
                    enemiesCount++;
                }
            }
        }

        if (IsKeyPressed(KEY_R)) {
            enemiesCount = 0;
        }

        if (IsKeyDown(KEY_RIGHT)) {
            playerPosition.x += 6.0f;
        }
        if (IsKeyDown(KEY_LEFT)) {
            playerPosition.x -= 6.0f;
        }
        if (IsKeyDown(KEY_DOWN)) {
            playerPosition.y += 6.0f;
        } else if (IsKeyDown(KEY_UP)) {
            playerPosition.y -= 6.0f;
        }
    }

    void updateVisual() {
        _window.BeginDrawing();
        _window.ClearBackground(RAYWHITE);

        _camera.BeginMode();

        DrawRectangle(-5, -5, screenWidth + 5, screenHeight + 5, GRAY);
        for (int i = 0; i < enemiesCount; i++) {
            DrawRectangle(enemies[i].position.x, enemies[i].position.y, 10, 10, enemies[i].color);
        }

        DrawCircleV(Vector2AddValue(playerPosition, 5.0f), 20, RED);
        _camera.EndMode();

        DrawText(TextFormat("enemies: %i", enemiesCount), 120, 10, 20, GREEN);

        DrawCircleV(GetMousePosition(), 4, DARKGRAY);
        //DrawTextEx(GetFontDefault(), TextFormat("[%i, %i]", GetMouseX(), GetMouseY()), Vector2Add(GetMousePosition(), (Vector2){-44, -24}), 20, 2, BLACK);

        _window.DrawFPS(10, 10);
        _window.EndDrawing();
    }

   private:
    const int screenWidth = 1280;
    const int screenHeight = 720;
    raylib::Window _window;
    bool isInitialized = false;
    raylib::Camera2D _camera = raylib::Camera2D({0, 0}, {0, 0}, 0.0f, 1.0f);

    Enemies* enemies = (Enemies*)malloc(MAX_ENEMIES * sizeof(Enemies));
    int enemiesCount = 0;
    raylib::Vector2 playerPosition = {200, 200};
};

int main(void) {
    benlib::pattern::Scheduler scheduler;

    auto app = std::make_shared<BasicApp>();
    scheduler.setUpdateFrequency(std::chrono::milliseconds(1));

    scheduler.addApp(app);

    while (scheduler.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}