#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Vec3 {
  double x, y, z, distance;
};

// --- Outils linéaires ---
inline double det3(const double M[3][3]) {
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
         M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
         M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

inline std::array<double, 3> solve3x3(const double A[3][3], const double b[3]) {
  double det = det3(A);
  if (std::fabs(det) < 1e-12) return {0, 0, 0};

  std::array<double, 3> x{};
  double M[3][3];
  for (int col = 0; col < 3; ++col) {
    std::memcpy(M, A, sizeof(M));
    for (int row = 0; row < 3; ++row) M[row][col] = b[row];
    x[col] = det3(M) / det;
  }
  return x;
}

// --- Résidu + Jacobienne ---
inline void residual_and_jacobian(const Vec3& a, double px, double py,
                                  double pz, double& r, double& jx, double& jy,
                                  double& jz) {
    double dx = px - a.x, dy = py - a.y, dz = pz - a.z;
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-9) dist = 1e-9;

    r = dist - a.distance;
    jx = dx / dist;
    jy = dy / dist;
    jz = dz / dist;
}

inline std::array<double, 3> barycentre(const std::vector<Vec3>& anchors) {
    double px = 0, py = 0, pz = 0;
    for (auto& a : anchors) {
        px += a.x;
        py += a.y;
        pz += a.z;
    }
    px /= anchors.size();
    py /= anchors.size();
    pz /= anchors.size();
    return {px, py, pz};
}

// Levenberg–Marquardt
std::array<double, 3> estimate_position_LM(const std::vector<Vec3>& anchors,
                                           int max_iter = 200,
                                           double tol = 1e-6,
                                           double lambda = 1e-3) {
    if (anchors.size() < 4)
        throw std::runtime_error("Au moins 4 balises nécessaires.");

    std::array<double, 3> bary = barycentre(anchors);

    for (int it = 0; it < max_iter; ++it) {
        double JtJ[3][3] = {};
        double Jtr[3] = {};
        double cost = 0.0;

        for (auto& a : anchors) {
            double r, jx, jy, jz;
            residual_and_jacobian(a, bary[0], bary[1], bary[2], r, jx, jy, jz);
            cost += r * r;

            JtJ[0][0] += jx * jx;
            JtJ[0][1] += jx * jy;
            JtJ[0][2] += jx * jz;
            JtJ[1][0] += jy * jx;
            JtJ[1][1] += jy * jy;
            JtJ[1][2] += jy * jz;
            JtJ[2][0] += jz * jx;
            JtJ[2][1] += jz * jy;
            JtJ[2][2] += jz * jz;

            Jtr[0] -= jx * r;
            Jtr[1] -= jy * r;
            Jtr[2] -= jz * r;
        }

        JtJ[0][0] += lambda;
        JtJ[1][1] += lambda;
        JtJ[2][2] += lambda;

        auto delta = solve3x3(JtJ, Jtr);
        bary[0] += delta[0];
        bary[1] += delta[1];
        bary[2] += delta[2];

        if (std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                    delta[2] * delta[2]) < tol)
        break;
    }
    return {bary[0], bary[1], bary[2]};
}

// RMS
double rms_error(const std::vector<Vec3>& anchors,
                 const std::array<double, 3>& pos) {
    double sum = 0.0;
    for (auto& a : anchors) {
        double dx = pos[0] - a.x;
        double dy = pos[1] - a.y;
        double dz = pos[2] - a.z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        double diff = dist - a.distance;
        sum += diff * diff;
    }
    return std::sqrt(sum / anchors.size());
}

int main() {
  std::vector<Vec3> anchors = {{0, 0, 0, 1.0},
                               {2, 0, 0, 1.4142135},
                               {0, 2, 0, 1.4142135},
                               {2, 2, 0, 2.0}};

    auto pos = estimate_position_LM(anchors, 400, 1e-9, 1e-6);
    double err = rms_error(anchors, pos);

    std::cout << "Position estimee: (" << pos[0] << ", " << pos[1] << ", "
                << pos[2] << ")\n";

    for (auto& a : anchors) {
        double dx = pos[0] - a.x;
        double dy = pos[1] - a.y;
        double dz = pos[2] - a.z;
        double dist_est = std::sqrt(dx * dx + dy * dy + dz * dz);
        std::cout << "Balise (" << a.x << "," << a.y << "," << a.z << ") : "
                << "Mesuree=" << a.distance << ", Estimee=" << dist_est
                << ", Diff=" << dist_est - a.distance << "\n";
    }

    std::cout << "Erreur RMS: " << err << "\n";
}
