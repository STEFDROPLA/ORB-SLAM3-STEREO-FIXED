#pragma once
#include <mutex>
#include <optional>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3 {
class Map; class KeyFrame;

struct ToFScan {
  double stamp_sec = 0.0;
  std::vector<float> ranges;   // meters
  std::vector<uint8_t> valid;  // optional same size
};

class ScaleSupervisor {
public:
  struct Params {
    int Nx = 1, Ny = 1;                // single-beam by default
    double fov_x_deg = 0.0, fov_y_deg = 0.0; // unused for single-beam
    int win_radius_px = 22;
    double incidence_min_dot = 0.30;
    int min_plane_inliers = 12;
    double ransac_thresh_m = 0.03;
    int min_good_rays = 1;
    int hist_len = 5;
    double rho2 = 0.05, sigma = 0.01;
    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero();
  };

  explicit ScaleSupervisor(Map* pMap, const Params& P);

  void UpdateToFScan(const ToFScan& s);
  // Phase 1: stubs (always return no-op/false)
  std::optional<double> ComputeLambdaForKeyFrame(KeyFrame*) { return std::nullopt; }
  bool MaybeApplyLocalScale(KeyFrame*) { return false; }

private:
  Map* mpMap;
  Params P_;
  mutable std::mutex mtx_scan_;
  std::optional<ToFScan> last_scan_;
};
} // namespace ORB_SLAM3
