#pragma once
#include <mutex>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

class Map; class KeyFrame;

struct ToFScan {
  double stamp_sec = 0.0;
  std::vector<float> ranges;     // meters
  std::vector<uint8_t> valid;    // optional 0/1, same size as ranges
};

class ScaleSupervisor {
public:
  struct Params {
    // Single-beam defaults (we’ll tune later)
    int Nx = 1, Ny = 1;
    double fov_x_deg = 0.0, fov_y_deg = 0.0; // unused for single-beam
    int    win_radius_px = 22;
    double incidence_min_dot = 0.30;
    int    min_plane_inliers = 12;
    double ransac_thresh_m = 0.03;
    int    min_good_rays = 1;
    int    hist_len = 5;
    double rho2 = 0.05, sigma = 0.01;
    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero();
  };

  explicit ScaleSupervisor(Map* pMap, const Params& P);

  // Feed latest ToF scan
  void UpdateToFScan(const ToFScan& s);

  // Phase 1 stubs (no behavior yet)
  // (C++11-friendly: return bool and optionally fill lambda_out later)
  bool ComputeLambdaForKeyFrame(KeyFrame* /*kf*/, double* /*lambda_out*/ = 0) { return false; }
  bool MaybeApplyLocalScale(KeyFrame* /*kf*/) { return false; }

private:
  Map* mpMap;
  Params P_;

  // C++11-friendly storage instead of std::optional
  mutable std::mutex mtx_scan_;
  bool   has_scan_ = false;
  ToFScan last_scan_;
};

} // namespace ORB_SLAM3
