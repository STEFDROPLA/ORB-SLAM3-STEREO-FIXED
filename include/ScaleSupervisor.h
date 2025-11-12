#pragma once
#include <mutex>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

class KeyFrame;

struct ToFScan {
  double stamp_sec = 0.0;
  std::vector<float> ranges;     // meters
  std::vector<uint8_t> valid;    // optional 0/1 mask, same size as ranges
};

class ScaleSupervisor {
public:
  struct Params {
    int Nx = 1, Ny = 1;                    // single-beam default
    double fov_x_deg = 0.0, fov_y_deg = 0.0; // unused for single-beam
    int    win_radius_px = 22;             // pixel window around projected ray
    double incidence_min_dot = 0.30;       // gating by incidence
    int    min_plane_inliers = 12;
    double ransac_thresh_m = 0.03;         // ~3 cm at ~10 m (loosely scales with depth)
    int    min_good_rays = 1;
    int    hist_len = 5;
    double rho2 = 0.05, sigma = 0.01;      // unused in phase 2
    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero(); // used in phase 3+
  };

  explicit ScaleSupervisor(const Params& P);

  // Feed latest ToF scan (kept for Phase 3; unused by Phase 2 logging)
  void UpdateToFScan(const ToFScan& s);

  // Phase 2: fit a plane near the ToF ray & LOG results; returns false (no lambda in phase 2)
  bool ComputeLambdaForKeyFrame(KeyFrame* kf, double* /*lambda_out*/ = 0);

  // Phase 2: still a stub (no scaling)
  bool MaybeApplyLocalScale(KeyFrame* /*kf*/) { return false; }

private:
  // ---- helpers for Phase 2 ----
  Eigen::Vector3d RayDirCamFrame(int ix, int iy) const;

  bool GatherNearby3D_FromMap(KeyFrame* pKF, const Eigen::Vector2d& px, int rad,
                              std::vector<Eigen::Vector3d>& pts_cam) const;

  bool RobustPlaneRANSAC(const std::vector<Eigen::Vector3d>& pts,
                         double thresh, int min_inliers,
                         Eigen::Vector3d& P0, Eigen::Vector3d& n, int& ninl) const;

  bool FitLocalPlaneFromMap(KeyFrame* pKF, const Eigen::Vector2d& px,
                            Eigen::Vector3d& P0, Eigen::Vector3d& n, int& inliers) const;

private:
  Params P_;

  // C++11-friendly (no std::optional)
  mutable std::mutex mtx_scan_;
  bool   has_scan_ = false;
  ToFScan last_scan_;
};

} // namespace ORB_SLAM3
