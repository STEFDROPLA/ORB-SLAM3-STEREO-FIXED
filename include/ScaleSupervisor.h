#pragma once
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

class KeyFrame;

struct ToFScan {
  double stamp_sec = 0.0;
  std::vector<float>  ranges;   // meters
  std::vector<uint8_t> valid;   // optional 0/1 mask, same size as ranges
};

class ScaleSupervisor {
public:
  struct Params {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // <-- IMPORTANT (holds Matrix3d/Vector3d)

    int Nx = 1, Ny = 1;                 // single-beam default
    double fov_x_deg = 0.0, fov_y_deg = 0.0; // unused for single-beam
    int    win_radius_px = 22;          // pixel window around projected ray
    double incidence_min_dot = 0.30;    // gating by incidence
    int    min_plane_inliers = 12;
    double ransac_thresh_m = 0.03;      // ~3 cm at ~10 m
    int    min_good_rays = 1;
    int    hist_len = 5;
    double rho2 = 0.05, sigma = 0.01;   // unused in phase 2

    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero(); // used in phase 3+
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW      // <-- IMPORTANT (stores Params by value)

  explicit ScaleSupervisor(const Params& P);

  // Feed latest ToF scan (kept for Phase 3; unused by Phase 2 logging)
  void UpdateToFScan(const ToFScan& s);

  bool ComputeLambdaForKeyFrame(KeyFrame* kf, double* lambda_out = 0);
  bool MaybeApplyLocalScale(KeyFrame* kf);

private:
  // --- helpers (float types to match ORB-SLAM3) ---
  Eigen::Vector3f RayDirCamFrame(int ix, int iy) const;

  // Use aligned_allocator for vectors of Eigen types
  using Vec3f  = Eigen::Vector3f;
  using VVec3f = std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>>;

  bool GatherNearby3D_FromMap(KeyFrame* pKF, const Eigen::Vector2f& px, int rad,
                              VVec3f& pts_cam) const;

  bool RobustPlaneRANSAC(const VVec3f& pts, float thresh, int min_inliers,
                         Vec3f& P0, Vec3f& n, int& ninl) const;

  bool FitLocalPlaneFromMap(KeyFrame* pKF, const Eigen::Vector2f& px,
                            Vec3f& P0, Vec3f& n, int& inliers) const;

  bool IntersectRayPlane(const Vec3f& O, const Vec3f& d_unit,
                         const Vec3f& P0, const Vec3f& n,
                         float& rhat) const;

private:
  Params P_;

  mutable std::mutex mtx_scan_;
  bool   has_scan_ = false;
  ToFScan last_scan_;
};

} // namespace ORB_SLAM3
