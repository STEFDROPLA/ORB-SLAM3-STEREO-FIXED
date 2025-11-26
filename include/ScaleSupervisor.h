// at the top of the header (add if missing)
#include <mutex>
#include <deque>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

class KeyFrame;

struct ToFScan {
  double stamp_sec = 0.0;
  std::vector<float> ranges;     // meters
  std::vector<uint8_t> valid;    // optional 0/1 mask
};

class ScaleSupervisor {
public:
  struct Params {
    int Nx = 1, Ny = 1;
    double fov_x_deg = 0.0, fov_y_deg = 0.0;
    int    win_radius_px = 22;
    double incidence_min_dot = 0.30;
    int    min_plane_inliers = 12;
    double ransac_thresh_m = 0.03;
    int    min_good_rays = 1;
    int    hist_len = 5;
    double rho2 = 0.05, sigma = 0.01;
    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero();

    // Phase 3 additions (make sure these exist)
    double max_tof_age_sec = 0.15;  // accept a ToF reading within 150 ms of KF
    double lambda_clip_min = 0.5;   // clamp λ
    double lambda_clip_max = 2.0;
  };

  explicit ScaleSupervisor(const Params& P);

  // ToF input
  void UpdateToFScan(const ToFScan& s);

  // Phase 3 public API
  bool ComputeLambdaForKeyFrame(KeyFrame* kf, double* lambda_out = nullptr);
  bool MaybeApplyLocalScale(KeyFrame* kf);         // safe no-op for now
  bool ComputeAndMaybeApply(KeyFrame* kf);         // wrapper: compute + maybe apply

private:
  // ---- helpers (declarations) ----
  Eigen::Vector3f RayDirCamFrame(int ix, int iy) const;

  bool GatherNearby3D_FromMap(KeyFrame* pKF, const Eigen::Vector2f& px, int rad,
                              std::vector<Eigen::Vector3f>& pts_cam) const;

  bool RobustPlaneRANSAC(const std::vector<Eigen::Vector3f>& pts, float thresh,
                         int min_inliers, Eigen::Vector3f& P0,
                         Eigen::Vector3f& n, int& ninl) const;

  bool FitLocalPlaneFromMap(KeyFrame* pKF, const Eigen::Vector2f& px,
                            Eigen::Vector3f& P0, Eigen::Vector3f& n,
                            int& inliers) const;

  bool IntersectRayPlane(const Eigen::Vector3f& O, const Eigen::Vector3f& d_unit,
                         const Eigen::Vector3f& P0, const Eigen::Vector3f& n,
                         float& rhat) const;

  // Phase 3 helpers
  bool DistanceAlongRayToPlane(const Eigen::Vector3f& d_cam,
                               const Eigen::Vector3f& P0,
                               const Eigen::Vector3f& n,
                               float& rhat) const;
  bool GetRayAndPlane(KeyFrame* pKF, Eigen::Vector3f& d_cam,
                      Eigen::Vector3f& P0, Eigen::Vector3f& n,
                      int& ninl) const;

  // Optional: closest scan search (used if you prefer buffer search)
  bool GetScanNear(double t_kf, ToFScan& out) const;

private:
  Params P_;

  // ToF storage & sync
  mutable std::mutex mtx_scan_;
  bool   has_scan_ = false;
  ToFScan last_scan_;

  // small buffer for nearest-time lookup
  std::deque<ToFScan> scan_buf_;
  size_t max_scans_ = 50;
  double max_dt_sync_ = 0.15;  // seconds
};

} // namespace ORB_SLAM3
