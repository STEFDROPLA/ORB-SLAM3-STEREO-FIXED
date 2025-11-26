#pragma once
#include <mutex>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

class KeyFrame;   // fwd decl
class MapPoint;   // fwd decl (not strictly needed here, but common in ORB-SLAM3)

// Simple ToF scan container (single- or multi-beam)
struct ToFScan {
  double stamp_sec = 0.0;            // acquisition time (seconds)
  std::vector<float>   ranges;       // meters (size Nx*Ny or 1)
  std::vector<uint8_t> valid;        // optional mask, same size as ranges
};

class ScaleSupervisor {
public:
  struct Params {
    // --- ToF layout / optics ---
    int    Nx = 1, Ny = 1;                 // 1x1 for single-beam
    double fov_x_deg = 0.0, fov_y_deg = 0.0; // for multi-pixel ToF, harmless for 1x1

    // --- Plane fit neighborhood & gating ---
    int    win_radius_px = 22;             // pixel window to gather local 3D points
    double incidence_min_dot = 0.70;       // require n·d_cam < -incidence_min_dot (nadir ~ -1)
    int    min_plane_inliers = 12;
    double ransac_thresh_m   = 0.03;       // base threshold; scaled by depth in fitter

    // --- Robustness / history knobs (reserved for future smoothing) ---
    int    min_good_rays = 1;
    int    hist_len      = 5;
    double rho2 = 0.05, sigma = 0.01;

    // --- Extrinsics: Camera <- ToF (ToF in camera frame) ---
    Eigen::Matrix3d R_cam_from_tof = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam_from_tof = Eigen::Vector3d::Zero(); // (unused in phase 3 math)

    // --- Timing / clamping ---
    double max_tof_age_sec = 0.050;        // accept ToF up to 50 ms from KF time
    double lambda_clip_min = 0.5;          // clamp lambda for safety
    double lambda_clip_max = 2.0;
  };

  explicit ScaleSupervisor(const Params& P);

  // Feed latest ToF scan (thread-safe)
  void UpdateToFScan(const ToFScan& s);

  // Phase 3: estimate local scale lambda for this KF (no map changes yet)
  // Returns true if a valid lambda was computed; writes it to lambda_out if provided.
  bool ComputeAndMaybeApply(KeyFrame* kf);



private:
  // ---- helpers (float math to match SLAM internals) ----

  // ToF ray (unit) in camera frame
  Eigen::Vector3f RayDirCamFrame(int ix, int iy) const;

  // Gather camera-frame 3D points near pixel px (±rad) from the KF’s map points
  bool GatherNearby3D_FromMap(KeyFrame* pKF,
                              const Eigen::Vector2f& px,
                              int rad,
                              std::vector<Eigen::Vector3f>& pts_cam) const;

  // Robust plane fit via RANSAC + LS refine
  bool RobustPlaneRANSAC(const std::vector<Eigen::Vector3f>& pts,
                         float thresh,
                         int min_inliers,
                         Eigen::Vector3f& P0,
                         Eigen::Vector3f& n,
                         int& ninl) const;

  // Fit a local plane using map points around the ToF-projected pixel
  bool FitLocalPlaneFromMap(KeyFrame* pKF,
                            const Eigen::Vector2f& px,
                            Eigen::Vector3f& P0,
                            Eigen::Vector3f& n,
                            int& inliers) const;

  // Ray/plane intersection from camera origin O=(0,0,0)
  bool IntersectRayPlane(const Eigen::Vector3f& O,
                         const Eigen::Vector3f& d_unit,
                         const Eigen::Vector3f& P0,
                         const Eigen::Vector3f& n,
                         float& rhat) const;

  // --- Phase 3 utilities ---
  // Pull a recent ToF range near this KF time (median over valid beams)
  bool LatestRangeForKF(double kf_stamp, float& r) const;

  // Distance along ray d_cam to plane (camera origin assumed)
  bool DistanceAlongRayToPlane(const Eigen::Vector3f& d_cam,
                               const Eigen::Vector3f& P0,
                               const Eigen::Vector3f& n,
                               float& rhat) const;

  // Build ToF ray & local ground-like plane; incidence-gated to favor nadir
  bool GetRayAndPlane(KeyFrame* pKF,
                      Eigen::Vector3f& d_cam,
                      Eigen::Vector3f& P0,
                      Eigen::Vector3f& n,
                      int& ninl) const;


  // --- buffering & sync (private) ---
  std::deque<ToFScan> scan_buf_;
  size_t max_scans_ = 100;        // keep last 100 scans
  double max_dt_sync_ = 0.05;     // accept ToF within ±50 ms of KF time

  bool GetScanNear(double t_kf, ToFScan& out) const;


private:
  Params P_;

  // last ToF
  mutable std::mutex mtx_scan_;
  bool   has_scan_ = false;
  ToFScan last_scan_;
};

} // namespace ORB_SLAM3
