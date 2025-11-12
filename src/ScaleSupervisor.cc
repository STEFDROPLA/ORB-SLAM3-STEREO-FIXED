#include "ScaleSupervisor.h"

#include "KeyFrame.h"
#include "MapPoint.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace ORB_SLAM3 {

// ---------- small helpers ----------
static inline float fpi()        { return 3.14159265358979323846f; }
static inline float fdeg2rad(float d){ return d * (fpi() / 180.0f); }

// ---------- ctor / ToF input ----------
ScaleSupervisor::ScaleSupervisor(const Params& P)
: P_(P)
{
}

void ScaleSupervisor::UpdateToFScan(const ToFScan& s)
{
  std::lock_guard<std::mutex> lk(mtx_scan_);
  last_scan_ = s;
  has_scan_  = true;
}

// ---------- geometry helpers (Phase 2) ----------

/**
 * Direction of ToF ray expressed in the CAMERA frame (unit length).
 * Single-beam: use ToF boresight (0,0,1) rotated by R_cam_from_tof.
 * Multi-pixel: uses Nx,Ny and FoV to compute per-pixel ray (still supported).
 */
Eigen::Vector3f ScaleSupervisor::RayDirCamFrame(int ix, int iy) const
{
  // Single-beam: straight ahead in ToF frame rotated to cam frame
  if (P_.Nx == 1 && P_.Ny == 1) {
    const Eigen::Matrix3f R = P_.R_cam_from_tof.cast<float>();
    Eigen::Vector3f u_to(0.f, 0.f, 1.f);
    Eigen::Vector3f u_cam = R * u_to;
    u_cam.normalize();
    return u_cam;
  }

  // Multi-pixel support (not used in single-beam, but kept clean)
  const float fx = fdeg2rad(static_cast<float>(P_.fov_x_deg));
  const float fy = fdeg2rad(static_cast<float>(P_.fov_y_deg));
  const float cx = 0.5f * static_cast<float>(P_.Nx);
  const float cy = 0.5f * static_cast<float>(P_.Ny);

  const float ax = ((ix + 0.5f) - cx) * (fx / static_cast<float>(P_.Nx));
  const float ay = ((iy + 0.5f) - cy) * (fy / static_cast<float>(P_.Ny));

  Eigen::Vector3f u_to(std::tan(ax), std::tan(ay), 1.f);
  u_to.normalize();

  const Eigen::Matrix3f R = P_.R_cam_from_tof.cast<float>();
  Eigen::Vector3f u_cam = R * u_to;
  u_cam.normalize();
  return u_cam;
}

/**
 * Collect MapPoints near a given pixel location (px) and return their
 * 3D coordinates in the **camera** frame of pKF. Pure float math.
 *
 * We use:
 *  - pKF->GetPose()  (Sophus::SE3f) to get Tcw
 *  - pKF->mvKeysUn   for keypoint pixels
 *  - pKF->GetMapPoint(i) for the 3D world pos (Eigen::Vector3f)
 *  - pKF intrinsics (fx,fy,cx,cy)
 */
bool ScaleSupervisor::GatherNearby3D_FromMap(KeyFrame* pKF,
                                             const Eigen::Vector2f& px,
                                             int rad,
                                             std::vector<Eigen::Vector3f>& pts_cam) const
{
  pts_cam.clear();

  const Sophus::SE3f  Tcw_se3 = pKF->GetPose();
  const Eigen::Matrix4f Tcw   = Tcw_se3.matrix();

  const float fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;

  const std::vector<cv::KeyPoint>& vKeys = pKF->mvKeysUn;
  const int N = static_cast<int>(vKeys.size());

  for (int i = 0; i < N; ++i) {
    const float u = vKeys[i].pt.x;
    const float v = vKeys[i].pt.y;

    // quick pixel-window pre-gate
    if (std::abs(u - px.x()) > rad || std::abs(v - px.y()) > rad) continue;

    MapPoint* pMP = pKF->GetMapPoint(i);
    if (!pMP || pMP->isBad()) continue;

    // ORB-SLAM3 MapPoint::GetWorldPos() returns Eigen::Vector3f
    const Eigen::Vector3f Xw = pMP->GetWorldPos();

    const Eigen::Vector4f Xc4 = Tcw * Eigen::Vector4f(Xw.x(), Xw.y(), Xw.z(), 1.f);
    Eigen::Vector3f Xc = Xc4.head<3>();

    if (Xc.z() <= 0.05f) continue; // behind camera / too close to plane at z=0

    // Reproject into this KF pixel plane (for a tighter neighborhood check)
    const float uu = fx * (Xc.x() / Xc.z()) + cx;
    const float vv = fy * (Xc.y() / Xc.z()) + cy;

    if (std::abs(uu - px.x()) > rad || std::abs(vv - px.y()) > rad) continue;

    pts_cam.push_back(Xc);
  }

  // a few points are needed to fit a plane robustly
  return static_cast<int>(pts_cam.size()) >= 6;
}

/**
 * Robust plane fitting in camera frame via RANSAC + LS refine (float math).
 * Returns plane point P0 (on plane), unit normal n, and inlier count ninl.
 */
bool ScaleSupervisor::RobustPlaneRANSAC(const std::vector<Eigen::Vector3f>& pts,
                                        float thresh,
                                        int min_inliers,
                                        Eigen::Vector3f& P0,
                                        Eigen::Vector3f& n,
                                        int& ninl) const
{
  if (static_cast<int>(pts.size()) < std::max(min_inliers, 6)) return false;

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> uni(0, static_cast<int>(pts.size()) - 1);

  int best = -1;
  Eigen::Vector3f bN(0.f,0.f,1.f), bP(0.f,0.f,0.f);

  // RANSAC
  for (int it = 0; it < 200; ++it) {
    const Eigen::Vector3f a = pts[uni(rng)];
    const Eigen::Vector3f b = pts[uni(rng)];
    const Eigen::Vector3f c = pts[uni(rng)];

    Eigen::Vector3f nc = (b - a).cross(c - a);
    const float nn = nc.norm();
    if (nn < 1e-6f) continue;
    nc /= nn;

    int inl = 0;
    for (size_t k = 0; k < pts.size(); ++k) {
      const float d = std::abs(nc.dot(pts[k] - a));
      if (d < thresh) ++inl;
    }
    if (inl > best) { best = inl; bN = nc; bP = a; }
  }

  if (best < min_inliers) return false;

  // LS refine on inliers
  std::vector<Eigen::Vector3f> inl; inl.reserve(best);
  for (size_t k = 0; k < pts.size(); ++k) {
    const float d = std::abs(bN.dot(pts[k] - bP));
    if (d < thresh) inl.push_back(pts[k]);
  }

  Eigen::Vector3f mu = Eigen::Vector3f::Zero();
  for (size_t k = 0; k < inl.size(); ++k) mu += inl[k];
  mu /= static_cast<float>(inl.size());

  Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
  for (size_t k = 0; k < inl.size(); ++k) {
    const Eigen::Vector3f q = inl[k] - mu;
    C += q * q.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
  n = es.eigenvectors().col(0);
  n.normalize();

  P0   = mu;
  ninl = static_cast<int>(inl.size());
  return true;
}

/**
 * Fit a local plane from MapPoints around pixel px in the current KF.
 * - Collects camera-frame 3D points near px
 * - Adapts RANSAC threshold with median depth
 * - Returns plane (P0, n) and inlier count
 */
bool ScaleSupervisor::FitLocalPlaneFromMap(KeyFrame* pKF,
                                           const Eigen::Vector2f& px,
                                           Eigen::Vector3f& P0,
                                           Eigen::Vector3f& n,
                                           int& inliers) const
{
  std::vector<Eigen::Vector3f> pts;
  if (!GatherNearby3D_FromMap(pKF, px, P_.win_radius_px, pts)) return false;

  // scale RANSAC threshold with median Z for mild depth-adaptivity
  float medz;
  {
    std::vector<float> zs; zs.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) zs.push_back(pts[i].z());
    std::nth_element(zs.begin(), zs.begin() + zs.size()/2, zs.end());
    medz = zs[zs.size()/2];
  }

  const float th = std::max(0.01f,
                            static_cast<float>(P_.ransac_thresh_m) * (medz / 10.0f));
  const int   min_inl = std::max(P_.min_plane_inliers, 8);

  return RobustPlaneRANSAC(pts, th, min_inl, P0, n, inliers);
}

/**
 * Ray/plane intersection (float). Returns distance rhat along the ray direction.
 * (Unused in Phase 2, but implemented for Phase 3.)
 */
bool ScaleSupervisor::IntersectRayPlane(const Eigen::Vector3f& O,
                                        const Eigen::Vector3f& d_unit,
                                        const Eigen::Vector3f& P0,
                                        const Eigen::Vector3f& n,
                                        float& rhat) const
{
  const float denom = n.dot(d_unit);
  if (std::abs(denom) < 1e-6f) return false;
  const float t = n.dot(P0 - O) / denom;
  if (t <= 0.f) return false;
  rhat = t;
  return true;
}

} // namespace ORB_SLAM3
