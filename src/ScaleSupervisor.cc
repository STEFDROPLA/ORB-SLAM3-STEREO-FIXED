#include "ScaleSupervisor.h"

#include "KeyFrame.h"
#include "MapPoint.h"
#include "Converter.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace ORB_SLAM3 {

static inline double deg2rad(double d){ return d*M_PI/180.0; }

ScaleSupervisor::ScaleSupervisor(const Params& P)
: P_(P) {}

void ScaleSupervisor::UpdateToFScan(const ToFScan& s) {
  std::lock_guard<std::mutex> lk(mtx_scan_);
  last_scan_ = s;
  has_scan_ = true;
}

// --------- helpers ---------

Eigen::Vector3d ScaleSupervisor::RayDirCamFrame(int ix, int iy) const {
  // Single-beam: ToF boresight (0,0,1) mapped into camera frame
  if (P_.Nx == 1 && P_.Ny == 1) {
    Vector3d u_to(0,0,1);
    Vector3d u_cam = P_.R_cam_from_tof * u_to;
    u_cam.normalize();
    return u_cam;
  }
  // Multi-pixel fallback
  const double fx = deg2rad(P_.fov_x_deg);
  const double fy = deg2rad(P_.fov_y_deg);
  const double cx = (P_.Nx) * 0.5;
  const double cy = (P_.Ny) * 0.5;
  const double ax = ((ix + 0.5) - cx) * (fx / P_.Nx);
  const double ay = ((iy + 0.5) - cy) * (fy / P_.Ny);
  Vector3d u_to(std::tan(ax), std::tan(ay), 1.0);
  u_to.normalize();
  Vector3d u_cam = P_.R_cam_from_tof * u_to;
  u_cam.normalize();
  return u_cam;
}

bool ScaleSupervisor::GatherNearby3D_FromMap(KeyFrame* pKF, const Vector2d& px, int rad,
                                             std::vector<Vector3d>& pts_cam) const
{
  pts_cam.clear();

  // Camera-from-world pose of this KF
  const Eigen::Matrix4d Tcw = ORB_SLAM3::Converter::toMatrix4d(pKF->GetPose());

  // Intrinsics & bounds
  const float fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
  const int w = pKF->imLeft.cols, h = pKF->imLeft.rows;

  const std::vector<cv::KeyPoint>& vKeys = pKF->mvKeysUn;
  const int N = vKeys.size();

  for (int i=0; i<N; ++i){
    const float u = vKeys[i].pt.x;
    const float v = vKeys[i].pt.y;
    if (std::abs(u - px.x()) > rad || std::abs(v - px.y()) > rad) continue;

    MapPoint* pMP = pKF->GetMapPoint(i);
    if (!pMP || pMP->isBad()) continue;

    const Eigen::Vector3d Xw = ORB_SLAM3::Converter::toVector3d(pMP->GetWorldPos());
    const Eigen::Vector4d Xc4 = Tcw * Eigen::Vector4d(Xw.x(),Xw.y(),Xw.z(),1.0);
    Vector3d Xc = Xc4.head<3>();
    if (Xc.z() <= 0.05) continue;

    const float uu = fx * (Xc.x() / Xc.z()) + cx;
    const float vv = fy * (Xc.y() / Xc.z()) + cy;
    if (!(uu >= 0 && uu < w && vv >= 0 && vv < h)) continue;
    if (std::abs(uu - px.x()) > rad || std::abs(vv - px.y()) > rad) continue;

    pts_cam.push_back(Xc);
  }
  return (int)pts_cam.size() >= 6;
}

bool ScaleSupervisor::RobustPlaneRANSAC(const std::vector<Vector3d>& pts,
                                        double thresh, int min_inliers,
                                        Vector3d& P0, Vector3d& n, int& ninl) const
{
  if ((int)pts.size() < std::max(min_inliers, 6)) return false;

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> uni(0, (int)pts.size()-1);

  int best_inl = -1;
  Vector3d best_n(0,0,1), best_P0(0,0,0);

  const int iters = 200;
  for (int it=0; it<iters; ++it){
    const Vector3d a = pts[uni(rng)];
    const Vector3d b = pts[uni(rng)];
    const Vector3d c = pts[uni(rng)];
    Vector3d n_cand = (b - a).cross(c - a);
    const double nn = n_cand.norm();
    if (nn < 1e-6) continue;
    n_cand /= nn;

    int inl = 0;
    for (size_t k=0; k<pts.size(); ++k){
      double d = std::abs(n_cand.dot(pts[k] - a));
      if (d < thresh) ++inl;
    }
    if (inl > best_inl){
      best_inl = inl;
      best_n   = n_cand;
      best_P0  = a;
    }
  }

  if (best_inl < min_inliers) return false;

  // refine with LS on inliers
  std::vector<Vector3d> inl_pts; inl_pts.reserve(best_inl);
  for (size_t k=0; k<pts.size(); ++k){
    double d = std::abs(best_n.dot(pts[k] - best_P0));
    if (d < thresh) inl_pts.push_back(pts[k]);
  }

  Vector3d mu = Vector3d::Zero();
  for (size_t k=0; k<inl_pts.size(); ++k) mu += inl_pts[k];
  mu /= (double)inl_pts.size();

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (size_t k=0; k<inl_pts.size(); ++k){
    Vector3d q = inl_pts[k] - mu;
    C += q*q.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(C);
  n = es.eigenvectors().col(0);
  n.normalize();
  P0 = mu;
  ninl = (int)inl_pts.size();
  return true;
}

bool ScaleSupervisor::FitLocalPlaneFromMap(KeyFrame* pKF, const Vector2d& px,
                                           Vector3d& P0, Vector3d& n, int& inliers) const
{
  std::vector<Vector3d> pts;
  if (!GatherNearby3D_FromMap(pKF, px, P_.win_radius_px, pts)) return false;

  // Scale distance threshold loosely with median Z
  double medz;
  {
    std::vector<double> zs; zs.reserve(pts.size());
    for (size_t i=0;i<pts.size();++i) zs.push_back(pts[i].z());
    std::nth_element(zs.begin(), zs.begin()+zs.size()/2, zs.end());
    medz = zs[zs.size()/2];
  }
  const double th = std::max(0.01, P_.ransac_thresh_m * (medz / 10.0));
  const int    min_inl = std::max(P_.min_plane_inliers, 8);

  return RobustPlaneRANSAC(pts, th, min_inl, P0, n, inliers);
}

// --------- Phase 2 main (log only) ---------

bool ScaleSupervisor::ComputeLambdaForKeyFrame(KeyFrame* pKF, double* /*lambda_out*/)
{
  // 1) Build ToF single-beam ray in camera frame
  const int ix = 0, iy = 0; // single pixel
  Vector3d d_cam = RayDirCamFrame(ix,iy);
  if (std::abs(d_cam.z()) < 1e-6) return false;

  // 2) Project that ray to a pixel (for neighborhood search)
  const float fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
  Vector2d px( fx * (d_cam.x()/d_cam.z()) + cx,  fy * (d_cam.y()/d_cam.z()) + cy );

  // 3) Fit a local plane from MapPoints near that pixel window
  Vector3d P0, n; int ninl=0;
  if (!FitLocalPlaneFromMap(pKF, px, P0, n, ninl)) {
    std::cout << "[ToF] KF " << pKF->mnId << " plane: NO_FIT (pts<inliers)\n";
    return false;
  }

  // 4) Gate by incidence angle (helps reject near-edge cases)
  const double cosang = std::abs(n.normalized().dot(d_cam.normalized()));
  if (cosang < P_.incidence_min_dot) {
    std::cout << "[ToF] KF " << pKF->mnId << " plane: REJECT (incidence=" << cosang
              << ", inliers="<< ninl << ")\n";
    return false;
  }

  // Phase 2: only LOG (no lambda yet, no scaling)
  std::cout << "[ToF] KF " << pKF->mnId << " plane: OK  (inliers=" << ninl
            << ", px=[" << px.x() << "," << px.y() << "], cos=" << cosang << ")\n";
  return false; // no lambda produced yet
}

} // namespace ORB_SLAM3
