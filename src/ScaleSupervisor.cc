#include "ScaleSupervisor.h"

using namespace ORB_SLAM3;

ScaleSupervisor::ScaleSupervisor(const Params& P)
: P_(P) {}

void ScaleSupervisor::UpdateToFScan(const ToFScan& s) {
  std::lock_guard<std::mutex> lk(mtx_scan_);
  last_scan_ = s;
  has_scan_ = true;
}
