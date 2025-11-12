#include "ScaleSupervisor.h"
using namespace ORB_SLAM3;

ScaleSupervisor::ScaleSupervisor(Map* pMap, const Params& P) : mpMap(pMap), P_(P) {}
void ScaleSupervisor::UpdateToFScan(const ToFScan& s) {
  std::lock_guard<std::mutex> lk(mtx_scan_);
  last_scan_ = s;
}
