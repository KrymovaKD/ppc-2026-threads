#pragma once

#include <cstdint>
#include <vector>

#include "krymova_k_lsd_sort_merge_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krymova_k_lsd_sort_merge_double {

class KrymovaKLsdSortMergeDoubleALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit KrymovaKLsdSortMergeDoubleALL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void LSDSort(double *arr, int size);
  static std::vector<double> SimpleMerge(const std::vector<double> &a, const std::vector<double> &b);

  static uint64_t DoubleToULL(double d);
  static double ULLToDouble(uint64_t ull);
};

}  // namespace krymova_k_lsd_sort_merge_double
