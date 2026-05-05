#include "krymova_k_lsd_sort_merge_double/stl/include/ops_stl.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "krymova_k_lsd_sort_merge_double/common/include/common.hpp"

namespace krymova_k_lsd_sort_merge_double {

KrymovaKLsdSortMergeDoubleSTL::KrymovaKLsdSortMergeDoubleSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool KrymovaKLsdSortMergeDoubleSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool KrymovaKLsdSortMergeDoubleSTL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

uint64_t KrymovaKLsdSortMergeDoubleSTL::DoubleToULL(double d) {
  uint64_t ull = 0;
  std::memcpy(&ull, &d, sizeof(double));

  if ((ull & 0x8000000000000000ULL) != 0U) {
    ull = ~ull;
  } else {
    ull |= 0x8000000000000000ULL;
  }

  return ull;
}

double KrymovaKLsdSortMergeDoubleSTL::ULLToDouble(uint64_t ull) {
  if ((ull & 0x8000000000000000ULL) != 0U) {
    ull &= 0x7FFFFFFFFFFFFFFFULL;
  } else {
    ull = ~ull;
  }

  double d = 0.0;
  std::memcpy(&d, &ull, sizeof(double));
  return d;
}

void KrymovaKLsdSortMergeDoubleSTL::LSDSortDoubleSequential(double *arr, int size) {
  if (size <= 1) {
    return;
  }

  const int k_bits_per_pass = 8;
  const int k_radix = 1 << k_bits_per_pass;
  const int k_passes = static_cast<int>(sizeof(double)) * 8 / k_bits_per_pass;

  std::vector<uint64_t> ull_arr(size);
  std::vector<uint64_t> ull_tmp(size);

  for (int i = 0; i < size; ++i) {
    ull_arr[i] = DoubleToULL(arr[i]);
  }

  std::vector<unsigned int> count(k_radix, 0U);

  for (int pass = 0; pass < k_passes; ++pass) {
    int shift = pass * k_bits_per_pass;

    std::fill(count.begin(), count.end(), 0U);

    for (int i = 0; i < size; ++i) {
      unsigned int digit = (ull_arr[i] >> shift) & (k_radix - 1);
      ++count[digit];
    }

    for (int i = 1; i < k_radix; ++i) {
      count[i] += count[i - 1];
    }

    for (int i = size - 1; i >= 0; --i) {
      unsigned int digit = (ull_arr[i] >> shift) & (k_radix - 1);
      ull_tmp[--count[digit]] = ull_arr[i];
    }

    ull_arr.swap(ull_tmp);
  }

  for (int i = 0; i < size; ++i) {
    arr[i] = ULLToDouble(ull_arr[i]);
  }
}

void KrymovaKLsdSortMergeDoubleSTL::LSDSortDoubleParallel(double *arr, int size, int num_threads) {
  if (size <= 1) {
    return;
  }

  const int k_bits_per_pass = 8;
  const int k_radix = 1 << k_bits_per_pass;
  const int k_passes = static_cast<int>(sizeof(double)) * 8 / k_bits_per_pass;

  std::vector<uint64_t> ull_arr(size);
  std::vector<uint64_t> ull_tmp(size);
  for (int i = 0; i < size; ++i) {
    ull_arr[i] = DoubleToULL(arr[i]);
  }

  if (num_threads <= 1) {
    LSDSortDoubleSequential(arr, size);
    return;
  }

  for (int pass = 0; pass < k_passes; ++pass) {
    int shift = pass * k_bits_per_pass;

    std::array<std::atomic<unsigned int>, k_radix> count{};
    for (int i = 0; i < k_radix; ++i) {
      count[i] = 0;
    }

    std::vector<std::thread> threads;
    int chunk_size = (size + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_size;
      int end = std::min(start + chunk_size, size);
      if (start >= size) {
        break;
      }

      threads.emplace_back([&, start, end, shift]() {
        for (int i = start; i < end; ++i) {
          unsigned int digit = (ull_arr[i] >> shift) & (k_radix - 1);
          count[digit].fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }
    threads.clear();
    std::vector<unsigned int> offsets(k_radix);
    unsigned int total = 0;
    for (int i = 0; i < k_radix; ++i) {
      offsets[i] = total;
      total += count[i].load();
    }

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_size;
      int end = std::min(start + chunk_size, size);
      if (start >= size) {
        break;
      }

      threads.emplace_back([&, start, end, shift, offsets]() {
        std::vector<unsigned int> local_offsets = offsets;

        for (int i = start; i < end; ++i) {
          unsigned int digit = (ull_arr[i] >> shift) & (k_radix - 1);
          ull_tmp[local_offsets[digit]++] = ull_arr[i];
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    ull_arr.swap(ull_tmp);
  }

  for (int i = 0; i < size; ++i) {
    arr[i] = ULLToDouble(ull_arr[i]);
  }
}

void KrymovaKLsdSortMergeDoubleSTL::MergeSections(double *left, const double *right, int left_size, int right_size) {
  std::vector<double> temp(left_size);
  std::copy(left, left + left_size, temp.begin());

  int left_index = 0;
  int right_index = 0;
  int dest_index = 0;

  while (left_index < left_size && right_index < right_size) {
    if (temp[left_index] <= right[right_index]) {
      left[dest_index++] = temp[left_index++];
    } else {
      left[dest_index++] = right[right_index++];
    }
  }

  while (left_index < left_size) {
    left[dest_index++] = temp[left_index++];
  }
}

void KrymovaKLsdSortMergeDoubleSTL::SortSectionsParallel(double *arr, int size, int portion, int num_threads) {
  int num_blocks = (size + portion - 1) / portion;
  int num_threads_used = std::min(num_threads, num_blocks);

  if (num_threads_used <= 1) {
    for (int i = 0; i < size; i += portion) {
      int current_size = std::min(portion, size - i);
      LSDSortDoubleSequential(arr + i, current_size);
    }
    return;
  }

  std::vector<std::thread> threads;
  int blocks_per_thread = (num_blocks + num_threads_used - 1) / num_threads_used;

  for (int t = 0; t < num_threads_used; ++t) {
    int start_block = t * blocks_per_thread;
    int end_block = std::min(start_block + blocks_per_thread, num_blocks);

    threads.emplace_back([&, start_block, end_block, portion, size, arr]() {
      for (int block = start_block; block < end_block; ++block) {
        int start = block * portion;
        int current_size = std::min(portion, size - start);
        LSDSortDoubleSequential(arr + start, current_size);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

void KrymovaKLsdSortMergeDoubleSTL::IterativeMergeSort(double *arr, int size, int portion, int num_threads) {
  if (size <= 1) {
    return;
  }

  SortSectionsParallel(arr, size, portion, num_threads);

  for (int merge_size = portion; merge_size < size; merge_size *= 2) {
    for (int i = 0; i < size; i += 2 * merge_size) {
      int left_size = merge_size;
      int right_size = std::min(merge_size, size - (i + merge_size));

      if (right_size <= 0) {
        continue;
      }

      double *left = arr + i;
      const double *right = arr + i + left_size;

      MergeSections(left, right, left_size, right_size);
    }
  }
}

bool KrymovaKLsdSortMergeDoubleSTL::RunImpl() {
  OutType &output = GetOutput();
  int size = static_cast<int>(output.size());

  if (size <= 1) {
    return true;
  }

  int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  num_threads = std::min(num_threads, size);

  int portion = std::max(1, size / num_threads);

  IterativeMergeSort(output.data(), size, portion, num_threads);

  return true;
}

bool KrymovaKLsdSortMergeDoubleSTL::PostProcessingImpl() {
  const OutType &output = GetOutput();

  for (size_t i = 1; i < output.size(); ++i) {
    if (output[i] < output[i - 1]) {
      return false;
    }
  }

  return true;
}

}  // namespace krymova_k_lsd_sort_merge_double
