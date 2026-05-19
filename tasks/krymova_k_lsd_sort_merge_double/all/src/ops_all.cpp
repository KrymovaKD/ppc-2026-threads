#include "krymova_k_lsd_sort_merge_double/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "krymova_k_lsd_sort_merge_double/common/include/common.hpp"

namespace krymova_k_lsd_sort_merge_double {

KrymovaKLsdSortMergeDoubleALL::KrymovaKLsdSortMergeDoubleALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool KrymovaKLsdSortMergeDoubleALL::ValidationImpl() {
  return true;
}

bool KrymovaKLsdSortMergeDoubleALL::PreProcessingImpl() {
  return true;
}

uint64_t KrymovaKLsdSortMergeDoubleALL::DoubleToULL(double d) {
  uint64_t ull = 0U;
  std::memcpy(&ull, &d, sizeof(double));
  return ((ull & 0x8000000000000000ULL) != 0U) ? ~ull : (ull | 0x8000000000000000ULL);
}

double KrymovaKLsdSortMergeDoubleALL::ULLToDouble(uint64_t ull) {
  if ((ull & 0x8000000000000000ULL) != 0U) {
    ull &= 0x7FFFFFFFFFFFFFFFULL;
  } else {
    ull = ~ull;
  }
  double d = 0.0;
  std::memcpy(&d, &ull, sizeof(double));
  return d;
}

void KrymovaKLsdSortMergeDoubleALL::LSDSort(double *arr, int size) {
  if (size <= 1) {
    return;
  }

  const int k_bits_per_pass = 8;
  const int k_radix = 1 << k_bits_per_pass;
  const int k_passes = 8;

  std::vector<uint64_t> ull_arr(static_cast<size_t>(size));
  std::vector<uint64_t> ull_tmp(static_cast<size_t>(size));
  std::vector<unsigned int> count(static_cast<size_t>(k_radix), 0U);

  for (int i = 0; i < size; ++i) {
    ull_arr[static_cast<size_t>(i)] = DoubleToULL(arr[i]);
  }

  for (int pass = 0; pass < k_passes; ++pass) {
    int shift = pass * k_bits_per_pass;
    std::ranges::fill(count.begin(), count.end(), 0U);

    for (int i = 0; i < size; ++i) {
      unsigned int digit = (ull_arr[static_cast<size_t>(i)] >> shift) & (k_radix - 1);
      ++count[digit];
    }

    for (int i = 1; i < k_radix; ++i) {
      count[static_cast<size_t>(i)] += count[static_cast<size_t>(i - 1)];
    }

    for (int i = size - 1; i >= 0; --i) {
      unsigned int digit = (ull_arr[static_cast<size_t>(i)] >> shift) & (k_radix - 1);
      ull_tmp[static_cast<size_t>(--count[digit])] = ull_arr[static_cast<size_t>(i)];
    }

    ull_arr.swap(ull_tmp);
  }

  for (int i = 0; i < size; ++i) {
    arr[i] = ULLToDouble(ull_arr[static_cast<size_t>(i)]);
  }
}

std::vector<double> KrymovaKLsdSortMergeDoubleALL::SimpleMerge(const std::vector<double> &a,
                                                               const std::vector<double> &b) {
  std::vector<double> res;
  res.reserve(a.size() + b.size());
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      res.push_back(a[i]);
      ++i;
    } else {
      res.push_back(b[j]);
      ++j;
    }
  }
  while (i < a.size()) {
    res.push_back(a[i]);
    ++i;
  }
  while (j < b.size()) {
    res.push_back(b[j]);
    ++j;
  }
  return res;
}

bool KrymovaKLsdSortMergeDoubleALL::RunSmallDataset(int total_size) {
  if (total_size > 0) {
    GetOutput() = GetInput();
    LSDSort(GetOutput().data(), total_size);
  } else {
    GetOutput().clear();
  }
  return true;
}

void KrymovaKLsdSortMergeDoubleALL::ComputeDistribution(int total_size, int size_comm, std::vector<int> &send_counts,
                                                        std::vector<int> &offsets) {
  int chunk = total_size / size_comm;
  int rem = total_size % size_comm;

  for (int i = 0; i < size_comm; ++i) {
    send_counts[static_cast<size_t>(i)] = chunk + (i < rem ? 1 : 0);
    if (i == 0) {
      offsets[0] = 0;
    } else {
      offsets[static_cast<size_t>(i)] = offsets[static_cast<size_t>(i - 1)] + send_counts[static_cast<size_t>(i - 1)];
    }
  }
}

void KrymovaKLsdSortMergeDoubleALL::ScatterData(int rank, const std::vector<int> &send_counts,
                                                const std::vector<int> &offsets, std::vector<double> &local_data) {
  const int local_size = send_counts[static_cast<size_t>(rank)];

  if (local_size == 0) {
    return;
  }

  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), send_counts.data(), offsets.data(), MPI_DOUBLE, local_data.data(), local_size,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, send_counts.data(), offsets.data(), MPI_DOUBLE, local_data.data(), local_size, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }
}

void KrymovaKLsdSortMergeDoubleALL::GatherResults(int rank, int size_comm, const std::vector<int> &send_counts,
                                                  std::vector<double> &local_data) {
  if (rank == 0) {
    std::vector<double> result;
    for (int i = 0; i < size_comm; ++i) {
      const int sz = send_counts[static_cast<size_t>(i)];
      if (sz == 0) {
        continue;
      }
      if (i == 0) {
        result = local_data;
      } else {
        std::vector<double> recv_buf(static_cast<size_t>(sz));
        MPI_Recv(recv_buf.data(), sz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result = SimpleMerge(result, recv_buf);
      }
    }
    GetOutput() = std::move(result);
  } else {
    const int local_size = send_counts[static_cast<size_t>(rank)];
    if (local_size > 0) {
      MPI_Send(local_data.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }
}

void KrymovaKLsdSortMergeDoubleALL::BroadcastResult(int rank) {
  int out_size = static_cast<int>(GetOutput().size());
  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (out_size <= 0) {
    if (rank != 0) {
      GetOutput().clear();
    }
    return;
  }

  if (rank != 0) {
    GetOutput().resize(static_cast<size_t>(out_size));
  }

  // Важно: проверяем, что указатель не nullptr
  if (out_size > 0 && GetOutput().data() != nullptr) {
    MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

bool KrymovaKLsdSortMergeDoubleALL::RunImpl() {
  int rank = 0;
  int size_comm = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size_comm);

  int total_size = static_cast<int>(GetInput().size());

  // Для маленьких данных - последовательная сортировка на rank 0
  if (total_size <= 100000) {
    if (rank == 0) {
      GetOutput() = GetInput();
      LSDSort(GetOutput().data(), total_size);
      int out_size = static_cast<int>(GetOutput().size());
      MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (out_size > 0) {
        MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
    } else {
      int out_size = 0;
      MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (out_size > 0) {
        GetOutput().resize(static_cast<size_t>(out_size));
        MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
    }
    return true;
  }

  // MPI версия для больших данных
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput().clear();
    return true;
  }

  std::vector<int> send_counts(static_cast<size_t>(size_comm));
  std::vector<int> offsets(static_cast<size_t>(size_comm));
  ComputeDistribution(total_size, size_comm, send_counts, offsets);

  const int local_size = send_counts[static_cast<size_t>(rank)];
  std::vector<double> local_data;

  if (local_size > 0) {
    local_data.resize(static_cast<size_t>(local_size));
  }

  ScatterData(rank, send_counts, offsets, local_data);

  if (local_size > 0) {
    LSDSort(local_data.data(), local_size);
  }

  GatherResults(rank, size_comm, send_counts, local_data);

  BroadcastResult(rank);

  return true;
}

bool KrymovaKLsdSortMergeDoubleALL::PostProcessingImpl() {
  const OutType &output = GetOutput();
  if (output.empty()) {
    return true;
  }
  for (size_t i = 1; i < output.size(); ++i) {
    if (output[i] < output[i - 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace krymova_k_lsd_sort_merge_double
