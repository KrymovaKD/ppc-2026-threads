#include "krymova_k_lsd_sort_merge_double/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <vector>

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
  uint64_t ull;
  std::memcpy(&ull, &d, sizeof(double));
  return (ull & 0x8000000000000000ULL) ? ~ull : (ull | 0x8000000000000000ULL);
}

double KrymovaKLsdSortMergeDoubleALL::ULLToDouble(uint64_t ull) {
  if (ull & 0x8000000000000000ULL) {
    ull &= 0x7FFFFFFFFFFFFFFFULL;
  } else {
    ull = ~ull;
  }
  double d;
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

  std::vector<uint64_t> ull_arr(size);
  std::vector<uint64_t> ull_tmp(size);
  std::vector<unsigned int> count(k_radix, 0U);

  for (int i = 0; i < size; ++i) {
    ull_arr[i] = DoubleToULL(arr[i]);
  }

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

std::vector<double> KrymovaKLsdSortMergeDoubleALL::SimpleMerge(const std::vector<double> &a,
                                                               const std::vector<double> &b) {
  std::vector<double> res;
  res.reserve(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(res));
  return res;
}

bool KrymovaKLsdSortMergeDoubleALL::RunImpl() {
  int rank, size_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size_comm);

  // fast path для одного процесса
  if (size_comm == 1) {
    auto &input = GetInput();
    std::vector<double> result = input;
    LSDSort(result.data(), static_cast<int>(result.size()));
    GetOutput() = std::move(result);
    return true;
  }

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput().clear();
    return true;
  }

  std::vector<int> send_counts(size_comm), displs(size_comm);
  int base = total_size / size_comm;
  int rem = total_size % size_comm;
  for (int i = 0; i < size_comm; ++i) {
    send_counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  std::vector<double> local_data(send_counts[rank]);

  MPI_Scatterv((rank == 0) ? GetInput().data() : nullptr, send_counts.data(), displs.data(), MPI_DOUBLE,
               local_data.data(), send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (send_counts[rank] > 0) {
    LSDSort(local_data.data(), send_counts[rank]);
  }

  std::vector<double> gathered;
  if (rank == 0) {
    gathered.resize(total_size);
  }

  MPI_Gatherv(local_data.data(), send_counts[rank], MPI_DOUBLE, (rank == 0) ? gathered.data() : nullptr,
              send_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<double> result;
    int pos = 0;
    for (int i = 0; i < size_comm; ++i) {
      if (send_counts[i] == 0) {
        continue;
      }
      std::vector<double> chunk(gathered.begin() + pos, gathered.begin() + pos + send_counts[i]);
      result = result.empty() ? std::move(chunk) : SimpleMerge(result, chunk);
      pos += send_counts[i];
    }
    GetOutput() = std::move(result);
  }

  int out_size = static_cast<int>(GetOutput().size());
  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetOutput().resize(out_size);
  }
  if (out_size > 0) {
    MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool KrymovaKLsdSortMergeDoubleALL::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  const OutType &output = GetOutput();
  for (size_t i = 1; i < output.size(); ++i) {
    if (output[i] < output[i - 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace krymova_k_lsd_sort_merge_double
