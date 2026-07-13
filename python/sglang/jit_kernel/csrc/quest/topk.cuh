/// \file topk.cuh
/// \brief Per-request radix top-k for Quest page scores.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_radix_sort.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kMaxNumScores = 8192;

struct QuestTopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ k_per_req;
  float* __restrict__ output_scores;
  int32_t* __restrict__ output_indices;
  int64_t score_stride;
  uint32_t num_scores;
  uint32_t output_width;
};

SGL_DEVICE uint32_t quest_topk_sort_key(float score) {
  // torch.topk orders NaN above +inf. Keep the original score for output so
  // the existing isfinite finalizer can discard it.
  uint32_t bits = __float_as_uint(score);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return 0xffffffffu;
  } else if (magnitude == 0) {
    bits = 0;
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int kItemsPerThread>
__global__ __launch_bounds__(kBlockSize) void quest_topk_kernel(
    const __grid_constant__ QuestTopKParams params) {
  using BlockRadixSort =
      cub::BlockRadixSort<uint32_t, kBlockSize, kItemsPerThread, int32_t>;
  static_assert(sizeof(typename BlockRadixSort::TempStorage) <= 48 * 1024);
  __shared__ typename BlockRadixSort::TempStorage sort_storage;

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const int64_t output_offset = static_cast<int64_t>(row) * params.output_width;
  float* const row_output_scores = params.output_scores + output_offset;
  int32_t* const row_output_indices = params.output_indices + output_offset;
  const float* const row_scores = params.scores + static_cast<int64_t>(row) * params.score_stride;
  const int32_t requested_topk = params.k_per_req[row];

  const auto pad_output = [&] {
    for (uint32_t pos = tx; pos < params.output_width; pos += kBlockSize) {
      row_output_scores[pos] = -std::numeric_limits<float>::infinity();
      row_output_indices[pos] = -1;
    }
  };

  // k_per_req deliberately stays device-resident. The assertion reports an
  // invalid producer without introducing a host synchronization, while this
  // guard keeps all memory accesses in bounds even in builds without asserts.
  if (requested_topk < 0 || static_cast<uint32_t>(requested_topk) > params.output_width) {
    pad_output();
    if (tx == 0) assert(requested_topk >= 0 && static_cast<uint32_t>(requested_topk) <= params.output_width);
    return;
  }

  const uint32_t topk = min(static_cast<uint32_t>(requested_topk), params.num_scores);
  if (topk == 0) {
    pad_output();
    return;
  }

  // Selecting every item does not need the radix passes. This also defines the
  // N < requested_topk case: emit all N inputs and pad the remaining capacity.
  if (params.num_scores <= static_cast<uint32_t>(requested_topk)) {
    for (uint32_t pos = tx; pos < params.output_width; pos += kBlockSize) {
      if (pos < params.num_scores) {
        row_output_scores[pos] = row_scores[pos];
        row_output_indices[pos] = static_cast<int32_t>(pos);
      } else {
        row_output_scores[pos] = -std::numeric_limits<float>::infinity();
        row_output_indices[pos] = -1;
      }
    }
    return;
  }

  uint32_t keys[kItemsPerThread];
  int32_t indices[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    // Striped global reads keep each warp coalesced. BlockRadixSort still sees
    // the complete key/index multiset and returns its output in blocked order.
    const uint32_t index = item * kBlockSize + tx;
    if (index < params.num_scores) {
      keys[item] = quest_topk_sort_key(row_scores[index]);
      indices[item] = static_cast<int32_t>(index);
    } else {
      // Every real non-NaN float, including -inf, has an ordered key greater
      // than zero. Padding therefore cannot enter the first min(K, N) slots.
      keys[item] = 0;
      indices[item] = -1;
    }
  }

  BlockRadixSort(sort_storage).SortDescending(keys, indices);

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const uint32_t pos = tx * kItemsPerThread + item;
    if (pos < topk) {
      const int32_t index = indices[item];
      row_output_scores[pos] = row_scores[index];
      row_output_indices[pos] = index;
    }
  }
  for (uint32_t pos = topk + tx; pos < params.output_width; pos += kBlockSize) {
    row_output_scores[pos] = -std::numeric_limits<float>::infinity();
    row_output_indices[pos] = -1;
  }
}

template <int kItemsPerThread>
void launch_quest_topk(
    const uint32_t batch_size,
    const DLDevice device,
    const QuestTopKParams& params) {
  host::LaunchKernel(batch_size, kBlockSize, device)(
      quest_topk_kernel<kItemsPerThread>, params);
}

struct QuestTopKKernel {
  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView k_per_req,
      const tvm::ffi::TensorView output_scores,
      const tvm::ffi::TensorView output_indices) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_scores"};
    auto K = SymbolicSize{"output_width"};
    auto S = SymbolicSize{"score_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, N})
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(k_per_req);
    TensorMatcher({B, K}).with_dtype<float>().with_device(device).verify(output_scores);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(output_indices);

    RuntimeCheck(N.unwrap() >= 0 && N.unwrap() <= kMaxNumScores, "num_scores must be in [0, 8192]");
    RuntimeCheck(K.unwrap() >= 0 && K.unwrap() <= kMaxTopK, "output_width must be in [0, 2048]");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = QuestTopKParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .k_per_req = static_cast<const int32_t*>(k_per_req.data_ptr()),
        .output_scores = static_cast<float*>(output_scores.data_ptr()),
        .output_indices = static_cast<int32_t*>(output_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .num_scores = static_cast<uint32_t>(N.unwrap()),
        .output_width = static_cast<uint32_t>(K.unwrap()),
    };
    if (N.unwrap() <= 1024) {
      launch_quest_topk<4>(batch_size, device.unwrap(), params);
    } else if (N.unwrap() <= 2048) {
      launch_quest_topk<8>(batch_size, device.unwrap(), params);
    } else if (N.unwrap() <= 4096) {
      launch_quest_topk<16>(batch_size, device.unwrap(), params);
    } else {
      launch_quest_topk<32>(batch_size, device.unwrap(), params);
    }
  }
};

}  // namespace
