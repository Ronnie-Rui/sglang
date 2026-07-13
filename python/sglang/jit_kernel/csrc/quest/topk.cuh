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
constexpr uint32_t kMaxDirectMetadataWidth = 1024;
constexpr uint32_t kInvalidLogicalPage = 0x7fffffffu;

struct QuestTopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ k_per_req;
  float* __restrict__ output_scores;
  int32_t* __restrict__ output_indices;
  int64_t score_stride;
  uint32_t num_scores;
  uint32_t output_width;
};

struct QuestTopKMetadataParams {
  const float* __restrict__ scores;
  const void* __restrict__ k_per_req;
  const void* __restrict__ recent_indices;
  const uint8_t* __restrict__ recent_valid;
  const uint8_t* __restrict__ sparse_mask;
  const void* __restrict__ seq_lens;
  const void* __restrict__ req_pool_indices;
  const void* __restrict__ req_to_token;
  int32_t* __restrict__ page_table;
  int32_t* __restrict__ valid_lengths;
  int32_t* __restrict__ cache_seqlens;
  int32_t* __restrict__ cu_seqlens;
  int64_t score_stride;
  int64_t recent_stride;
  int64_t req_to_token_stride_r;
  int64_t req_to_token_stride_t;
  int64_t page_table_stride_b;
  int64_t page_table_stride_p;
  uint32_t batch_size;
  uint32_t num_scores;
  uint32_t topk_width;
  uint32_t recent_width;
  uint32_t output_width;
  uint32_t page_size;
  bool k_per_req_i32;
  bool recent_indices_i32;
  bool seq_lens_i32;
  bool req_pool_indices_i32;
  bool req_to_token_i32;
  bool update_lengths;
};

SGL_DEVICE int64_t quest_load_index(const void* ptr, bool is_int32, int64_t offset) {
  return is_int32 ? static_cast<int64_t>(static_cast<const int32_t*>(ptr)[offset])
                  : static_cast<const int64_t*>(ptr)[offset];
}

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

SGL_DEVICE bool quest_score_is_finite(float score) {
  return (__float_as_uint(score) & 0x7fffffffu) < 0x7f800000u;
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

template <int kScoreItemsPerThread, int kSelectedItemsPerThread>
__global__ __launch_bounds__(kBlockSize) void quest_topk_to_metadata_kernel(
    const __grid_constant__ QuestTopKMetadataParams params) {
  using ScoreSort = cub::BlockRadixSort<uint32_t, kBlockSize, kScoreItemsPerThread, int32_t>;
  using SelectedSort = cub::BlockRadixSort<uint32_t, kBlockSize, kSelectedItemsPerThread>;
  union SharedStorage {
    typename ScoreSort::TempStorage score_sort;
    typename SelectedSort::TempStorage selected_sort;
    uint32_t candidates[kBlockSize * kSelectedItemsPerThread];
  };
  static_assert(sizeof(SharedStorage) <= 48 * 1024);
  __shared__ SharedStorage shared;

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const float* const row_scores = params.scores + static_cast<int64_t>(row) * params.score_stride;
  const int64_t requested_topk_raw = quest_load_index(params.k_per_req, params.k_per_req_i32, row);
  const bool valid_requested_topk =
      requested_topk_raw >= 0 && static_cast<uint64_t>(requested_topk_raw) <= params.topk_width;
  if (!valid_requested_topk && tx == 0) {
    assert(requested_topk_raw >= 0 && static_cast<uint64_t>(requested_topk_raw) <= params.topk_width);
  }
  const uint32_t requested_topk =
      valid_requested_topk ? static_cast<uint32_t>(requested_topk_raw) : 0;
  const uint32_t selected_topk = min(requested_topk, params.num_scores);

  uint32_t score_keys[kScoreItemsPerThread];
  int32_t score_indices[kScoreItemsPerThread];
#pragma unroll
  for (int item = 0; item < kScoreItemsPerThread; ++item) {
    const uint32_t index = item * kBlockSize + tx;
    if (index < params.num_scores) {
      score_keys[item] = quest_topk_sort_key(row_scores[index]);
      score_indices[item] = static_cast<int32_t>(index);
    } else {
      score_keys[item] = 0;
      score_indices[item] = -1;
    }
  }
  ScoreSort(shared.score_sort).SortDescending(score_keys, score_indices);
  __syncthreads();

  // Stage only the fixed-width selected prefix. The following logical-page
  // sort is sized by K + recent rather than by N (up to 8192).
#pragma unroll
  for (int item = 0; item < kScoreItemsPerThread; ++item) {
    const uint32_t pos = tx * kScoreItemsPerThread + item;
    if (pos < params.output_width) {
      uint32_t logical_page = kInvalidLogicalPage;
      if (pos < params.topk_width) {
        if (pos < selected_topk) {
          const int32_t index = score_indices[item];
          const float score = row_scores[index];
          if (quest_score_is_finite(score)) logical_page = static_cast<uint32_t>(index);
        }
      } else {
        const uint32_t recent_pos = pos - params.topk_width;
        const int64_t recent_offset = static_cast<int64_t>(row) * params.recent_stride + recent_pos;
        if (params.recent_valid[recent_offset]) {
          const int64_t recent_page =
              quest_load_index(params.recent_indices, params.recent_indices_i32, recent_offset);
          if (recent_page >= 0 && recent_page < kInvalidLogicalPage) {
            logical_page = static_cast<uint32_t>(recent_page);
          }
        }
      }
      shared.candidates[pos] = logical_page;
    }
  }
  __syncthreads();

  uint32_t selected_keys[kSelectedItemsPerThread];
#pragma unroll
  for (int item = 0; item < kSelectedItemsPerThread; ++item) {
    const uint32_t pos = tx * kSelectedItemsPerThread + item;
    selected_keys[item] = pos < params.output_width ? shared.candidates[pos] : kInvalidLogicalPage;
  }
  __syncthreads();
  SelectedSort(shared.selected_sort).Sort(selected_keys);
  __syncthreads();

  // Reuse shared storage after both radix sorts to count the finite prefix.
  if (tx == 0) shared.candidates[0] = 0;
  __syncthreads();
  int32_t local_valid = 0;
#pragma unroll
  for (int item = 0; item < kSelectedItemsPerThread; ++item) {
    const uint32_t pos = tx * kSelectedItemsPerThread + item;
    local_valid += static_cast<int32_t>(pos < params.output_width && selected_keys[item] != kInvalidLogicalPage);
  }
  if (local_valid) atomicAdd(reinterpret_cast<int32_t*>(&shared.candidates[0]), local_valid);
  __syncthreads();
  const int32_t valid_length = static_cast<int32_t>(shared.candidates[0]);

  const bool use_sparse = params.sparse_mask[row] != 0;
  const bool active = use_sparse && valid_length > 0;
  const int64_t req_idx = active
      ? quest_load_index(params.req_pool_indices, params.req_pool_indices_i32, row)
      : 0;
#pragma unroll
  for (int item = 0; item < kSelectedItemsPerThread; ++item) {
    const uint32_t pos = tx * kSelectedItemsPerThread + item;
    const uint32_t logical_page = selected_keys[item];
    if (active && pos < params.output_width && logical_page != kInvalidLogicalPage) {
      const int64_t token_offset =
          req_idx * params.req_to_token_stride_r +
          static_cast<int64_t>(logical_page) * params.page_size * params.req_to_token_stride_t;
      const int64_t first_token =
          quest_load_index(params.req_to_token, params.req_to_token_i32, token_offset);
      params.page_table[
          static_cast<int64_t>(row) * params.page_table_stride_b +
          static_cast<int64_t>(pos) * params.page_table_stride_p] =
          static_cast<int32_t>(first_token / params.page_size);
    }
  }

  if (tx == 0) {
    params.valid_lengths[row] = active ? valid_length : 0;
    if (params.update_lengths) {
      const int64_t seq_len = quest_load_index(params.seq_lens, params.seq_lens_i32, row);
      const int64_t last_page_length = seq_len > 0 ? (seq_len - 1) % params.page_size + 1 : 0;
      const int64_t sparse_seq_len =
          valid_length > 0 ? (static_cast<int64_t>(valid_length) - 1) * params.page_size + last_page_length : 0;
      const int32_t cache_seq_len = static_cast<int32_t>(active ? sparse_seq_len : seq_len);
      params.cache_seqlens[row] = cache_seq_len;
      if (params.batch_size == 1) {
        params.cu_seqlens[0] = 0;
        params.cu_seqlens[1] = cache_seq_len;
      }
    }
  }
}

__global__ void quest_prefix_cache_seqlens_kernel(
    const int32_t* __restrict__ cache_seqlens,
    int32_t* __restrict__ cu_seqlens,
    uint32_t batch_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t cumulative = 0;
  for (uint32_t row = 0; row < batch_size; ++row) {
    cu_seqlens[row] = cumulative;
    cumulative += cache_seqlens[row];
  }
  cu_seqlens[batch_size] = cumulative;
}

template <int kItemsPerThread>
void launch_quest_topk(
    const uint32_t batch_size,
    const DLDevice device,
    const QuestTopKParams& params) {
  host::LaunchKernel(batch_size, kBlockSize, device)(
      quest_topk_kernel<kItemsPerThread>, params);
}

template <int kScoreItemsPerThread, int kSelectedItemsPerThread>
void launch_quest_topk_to_metadata(
    const DLDevice device,
    const QuestTopKMetadataParams& params) {
  host::LaunchKernel(params.batch_size, kBlockSize, device)(
      quest_topk_to_metadata_kernel<kScoreItemsPerThread, kSelectedItemsPerThread>, params);
}

template <int kScoreItemsPerThread>
void dispatch_quest_topk_to_metadata_by_width(
    const DLDevice device,
    const QuestTopKMetadataParams& params) {
  if (params.output_width <= 256) {
    launch_quest_topk_to_metadata<kScoreItemsPerThread, 1>(device, params);
  } else if (params.output_width <= 512) {
    launch_quest_topk_to_metadata<kScoreItemsPerThread, 2>(device, params);
  } else {
    launch_quest_topk_to_metadata<kScoreItemsPerThread, 4>(device, params);
  }
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

struct QuestTopKToMetadataKernel {
  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView k_per_req,
      const tvm::ffi::TensorView recent_indices,
      const tvm::ffi::TensorView recent_valid_u8,
      const tvm::ffi::TensorView sparse_mask_u8,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView valid_lengths,
      const tvm::ffi::TensorView cache_seqlens,
      const tvm::ffi::TensorView cu_seqlens,
      const int topk_width,
      const int page_size,
      const bool update_lengths) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_scores"};
    auto R = SymbolicSize{"recent_width"};
    auto P = SymbolicSize{"page_table_width"};
    auto BP1 = SymbolicSize{"batch_size_plus_one"};
    auto NumReqs = SymbolicSize{"num_request_slots"};
    auto MaxReqLen = SymbolicSize{"max_request_length"};
    auto ScoreStride = SymbolicSize{"score_stride"};
    auto RecentStride = SymbolicSize{"recent_stride"};
    auto ReqStride = SymbolicSize{"req_to_token_stride_r"};
    auto ReqTokenStride = SymbolicSize{"req_to_token_stride_t"};
    auto PageStride = SymbolicSize{"page_table_stride_b"};
    auto PageItemStride = SymbolicSize{"page_table_stride_p"};
    auto k_dtype = SymbolicDType{};
    auto recent_dtype = SymbolicDType{};
    auto seq_dtype = SymbolicDType{};
    auto req_dtype = SymbolicDType{};
    auto table_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, N})
        .with_strides({ScoreStride, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B})
        .with_strides({1})
        .with_dtype<int32_t, int64_t>(k_dtype)
        .with_device(device)
        .verify(k_per_req);
    TensorMatcher({B, R})
        .with_strides({RecentStride, 1})
        .with_dtype<int32_t, int64_t>(recent_dtype)
        .with_device(device)
        .verify(recent_indices);
    TensorMatcher({B, R})
        .with_strides({RecentStride, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(recent_valid_u8);
    TensorMatcher({B})
        .with_strides({1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(sparse_mask_u8);
    TensorMatcher({B})
        .with_strides({1})
        .with_dtype<int32_t, int64_t>(seq_dtype)
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({B})
        .with_strides({1})
        .with_dtype<int32_t, int64_t>(req_dtype)
        .with_device(device)
        .verify(req_pool_indices);
    TensorMatcher({NumReqs, MaxReqLen})
        .with_strides({ReqStride, ReqTokenStride})
        .with_dtype<int32_t, int64_t>(table_dtype)
        .with_device(device)
        .verify(req_to_token);
    TensorMatcher({B, P})
        .with_strides({PageStride, PageItemStride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B})
        .with_strides({1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(valid_lengths)
        .verify(cache_seqlens);
    TensorMatcher({BP1})
        .with_strides({1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(cu_seqlens);

    RuntimeCheck(page_size > 0, "page_size must be positive");
    RuntimeCheck(B.unwrap() >= 0, "batch_size cannot be negative");
    RuntimeCheck(BP1.unwrap() == B.unwrap() + 1, "cu_seqlens must have batch_size + 1 elements");
    RuntimeCheck(N.unwrap() >= 0 && N.unwrap() <= kMaxNumScores, "num_scores must be in [0, 8192]");
    RuntimeCheck(topk_width >= 0 && topk_width <= kMaxTopK, "topk width must be in [0, 2048]");
    const int64_t output_width_i64 = static_cast<int64_t>(topk_width) + R.unwrap();
    RuntimeCheck(
        output_width_i64 > 0 && output_width_i64 <= kMaxDirectMetadataWidth,
        "direct metadata width must be in [1, 1024]");
    RuntimeCheck(output_width_i64 <= N.unwrap(), "direct metadata width cannot exceed num_scores");
    RuntimeCheck(P.unwrap() >= output_width_i64, "page table is narrower than direct metadata output");
    RuntimeCheck(MaxReqLen.unwrap() >= page_size, "request table cannot hold one page");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = QuestTopKMetadataParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .k_per_req = k_per_req.data_ptr(),
        .recent_indices = recent_indices.data_ptr(),
        .recent_valid = static_cast<const uint8_t*>(recent_valid_u8.data_ptr()),
        .sparse_mask = static_cast<const uint8_t*>(sparse_mask_u8.data_ptr()),
        .seq_lens = seq_lens.data_ptr(),
        .req_pool_indices = req_pool_indices.data_ptr(),
        .req_to_token = req_to_token.data_ptr(),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .valid_lengths = static_cast<int32_t*>(valid_lengths.data_ptr()),
        .cache_seqlens = static_cast<int32_t*>(cache_seqlens.data_ptr()),
        .cu_seqlens = static_cast<int32_t*>(cu_seqlens.data_ptr()),
        .score_stride = ScoreStride.unwrap(),
        .recent_stride = RecentStride.unwrap(),
        .req_to_token_stride_r = ReqStride.unwrap(),
        .req_to_token_stride_t = ReqTokenStride.unwrap(),
        .page_table_stride_b = PageStride.unwrap(),
        .page_table_stride_p = PageItemStride.unwrap(),
        .batch_size = batch_size,
        .num_scores = static_cast<uint32_t>(N.unwrap()),
        .topk_width = static_cast<uint32_t>(topk_width),
        .recent_width = static_cast<uint32_t>(R.unwrap()),
        .output_width = static_cast<uint32_t>(output_width_i64),
        .page_size = static_cast<uint32_t>(page_size),
        .k_per_req_i32 = k_dtype.is_type<int32_t>(),
        .recent_indices_i32 = recent_dtype.is_type<int32_t>(),
        .seq_lens_i32 = seq_dtype.is_type<int32_t>(),
        .req_pool_indices_i32 = req_dtype.is_type<int32_t>(),
        .req_to_token_i32 = table_dtype.is_type<int32_t>(),
        .update_lengths = update_lengths,
    };

    if (N.unwrap() <= 1024) {
      dispatch_quest_topk_to_metadata_by_width<4>(device.unwrap(), params);
    } else if (N.unwrap() <= 2048) {
      dispatch_quest_topk_to_metadata_by_width<8>(device.unwrap(), params);
    } else if (N.unwrap() <= 4096) {
      dispatch_quest_topk_to_metadata_by_width<16>(device.unwrap(), params);
    } else {
      dispatch_quest_topk_to_metadata_by_width<32>(device.unwrap(), params);
    }

    if (update_lengths && batch_size > 1) {
      LaunchKernel(1, 1, device.unwrap())(
          quest_prefix_cache_seqlens_kernel,
          static_cast<const int32_t*>(cache_seqlens.data_ptr()),
          static_cast<int32_t*>(cu_seqlens.data_ptr()),
          batch_size);
    }
  }
};

}  // namespace
