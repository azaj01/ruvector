//! SIMD-optimized distance implementations
//!
//! Provides AVX2 and ARM NEON implementations of distance functions.
//! AVX-512 requires nightly Rust and is gated behind a feature flag.
//! Includes zero-copy raw pointer variants for maximum performance in index operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar;

// ============================================================================
// Pointer-based Zero-Copy SIMD Implementations
// ============================================================================

/// Check if pointer is aligned to N bytes
#[inline]
fn is_aligned_to(ptr: *const f32, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// Check if both pointers are 32-byte aligned (AVX2)
#[inline]
fn is_avx2_aligned(a: *const f32, b: *const f32) -> bool {
    is_aligned_to(a, 32) && is_aligned_to(b, 32)
}

// ============================================================================
// AVX2 Pointer-based Implementations (Zero-Copy)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Euclidean distance using raw pointers (AVX2, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
pub unsafe fn l2_distance_ptr_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    let use_aligned = is_avx2_aligned(a, b);

    if use_aligned {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_load_ps(a.add(offset));
            let vb = _mm256_load_ps(b.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
    } else {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
    }

    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    for i in (chunks * 8)..len {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Cosine distance using raw pointers (AVX2, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
pub unsafe fn cosine_distance_ptr_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = len / 8;
    let use_aligned = is_avx2_aligned(a, b);

    if use_aligned {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_load_ps(a.add(offset));
            let vb = _mm256_load_ps(b.add(offset));

            dot = _mm256_fmadd_ps(va, vb, dot);
            norm_a = _mm256_fmadd_ps(va, va, norm_a);
            norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
        }
    } else {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));

            dot = _mm256_fmadd_ps(va, vb, dot);
            norm_a = _mm256_fmadd_ps(va, va, norm_a);
            norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
        }
    }

    let dot_sum = horizontal_sum_256(dot);
    let norm_a_sum = horizontal_sum_256(norm_a);
    let norm_b_sum = horizontal_sum_256(norm_b);

    let mut dot_total = dot_sum;
    let mut norm_a_total = norm_a_sum;
    let mut norm_b_total = norm_b_sum;

    // Handle remainder
    for i in (chunks * 8)..len {
        let a_val = *a.add(i);
        let b_val = *b.add(i);
        dot_total += a_val * b_val;
        norm_a_total += a_val * a_val;
        norm_b_total += b_val * b_val;
    }

    let denominator = (norm_a_total * norm_b_total).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_total / denominator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Inner product using raw pointers (AVX2, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
pub unsafe fn inner_product_ptr_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    let use_aligned = is_avx2_aligned(a, b);

    if use_aligned {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_load_ps(a.add(offset));
            let vb = _mm256_load_ps(b.add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
    } else {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
    }

    let mut result = horizontal_sum_256(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        result += *a.add(i) * *b.add(i);
    }

    -result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
/// Manhattan distance using raw pointers (AVX2, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
pub unsafe fn manhattan_distance_ptr_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let sign_mask = _mm256_set1_ps(-0.0);
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    let use_aligned = is_avx2_aligned(a, b);

    if use_aligned {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_load_ps(a.add(offset));
            let vb = _mm256_load_ps(b.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            sum = _mm256_add_ps(sum, abs_diff);
        }
    } else {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            sum = _mm256_add_ps(sum, abs_diff);
        }
    }

    let mut result = horizontal_sum_256(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        result += (*a.add(i) - *b.add(i)).abs();
    }

    result
}

// ============================================================================
// Scalar Pointer-based Implementations (Zero-Copy Fallback)
// ============================================================================

/// Euclidean distance using raw pointers (scalar fallback, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn l2_distance_ptr_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut sum = 0.0f32;
    for i in 0..len {
        let diff = *a.add(i) - *b.add(i);
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Cosine distance using raw pointers (scalar fallback, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn cosine_distance_ptr_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        let a_val = *a.add(i);
        let b_val = *b.add(i);
        dot += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }

    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot / denominator)
}

/// Inner product using raw pointers (scalar fallback, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn inner_product_ptr_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut sum = 0.0f32;
    for i in 0..len {
        sum += *a.add(i) * *b.add(i);
    }
    -sum
}

/// Manhattan distance using raw pointers (scalar fallback, zero-copy)
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn manhattan_distance_ptr_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut sum = 0.0f32;
    for i in 0..len {
        sum += (*a.add(i) - *b.add(i)).abs();
    }
    sum
}

// ============================================================================
// Public Pointer-based Wrappers with Runtime Dispatch
// ============================================================================

/// Euclidean (L2) distance with zero-copy pointer access
///
/// Automatically selects the best SIMD implementation available:
/// - AVX2 (8 floats per iteration)
/// - Scalar fallback
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
/// - No overlap between memory regions is allowed
#[inline]
pub unsafe fn l2_distance_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return l2_distance_ptr_avx2(a, b, len);
        }
    }

    l2_distance_ptr_scalar(a, b, len)
}

/// Cosine distance with zero-copy pointer access
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn cosine_distance_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return cosine_distance_ptr_avx2(a, b, len);
        }
    }

    cosine_distance_ptr_scalar(a, b, len)
}

/// Inner product with zero-copy pointer access
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn inner_product_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return inner_product_ptr_avx2(a, b, len);
        }
    }

    inner_product_ptr_scalar(a, b, len)
}

/// Manhattan distance with zero-copy pointer access
///
/// # Safety
/// - `a` and `b` must be valid for reads of `len` elements
/// - `len` must be > 0
#[inline]
pub unsafe fn manhattan_distance_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return manhattan_distance_ptr_avx2(a, b, len);
        }
    }

    manhattan_distance_ptr_scalar(a, b, len)
}

// ============================================================================
// Batch Distance Functions for Index Operations
// ============================================================================

/// Batch L2 distance calculation for index operations
///
/// Computes distances from a query vector to multiple vectors in parallel.
///
/// # Safety
/// - `query` must be valid for reads of `len` elements
/// - All pointers in `vectors` must be valid for reads of `len` elements
/// - `results` must have length >= `vectors.len()`
/// - `len` must be > 0
#[inline]
pub unsafe fn l2_distances_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = l2_distance_ptr(query, vec_ptr, len);
    }
}

/// Batch cosine distance calculation for index operations
///
/// # Safety
/// - `query` must be valid for reads of `len` elements
/// - All pointers in `vectors` must be valid for reads of `len` elements
/// - `results` must have length >= `vectors.len()`
/// - `len` must be > 0
#[inline]
pub unsafe fn cosine_distances_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = cosine_distance_ptr(query, vec_ptr, len);
    }
}

/// Batch inner product calculation for index operations
///
/// # Safety
/// - `query` must be valid for reads of `len` elements
/// - All pointers in `vectors` must be valid for reads of `len` elements
/// - `results` must have length >= `vectors.len()`
/// - `len` must be > 0
#[inline]
pub unsafe fn inner_product_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = inner_product_ptr(query, vec_ptr, len);
    }
}

/// Batch manhattan distance calculation for index operations
///
/// # Safety
/// - `query` must be valid for reads of `len` elements
/// - All pointers in `vectors` must be valid for reads of `len` elements
/// - `results` must have length >= `vectors.len()`
/// - `len` must be > 0
#[inline]
pub unsafe fn manhattan_distances_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = manhattan_distance_ptr(query, vec_ptr, len);
    }
}

/// Batch L2 distance calculation (sequential, SIMD-optimized)
///
/// # Safety
/// - `query` must be valid for reads of `len` elements
/// - All pointers in `vectors` must be valid for reads of `len` elements
/// - `results` must have length >= `vectors.len()`
/// - `len` must be > 0
#[inline]
pub unsafe fn l2_distances_batch_parallel(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    // Sequential loop with SIMD-optimized inner distance
    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = l2_distance_ptr(query, vec_ptr, len);
    }
}

/// Batch cosine distance calculation (sequential, SIMD-optimized)
///
/// # Safety
/// - Same safety requirements as `l2_distances_batch_parallel`
#[inline]
pub unsafe fn cosine_distances_batch_parallel(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    results: &mut [f32],
) {
    debug_assert!(results.len() >= vectors.len());
    debug_assert!(!query.is_null() && len > 0);

    // Sequential loop with SIMD-optimized inner distance
    for (i, &vec_ptr) in vectors.iter().enumerate() {
        results[i] = cosine_distance_ptr(query, vec_ptr, len);
    }
}

// ============================================================================
// AVX2 Implementations (Slice-based)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    for i in (chunks * 8)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        dot = _mm256_fmadd_ps(va, vb, dot);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    // Horizontal sums
    let dot_sum = horizontal_sum_256(dot);
    let norm_a_sum = horizontal_sum_256(norm_a);
    let norm_b_sum = horizontal_sum_256(norm_b);

    let mut dot_total = dot_sum;
    let mut norm_a_total = norm_a_sum;
    let mut norm_b_total = norm_b_sum;

    for i in (chunks * 8)..n {
        dot_total += a[i] * b[i];
        norm_a_total += a[i] * a[i];
        norm_b_total += b[i] * b[i];
    }

    let denominator = (norm_a_total * norm_b_total).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_total / denominator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum_256(sum);

    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    -result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let sign_mask = _mm256_set1_ps(-0.0); // Sign bit mask
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff); // Clear sign bit
        sum = _mm256_add_ps(sum, abs_diff);
    }

    let mut result = horizontal_sum_256(sum);

    for i in (chunks * 8)..n {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_256(v: __m256) -> f32 {
    let sum_high = _mm256_extractf128_ps(v, 1);
    let sum_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

// ============================================================================
// ARM NEON Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));

        dot = vfmaq_f32(dot, va, vb);
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
    }

    let mut dot_sum = vaddvq_f32(dot);
    let mut norm_a_sum = vaddvq_f32(norm_a);
    let mut norm_b_sum = vaddvq_f32(norm_b);

    for i in (chunks * 4)..n {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    let denominator = (norm_a_sum * norm_b_sum).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_sum / denominator)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }

    -result
}

// ============================================================================
// Public Wrapper Functions
// ============================================================================

// AVX2 wrappers
#[cfg(target_arch = "x86_64")]
pub fn euclidean_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { euclidean_distance_avx2(a, b) }
    } else {
        scalar::euclidean_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn euclidean_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn cosine_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { cosine_distance_avx2(a, b) }
    } else {
        scalar::cosine_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn cosine_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn inner_product_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { inner_product_avx2(a, b) }
    } else {
        scalar::inner_product_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn inner_product_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::inner_product_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn manhattan_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { manhattan_distance_avx2(a, b) }
    } else {
        scalar::manhattan_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn manhattan_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::manhattan_distance(a, b)
}

// NEON wrappers
#[cfg(target_arch = "aarch64")]
pub fn euclidean_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { euclidean_distance_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn euclidean_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean_distance(a, b)
}

#[cfg(target_arch = "aarch64")]
pub fn cosine_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_distance_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn cosine_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

#[cfg(target_arch = "aarch64")]
pub fn inner_product_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { inner_product_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn inner_product_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::inner_product_distance(a, b)
}

// ============================================================================
// Optimized Pre-Normalized Cosine Distance (Just Dot Product)
// When vectors are already normalized, cosine distance = 1 - dot_product
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Cosine distance for pre-normalized vectors (AVX2)
pub unsafe fn cosine_distance_normalized_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut dot = _mm256_setzero_ps();
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        dot = _mm256_fmadd_ps(va, vb, dot);
    }

    let mut result = horizontal_sum_256(dot);

    for i in (chunks * 8)..len {
        result += *a.add(i) * *b.add(i);
    }

    1.0 - result
}

/// Cosine distance for pre-normalized vectors (scalar)
#[inline]
pub unsafe fn cosine_distance_normalized_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(!a.is_null() && !b.is_null() && len > 0);

    let mut dot = 0.0f32;
    for i in 0..len {
        dot += *a.add(i) * *b.add(i);
    }

    1.0 - dot
}

/// Pre-normalized cosine distance (auto-dispatched)
#[inline]
pub unsafe fn cosine_distance_normalized_ptr(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return cosine_distance_normalized_avx2(a, b, len);
        }
    }

    cosine_distance_normalized_scalar(a, b, len)
}

/// Pre-normalized cosine distance (slice version)
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe { cosine_distance_normalized_ptr(a.as_ptr(), b.as_ptr(), a.len()) }
}

// ============================================================================
// Batch Operations for Multiple Vectors (Efficient for K-NN)
// ============================================================================

/// Compute top-k nearest neighbors with L2 distance
#[inline]
pub unsafe fn l2_topk_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, &ptr)| (i, l2_distance_ptr(query, ptr, len)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Compute top-k nearest neighbors with normalized cosine distance
#[inline]
pub unsafe fn cosine_topk_normalized_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, &ptr)| (i, cosine_distance_normalized_ptr(query, ptr, len)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_euclidean() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let scalar = scalar::euclidean_distance(&a, &b);
        let simd = euclidean_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_cosine() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.01).collect();

        let scalar = scalar::cosine_distance(&a, &b);
        let simd = cosine_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_inner_product() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.01).collect();

        let scalar = scalar::inner_product_distance(&a, &b);
        let simd = inner_product_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-3, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_manhattan() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let scalar = scalar::manhattan_distance(&a, &b);
        let simd = manhattan_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_remainder_handling() {
        // Test with non-aligned sizes
        for size in [1, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

            let scalar = scalar::euclidean_distance(&a, &b);
            let simd = euclidean_distance_avx2_wrapper(&a, &b);

            assert!(
                (scalar - simd).abs() < 1e-3,
                "size={}, scalar={}, simd={}",
                size,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_ptr_l2_distance() {
        let a: Vec<f32> = vec![0.0, 0.0, 0.0];
        let b: Vec<f32> = vec![3.0, 4.0, 0.0];

        let dist = unsafe { l2_distance_ptr(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((dist - 5.0).abs() < 1e-5, "Expected 5.0, got {}", dist);
    }

    #[test]
    fn test_ptr_cosine_distance() {
        let a: Vec<f32> = vec![1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0];

        let dist = unsafe { cosine_distance_ptr(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!(dist.abs() < 1e-5, "Expected ~0.0, got {}", dist);
    }

    #[test]
    fn test_ptr_inner_product() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];

        let dist = unsafe { inner_product_ptr(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((dist - (-32.0)).abs() < 1e-5, "Expected -32.0, got {}", dist);
    }

    #[test]
    fn test_ptr_manhattan_distance() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 6.0, 8.0];

        let dist = unsafe { manhattan_distance_ptr(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((dist - 12.0).abs() < 1e-5, "Expected 12.0, got {}", dist);
    }
}
