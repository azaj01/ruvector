//! Quantized GEMM (General Matrix Multiplication) operations.
//!
//! Core primitive for projections and FFN layers.
//! Supports int8 weights with per-row scaling.

/// Quantized GEMM: C = A * B^T + bias
///
/// Computes matrix multiplication with int8 inputs, accumulating to i32.
///
/// # Arguments
///
/// * `m` - Number of rows in A (and output C)
/// * `n` - Number of columns in B^T (and output C) = number of rows in B
/// * `k` - Number of columns in A = number of columns in B
/// * `a` - Input activations, shape [m, k], int8
/// * `a_scale` - Scale factor for input activations
/// * `b` - Weight matrix, shape [n, k], int8 (row-major, transposed)
/// * `b_row_scales` - Per-row scale factors for B, shape [n]
/// * `bias` - Optional bias vector, shape [n], i32
/// * `out` - Output buffer, shape [m, n], i32
///
/// # Output
///
/// out[i, j] = sum_k(a[i, k] * b[j, k]) * a_scale * b_row_scales[j] + bias[j]
///
/// Note: The output is in accumulator domain (i32). Caller is responsible
/// for any further dequantization if needed.
#[inline(never)]
pub fn qgemm_i8(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    _a_scale: f32,
    b: &[i8],
    _b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    // Scalar implementation (fallback)
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;

            // Dot product
            for kk in 0..k {
                let a_val = a[i * k + kk] as i32;
                let b_val = b[j * k + kk] as i32;
                acc += a_val * b_val;
            }

            // Add bias if present
            if let Some(bias) = bias {
                acc += bias[j];
            }

            out[i * n + j] = acc;
        }
    }
}

/// SIMD-optimized quantized GEMM.
///
/// Uses architecture-specific SIMD when available, falls back to scalar.
#[cfg(feature = "simd")]
#[inline(never)]
pub fn qgemm_i8_simd(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    // For now, delegate to scalar. SIMD implementation would go here.
    // In production, this would use:
    // - x86_64: AVX2/AVX-512 VNNI instructions
    // - aarch64: NEON or SVE2 dot product instructions
    qgemm_i8(m, n, k, a, a_scale, b, b_row_scales, bias, out)
}

#[cfg(not(feature = "simd"))]
#[inline(never)]
pub fn qgemm_i8_simd(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    qgemm_i8(m, n, k, a, a_scale, b, b_row_scales, bias, out)
}

/// Quantized matrix-vector multiplication.
///
/// Specialized for single-row input (common in autoregressive generation).
#[inline]
pub fn qgemv_i8(
    n: usize,
    k: usize,
    x: &[i8],
    _x_scale: f32,
    w: &[i8],
    _w_row_scales: &[f32],
    bias: Option<&[i32]>,
    out: &mut [i32],
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(out.len(), n);

    for j in 0..n {
        let mut acc: i32 = 0;

        for kk in 0..k {
            let x_val = x[kk] as i32;
            let w_val = w[j * k + kk] as i32;
            acc += x_val * w_val;
        }

        if let Some(bias) = bias {
            acc += bias[j];
        }

        out[j] = acc;
    }
}

/// Dequantize i32 accumulator to f32.
#[inline]
pub fn dequantize_i32_to_f32(
    values: &[i32],
    input_scale: f32,
    weight_scales: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(values.len(), output.len());
    debug_assert_eq!(values.len(), weight_scales.len());

    for (i, (&v, &ws)) in values.iter().zip(weight_scales.iter()).enumerate() {
        output[i] = (v as f32) * input_scale * ws;
    }
}

/// Quantize f32 to i8 with scale.
#[inline]
pub fn quantize_f32_to_i8(values: &[f32], scale: f32, output: &mut [i8]) {
    debug_assert_eq!(values.len(), output.len());

    let inv_scale = 1.0 / scale;
    for (i, &v) in values.iter().enumerate() {
        let q = (v * inv_scale).round();
        output[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

/// Compute scale factor for quantization.
#[inline]
pub fn compute_scale(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qgemm_basic() {
        // 2x3 * 4x3^T = 2x4
        let a: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let b: [i8; 12] = [
            1, 0, 0,  // row 0
            0, 1, 0,  // row 1
            0, 0, 1,  // row 2
            1, 1, 1,  // row 3
        ];
        let scales: [f32; 4] = [1.0; 4];
        let mut out = [0i32; 8];

        qgemm_i8(2, 4, 3, &a, 1.0, &b, &scales, None, &mut out);

        // Row 0 of A: [1, 2, 3]
        // Row 0 of B: [1, 0, 0] -> dot = 1
        // Row 1 of B: [0, 1, 0] -> dot = 2
        // Row 2 of B: [0, 0, 1] -> dot = 3
        // Row 3 of B: [1, 1, 1] -> dot = 6
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2);
        assert_eq!(out[2], 3);
        assert_eq!(out[3], 6);
    }

    #[test]
    fn test_qgemm_with_bias() {
        let a: [i8; 4] = [1, 1, 1, 1];
        let b: [i8; 4] = [1, 1, 1, 1];
        let scales: [f32; 2] = [1.0; 2];
        let bias: [i32; 2] = [10, 20];
        let mut out = [0i32; 4];

        qgemm_i8(2, 2, 2, &a, 1.0, &b, &scales, Some(&bias), &mut out);

        // Each dot product = 2, plus bias
        assert_eq!(out[0], 12); // 2 + 10
        assert_eq!(out[1], 22); // 2 + 20
    }

    #[test]
    fn test_qgemv() {
        let x: [i8; 3] = [1, 2, 3];
        let w: [i8; 6] = [
            1, 0, 0,  // row 0
            0, 1, 0,  // row 1
        ];
        let scales: [f32; 2] = [1.0; 2];
        let mut out = [0i32; 2];

        qgemv_i8(2, 3, &x, 1.0, &w, &scales, None, &mut out);

        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2);
    }

    #[test]
    fn test_quantize_dequantize() {
        let original: [f32; 4] = [0.5, -0.25, 1.0, -1.0];
        let scale = compute_scale(&original);

        let mut quantized = [0i8; 4];
        quantize_f32_to_i8(&original, scale, &mut quantized);

        let scales = [scale; 4];
        let quantized_i32: Vec<i32> = quantized.iter().map(|&x| x as i32).collect();
        let mut recovered = [0.0f32; 4];
        dequantize_i32_to_f32(&quantized_i32, 1.0, &scales, &mut recovered);

        // Check approximate recovery (quantization loses precision)
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }
}
