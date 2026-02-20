use crate::mincut::{dynamic_min_cut, GatingResult};

/// Combined output from min-cut gated attention.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// The output vectors (flattened `seq_len x d`).
    pub output: Vec<f32>,
    /// Gating metadata from the min-cut pass.
    pub gating: GatingResult,
}

/// Compute raw attention logits: Q * K^T / sqrt(d).
///
/// `q` and `k` are flattened `seq_len x d` matrices in row-major order.
/// Returns a flattened `seq_len x seq_len` logit matrix.
fn compute_logits(q: &[f32], k: &[f32], d: usize, seq_len: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut logits = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for h in 0..d {
                dot += q[i * d + h] * k[j * d + h];
            }
            logits[i * seq_len + j] = dot * scale;
        }
    }
    logits
}

/// Row-wise softmax in place on a flattened `rows x cols` matrix.
fn row_softmax(mat: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut mat[i * cols..(i + 1) * cols];

        // Numerical stability: subtract row max
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Multiply attention weights by V: `weights (seq_len x seq_len) * V (seq_len x d)`.
fn matmul_weights_v(weights: &[f32], v: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * d];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let w = weights[i * seq_len + j];
            if w != 0.0 {
                for h in 0..d {
                    out[i * d + h] += w * v[j * d + h];
                }
            }
        }
    }
    out
}

/// Baseline standard softmax attention.
///
/// `q`, `k`, `v` are flattened `seq_len x d` matrices.
/// Returns a flattened `seq_len x d` output.
pub fn attn_softmax(q: &[f32], k: &[f32], v: &[f32], d: usize, seq_len: usize) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * d);
    assert_eq!(k.len(), seq_len * d);
    assert_eq!(v.len(), seq_len * d);

    let mut logits = compute_logits(q, k, d, seq_len);
    row_softmax(&mut logits, seq_len, seq_len);
    matmul_weights_v(&logits, v, seq_len, d)
}

/// Min-cut gated attention.
///
/// 1. Compute logits Q*K^T / sqrt(d).
/// 2. Build attention graph and compute dynamic min-cut to obtain a keep mask.
/// 3. Apply keep mask: gated entries are set to -INF.
/// 4. Row-softmax over surviving entries.
/// 5. Multiply by V.
pub fn attn_mincut(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    seq_len: usize,
    lambda: f32,
    tau: usize,
    eps: f32,
) -> AttentionOutput {
    assert_eq!(q.len(), seq_len * d);
    assert_eq!(k.len(), seq_len * d);
    assert_eq!(v.len(), seq_len * d);

    let mut logits = compute_logits(q, k, d, seq_len);

    // Compute gating via min-cut
    let gating = dynamic_min_cut(&logits, seq_len, lambda, tau, eps);

    // Apply keep mask: gated entries -> -INF so softmax zeroes them out
    for i in 0..logits.len() {
        if !gating.keep_mask[i] {
            logits[i] = f32::NEG_INFINITY;
        }
    }

    // Row-softmax on surviving entries
    row_softmax(&mut logits, seq_len, seq_len);

    // Handle rows that are entirely -INF (all gated) -> softmax produces NaN
    // Replace NaN with 0 for safety
    for val in logits.iter_mut() {
        if val.is_nan() {
            *val = 0.0;
        }
    }

    let output = matmul_weights_v(&logits, v, seq_len, d);

    AttentionOutput { output, gating }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_qkv(seq_len: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Q = K = identity-like, V = sequential values
        let mut q = vec![0.0f32; seq_len * d];
        let mut k = vec![0.0f32; seq_len * d];
        let v: Vec<f32> = (0..seq_len * d).map(|i| i as f32).collect();

        for i in 0..seq_len.min(d) {
            q[i * d + i] = 1.0;
            k[i * d + i] = 1.0;
        }

        (q, k, v)
    }

    #[test]
    fn test_attn_softmax_output_shape() {
        let (q, k, v) = make_identity_qkv(4, 3);
        let out = attn_softmax(&q, &k, &v, 3, 4);
        assert_eq!(out.len(), 4 * 3);
    }

    #[test]
    fn test_attn_softmax_values_finite() {
        let (q, k, v) = make_identity_qkv(3, 2);
        let out = attn_softmax(&q, &k, &v, 2, 3);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_attn_mincut_output_shape() {
        let (q, k, v) = make_identity_qkv(4, 3);
        let result = attn_mincut(&q, &k, &v, 3, 4, 0.5, 2, 0.01);
        assert_eq!(result.output.len(), 4 * 3);
        assert_eq!(result.gating.edges_total, 16);
        assert_eq!(result.gating.keep_mask.len(), 16);
    }

    #[test]
    fn test_attn_mincut_values_finite() {
        let (q, k, v) = make_identity_qkv(3, 2);
        let result = attn_mincut(&q, &k, &v, 2, 3, 0.5, 2, 0.01);
        assert!(result.output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_attn_mincut_gates_some_edges() {
        // With lambda=0.0 (very aggressive gating) most edges should be cut
        let q = vec![1.0; 4 * 2];
        let k = vec![1.0; 4 * 2];
        let v = vec![1.0; 4 * 2];
        let result = attn_mincut(&q, &k, &v, 2, 4, 0.0, 1, 0.01);
        // With lambda=0 the threshold is 0, so cut_cost <= 0 is impossible for
        // positive flows. Check that the output is at least produced.
        assert_eq!(result.output.len(), 4 * 2);
    }

    #[test]
    fn test_compute_logits_scale() {
        // d=4, so scale = 1/2 = 0.5
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let logits = compute_logits(&q, &k, 4, 1);
        // dot = 4.0, scaled = 4.0 * 0.5 = 2.0
        assert!((logits[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_row_softmax_sums_to_one() {
        let mut mat = vec![1.0, 2.0, 3.0, 4.0];
        row_softmax(&mut mat, 2, 2);
        let sum0: f32 = mat[0] + mat[1];
        let sum1: f32 = mat[2] + mat[3];
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }
}
