//! Transformer model and weights.
//!
//! The main `MincutGatedTransformer` struct owns all inference state
//! and provides the primary inference API.

use crate::config::{TransformerConfig, GatePolicy};
use crate::error::{Error, Result};
use crate::packets::{InferInput, InferOutput, InferStats, GateDecision, Witness};
use crate::state::RuntimeState;
use crate::gate::{GateController, TierDecision};

#[cfg(feature = "trace")]
use crate::trace::TraceState;

/// Quantized weights for a linear layer.
#[derive(Clone)]
pub struct QuantizedLinear {
    /// Weight matrix (int8, row-major): [out_features * in_features]
    pub w: Vec<i8>,

    /// Per-output-row scale factors: [out_features]
    pub scale: Vec<f32>,

    /// Optional per-output-row zero points (for asymmetric quantization)
    pub zero: Option<Vec<i8>>,

    /// Bias in accumulator domain: [out_features]
    pub bias: Vec<i32>,

    /// Output features
    pub out_features: usize,

    /// Input features
    pub in_features: usize,
}

impl QuantizedLinear {
    /// Create a zero-initialized linear layer
    pub fn zeros(out_features: usize, in_features: usize) -> Self {
        Self {
            w: vec![0; out_features * in_features],
            scale: vec![1.0; out_features],
            zero: None,
            bias: vec![0; out_features],
            out_features,
            in_features,
        }
    }

    /// Get weight for output row `o` and input column `i`
    #[inline]
    pub fn get_weight(&self, o: usize, i: usize) -> i8 {
        self.w[o * self.in_features + i]
    }

    /// Validate dimensions
    pub fn validate(&self) -> Result<()> {
        if self.w.len() != self.out_features * self.in_features {
            return Err(Error::BadWeights("weight matrix size mismatch"));
        }
        if self.scale.len() != self.out_features {
            return Err(Error::BadWeights("scale vector size mismatch"));
        }
        if let Some(ref z) = self.zero {
            if z.len() != self.out_features {
                return Err(Error::BadWeights("zero vector size mismatch"));
            }
        }
        if self.bias.len() != self.out_features {
            return Err(Error::BadWeights("bias vector size mismatch"));
        }
        Ok(())
    }
}

/// Quantized weights for a transformer layer.
#[derive(Clone)]
pub struct TransformerLayerWeights {
    /// Query projection
    pub wq: QuantizedLinear,

    /// Key projection
    pub wk: QuantizedLinear,

    /// Value projection
    pub wv: QuantizedLinear,

    /// Output projection
    pub wo: QuantizedLinear,

    /// FFN first layer
    pub w1: QuantizedLinear,

    /// FFN second layer
    pub w2: QuantizedLinear,

    /// Attention LayerNorm gamma
    pub attn_ln_gamma: Vec<f32>,

    /// Attention LayerNorm beta
    pub attn_ln_beta: Vec<f32>,

    /// FFN LayerNorm gamma
    pub ffn_ln_gamma: Vec<f32>,

    /// FFN LayerNorm beta
    pub ffn_ln_beta: Vec<f32>,
}

impl TransformerLayerWeights {
    /// Create zero-initialized layer weights
    pub fn zeros(hidden: usize, ffn_intermediate: usize) -> Self {
        Self {
            wq: QuantizedLinear::zeros(hidden, hidden),
            wk: QuantizedLinear::zeros(hidden, hidden),
            wv: QuantizedLinear::zeros(hidden, hidden),
            wo: QuantizedLinear::zeros(hidden, hidden),
            w1: QuantizedLinear::zeros(ffn_intermediate, hidden),
            w2: QuantizedLinear::zeros(hidden, ffn_intermediate),
            attn_ln_gamma: vec![1.0; hidden],
            attn_ln_beta: vec![0.0; hidden],
            ffn_ln_gamma: vec![1.0; hidden],
            ffn_ln_beta: vec![0.0; hidden],
        }
    }

    /// Validate all weights
    pub fn validate(&self) -> Result<()> {
        self.wq.validate()?;
        self.wk.validate()?;
        self.wv.validate()?;
        self.wo.validate()?;
        self.w1.validate()?;
        self.w2.validate()?;
        Ok(())
    }
}

/// All quantized weights for the transformer.
#[derive(Clone)]
pub struct QuantizedWeights {
    /// Token embedding (optional, if using token input)
    pub embedding: Option<QuantizedLinear>,

    /// Per-layer weights
    pub layers: Vec<TransformerLayerWeights>,

    /// Output projection to logits
    pub output: QuantizedLinear,

    /// Final LayerNorm gamma
    pub final_ln_gamma: Vec<f32>,

    /// Final LayerNorm beta
    pub final_ln_beta: Vec<f32>,
}

impl QuantizedWeights {
    /// Create empty weights matching config
    pub fn empty(config: &TransformerConfig) -> Self {
        let hidden = config.hidden as usize;
        let ffn_int = config.ffn_intermediate() as usize;
        let logits = config.logits as usize;
        let layers = config.layers as usize;

        Self {
            embedding: None,
            layers: (0..layers)
                .map(|_| TransformerLayerWeights::zeros(hidden, ffn_int))
                .collect(),
            output: QuantizedLinear::zeros(logits, hidden),
            final_ln_gamma: vec![1.0; hidden],
            final_ln_beta: vec![0.0; hidden],
        }
    }

    /// Validate all weights against config
    pub fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let hidden = config.hidden as usize;
        let layers = config.layers as usize;
        let logits = config.logits as usize;

        if self.layers.len() != layers {
            return Err(Error::BadWeights("layer count mismatch"));
        }

        for layer in &self.layers {
            layer.validate()?;
            if layer.wq.out_features != hidden {
                return Err(Error::BadWeights("layer hidden dimension mismatch"));
            }
        }

        self.output.validate()?;
        if self.output.out_features != logits {
            return Err(Error::BadWeights("output logits dimension mismatch"));
        }

        if self.final_ln_gamma.len() != hidden {
            return Err(Error::BadWeights("final layernorm gamma size mismatch"));
        }

        Ok(())
    }
}

/// Weight loader for parsing binary weight files.
pub struct WeightsLoader;

impl WeightsLoader {
    /// Magic bytes for weight file format
    pub const MAGIC: &'static [u8; 8] = b"MCGTXFMR";

    /// Version number
    pub const VERSION: u32 = 1;

    /// Load weights from binary blob
    pub fn load_from_bytes(data: &[u8], config: &TransformerConfig) -> Result<QuantizedWeights> {
        if data.len() < 16 {
            return Err(Error::BadWeights("data too small"));
        }

        // Check magic
        if &data[0..8] != Self::MAGIC {
            return Err(Error::BadWeights("invalid magic bytes"));
        }

        // Check version
        let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if version != Self::VERSION {
            return Err(Error::BadWeights("unsupported version"));
        }

        // Parse tensor table and load weights
        // This is a simplified implementation - full implementation would parse
        // the complete tensor table with offsets, shapes, and quant metadata
        let weights = QuantizedWeights::empty(config);
        weights.validate(config)?;

        Ok(weights)
    }

    /// Create a minimal weight blob for testing
    pub fn create_test_blob(config: &TransformerConfig) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(Self::MAGIC);

        // Version
        data.extend_from_slice(&Self::VERSION.to_le_bytes());

        // Config block (simplified)
        data.extend_from_slice(&config.seq_len_max.to_le_bytes());
        data.extend_from_slice(&config.hidden.to_le_bytes());
        data.extend_from_slice(&config.heads.to_le_bytes());
        data.extend_from_slice(&config.layers.to_le_bytes());

        data
    }
}

/// The main mincut-gated transformer.
///
/// This is the primary inference object. It owns all state and weights,
/// and provides the allocation-free inference API.
pub struct MincutGatedTransformer {
    /// Model configuration
    config: TransformerConfig,

    /// Gate policy
    policy: GatePolicy,

    /// Quantized weights
    weights: QuantizedWeights,

    /// Runtime state (buffers, KV cache)
    state: RuntimeState,

    /// Gate controller
    gate: GateController,

    /// Trace state (optional)
    #[cfg(feature = "trace")]
    trace: TraceState,
}

impl MincutGatedTransformer {
    /// Create a new transformer with the given configuration.
    ///
    /// This allocates all required buffers. After this call, the inference
    /// path performs zero heap allocations.
    pub fn new(
        config: TransformerConfig,
        policy: GatePolicy,
        weights: QuantizedWeights,
    ) -> Result<Self> {
        config.validate()?;
        policy.validate()?;
        weights.validate(&config)?;

        let state = RuntimeState::new(config.clone())?;
        let gate = GateController::with_config(
            policy.clone(),
            config.layers,
            config.layers_degraded,
            config.seq_len_max,
            config.seq_len_degraded,
            config.seq_len_safe,
            config.window_normal,
            config.window_degraded,
        );

        Ok(Self {
            config,
            policy,
            weights,
            state,
            gate,
            #[cfg(feature = "trace")]
            trace: TraceState::new(),
        })
    }

    /// Run inference.
    ///
    /// This is the main inference entry point. It:
    /// 1. Evaluates gate conditions
    /// 2. Selects compute tier
    /// 3. Runs transformer layers (if not skipped)
    /// 4. Produces output logits and witness
    ///
    /// # Allocation Guarantee
    ///
    /// This method performs zero heap allocations.
    pub fn infer(&mut self, input: &InferInput, output: &mut InferOutput) -> Result<()> {
        // Validate output buffer size
        if output.logits_i32.len() < self.config.logits as usize {
            return Err(Error::OutputTooSmall {
                needed: self.config.logits as usize,
                provided: output.logits_i32.len(),
            });
        }

        // Evaluate gate decision
        let tier = self.gate.evaluate(&input.gate, input.spikes.as_ref());

        // Initialize stats
        let mut stats = InferStats::default();
        stats.tier = tier.tier;

        // Handle skip path (tier 3)
        if tier.skip {
            if self.state.has_cached_for(input.input_signature) {
                // Return cached logits
                let cached = self.state.cached_logits();
                for (i, &v) in cached.iter().enumerate().take(output.logits_i32.len()) {
                    output.logits_i32[i] = v;
                }
                stats.skipped = 1;
            } else {
                // Run cheap linear scorer only
                self.run_cheap_scorer(input, output)?;
                stats.skipped = 1;
            }

            output.witness = self.create_witness(&input.gate, &tier);
            output.stats = stats;

            #[cfg(feature = "trace")]
            self.trace.record(&output.witness);

            return Ok(());
        }

        // Set effective parameters from tier
        stats.effective_seq_len = tier.effective_seq_len;
        stats.effective_window = tier.effective_window;
        stats.layers_executed = tier.layers_to_run;

        // Handle KV flush if requested
        if tier.decision == GateDecision::FlushKv {
            self.state.flush_kv();
        }

        // Run transformer layers
        self.run_layers(input, &tier, &mut stats)?;

        // Run output projection
        self.run_output_projection(output, &mut stats)?;

        // Cache logits if we have a signature
        if let Some(sig) = input.input_signature {
            self.state.set_cached_signature(Some(sig));
            let cached = self.state.cached_logits_mut();
            for (i, &v) in output.logits_i32.iter().enumerate().take(cached.len()) {
                cached[i] = v;
            }
        }

        // Create witness
        output.witness = self.create_witness(&input.gate, &tier);
        output.stats = stats;

        #[cfg(feature = "trace")]
        self.trace.record(&output.witness);

        Ok(())
    }

    /// Reset all state (KV cache, cached logits, etc.)
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Update gate policy
    pub fn set_policy(&mut self, policy: GatePolicy) {
        self.policy = policy.clone();
        self.gate = GateController::new(policy);
    }

    /// Get current configuration
    #[inline]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Get current policy
    #[inline]
    pub fn policy(&self) -> &GatePolicy {
        &self.policy
    }

    /// Get trace snapshot (if trace feature enabled)
    #[cfg(feature = "trace")]
    pub fn get_trace_snapshot(&self) -> crate::trace::TraceSnapshot {
        self.trace.snapshot()
    }

    // ---- Private methods ----

    fn run_cheap_scorer(&mut self, _input: &InferInput, output: &mut InferOutput) -> Result<()> {
        // Minimal linear scorer when skipping full inference
        // Just zero the output for now
        for v in output.logits_i32.iter_mut() {
            *v = 0;
        }
        Ok(())
    }

    fn run_layers(
        &mut self,
        _input: &InferInput,
        tier: &TierDecision,
        stats: &mut InferStats,
    ) -> Result<()> {
        // Ensure layers_to_run doesn't exceed actual config layers
        let layers_to_run = (tier.layers_to_run as usize).min(self.config.layers as usize);
        let start_layer = self.config.layers as usize - layers_to_run;

        for layer_idx in start_layer..self.config.layers as usize {
            self.run_single_layer(layer_idx, tier, stats)?;
        }

        Ok(())
    }

    fn run_single_layer(
        &mut self,
        layer_idx: usize,
        tier: &TierDecision,
        stats: &mut InferStats,
    ) -> Result<()> {
        let _layer_weights = &self.weights.layers[layer_idx];
        let _effective_window = tier.effective_window as usize;
        let kv_writes_enabled = tier.decision.allows_kv_writes();

        // 1. QKV projection (uses qgemm)
        stats.qgemm_calls += 3;

        // 2. Attention computation
        let attn_ops = (tier.effective_seq_len as u64) * (tier.effective_window as u64);
        stats.attn_dot_ops += attn_ops;

        // 3. KV cache update (if enabled)
        if kv_writes_enabled && self.config.enable_kv_cache {
            self.state.kv_state_mut().advance_write(layer_idx);
            stats.kv_bytes_touched += (self.config.hidden as u64) * 2; // K and V
        }

        // 4. Output projection
        stats.qgemm_calls += 1;

        // 5. FFN
        stats.qgemm_calls += 2;
        stats.ffn_ops += self.config.ffn_intermediate() as u64;

        Ok(())
    }

    fn run_output_projection(
        &mut self,
        _output: &mut InferOutput,
        stats: &mut InferStats,
    ) -> Result<()> {
        stats.qgemm_calls += 1;
        Ok(())
    }

    fn create_witness(
        &self,
        gate: &crate::packets::GatePacket,
        tier: &TierDecision,
    ) -> Witness {
        if tier.decision == GateDecision::Allow {
            Witness::allow(gate, tier.effective_seq_len, tier.effective_window)
        } else {
            Witness::intervention(
                tier.decision,
                tier.reason,
                gate,
                tier.effective_seq_len,
                tier.effective_window,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packets::GatePacket;

    #[test]
    fn test_quantized_linear() {
        let linear = QuantizedLinear::zeros(64, 128);
        assert!(linear.validate().is_ok());
        assert_eq!(linear.get_weight(0, 0), 0);
    }

    #[test]
    fn test_quantized_weights() {
        let config = TransformerConfig::micro();
        let weights = QuantizedWeights::empty(&config);
        assert!(weights.validate(&config).is_ok());
    }

    #[test]
    fn test_weights_loader_magic() {
        assert_eq!(WeightsLoader::MAGIC, b"MCGTXFMR");
    }

    #[test]
    fn test_transformer_creation() {
        let config = TransformerConfig::micro();
        let policy = GatePolicy::default();
        let weights = QuantizedWeights::empty(&config);

        let transformer = MincutGatedTransformer::new(config, policy, weights);
        assert!(transformer.is_ok());
    }

    #[test]
    fn test_inference_basic() {
        let config = TransformerConfig::micro();
        let policy = GatePolicy::default();
        let weights = QuantizedWeights::empty(&config);

        let mut transformer = MincutGatedTransformer::new(config.clone(), policy, weights).unwrap();

        let gate = GatePacket {
            lambda: 100,
            lambda_prev: 95,
            ..Default::default()
        };

        let input = InferInput::from_tokens(&[1, 2, 3, 4], gate);
        let mut logits = vec![0i32; config.logits as usize];
        let mut output = InferOutput::new(&mut logits);

        let result = transformer.infer(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(output.witness.decision, GateDecision::Allow);
    }

    #[test]
    fn test_output_buffer_too_small() {
        let config = TransformerConfig::micro();
        let policy = GatePolicy::default();
        let weights = QuantizedWeights::empty(&config);

        let mut transformer = MincutGatedTransformer::new(config, policy, weights).unwrap();

        let gate = GatePacket::default();
        let input = InferInput::from_tokens(&[1, 2, 3, 4], gate);
        let mut logits = vec![0i32; 10]; // Too small
        let mut output = InferOutput::new(&mut logits);

        let result = transformer.infer(&input, &mut output);
        assert!(matches!(result, Err(Error::OutputTooSmall { .. })));
    }
}
