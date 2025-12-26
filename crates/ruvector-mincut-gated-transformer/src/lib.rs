//! # Mincut Gated Transformer
//!
//! Ultra low latency transformer inference designed for continuous systems.
//! Governed by a coherence controller driven by dynamic minimum cut signals
//! and optionally a spiking scheduler that skips work when nothing meaningful
//! is happening.
//!
//! ## Primary Outcomes
//!
//! 1. **Deterministic, bounded inference** - Same inputs yield same outputs
//! 2. **Allocation-free hot path** - Zero heap allocations after initialization
//! 3. **Predictable tail latency** - Bounded p99 latency guarantees
//! 4. **Explainable interventions** - Every gate decision produces a witness
//! 5. **Easy integration** - Works with RuVector, ruvector-mincut, and agent orchestration
//!
//! ## Core Concepts
//!
//! The system has three roles:
//!
//! 1. **Transformer Kernel** - Produces logits or scores under fixed compute budgets
//! 2. **Spike Scheduler** (optional) - Decides whether to run and selects compute tier
//! 3. **Mincut Gate** (authoritative) - Decides what state changes are allowed
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvector_mincut_gated_transformer::{
//!     MincutGatedTransformer, TransformerConfig, GatePolicy,
//!     GatePacket, InferInput, InferOutput,
//! };
//!
//! // Create configuration
//! let config = TransformerConfig::micro();
//! let policy = GatePolicy::default();
//!
//! // Load weights (pseudo-code)
//! # let weights = ruvector_mincut_gated_transformer::QuantizedWeights::empty(&config);
//!
//! // Create transformer
//! let mut transformer = MincutGatedTransformer::new(config, policy, weights).unwrap();
//!
//! // Create gate packet from mincut signals
//! let gate = GatePacket {
//!     lambda: 100,
//!     lambda_prev: 95,
//!     boundary_edges: 5,
//!     boundary_concentration_q15: 8192,
//!     partition_count: 3,
//!     flags: 0,
//! };
//!
//! // Prepare input
//! let input = InferInput {
//!     tokens: Some(&[1, 2, 3, 4]),
//!     embedding_q: None,
//!     embedding_scale: 1.0,
//!     input_signature: None,
//!     gate,
//!     spikes: None,
//! };
//!
//! // Allocate output buffer
//! let mut logits = vec![0i32; 1024];
//! let mut output = InferOutput::new(&mut logits);
//!
//! // Run inference
//! transformer.infer(&input, &mut output).unwrap();
//!
//! // Check witness for allowed actions
//! if output.witness.external_writes_enabled == 1 {
//!     // Safe to persist memory
//! }
//! ```

#![cfg_attr(feature = "no_std_gateway", no_std)]

#[cfg(feature = "no_std_gateway")]
extern crate alloc;

pub mod config;
pub mod error;
pub mod packets;
pub mod state;
pub mod model;
pub mod gate;
pub mod spike;
pub mod kernel;
pub mod attention;
pub mod ffn;

#[cfg(feature = "trace")]
pub mod trace;

// Re-exports for convenient access
pub use config::{TransformerConfig, GatePolicy};
pub use error::{Error, Result};
pub use packets::{
    GatePacket, SpikePacket, GateDecision, GateReason, Witness, InferInput, InferOutput, InferStats,
};
pub use state::RuntimeState;
pub use model::{MincutGatedTransformer, QuantizedWeights, WeightsLoader};
pub use gate::{GateController, TierDecision};
pub use spike::SpikeScheduler;

#[cfg(feature = "trace")]
pub use trace::{TraceState, TraceSnapshot, TraceCounters};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        MincutGatedTransformer, TransformerConfig, GatePolicy,
        GatePacket, SpikePacket, GateDecision, GateReason, Witness,
        InferInput, InferOutput, InferStats,
        QuantizedWeights, WeightsLoader,
        Error, Result,
    };

    #[cfg(feature = "trace")]
    pub use crate::{TraceSnapshot, TraceCounters};
}

/// Supported model configurations
pub mod configs {
    use super::TransformerConfig;

    /// Baseline CPU configuration
    /// - Sequence length: 64
    /// - Hidden size: 256
    /// - Heads: 4
    /// - Layers: 4
    pub fn baseline() -> TransformerConfig {
        TransformerConfig::baseline()
    }

    /// Micro configuration for WASM and edge gateways
    /// - Sequence length: 32
    /// - Hidden size: 128
    /// - Heads: 4
    /// - Layers: 2
    pub fn micro() -> TransformerConfig {
        TransformerConfig::micro()
    }
}
