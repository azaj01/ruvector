//! Attention mechanisms for the transformer.
//!
//! Provides sliding window attention as the default, with optional
//! linear attention for longer sequences.

pub mod window;

#[cfg(feature = "linear_attention")]
pub mod linear;

pub use window::{SlidingWindowAttention, WindowAttentionConfig};
