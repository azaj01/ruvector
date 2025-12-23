//! WASM bindings and optimizations for agentic chip
//!
//! Provides:
//! - SIMD-accelerated boundary computation
//! - Agentic chip interface
//! - Inter-core messaging

pub mod simd;
pub mod agentic;

pub use simd::*;
pub use agentic::*;
