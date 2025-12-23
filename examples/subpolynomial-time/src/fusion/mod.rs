//! Vector-Graph Fusion Module
//!
//! Unified retrieval substrate combining vector similarity and graph relations
//! with minimum-cut brittleness detection for robust knowledge retrieval.

mod fusion_graph;
mod structural_monitor;
mod optimizer;

pub use fusion_graph::{
    FusionGraph, FusionNode, FusionEdge, FusionConfig,
    EdgeOrigin, RelationType, FusionResult,
};
pub use structural_monitor::{
    StructuralMonitor, MonitorState, BrittlenessSignal,
    Trigger, TriggerType, MonitorConfig as StructuralMonitorConfig,
};
pub use optimizer::{
    Optimizer, OptimizerAction, MaintenancePlan, MaintenanceTask,
    OptimizationResult, LearningGate,
};
