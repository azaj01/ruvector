# ruvector-domain-expansion

[![Crates.io](https://img.shields.io/crates/v/ruvector-domain-expansion.svg)](https://crates.io/crates/ruvector-domain-expansion)
[![docs.rs](https://docs.rs/ruvector-domain-expansion/badge.svg)](https://docs.rs/ruvector-domain-expansion)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)

**Cross-domain transfer learning — train on one problem, get better at a different one automatically.**

```toml
ruvector-domain-expansion = "0.1"
```

Most AI systems learn one task at a time. If you train a model to write Rust code, it doesn't help it plan workflows. `ruvector-domain-expansion` changes that: knowledge learned in one domain (say, code synthesis) automatically transfers to other domains (planning, tool orchestration) — and it proves the transfer actually helped before committing it. This is how generalization works. Part of the [RuVector](https://github.com/ruvnet/ruvector) ecosystem.

| | ruvector-domain-expansion | Traditional Fine-Tuning |
|---|---|---|
| **Learning scope** | Learns across domains — code, planning, tool use | One task at a time |
| **Transfer** | Automatic: priors from Domain 1 seed Domain 2 | Manual: retrain from scratch per domain |
| **Verification** | Transfer only accepted if it helps target without hurting source | No verification — hope it works |
| **Strategy selection** | Thompson Sampling picks the best approach per context | Fixed strategy for all inputs |
| **Population search** | 8 policy variants evolve in parallel, best survives | Single model, single strategy |
| **Curiosity** | Explores under-visited areas automatically | Only learns from data you provide |

## Quick Start

```rust
use ruvector_domain_expansion::{
    DomainExpansionEngine, DomainId, ContextBucket, ArmId,
};

let mut engine = DomainExpansionEngine::new();
// Three domains registered automatically: rust_synthesis, structured_planning, tool_orchestration

// Generate training tasks
let domain = DomainId("rust_synthesis".into());
let tasks = engine.generate_tasks(&domain, 10, 0.5); // 10 tasks, medium difficulty

// Select strategy using Thompson Sampling
let bucket = ContextBucket { difficulty_tier: "medium".into(), category: "algorithm".into() };
let arm = engine.select_arm(&domain, &bucket).unwrap();

// Evaluate and learn
let eval = engine.evaluate_and_record(&domain, &tasks[0], &solution, bucket, arm);

// Transfer knowledge to a new domain
let target = DomainId("structured_planning".into());
engine.initiate_transfer(&domain, &target);
```

## Key Features

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Meta Thompson Sampling** | Picks the best strategy per context using uncertainty-aware selection | Explores when unsure, exploits when confident — no manual tuning |
| **Cross-Domain Transfer** | Extracts compact priors from one domain, seeds another | New domains learn faster by starting with knowledge from related domains |
| **Transfer Verification** | Accepts a transfer only if target improves without source regressing | Guarantees generalization — no silent regressions |
| **Population-Based Search** | Evolves 8 policy kernel variants in parallel | Finds optimal strategies faster than single-model training |
| **Curiosity-Driven Exploration** | UCB-style bonus for under-visited contexts | Automatically explores blind spots instead of getting stuck |
| **Pareto Front Tracking** | Tracks non-dominated kernels across accuracy, cost, and robustness | See the best tradeoffs, not just the single "best" model |
| **Plateau Detection** | Detects when learning stalls and recommends actions | Automatically switches strategies instead of wasting compute |
| **Counterexample Tracking** | Records failed solutions to inform future decisions | Learns from mistakes, not just successes |
| **Cost Curve & Scoreboard** | Tracks convergence speed per domain with acceleration metrics | Proves that transfer actually accelerated learning |
| **RVF Integration** | Package trained models as cognitive containers (optional `rvf` feature) | Ship a trained domain expansion engine as a single `.rvf` file |

## Three Built-In Domains

| Domain | What It Generates | What It Evaluates |
|--------|------------------|-------------------|
| **Rust Synthesis** | Rust function specifications (transforms, filters, searches) | Correctness, efficiency, idiomatic style |
| **Structured Planning** | Multi-step plans with dependencies and resource constraints | Feasibility, completeness, dependency ordering |
| **Tool Orchestration** | Tool coordination tasks (parallel execution, error handling) | Correct sequencing, parallelism, failure recovery |

All three domains share a common embedding space, enabling cross-domain similarity and transfer.

## How Transfer Works

```
Domain 1 (Rust Synthesis)          Domain 2 (Planning)
┌─────────────────────┐            ┌─────────────────────┐
│ Train on 100 tasks   │            │ Start from scratch   │
│ Extract posteriors   │───prior──▶│ Seed with priors     │
│ Score: 0.85          │            │ Score after 45 runs: │
│                      │            │   0.70 (vs 0.30      │
│                      │            │   without transfer)  │
└─────────────────────┘            └─────────────────────┘
                                           │
                                    Verification Gate:
                                    ✓ Target improved
                                    ✓ Source didn't regress
                                    ✓ Acceleration > 1.0
                                    → Transfer PROMOTED
```

## Feature Flags

| Flag | Default | What It Enables |
|------|---------|-----------------|
| `rvf` | No | RVF cognitive container integration — serialize engines to `.rvf` format |

```toml
[dependencies]
ruvector-domain-expansion = { version = "0.1", features = ["rvf"] }
```

## API Overview

### Core Types

| Type | Description |
|------|-------------|
| `DomainExpansionEngine` | Main orchestrator — manages domains, transfer, population search |
| `Domain` (trait) | Implement to add custom domains — generate tasks, evaluate, embed |
| `DomainId` | Unique identifier for a domain |
| `Task` | A problem instance with difficulty, constraints, and spec |
| `Solution` | A candidate answer with content and structured data |
| `Evaluation` | Score (0.0–1.0) with correctness, efficiency, and elegance breakdown |

### Transfer & Strategy

| Type | Description |
|------|-------------|
| `MetaThompsonEngine` | Thompson Sampling with Beta priors across context buckets |
| `TransferPrior` | Compact posterior summary extracted from a trained domain |
| `TransferVerification` | Result of verifying a transfer — promotable only if both domains benefit |
| `PolicyKernel` | A strategy configuration with tunable knobs |
| `PopulationSearch` | Evolutionary search across policy kernel variants |

### Meta-Learning

| Type | Description |
|------|-------------|
| `MetaLearningEngine` | Regret tracking, plateau detection, Pareto front, curiosity bonuses |
| `CostCurve` | Convergence trajectory per domain |
| `AccelerationScoreboard` | Measures how much faster transfer makes learning |
| `ParetoFront` | Non-dominated set of kernels across accuracy/cost/robustness |

## Related Crates

- **[sona](../sona/README.md)** — Self-optimizing neural architecture (LoRA + EWC++)
- **[ruvector-gnn](../ruvector-gnn/README.md)** — GNN layer for self-learning search
- **[ruvector-graph-transformer](../ruvector-graph-transformer/README.md)** — 8 verified graph transformer modules
- **[rvf](../rvf/README.md)** — Cognitive container format

## License

**MIT License** — see [LICENSE](../../LICENSE) for details.

---

Part of [RuVector](https://github.com/ruvnet/ruvector) — the self-learning vector database.
