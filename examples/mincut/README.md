# Exotic MinCut Examples

Advanced examples demonstrating cutting-edge applications of dynamic minimum cut algorithms combined with temporal intelligence, self-organizing systems, and neural optimization.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXOTIC MINCUT APPLICATIONS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Temporal      │    │   Strange       │    │   Causal        │        │
│   │   Attractors    │    │   Loop          │    │   Discovery     │        │
│   │                 │    │                 │    │                 │        │
│   │  Networks that  │    │  Self-aware     │    │  Find cause &   │        │
│   │  evolve toward  │    │  swarms that    │    │  effect in      │        │
│   │  stable states  │    │  reorganize     │    │  dynamic graphs │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Time          │    │   Morpho-       │    │   Neural        │        │
│   │   Crystal       │    │   genetic       │    │   Optimizer     │        │
│   │                 │    │                 │    │                 │        │
│   │  Periodic       │    │  Bio-inspired   │    │  Learn optimal  │        │
│   │  coordination   │    │  network        │    │  graph configs  │        │
│   │  patterns       │    │  growth         │    │  over time      │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Examples

### 1. Temporal Attractors (`temporal_attractors/`)

Networks that naturally evolve toward stable "attractor" states.

```
Time →
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│Chaos│ ──► │Weak │ ──► │Strong│ ──► │Stable│
│mc=1 │     │mc=2 │     │mc=4  │     │mc=6  │
└─────┘     └─────┘     └─────┘     └─────┘
                                    ATTRACTOR
```

**Key Concepts:**
- Optimal attractor: Network strengthens over time
- Fragmented attractor: Network splits into clusters
- Oscillating attractor: Periodic connectivity patterns

**Run:** `cargo run --example temporal_attractors`

---

### 2. Strange Loop Swarms (`strange_loop/`)

Self-aware swarms that observe and modify themselves based on their own structure.

```
┌──────────────────────────────────────────┐
│              STRANGE LOOP                │
│                                          │
│   Observe ──► Model ──► Decide ──► Act   │
│      ▲                              │    │
│      └──────────────────────────────┘    │
│                                          │
│   "I see I am weak, so I strengthen"    │
└──────────────────────────────────────────┘
```

**Key Concepts:**
- Self-reference: System analyzes itself
- Feedback loop: Actions change what is observed
- Emergent intelligence: Simple rules → complex behavior

**Run:** `cargo run --example strange_loop`

---

### 3. Causal Discovery (`causal_discovery/`)

Discover cause-and-effect relationships in dynamic networks.

```
Event A          Event B          Event C
(edge cut)       (mincut drops)   (partition)
    │                 │                │
    ├────200ms────────┤                │
    │                 ├────500ms───────┤
    │                                  │
    └──────────700ms───────────────────┘

Discovered: A causes B causes C
```

**Key Concepts:**
- Granger causality: Predict B from A
- Temporal windows: Detect patterns within time bounds
- Latency analysis: Measure cause-effect delays

**Run:** `cargo run --example causal_discovery`

---

### 4. Time Crystal Coordination (`time_crystal/`)

Periodic, self-sustaining coordination patterns inspired by physics.

```
Phase 1       Phase 2       Phase 3       Phase 1...
  Ring          Star          Mesh          Ring
  ●─●           ●             ●─●─●         ●─●
  │ │          /│\            │╲│╱│         │ │
  ●─●         ● ● ●           ●─●─●         ●─●
 mc=2         mc=1            mc=6         mc=2

    └─────────────── REPEATS FOREVER ───────────────┘
```

**Key Concepts:**
- Phase transitions: Topology changes periodically
- Stability verification: Check expected vs actual mincut
- Self-sustaining: Pattern continues without external input

**Run:** `cargo run --example time_crystal`

---

### 5. Morphogenetic Networks (`morphogenetic/`)

Networks that grow like biological organisms.

```
Seed        Sprout       Branch       Mature
  ●      →   ●─●    →    ●─●─●   →   ●─●─●
                         │   │       │ │ │
                         ●   ●       ●─●─●
                                     │   │
                                     ●───●
```

**Key Concepts:**
- Growth signals: Diffuse like chemical gradients
- Local rules: IF weak THEN grow, IF crowded THEN branch
- Maturity: Network reaches stable adult form

**Run:** `cargo run --example morphogenetic`

---

### 6. Neural Graph Optimizer (`neural_optimizer/`)

Learn to predict and optimize graph configurations.

```
┌─────────────────────────────────────────────┐
│         NEURAL OPTIMIZATION LOOP            │
│                                             │
│   ┌─────────┐    ┌─────────┐    ┌────────┐ │
│   │ Observe │───►│ Predict │───►│  Act   │ │
│   │  Graph  │    │ MinCut  │    │ Modify │ │
│   └─────────┘    └─────────┘    └────────┘ │
│        ▲                             │      │
│        └─────────── Learn ───────────┘      │
└─────────────────────────────────────────────┘
```

**Key Concepts:**
- Feature extraction: Convert graph to vectors
- Policy network: Choose optimal actions
- Reinforcement learning: Improve through experience

**Run:** `cargo run --example neural_optimizer`

---

## Benchmarks

Run the comprehensive benchmark suite:

```bash
cargo run --release --example benchmarks
```

**Benchmark Categories:**
- Temporal evolution performance
- Self-observation overhead
- Causality detection speed
- Phase transition timing
- Growth cycle efficiency
- Neural inference latency
- Scaling analysis (100 → 10,000 nodes)

---

## Performance Characteristics

| Example | Nodes | Edges | Update Time | Memory |
|---------|-------|-------|-------------|--------|
| Temporal Attractors | 1,000 | 2,000 | ~50 μs | ~1 MB |
| Strange Loop | 500 | 1,500 | ~100 μs | ~500 KB |
| Causal Discovery | 1,000 events | - | ~10 μs/event | ~100 KB |
| Time Crystal | 100 | 300 | ~20 μs/phase | ~200 KB |
| Morphogenetic | 10→100 | 20→200 | ~200 μs/cycle | ~500 KB |
| Neural Optimizer | 500 | 1,000 | ~1 ms/step | ~2 MB |

---

## Use Cases

### Swarm Robotics
- **Temporal Attractors**: Swarm converges to optimal formation
- **Strange Loop**: Self-healing swarm topology
- **Time Crystal**: Periodic task scheduling

### Distributed Systems
- **Causal Discovery**: Debug cascading failures
- **Morphogenetic**: Auto-scaling infrastructure
- **Neural Optimizer**: Learned load balancing

### AI/ML Training
- **Strange Loop**: Self-improving agents
- **Neural Optimizer**: Hyperparameter optimization
- **Causal Discovery**: Feature importance

---

## Dependencies

```toml
[dependencies]
ruvector-mincut = { version = "0.2", features = ["monitoring", "approximate"] }
```

Optional for streaming integration:
```toml
midstreamer-quic = "0.1"
nanosecond-scheduler = "0.1"
temporal-attractor-studio = "0.1"
```

---

## Further Reading

- **Temporal Attractors**: [Dynamical Systems Theory](https://en.wikipedia.org/wiki/Attractor)
- **Strange Loops**: [Hofstadter, "Gödel, Escher, Bach"](https://en.wikipedia.org/wiki/Strange_loop)
- **Causal Discovery**: [Granger Causality](https://en.wikipedia.org/wiki/Granger_causality)
- **Time Crystals**: [Wilczek, 2012](https://en.wikipedia.org/wiki/Time_crystal)
- **Morphogenesis**: [Turing Patterns](https://en.wikipedia.org/wiki/Turing_pattern)
- **Neural Optimization**: [Neural Combinatorial Optimization](https://arxiv.org/abs/1611.09940)

---

## License

MIT OR Apache-2.0

---

<div align="center">

**Built with RuVector MinCut + Midstream**

[ruv.io](https://ruv.io) | [GitHub](https://github.com/ruvnet/ruvector)

</div>
