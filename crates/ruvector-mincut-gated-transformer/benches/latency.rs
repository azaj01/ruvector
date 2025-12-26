//! Latency benchmarks for mincut gated transformer.
//!
//! Tests inference latency across different tiers and configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvector_mincut_gated_transformer::{
    MincutGatedTransformer, TransformerConfig, GatePolicy, GatePacket, SpikePacket,
    InferInput, InferOutput, QuantizedWeights,
};

fn create_transformer(config: TransformerConfig) -> MincutGatedTransformer {
    let policy = GatePolicy::default();
    let weights = QuantizedWeights::empty(&config);
    MincutGatedTransformer::new(config, policy, weights).unwrap()
}

fn bench_tier0_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier0_inference");

    for seq_len in [16, 32, 64].iter() {
        let mut config = TransformerConfig::baseline();
        config.seq_len_max = *seq_len;
        let mut transformer = create_transformer(config.clone());

        let gate = GatePacket {
            lambda: 100,
            lambda_prev: 95,
            boundary_edges: 5,
            boundary_concentration_q15: 8192,
            partition_count: 3,
            flags: 0,
        };

        let tokens: Vec<u32> = (0..*seq_len as u32).collect();
        let input = InferInput::from_tokens(&tokens, gate);
        let mut logits = vec![0i32; config.logits as usize];

        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let mut output = InferOutput::new(&mut logits);
                    transformer.infer(black_box(&input), &mut output).unwrap();
                    black_box(&output.witness)
                })
            },
        );
    }

    group.finish();
}

fn bench_tier1_degraded(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_degraded");

    let config = TransformerConfig::baseline();
    let mut transformer = create_transformer(config.clone());

    // Gate packet that triggers tier 1 (ReduceScope)
    let gate = GatePacket {
        lambda: 100,
        lambda_prev: 95,
        boundary_edges: 30, // Above default max of 20
        boundary_concentration_q15: 8192,
        partition_count: 3,
        flags: 0,
    };

    let tokens: Vec<u32> = (0..64).collect();
    let input = InferInput::from_tokens(&tokens, gate);
    let mut logits = vec![0i32; config.logits as usize];

    group.bench_function("baseline_64_degraded", |b| {
        b.iter(|| {
            let mut output = InferOutput::new(&mut logits);
            transformer.infer(black_box(&input), &mut output).unwrap();
            black_box(&output.witness)
        })
    });

    group.finish();
}

fn bench_tier2_safe(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier2_safe");

    let config = TransformerConfig::baseline();
    let mut transformer = create_transformer(config.clone());

    // Gate packet that triggers tier 2 (FreezeWrites via force flag)
    let gate = GatePacket {
        lambda: 100,
        lambda_prev: 95,
        boundary_edges: 5,
        boundary_concentration_q15: 8192,
        partition_count: 3,
        flags: GatePacket::FLAG_FORCE_SAFE,
    };

    let tokens: Vec<u32> = (0..64).collect();
    let input = InferInput::from_tokens(&tokens, gate);
    let mut logits = vec![0i32; config.logits as usize];

    group.bench_function("baseline_64_safe", |b| {
        b.iter(|| {
            let mut output = InferOutput::new(&mut logits);
            transformer.infer(black_box(&input), &mut output).unwrap();
            black_box(&output.witness)
        })
    });

    group.finish();
}

fn bench_tier3_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier3_skip");

    let config = TransformerConfig::baseline();
    let mut transformer = create_transformer(config.clone());

    // Gate packet that triggers skip
    let gate = GatePacket {
        lambda: 100,
        flags: GatePacket::FLAG_SKIP,
        ..Default::default()
    };

    let tokens: Vec<u32> = (0..64).collect();
    let input = InferInput::from_tokens(&tokens, gate);
    let mut logits = vec![0i32; config.logits as usize];

    group.bench_function("baseline_64_skip", |b| {
        b.iter(|| {
            let mut output = InferOutput::new(&mut logits);
            transformer.infer(black_box(&input), &mut output).unwrap();
            black_box(&output.witness)
        })
    });

    group.finish();
}

fn bench_spike_inactive_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_inactive");

    let config = TransformerConfig::baseline();
    let mut transformer = create_transformer(config.clone());

    let gate = GatePacket {
        lambda: 100,
        lambda_prev: 95,
        ..Default::default()
    };

    let spike = SpikePacket {
        fired: 0, // Not fired - should skip
        ..Default::default()
    };

    let tokens: Vec<u32> = (0..64).collect();
    let input = InferInput::from_tokens(&tokens, gate).with_spikes(spike);
    let mut logits = vec![0i32; config.logits as usize];

    group.bench_function("baseline_64_spike_inactive", |b| {
        b.iter(|| {
            let mut output = InferOutput::new(&mut logits);
            transformer.infer(black_box(&input), &mut output).unwrap();
            black_box(&output.witness)
        })
    });

    group.finish();
}

fn bench_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_sweep");

    for window in [4, 8, 16, 32].iter() {
        let mut config = TransformerConfig::baseline();
        config.window_normal = *window;
        config.window_degraded = window / 2;
        let mut transformer = create_transformer(config.clone());

        let gate = GatePacket {
            lambda: 100,
            lambda_prev: 95,
            boundary_edges: 5,
            ..Default::default()
        };

        let tokens: Vec<u32> = (0..64).collect();
        let input = InferInput::from_tokens(&tokens, gate);
        let mut logits = vec![0i32; config.logits as usize];

        group.bench_with_input(
            BenchmarkId::from_parameter(window),
            window,
            |b, _| {
                b.iter(|| {
                    let mut output = InferOutput::new(&mut logits);
                    transformer.infer(black_box(&input), &mut output).unwrap();
                    black_box(&output.witness)
                })
            },
        );
    }

    group.finish();
}

fn bench_micro_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_config");

    let config = TransformerConfig::micro();
    let mut transformer = create_transformer(config.clone());

    let gate = GatePacket {
        lambda: 100,
        lambda_prev: 95,
        boundary_edges: 5,
        ..Default::default()
    };

    let tokens: Vec<u32> = (0..32).collect();
    let input = InferInput::from_tokens(&tokens, gate);
    let mut logits = vec![0i32; config.logits as usize];

    group.bench_function("micro_32", |b| {
        b.iter(|| {
            let mut output = InferOutput::new(&mut logits);
            transformer.infer(black_box(&input), &mut output).unwrap();
            black_box(&output.witness)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tier0_inference,
    bench_tier1_degraded,
    bench_tier2_safe,
    bench_tier3_skip,
    bench_spike_inactive_skip,
    bench_window_sizes,
    bench_micro_config,
);

criterion_main!(benches);
