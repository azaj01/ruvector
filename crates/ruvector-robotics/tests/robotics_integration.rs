//! Integration tests for the ruvector-robotics subsystem.
//!
//! Tests the full integration across bridge types, perception pipeline,
//! cognitive loop, MCP tool registration, and swarm coordination.
//!
//! Run with: cargo test --test robotics_integration -p ruvector-robotics

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ruvector_robotics::bridge::{
    Obstacle, Point3D, PointCloud, RobotState, SceneEdge, SceneGraph, SceneObject, SpatialIndex,
    Trajectory,
};
use ruvector_robotics::cognitive::behavior_tree::{
    BehaviorNode, BehaviorStatus, BehaviorTree, DecoratorType,
};
// Perception config types are available via ruvector_robotics::perception::{ObstacleConfig, PerceptionConfig, SceneGraphConfig}
// but not used directly in these integration tests.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random f32 in [0, 1).
fn pseudo_random_f32(seed: u64, index: usize) -> f32 {
    let h = seed
        .wrapping_mul(index as u64 + 1)
        .wrapping_mul(0x5DEECE66D)
        .wrapping_add(0xB);
    ((h % 10000) as f32) / 10000.0
}

fn generate_point_cloud_around(center: Point3D, n: usize, spread: f32) -> Vec<Point3D> {
    (0..n)
        .map(|i| {
            Point3D::new(
                center.x + (pseudo_random_f32(42, i * 3) - 0.5) * spread,
                center.y + (pseudo_random_f32(42, i * 3 + 1) - 0.5) * spread,
                center.z + (pseudo_random_f32(42, i * 3 + 2) - 0.5) * spread,
            )
        })
        .collect()
}

fn cluster_point_cloud(
    points: &[Point3D],
    cell_size: f32,
    min_cluster_size: usize,
) -> Vec<Obstacle> {
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        let key = (
            (p.x / cell_size) as i32,
            (p.y / cell_size) as i32,
            (p.z / cell_size) as i32,
        );
        grid.entry(key).or_default().push(i);
    }

    let mut obstacles = Vec::new();
    let mut id = 0u64;
    for indices in grid.values() {
        if indices.len() >= min_cluster_size {
            let cx =
                indices.iter().map(|&i| points[i].x as f64).sum::<f64>() / indices.len() as f64;
            let cy =
                indices.iter().map(|&i| points[i].y as f64).sum::<f64>() / indices.len() as f64;
            let cz =
                indices.iter().map(|&i| points[i].z as f64).sum::<f64>() / indices.len() as f64;
            let radius = indices
                .iter()
                .map(|&i| {
                    let dx = points[i].x as f64 - cx;
                    let dy = points[i].y as f64 - cy;
                    let dz = points[i].z as f64 - cz;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(0.0f64, f64::max);

            obstacles.push(Obstacle {
                id,
                position: [cx, cy, cz],
                distance: 0.0,
                radius,
                label: String::new(),
                confidence: 0.9,
            });
            id += 1;
        }
    }
    obstacles
}

fn build_scene_graph(objects: &[SceneObject], edge_threshold: f64) -> SceneGraph {
    let mut edges = Vec::new();
    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let dx = objects[i].center[0] - objects[j].center[0];
            let dy = objects[i].center[1] - objects[j].center[1];
            let dz = objects[i].center[2] - objects[j].center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist <= edge_threshold {
                edges.push(SceneEdge {
                    from: i,
                    to: j,
                    distance: dist,
                    relation: "near".to_string(),
                });
            }
        }
    }
    SceneGraph::new(objects.to_vec(), edges, 0)
}

fn predict_trajectory_linear(
    state: &RobotState,
    horizon: usize,
    dt_us: i64,
) -> Trajectory {
    let dt = dt_us as f64 / 1_000_000.0;
    let waypoints: Vec<[f64; 3]> = (1..=horizon)
        .map(|step| {
            let t = step as f64 * dt;
            [
                state.position[0] + state.velocity[0] * t,
                state.position[1] + state.velocity[1] * t,
                state.position[2] + state.velocity[2] * t,
            ]
        })
        .collect();
    let timestamps: Vec<i64> = (1..=horizon)
        .map(|step| state.timestamp_us + (step as i64) * dt_us)
        .collect();
    Trajectory::new(waypoints, timestamps, 0.95)
}

fn compute_attention_weights(
    objects: &[SceneObject],
    robot_pos: &[f64; 3],
    temperature: f64,
) -> Vec<f64> {
    if objects.is_empty() {
        return Vec::new();
    }
    let scores: Vec<f64> = objects
        .iter()
        .map(|obj| {
            let dx = obj.center[0] - robot_pos[0];
            let dy = obj.center[1] - robot_pos[1];
            let dz = obj.center[2] - robot_pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
            let vel_mag = obj
                .velocity
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .unwrap_or(0.0);
            (1.0 / dist + vel_mag) / temperature
        })
        .collect();
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Episodic memory
// ---------------------------------------------------------------------------

struct EpisodicMemory {
    embeddings: Vec<(u64, Vec<f32>)>,
}

impl EpisodicMemory {
    fn new() -> Self {
        Self {
            embeddings: Vec::new(),
        }
    }

    fn store(&mut self, id: u64, embedding: Vec<f32>) {
        self.embeddings.push((id, embedding));
    }

    fn recall_similar(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        let mut sims: Vec<(u64, f32)> = self
            .embeddings
            .iter()
            .map(|(id, ep)| {
                let dot: f32 = query.iter().zip(ep.iter()).map(|(a, b)| a * b).sum();
                let ep_norm: f32 = ep.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                (*id, dot / (q_norm * ep_norm))
            })
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);
        sims
    }
}

// ---------------------------------------------------------------------------
// Anomaly detector
// ---------------------------------------------------------------------------

struct AnomalyDetector {
    window: Vec<f64>,
    window_size: usize,
    threshold: f64,
}

impl AnomalyDetector {
    fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            threshold,
        }
    }

    fn train(&mut self, values: &[f64]) {
        for &v in values {
            if self.window.len() >= self.window_size {
                self.window.remove(0);
            }
            self.window.push(v);
        }
    }

    fn is_anomaly(&self, value: f64) -> bool {
        if self.window.len() < 2 {
            return false;
        }
        let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
        let variance = self
            .window
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / self.window.len() as f64;
        let std_dev = variance.sqrt().max(1e-10);
        ((value - mean).abs() / std_dev) > self.threshold
    }
}

// ---------------------------------------------------------------------------
// Skill learner
// ---------------------------------------------------------------------------

struct SkillLearner {
    demonstrations: Vec<Vec<[f64; 3]>>,
}

impl SkillLearner {
    fn new() -> Self {
        Self {
            demonstrations: Vec::new(),
        }
    }

    fn add_demonstration(&mut self, trajectory: Vec<[f64; 3]>) {
        self.demonstrations.push(trajectory);
    }

    fn reproduce(&self) -> Option<Vec<[f64; 3]>> {
        if self.demonstrations.is_empty() {
            return None;
        }
        let min_len = self
            .demonstrations
            .iter()
            .map(|d| d.len())
            .min()
            .unwrap_or(0);
        if min_len == 0 {
            return None;
        }
        let n = self.demonstrations.len() as f64;
        let trajectory: Vec<[f64; 3]> = (0..min_len)
            .map(|i| {
                let mut avg = [0.0f64; 3];
                for demo in &self.demonstrations {
                    avg[0] += demo[i][0];
                    avg[1] += demo[i][1];
                    avg[2] += demo[i][2];
                }
                [avg[0] / n, avg[1] / n, avg[2] / n]
            })
            .collect();
        Some(trajectory)
    }
}

// ---------------------------------------------------------------------------
// Swarm coordinator
// ---------------------------------------------------------------------------

fn assign_tasks_greedy(
    robots: &[([f64; 3], f64)], // (position, capability)
    tasks: &[([f64; 3], f64)],  // (position, priority)
) -> Vec<(usize, usize)> {
    let mut assignments = Vec::new();
    let mut available: Vec<bool> = vec![true; tasks.len()];
    let mut load: Vec<usize> = vec![0; robots.len()];

    for _ in 0..tasks.len() {
        let mut best_cost = f64::MAX;
        let mut best_ri = 0;
        let mut best_ti = 0;

        for (ri, (rpos, cap)) in robots.iter().enumerate() {
            for (ti, (tpos, prio)) in tasks.iter().enumerate() {
                if !available[ti] {
                    continue;
                }
                let dx = rpos[0] - tpos[0];
                let dy = rpos[1] - tpos[1];
                let dist = (dx * dx + dy * dy).sqrt();
                let cost = dist / cap.max(0.01) + (load[ri] as f64) * 5.0 - prio;
                if cost < best_cost {
                    best_cost = cost;
                    best_ri = ri;
                    best_ti = ti;
                }
            }
        }

        if best_cost < f64::MAX {
            available[best_ti] = false;
            load[best_ri] += 1;
            assignments.push((best_ri, best_ti));
        }
    }
    assignments
}

// ---------------------------------------------------------------------------
// Decision engine
// ---------------------------------------------------------------------------

struct DecisionOption {
    name: String,
    score: f64,
    risk: f64,
}

fn select_best_decision(options: &[DecisionOption], risk_tolerance: f64) -> Option<&DecisionOption> {
    options
        .iter()
        .filter(|o| o.risk <= risk_tolerance)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
}

// ---------------------------------------------------------------------------
// MCP tool registry
// ---------------------------------------------------------------------------

type ToolFn = Box<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

struct McpToolRegistry {
    tools: HashMap<String, (String, ToolFn)>,
}

impl McpToolRegistry {
    fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, description: &str, handler: ToolFn) {
        self.tools
            .insert(name.to_string(), (description.to_string(), handler));
    }

    fn tool_count(&self) -> usize {
        self.tools.len()
    }

    fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    fn call(&self, name: &str, args: &str) -> Result<String, String> {
        match self.tools.get(name) {
            Some((_, handler)) => handler(args),
            None => Err(format!("Tool not found: {}", name)),
        }
    }
}

fn build_robotics_tool_registry() -> McpToolRegistry {
    let mut r = McpToolRegistry::new();
    let tools: Vec<(&str, &str)> = vec![
        ("get_robot_state", "Get the current robot state"),
        ("move_to", "Move to a target position"),
        ("stop", "Emergency stop"),
        ("get_point_cloud", "Get the latest point cloud"),
        ("detect_obstacles", "Run obstacle detection"),
        ("get_scene_graph", "Get the current scene graph"),
        ("predict_trajectory", "Predict trajectory for an object"),
        ("set_velocity", "Set the robot's target velocity"),
        ("get_map", "Get the current occupancy grid map"),
        ("plan_path", "Plan a path to target"),
        ("get_battery", "Get robot battery status"),
        ("recall_memory", "Recall episodic memory"),
        ("store_memory", "Store an episode to memory"),
        ("get_swarm_status", "Get status of all robots"),
        ("assign_task", "Assign a task to a robot"),
    ];
    for (name, desc) in tools {
        r.register(
            name,
            desc,
            Box::new(move |_args| Ok(format!(r#"{{"tool":"{}","status":"ok"}}"#, name))),
        );
    }
    r
}

// ===========================================================================
// TESTS
// ===========================================================================

/// Test 1: End-to-end perception pipeline.
#[test]
fn test_perception_pipeline_end_to_end() {
    let mut points = generate_point_cloud_around(Point3D::new(2.0, 3.0, 0.0), 50, 0.8);
    points.extend(generate_point_cloud_around(
        Point3D::new(8.0, 7.0, 0.0),
        40,
        0.6,
    ));
    points.extend(generate_point_cloud_around(
        Point3D::new(5.0, 5.0, 0.0),
        30,
        0.5,
    ));
    let cloud = PointCloud::new(points.clone(), 1_000_000);
    assert_eq!(cloud.len(), 120);

    // Detect obstacles
    let obstacles = cluster_point_cloud(&cloud.points, 0.5, 3);
    assert!(
        !obstacles.is_empty(),
        "Should detect at least one obstacle cluster"
    );

    // Build scene graph
    let scene_objects: Vec<SceneObject> = obstacles
        .iter()
        .map(|obs| SceneObject::new(obs.id as usize, obs.position, [obs.radius; 3]))
        .collect();
    let graph = build_scene_graph(&scene_objects, 10.0);
    assert_eq!(graph.objects.len(), scene_objects.len());
    if scene_objects.len() > 1 {
        assert!(!graph.edges.is_empty(), "Should have edges between nearby objects");
    }

    // Predict trajectory
    let robot = RobotState {
        position: [0.0, 0.0, 0.0],
        velocity: [1.0, 0.5, 0.0],
        acceleration: [0.0; 3],
        timestamp_us: 1_000_000,
    };
    let trajectory = predict_trajectory_linear(&robot, 10, 100_000);
    assert_eq!(trajectory.len(), 10);
    let last = &trajectory.waypoints[9];
    assert!(
        (last[0] - 1.0).abs() < 1e-6,
        "Final x should be ~1.0, got {}",
        last[0]
    );
}

/// Test 2: Cognitive perceive-think-act cycle.
#[test]
fn test_cognitive_perceive_think_act() {
    let robot = RobotState {
        position: [5.0, 5.0, 0.0],
        velocity: [1.0, 0.0, 0.0],
        acceleration: [0.0; 3],
        timestamp_us: 0,
    };

    let mut points = generate_point_cloud_around(Point3D::new(6.0, 5.0, 0.0), 20, 0.3);
    points.extend(generate_point_cloud_around(
        Point3D::new(10.0, 10.0, 0.0),
        15,
        0.3,
    ));
    let obstacles = cluster_point_cloud(&points, 0.3, 3);

    let scene_objects: Vec<SceneObject> = obstacles
        .iter()
        .map(|obs| {
            let mut so = SceneObject::new(obs.id as usize, obs.position, [0.5; 3]);
            so.velocity = Some([0.0; 3]);
            so
        })
        .collect();

    let weights = compute_attention_weights(&scene_objects, &robot.position, 1.0);
    assert_eq!(weights.len(), scene_objects.len());
    let weight_sum: f64 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-6,
        "Weights should sum to 1.0, got {}",
        weight_sum
    );

    let min_dist = obstacles
        .iter()
        .map(|obs| {
            let dx = obs.position[0] - robot.position[0];
            let dy = obs.position[1] - robot.position[1];
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::MAX, f64::min);

    let action = if min_dist < 1.0 {
        "emergency_stop"
    } else if min_dist < 3.0 {
        "slow_down"
    } else {
        "proceed"
    };

    assert!(
        action == "emergency_stop" || action == "slow_down",
        "Should slow down or stop with obstacle at distance {}",
        min_dist
    );
}

/// Test 3: Episodic memory store and recall.
#[test]
fn test_episodic_memory_store_recall() {
    let mut memory = EpisodicMemory::new();

    for i in 0..20 {
        let mut embedding = vec![0.0f32; 32];
        embedding[i % 32] = 1.0;
        embedding[(i + 1) % 32] = 0.5;
        memory.store(i as u64, embedding);
    }

    let mut query = vec![0.0f32; 32];
    query[5] = 0.9;
    query[6] = 0.4;

    let results = memory.recall_similar(&query, 3);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 5, "Most similar should be episode 5");
    assert!(results[0].1 > 0.8, "Similarity should be high");
}

/// Test 4: Behavior tree patrol execution.
#[test]
fn test_behavior_tree_patrol() {
    let tree_node = BehaviorNode::Selector(vec![
        BehaviorNode::Sequence(vec![
            BehaviorNode::Condition("battery_ok".into()),
            BehaviorNode::Action("patrol".into()),
        ]),
        BehaviorNode::Action("return_to_base".into()),
    ]);

    let mut tree = BehaviorTree::new(tree_node);
    tree.set_condition("battery_ok", true);
    tree.set_action_result("patrol", BehaviorStatus::Running);
    tree.set_action_result("return_to_base", BehaviorStatus::Running);

    // First 5 ticks: patrol is Running
    for _tick in 0..5 {
        let status = tree.tick();
        assert_eq!(status, BehaviorStatus::Running);
    }

    // Switch patrol to Success
    tree.set_action_result("patrol", BehaviorStatus::Success);
    for _tick in 5..10 {
        let status = tree.tick();
        assert_eq!(status, BehaviorStatus::Success);
    }
}

/// Test 5: Swarm task assignment with 5 robots and 10 tasks.
#[test]
fn test_swarm_task_assignment() {
    let robots: Vec<([f64; 3], f64)> = (0..5)
        .map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 5.0;
            ([10.0 * angle.cos(), 10.0 * angle.sin(), 0.0], 0.5 + (i as f64) * 0.1)
        })
        .collect();

    let tasks: Vec<([f64; 3], f64)> = (0..10)
        .map(|i| {
            (
                [
                    pseudo_random_f32(500, i * 2) as f64 * 20.0 - 10.0,
                    pseudo_random_f32(500, i * 2 + 1) as f64 * 20.0 - 10.0,
                    0.0,
                ],
                pseudo_random_f32(600, i) as f64 * 10.0,
            )
        })
        .collect();

    let assignments = assign_tasks_greedy(&robots, &tasks);
    assert_eq!(assignments.len(), 10, "All tasks should be assigned");

    let assigned_tasks: Vec<usize> = assignments.iter().map(|(_, t)| *t).collect();
    for ti in 0..10 {
        assert!(assigned_tasks.contains(&ti), "Task {} should be assigned", ti);
    }

    for ri in 0..5 {
        let count = assignments.iter().filter(|(r, _)| *r == ri).count();
        assert!(count >= 1, "Robot {} should have at least 1 task", ri);
    }
}

/// Test 6: World model update and predict.
#[test]
fn test_world_model_update_predict() {
    let mut current_state = RobotState::default();

    for i in 0..20 {
        current_state = RobotState {
            position: [i as f64 * 0.5, 0.0, 0.0],
            velocity: [0.5, 0.0, 0.0],
            acceleration: [0.0; 3],
            timestamp_us: (i as i64) * 100_000,
        };
    }

    let traj = predict_trajectory_linear(&current_state, 30, 100_000);
    assert_eq!(traj.len(), 30);

    let step_20 = &traj.waypoints[19];
    let expected_x = current_state.position[0] + current_state.velocity[0] * 2.0;
    assert!(
        (step_20[0] - expected_x).abs() < 1e-6,
        "Predicted x at t+2s should be {}, got {}",
        expected_x,
        step_20[0]
    );
}

/// Test 7: Skill learning from demonstrations.
#[test]
fn test_skill_learning_from_demo() {
    let mut learner = SkillLearner::new();

    learner.add_demonstration(vec![
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.2],
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
    ]);
    learner.add_demonstration(vec![
        [0.0, 0.1, 0.0],
        [0.5, 0.1, 0.25],
        [1.0, 0.1, 0.55],
        [1.0, 0.1, 0.0],
    ]);
    learner.add_demonstration(vec![
        [0.0, -0.1, 0.0],
        [0.5, -0.1, 0.15],
        [1.0, -0.1, 0.45],
        [1.0, -0.1, 0.0],
    ]);

    let reproduced = learner.reproduce().expect("Should reproduce from 3 demos");
    assert_eq!(reproduced.len(), 4);
    assert!((reproduced[0][0] - 0.0).abs() < 1e-6);
    assert!((reproduced[0][1] - 0.0).abs() < 1e-6);

    let avg_z = (0.2 + 0.25 + 0.15) / 3.0;
    assert!(
        (reproduced[1][2] - avg_z).abs() < 1e-6,
        "Expected avg z={}, got {}",
        avg_z,
        reproduced[1][2]
    );
    assert!((reproduced[3][2] - 0.0).abs() < 1e-6, "End z should be 0");
}

/// Test 8: Anomaly detection accuracy.
#[test]
fn test_anomaly_detection_accuracy() {
    let mut detector = AnomalyDetector::new(100, 3.0);

    let normal_data: Vec<f64> = (0..100)
        .map(|i| 5.0 + (i as f64 % 7.0) * 0.1 - 0.3)
        .collect();
    detector.train(&normal_data);

    assert!(!detector.is_anomaly(5.0), "5.0 should not be anomalous");
    assert!(!detector.is_anomaly(5.3), "5.3 should not be anomalous");
    assert!(!detector.is_anomaly(4.7), "4.7 should not be anomalous");

    assert!(detector.is_anomaly(100.0), "100.0 should be anomalous");
    assert!(detector.is_anomaly(-50.0), "-50.0 should be anomalous");
    assert!(detector.is_anomaly(20.0), "20.0 should be anomalous");
}

/// Test 9: Attention focuses on nearest/fastest objects.
#[test]
fn test_attention_focuses_on_nearest() {
    let robot_pos = [0.0, 0.0, 0.0];

    let objects = vec![
        {
            let mut o = SceneObject::new(0, [10.0, 0.0, 0.0], [0.5; 3]);
            o.velocity = Some([0.0; 3]);
            o
        },
        {
            let mut o = SceneObject::new(1, [1.0, 0.0, 0.0], [0.5; 3]);
            o.velocity = Some([0.0; 3]);
            o
        },
        {
            let mut o = SceneObject::new(2, [5.0, 0.0, 0.0], [0.5; 3]);
            o.velocity = Some([0.0; 3]);
            o
        },
    ];

    let weights = compute_attention_weights(&objects, &robot_pos, 1.0);
    assert!(
        weights[1] > weights[0],
        "Near > far: {} vs {}",
        weights[1],
        weights[0]
    );
    assert!(
        weights[1] > weights[2],
        "Near > medium: {} vs {}",
        weights[1],
        weights[2]
    );

    // Fast moving far object should dominate
    let objects_fast = vec![
        {
            let mut o = SceneObject::new(0, [10.0, 0.0, 0.0], [0.5; 3]);
            o.velocity = Some([10.0, 0.0, 0.0]);
            o
        },
        {
            let mut o = SceneObject::new(1, [2.0, 0.0, 0.0], [0.5; 3]);
            o.velocity = Some([0.0; 3]);
            o
        },
    ];

    let weights_fast = compute_attention_weights(&objects_fast, &robot_pos, 1.0);
    assert!(
        weights_fast[0] > weights_fast[1],
        "Fast far > slow near: {} vs {}",
        weights_fast[0],
        weights_fast[1]
    );
}

/// Test 10: Decision engine selects optimal action.
#[test]
fn test_decision_engine_selects_best() {
    let options = vec![
        DecisionOption {
            name: "proceed_fast".to_string(),
            score: 8.0,
            risk: 0.7,
        },
        DecisionOption {
            name: "proceed_slow".to_string(),
            score: 6.0,
            risk: 0.2,
        },
        DecisionOption {
            name: "stop".to_string(),
            score: 3.0,
            risk: 0.0,
        },
        DecisionOption {
            name: "detour".to_string(),
            score: 7.0,
            risk: 0.4,
        },
    ];

    let best = select_best_decision(&options, 1.0).unwrap();
    assert_eq!(best.name, "proceed_fast");

    let best_moderate = select_best_decision(&options, 0.5).unwrap();
    assert_eq!(best_moderate.name, "detour");

    let best_safe = select_best_decision(&options, 0.1).unwrap();
    assert_eq!(best_safe.name, "stop");

    let best_zero = select_best_decision(&options, 0.0).unwrap();
    assert_eq!(best_zero.name, "stop");
}

/// Test 11: MCP tool listing -- verify all 15 tools are registered.
#[test]
fn test_mcp_tool_listing() {
    let registry = build_robotics_tool_registry();
    assert_eq!(registry.tool_count(), 15, "Should have 15 tools");

    let expected = [
        "get_robot_state",
        "move_to",
        "stop",
        "get_point_cloud",
        "detect_obstacles",
        "get_scene_graph",
        "predict_trajectory",
        "set_velocity",
        "get_map",
        "plan_path",
        "get_battery",
        "recall_memory",
        "store_memory",
        "get_swarm_status",
        "assign_task",
    ];
    for name in &expected {
        assert!(registry.has_tool(name), "Tool '{}' should be registered", name);
    }
}

/// Test 12: MCP tool execution.
#[test]
fn test_mcp_tool_execution() {
    let registry = build_robotics_tool_registry();

    let tool_names = [
        "get_robot_state",
        "move_to",
        "stop",
        "get_point_cloud",
        "detect_obstacles",
        "get_scene_graph",
        "predict_trajectory",
        "set_velocity",
        "get_map",
        "plan_path",
        "get_battery",
        "recall_memory",
        "store_memory",
        "get_swarm_status",
        "assign_task",
    ];

    for name in &tool_names {
        let result = registry.call(name, "{}");
        assert!(result.is_ok(), "Tool '{}' should succeed: {:?}", name, result);
        let response = result.unwrap();
        assert!(!response.is_empty(), "Tool '{}' should return non-empty", name);
    }

    let result = registry.call("nonexistent", "{}");
    assert!(result.is_err(), "Nonexistent tool should fail");
}

/// Test 13: Full pipeline stress test -- 100 frames.
#[test]
fn test_full_pipeline_100_frames() {
    let start = Instant::now();
    let mut total_obstacles = 0usize;

    for frame in 0..100 {
        let robot = RobotState {
            position: [frame as f64 * 0.1, 0.0, 0.0],
            velocity: [0.1, 0.0, 0.0],
            acceleration: [0.0; 3],
            timestamp_us: (frame as i64) * 33_333,
        };

        let center = Point3D::new(
            robot.position[0] as f32 + 3.0,
            pseudo_random_f32(frame as u64, 0) * 4.0 - 2.0,
            0.0,
        );
        let points = generate_point_cloud_around(center, 100, 1.0);

        let obstacles = cluster_point_cloud(&points, 0.5, 3);
        total_obstacles += obstacles.len();

        if !obstacles.is_empty() {
            let objects: Vec<SceneObject> = obstacles
                .iter()
                .map(|obs| SceneObject::new(obs.id as usize, obs.position, [0.5; 3]))
                .collect();
            let _weights = compute_attention_weights(&objects, &robot.position, 1.0);
        }

        let _traj = predict_trajectory_linear(&robot, 5, 33_333);
    }

    let elapsed = start.elapsed();
    assert!(total_obstacles > 0, "Should detect obstacles across 100 frames");
    assert!(
        elapsed.as_secs() < 5,
        "100 frames should complete in < 5s, took {:?}",
        elapsed
    );
}

/// Test 14: Concurrent spatial search from multiple threads.
#[test]
fn test_concurrent_spatial_search() {
    let points: Vec<Point3D> = (0..5000)
        .map(|i| {
            Point3D::new(
                pseudo_random_f32(42, i * 3) * 10.0,
                pseudo_random_f32(42, i * 3 + 1) * 10.0,
                pseudo_random_f32(42, i * 3 + 2) * 10.0,
            )
        })
        .collect();

    // Build a shared spatial index
    let cloud = PointCloud::new(points, 0);
    let mut index = SpatialIndex::new(3);
    index.insert_point_cloud(&cloud);
    let shared_index = Arc::new(index);
    let results = Arc::new(Mutex::new(Vec::new()));
    let k = 5usize;

    let mut handles = Vec::new();
    for thread_id in 0..4 {
        let idx = Arc::clone(&shared_index);
        let res = Arc::clone(&results);

        let handle = std::thread::spawn(move || {
            let query = [
                (thread_id as f32) * 2.5,
                (thread_id as f32) * 2.5,
                5.0_f32,
            ];
            let neighbors = idx.search_nearest(&query, k).unwrap();
            let mut r = res.lock().unwrap();
            r.push((thread_id, neighbors));
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let final_results = results.lock().unwrap();
    assert_eq!(final_results.len(), 4, "All 4 threads should complete");

    for (tid, neighbors) in final_results.iter() {
        assert_eq!(
            neighbors.len(),
            k,
            "Thread {} should return {} neighbors",
            tid,
            k
        );
        // Results should be distance-sorted
        for window in neighbors.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Thread {} results should be distance-sorted",
                tid
            );
        }
    }
}

/// Test 15: Edge cases -- empty inputs and boundary conditions.
#[test]
fn test_edge_cases() {
    // Empty point cloud
    let empty: Vec<Point3D> = Vec::new();
    let obstacles = cluster_point_cloud(&empty, 0.5, 3);
    assert!(obstacles.is_empty(), "Empty cloud => no obstacles");

    // Empty scene graph
    let graph = build_scene_graph(&[], 10.0);
    assert!(graph.objects.is_empty());
    assert!(graph.edges.is_empty());

    // Single point does not form cluster
    let single = vec![Point3D::new(1.0, 1.0, 1.0)];
    let obs = cluster_point_cloud(&single, 0.5, 3);
    assert!(obs.is_empty(), "Single point => no cluster");

    // Attention on empty objects
    let weights = compute_attention_weights(&[], &[0.0; 3], 1.0);
    assert!(weights.is_empty());

    // Trajectory from zero velocity
    let stationary = RobotState::default();
    let traj = predict_trajectory_linear(&stationary, 10, 100_000);
    assert_eq!(traj.len(), 10);
    for wp in &traj.waypoints {
        assert_eq!(*wp, [0.0, 0.0, 0.0], "Stationary robot stays at origin");
    }

    // Empty spatial index
    let index = SpatialIndex::new(3);
    assert!(index.is_empty());
    let result = index.search_nearest(&[0.0_f32, 0.0, 0.0], 5);
    assert!(result.is_err(), "Search on empty index should fail");

    // Skill learner with no demos
    let learner = SkillLearner::new();
    assert!(learner.reproduce().is_none());

    // Decision engine with no matching options
    let options = vec![DecisionOption {
        name: "risky".to_string(),
        score: 10.0,
        risk: 1.0,
    }];
    assert!(select_best_decision(&options, 0.5).is_none());

    // Empty memory recall
    let memory = EpisodicMemory::new();
    let results = memory.recall_similar(&[1.0, 2.0], 5);
    assert!(results.is_empty());

    // Anomaly detector with insufficient data
    let detector = AnomalyDetector::new(100, 3.0);
    assert!(!detector.is_anomaly(999.0));

    // Behavior tree with decorator
    let node = BehaviorNode::Decorator(
        DecoratorType::Inverter,
        Box::new(BehaviorNode::Action("a".into())),
    );
    let mut tree = BehaviorTree::new(node);
    tree.set_action_result("a", BehaviorStatus::Success);
    assert_eq!(tree.tick(), BehaviorStatus::Failure, "Inverter should flip Success to Failure");
}
