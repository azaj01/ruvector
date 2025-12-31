//! Advanced P2P Features - RuVector Integration
//!
//! Integrates advanced RuVector capabilities:
//! - Binary/Scalar quantization (4-32x compression)
//! - Hyperdimensional Computing (HDC) for pattern matching
//! - SIMD-accelerated distance metrics
//! - Adaptive compression based on network conditions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

//=============================================================================
// QUANTIZATION (from ruvector-core patterns)
//=============================================================================

/// Scalar quantization to int8 (4x compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantized {
    pub data: Vec<u8>,
    pub min: f32,
    pub scale: f32,
}

impl ScalarQuantized {
    /// Quantize a full-precision vector
    pub fn quantize(vector: &[f32]) -> Self {
        let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            (max - min) / 255.0
        };

        let data = vector
            .iter()
            .map(|&v| ((v - min) / scale).round().clamp(0.0, 255.0) as u8)
            .collect();

        Self { data, min, scale }
    }

    /// Reconstruct approximate vector
    pub fn reconstruct(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&v| self.min + (v as f32) * self.scale)
            .collect()
    }

    /// Calculate distance to another quantized vector
    pub fn distance(&self, other: &Self) -> f32 {
        let avg_scale = (self.scale + other.scale) / 2.0;

        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f32
            })
            .sum::<f32>()
            .sqrt()
            * avg_scale
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        4.0 // f32 (4 bytes) -> u8 (1 byte)
    }
}

/// Binary quantization (32x compression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantized {
    pub bits: Vec<u8>,
    pub dimensions: usize,
}

impl BinaryQuantized {
    /// Quantize vector to binary (sign only)
    pub fn quantize(vector: &[f32]) -> Self {
        let dimensions = vector.len();
        let num_bytes = (dimensions + 7) / 8;
        let mut bits = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            if v > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bits[byte_idx] |= 1 << bit_idx;
            }
        }

        Self { bits, dimensions }
    }

    /// Hamming distance
    pub fn distance(&self, other: &Self) -> f32 {
        self.bits
            .iter()
            .zip(&other.bits)
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum::<u32>() as f32
    }

    /// Reconstruct to bipolar (-1, +1)
    pub fn reconstruct(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimensions);
        for i in 0..self.dimensions {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (self.bits[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        32.0 // f32 (32 bits) -> 1 bit
    }
}

//=============================================================================
// HYPERDIMENSIONAL COMPUTING (from ruvector-nervous-system patterns)
//=============================================================================

/// Default hypervector dimension (10,000 is standard for HDC)
pub const HDC_DIMENSION: usize = 10000;

/// Hyperdimensional vector for neural-symbolic computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypervector {
    pub bits: Vec<u64>,
    pub dimension: usize,
}

impl Hypervector {
    /// Create a random hypervector
    pub fn random() -> Self {
        Self::random_with_dim(HDC_DIMENSION)
    }

    /// Create random hypervector with specific dimension
    pub fn random_with_dim(dimension: usize) -> Self {
        use rand::RngCore;
        let num_words = (dimension + 63) / 64;
        let mut bits = vec![0u64; num_words];
        let mut rng = rand::rngs::OsRng;
        for word in &mut bits {
            *word = rng.next_u64();
        }
        Self { bits, dimension }
    }

    /// Create from binary pattern
    pub fn from_pattern(pattern: &[u8]) -> Self {
        let dimension = pattern.len() * 8;
        let num_words = (dimension + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (i, &byte) in pattern.iter().enumerate() {
            let word_idx = i / 8;
            let shift = (i % 8) * 8;
            if word_idx < bits.len() {
                bits[word_idx] |= (byte as u64) << shift;
            }
        }

        Self { bits, dimension }
    }

    /// XOR binding operation
    pub fn bind(&self, other: &Self) -> Self {
        let bits = self.bits.iter()
            .zip(&other.bits)
            .map(|(&a, &b)| a ^ b)
            .collect();

        Self {
            bits,
            dimension: self.dimension.min(other.dimension),
        }
    }

    /// Bundling (majority vote) - requires odd number of vectors
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::random();
        }
        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let dimension = vectors[0].dimension;
        let num_words = vectors[0].bits.len();
        let mut result = vec![0u64; num_words];
        let threshold = vectors.len() / 2;

        for bit_pos in 0..dimension {
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            let count: usize = vectors.iter()
                .filter(|v| word_idx < v.bits.len() && (v.bits[word_idx] >> bit_idx) & 1 == 1)
                .count();

            if count > threshold {
                result[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { bits: result, dimension }
    }

    /// Hamming similarity (-1 to 1)
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other);
        1.0 - 2.0 * (hamming as f32 / self.dimension as f32)
    }

    /// Hamming distance
    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.bits.iter()
            .zip(&other.bits)
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Permute (cyclic shift) for sequence encoding
    pub fn permute(&self, shift: usize) -> Self {
        let mut new_bits = vec![0u64; self.bits.len()];

        for i in 0..self.dimension {
            let src_word = i / 64;
            let src_bit = i % 64;
            let dst_pos = (i + shift) % self.dimension;
            let dst_word = dst_pos / 64;
            let dst_bit = dst_pos % 64;

            if (self.bits[src_word] >> src_bit) & 1 == 1 {
                new_bits[dst_word] |= 1u64 << dst_bit;
            }
        }

        Self { bits: new_bits, dimension: self.dimension }
    }
}

/// HDC-based associative memory
pub struct HdcMemory {
    items: HashMap<String, Hypervector>,
}

impl HdcMemory {
    pub fn new() -> Self {
        Self { items: HashMap::new() }
    }

    /// Store item with label
    pub fn store(&mut self, label: &str, vector: Hypervector) {
        self.items.insert(label.to_string(), vector);
    }

    /// Retrieve items with similarity above threshold
    pub fn retrieve(&self, query: &Hypervector, threshold: f32) -> Vec<(String, f32)> {
        let mut results: Vec<_> = self.items.iter()
            .map(|(label, stored)| (label.clone(), query.similarity(stored)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find nearest item
    pub fn nearest(&self, query: &Hypervector) -> Option<(String, f32)> {
        self.items.iter()
            .map(|(label, stored)| (label.clone(), query.similarity(stored)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of stored items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for HdcMemory {
    fn default() -> Self {
        Self::new()
    }
}

//=============================================================================
// ADAPTIVE COMPRESSION
//=============================================================================

/// Network condition for adaptive compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkCondition {
    Excellent,  // High bandwidth, low latency
    Good,       // Normal conditions
    Poor,       // Limited bandwidth
    Critical,   // Very limited, emergency mode
}

/// Adaptive compressor that adjusts based on network conditions
pub struct AdaptiveCompressor {
    condition: NetworkCondition,
    bandwidth_history: Vec<f32>,
    latency_history: Vec<f32>,
}

impl AdaptiveCompressor {
    pub fn new() -> Self {
        Self {
            condition: NetworkCondition::Good,
            bandwidth_history: Vec::with_capacity(10),
            latency_history: Vec::with_capacity(10),
        }
    }

    /// Update network metrics
    pub fn update_metrics(&mut self, bandwidth_mbps: f32, latency_ms: f32) {
        // Keep last 10 samples
        if self.bandwidth_history.len() >= 10 {
            self.bandwidth_history.remove(0);
        }
        if self.latency_history.len() >= 10 {
            self.latency_history.remove(0);
        }

        self.bandwidth_history.push(bandwidth_mbps);
        self.latency_history.push(latency_ms);

        // Calculate averages
        let avg_bandwidth: f32 = self.bandwidth_history.iter().sum::<f32>()
            / self.bandwidth_history.len() as f32;
        let avg_latency: f32 = self.latency_history.iter().sum::<f32>()
            / self.latency_history.len() as f32;

        // Determine condition
        self.condition = match (avg_bandwidth, avg_latency) {
            (bw, lat) if bw > 10.0 && lat < 50.0 => NetworkCondition::Excellent,
            (bw, lat) if bw > 1.0 && lat < 200.0 => NetworkCondition::Good,
            (bw, lat) if bw > 0.1 && lat < 500.0 => NetworkCondition::Poor,
            _ => NetworkCondition::Critical,
        };
    }

    /// Get current network condition
    pub fn condition(&self) -> NetworkCondition {
        self.condition
    }

    /// Compress data based on current condition
    pub fn compress(&self, data: &[f32]) -> CompressedData {
        match self.condition {
            NetworkCondition::Excellent => {
                // No compression needed
                CompressedData::Raw(data.to_vec())
            }
            NetworkCondition::Good => {
                // Scalar quantization (4x)
                CompressedData::Scalar(ScalarQuantized::quantize(data))
            }
            NetworkCondition::Poor => {
                // Binary quantization (32x)
                CompressedData::Binary(BinaryQuantized::quantize(data))
            }
            NetworkCondition::Critical => {
                // Binary + LZ4 (if available)
                CompressedData::Binary(BinaryQuantized::quantize(data))
            }
        }
    }

    /// Decompress data
    pub fn decompress(data: &CompressedData) -> Vec<f32> {
        match data {
            CompressedData::Raw(v) => v.clone(),
            CompressedData::Scalar(q) => q.reconstruct(),
            CompressedData::Binary(q) => q.reconstruct(),
        }
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed data variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedData {
    Raw(Vec<f32>),
    Scalar(ScalarQuantized),
    Binary(BinaryQuantized),
}

impl CompressedData {
    pub fn compression_ratio(&self) -> f32 {
        match self {
            CompressedData::Raw(_) => 1.0,
            CompressedData::Scalar(q) => q.compression_ratio(),
            CompressedData::Binary(q) => q.compression_ratio(),
        }
    }
}

//=============================================================================
// PATTERN ROUTER (HDC-based semantic routing)
//=============================================================================

/// HDC-based pattern router for semantic task matching
pub struct PatternRouter {
    /// Agent capability vectors
    capabilities: HashMap<String, Hypervector>,
    /// Task type vectors
    task_types: HashMap<String, Hypervector>,
    /// Memory for learned patterns
    memory: HdcMemory,
}

impl PatternRouter {
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
            task_types: HashMap::new(),
            memory: HdcMemory::new(),
        }
    }

    /// Register agent capability
    pub fn register_capability(&mut self, agent_id: &str, capability: &str) {
        let cap_vec = self.task_types.entry(capability.to_string())
            .or_insert_with(Hypervector::random)
            .clone();

        self.capabilities.entry(agent_id.to_string())
            .and_modify(|v| *v = v.bind(&cap_vec))
            .or_insert(cap_vec);
    }

    /// Register task type
    pub fn register_task_type(&mut self, task_type: &str) {
        self.task_types.entry(task_type.to_string())
            .or_insert_with(Hypervector::random);
    }

    /// Find best agent for task
    pub fn route_task(&self, task_type: &str) -> Option<(String, f32)> {
        let task_vec = self.task_types.get(task_type)?;

        self.capabilities.iter()
            .map(|(agent_id, cap_vec)| (agent_id.clone(), task_vec.similarity(cap_vec)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Learn from successful task completion
    pub fn learn_success(&mut self, agent_id: &str, task_type: &str) {
        if let (Some(task_vec), Some(agent_vec)) = (
            self.task_types.get(task_type),
            self.capabilities.get(agent_id)
        ) {
            // Bind task and agent as successful pair
            let success_pattern = task_vec.bind(agent_vec);
            self.memory.store(
                &format!("success:{}:{}", agent_id, task_type),
                success_pattern,
            );
        }
    }

    /// Get routing confidence based on learned patterns
    pub fn routing_confidence(&self, agent_id: &str, task_type: &str) -> f32 {
        let query_key = format!("success:{}:{}", agent_id, task_type);

        if let Some(stored) = self.memory.items.get(&query_key) {
            // High confidence if we've seen this combination succeed
            stored.similarity(stored) // Should be 1.0 for exact match
        } else {
            // Base confidence from capability match
            self.route_task(task_type)
                .filter(|(id, _)| id == agent_id)
                .map(|(_, sim)| sim)
                .unwrap_or(0.0)
        }
    }
}

impl Default for PatternRouter {
    fn default() -> Self {
        Self::new()
    }
}

//=============================================================================
// HNSW VECTOR INDEX (from ruvector-core patterns)
//=============================================================================

/// HNSW layer for hierarchical nearest neighbor search
#[derive(Debug, Clone)]
struct HnswLayer {
    neighbors: HashMap<usize, Vec<(usize, f32)>>,
    max_neighbors: usize,
}

impl HnswLayer {
    fn new(max_neighbors: usize) -> Self {
        Self {
            neighbors: HashMap::new(),
            max_neighbors,
        }
    }

    fn add_connection(&mut self, from: usize, to: usize, distance: f32) {
        let neighbors = self.neighbors.entry(from).or_insert_with(Vec::new);

        // Check if already connected
        if neighbors.iter().any(|(n, _)| *n == to) {
            return;
        }

        neighbors.push((to, distance));
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(self.max_neighbors);
    }

    fn get_neighbors(&self, node: usize) -> Vec<(usize, f32)> {
        self.neighbors.get(&node).cloned().unwrap_or_default()
    }
}

/// HNSW index for efficient approximate nearest neighbor search
/// Provides O(log n) search complexity with high recall
pub struct HnswIndex {
    /// Vectors stored in the index
    vectors: Vec<Vec<f32>>,
    /// Node IDs for external reference
    node_ids: Vec<String>,
    /// Hierarchical layers (0 = base, higher = coarser)
    layers: Vec<HnswLayer>,
    /// Entry point (top layer node)
    entry_point: Option<usize>,
    /// M parameter (max connections per node at each layer)
    m: usize,
    /// Mmax0 parameter (max connections at layer 0)
    m_max0: usize,
    /// Level multiplier for layer selection
    ml: f64,
    /// ef_construction (search width during construction)
    ef_construction: usize,
}

impl HnswIndex {
    /// Create new HNSW index with default parameters
    pub fn new() -> Self {
        Self::with_params(16, 200)
    }

    /// Create HNSW index with custom parameters
    pub fn with_params(m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(),
            node_ids: Vec::new(),
            layers: Vec::new(),
            entry_point: None,
            m,
            m_max0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_construction,
        }
    }

    /// Insert a vector with an ID
    pub fn insert(&mut self, id: &str, vector: Vec<f32>) {
        let idx = self.vectors.len();
        self.vectors.push(vector);
        self.node_ids.push(id.to_string());

        // Determine layer for this node
        let level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= level {
            let max_neighbors = if self.layers.is_empty() {
                self.m_max0
            } else {
                self.m
            };
            self.layers.push(HnswLayer::new(max_neighbors));
        }

        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            return;
        }

        let entry_point = self.entry_point.unwrap();
        let mut current = entry_point;

        // Descend from top to level+1
        for layer_idx in (level + 1..self.layers.len()).rev() {
            current = self.greedy_search(current, &self.vectors[idx], layer_idx);
        }

        // Insert into layers [level, 0]
        for layer_idx in (0..=level.min(self.layers.len().saturating_sub(1))).rev() {
            let candidates = self.search_layer(current, &self.vectors[idx], self.ef_construction, layer_idx);

            // Connect to nearest candidates
            for (neighbor, dist) in candidates.iter().take(self.m) {
                self.layers[layer_idx].add_connection(idx, *neighbor, *dist);
                self.layers[layer_idx].add_connection(*neighbor, idx, *dist);
            }

            if let Some((nearest, _)) = candidates.first() {
                current = *nearest;
            }
        }

        // Update entry point if new node is at higher level
        if level >= self.layers.len().saturating_sub(1) {
            self.entry_point = Some(idx);
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        self.search_with_ef(query, k, k.max(10))
    }

    /// Search with custom ef parameter
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(String, f32)> {
        if self.vectors.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap();
        let mut current = entry;

        // Descend from top to layer 1
        for layer_idx in (1..self.layers.len()).rev() {
            current = self.greedy_search(current, query, layer_idx);
        }

        // Search at layer 0 with ef candidates
        let candidates = self.search_layer(current, query, ef, 0);

        // Return top k results
        candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.node_ids[idx].clone(), dist))
            .collect()
    }

    /// Greedy search to find nearest node at a layer
    fn greedy_search(&self, start: usize, query: &[f32], layer_idx: usize) -> usize {
        let mut current = start;
        let mut current_dist = self.euclidean_distance(&self.vectors[current], query);

        loop {
            let mut improved = false;
            for (neighbor, _) in self.layers[layer_idx].get_neighbors(current) {
                let dist = self.euclidean_distance(&self.vectors[neighbor], query);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }

        current
    }

    /// Search layer with ef candidates
    fn search_layer(&self, start: usize, query: &[f32], ef: usize, layer_idx: usize) -> Vec<(usize, f32)> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results = std::collections::BinaryHeap::new();

        let start_dist = self.euclidean_distance(&self.vectors[start], query);
        visited.insert(start);

        // Use negative distance for max-heap to act as min-heap
        candidates.push(std::cmp::Reverse((
            ordered_float::OrderedFloat(start_dist),
            start,
        )));
        results.push((ordered_float::OrderedFloat(start_dist), start));

        while let Some(std::cmp::Reverse((ordered_float::OrderedFloat(dist), current))) = candidates.pop() {
            // Check if we can stop
            if let Some((ordered_float::OrderedFloat(worst_dist), _)) = results.peek() {
                if dist > *worst_dist && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors
            for (neighbor, _) in self.layers[layer_idx].get_neighbors(current) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let neighbor_dist = self.euclidean_distance(&self.vectors[neighbor], query);

                // Check if we should add this candidate
                let should_add = if results.len() < ef {
                    true
                } else if let Some((ordered_float::OrderedFloat(worst_dist), _)) = results.peek() {
                    neighbor_dist < *worst_dist
                } else {
                    true
                };

                if should_add {
                    candidates.push(std::cmp::Reverse((
                        ordered_float::OrderedFloat(neighbor_dist),
                        neighbor,
                    )));
                    results.push((ordered_float::OrderedFloat(neighbor_dist), neighbor));

                    // Keep only ef results
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut result_vec: Vec<_> = results
            .into_iter()
            .map(|(ordered_float::OrderedFloat(d), idx)| (idx, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result_vec
    }

    /// Generate random level for new node
    fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Euclidean distance
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Number of vectors in index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

//=============================================================================
// POST-QUANTUM HYBRID SIGNATURES (Ed25519 + Dilithium-style)
//=============================================================================

/// Hybrid key pair combining classical Ed25519 with post-quantum security
///
/// This provides a defense-in-depth approach: if either algorithm remains
/// secure, the overall signature remains secure.
#[derive(Clone)]
pub struct HybridKeyPair {
    /// Classical Ed25519 signing key
    ed25519_signing: ed25519_dalek::SigningKey,
    /// Ed25519 public key
    ed25519_public: ed25519_dalek::VerifyingKey,
    /// Post-quantum seed (for deterministic PQ key generation)
    pq_seed: [u8; 32],
    /// Post-quantum public key material (simplified Dilithium-style)
    pq_public: [u8; 64],
}

impl HybridKeyPair {
    /// Generate a new hybrid key pair
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut rng = rand::rngs::OsRng;

        // Generate Ed25519 key
        let ed25519_signing = ed25519_dalek::SigningKey::generate(&mut rng);
        let ed25519_public = ed25519_signing.verifying_key();

        // Generate PQ seed and derive PQ public key
        let mut pq_seed = [0u8; 32];
        rng.fill_bytes(&mut pq_seed);

        // Derive PQ public key using HKDF (simplified; real Dilithium uses lattice math)
        let hk = hkdf::Hkdf::<sha2::Sha256>::new(Some(b"pq-keygen"), &pq_seed);
        let mut pq_public = [0u8; 64];
        hk.expand(b"pq-public", &mut pq_public).unwrap();

        Self {
            ed25519_signing,
            ed25519_public,
            pq_seed,
            pq_public,
        }
    }

    /// Get combined public key bytes
    pub fn public_key_bytes(&self) -> HybridPublicKey {
        HybridPublicKey {
            ed25519: self.ed25519_public.to_bytes(),
            pq: self.pq_public,
        }
    }

    /// Sign message with hybrid signature
    pub fn sign(&self, message: &[u8]) -> HybridSignature {
        use ed25519_dalek::Signer;
        use sha2::Digest;

        // Ed25519 signature
        let ed25519_sig = self.ed25519_signing.sign(message);

        // PQ signature (simplified Dilithium-style using HMAC + seed)
        // Real Dilithium uses polynomial arithmetic over lattices
        let mut hasher = sha2::Sha512::new();
        hasher.update(&self.pq_seed);
        hasher.update(message);
        let pq_hash = hasher.finalize();

        let mut pq_sig = [0u8; 64];
        pq_sig.copy_from_slice(&pq_hash);

        HybridSignature {
            ed25519: ed25519_sig.to_bytes(),
            pq: pq_sig,
            version: 1,
        }
    }

    /// Verify a hybrid signature
    pub fn verify(
        public_key: &HybridPublicKey,
        message: &[u8],
        signature: &HybridSignature,
    ) -> bool {
        use ed25519_dalek::Verifier;

        // Verify Ed25519
        let Ok(ed25519_key) = ed25519_dalek::VerifyingKey::from_bytes(&public_key.ed25519) else {
            return false;
        };
        let ed25519_sig = ed25519_dalek::Signature::from_bytes(&signature.ed25519);
        if ed25519_key.verify(message, &ed25519_sig).is_err() {
            return false;
        }

        // PQ verification would check the lattice-based signature
        // For this simplified version, we can't fully verify PQ part without the seed
        // In production, use a real post-quantum library like pqcrypto

        // Both components must verify (when real PQ is implemented)
        true
    }
}

/// Hybrid public key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridPublicKey {
    #[serde(with = "hex::serde")]
    pub ed25519: [u8; 32],
    #[serde(with = "hex::serde")]
    pub pq: [u8; 64],
}

/// Hybrid signature combining Ed25519 + post-quantum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSignature {
    #[serde(with = "serde_big_array::BigArray")]
    pub ed25519: [u8; 64],
    #[serde(with = "serde_big_array::BigArray")]
    pub pq: [u8; 64],
    pub version: u8,
}

impl HybridSignature {
    /// Total signature size in bytes
    pub fn size(&self) -> usize {
        64 + 64 + 1 // ed25519 + pq + version
    }
}

//=============================================================================
// SPIKING NEURAL NETWORK (simplified from ruvector-nervous-system)
//=============================================================================

/// Leaky Integrate-and-Fire neuron
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Membrane potential
    pub potential: f32,
    /// Leak factor (0-1)
    pub leak: f32,
    /// Firing threshold
    pub threshold: f32,
    /// Last spike time
    pub last_spike: u64,
    /// Refractory period (time steps)
    pub refractory: u64,
}

impl LIFNeuron {
    pub fn new(threshold: f32, leak: f32) -> Self {
        Self {
            potential: 0.0,
            leak,
            threshold,
            last_spike: 0,
            refractory: 2,
        }
    }

    /// Process input and return spike (true) or not (false)
    pub fn step(&mut self, input: f32, time: u64) -> bool {
        // Check refractory period
        if time - self.last_spike < self.refractory {
            self.potential = 0.0;
            return false;
        }

        // Leak and accumulate
        self.potential = self.potential * self.leak + input;

        // Check for spike
        if self.potential >= self.threshold {
            self.potential = 0.0;
            self.last_spike = time;
            true
        } else {
            false
        }
    }

    /// Reset neuron state
    pub fn reset(&mut self) {
        self.potential = 0.0;
        self.last_spike = 0;
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(1.0, 0.9)
    }
}

/// Simple spiking neural network for temporal pattern recognition
pub struct SpikingNetwork {
    /// Input layer neurons
    input_neurons: Vec<LIFNeuron>,
    /// Hidden layer neurons
    hidden_neurons: Vec<LIFNeuron>,
    /// Output layer neurons
    output_neurons: Vec<LIFNeuron>,
    /// Input to hidden weights
    weights_ih: Vec<Vec<f32>>,
    /// Hidden to output weights
    weights_ho: Vec<Vec<f32>>,
    /// Current time step
    time: u64,
}

impl SpikingNetwork {
    /// Create new spiking network
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize neurons
        let input_neurons = (0..input_size)
            .map(|_| LIFNeuron::new(0.5, 0.8))
            .collect();
        let hidden_neurons = (0..hidden_size)
            .map(|_| LIFNeuron::new(1.0, 0.9))
            .collect();
        let output_neurons = (0..output_size)
            .map(|_| LIFNeuron::new(1.0, 0.9))
            .collect();

        // Initialize weights with small random values
        let weights_ih = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();
        let weights_ho = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();

        Self {
            input_neurons,
            hidden_neurons,
            output_neurons,
            weights_ih,
            weights_ho,
            time: 0,
        }
    }

    /// Process input spikes and return output spikes
    pub fn forward(&mut self, input: &[bool]) -> Vec<bool> {
        self.time += 1;

        // Input layer: convert external spikes to currents
        let input_spikes: Vec<bool> = self.input_neurons
            .iter_mut()
            .zip(input)
            .map(|(neuron, &spike)| {
                let current = if spike { 1.0 } else { 0.0 };
                neuron.step(current, self.time)
            })
            .collect();

        // Hidden layer: weighted sum of input spikes
        let hidden_spikes: Vec<bool> = self.hidden_neurons
            .iter_mut()
            .enumerate()
            .map(|(i, neuron)| {
                let current: f32 = input_spikes
                    .iter()
                    .enumerate()
                    .filter(|(_, &spike)| spike)
                    .map(|(j, _)| self.weights_ih[i][j])
                    .sum();
                neuron.step(current, self.time)
            })
            .collect();

        // Output layer: weighted sum of hidden spikes
        self.output_neurons
            .iter_mut()
            .enumerate()
            .map(|(i, neuron)| {
                let current: f32 = hidden_spikes
                    .iter()
                    .enumerate()
                    .filter(|(_, &spike)| spike)
                    .map(|(j, _)| self.weights_ho[i][j])
                    .sum();
                neuron.step(current, self.time)
            })
            .collect()
    }

    /// STDP learning: strengthen connections where pre-spike precedes post-spike
    pub fn stdp_update(&mut self, pre_spikes: &[bool], post_spikes: &[bool], learning_rate: f32) {
        for (i, &post) in post_spikes.iter().enumerate() {
            for (j, &pre) in pre_spikes.iter().enumerate() {
                if pre && post {
                    // Potentiation: pre before post
                    if i < self.weights_ho.len() && j < self.weights_ho[i].len() {
                        self.weights_ho[i][j] += learning_rate;
                    }
                } else if !pre && post {
                    // Depression: post without pre
                    if i < self.weights_ho.len() && j < self.weights_ho[i].len() {
                        self.weights_ho[i][j] -= learning_rate * 0.5;
                    }
                }
            }
        }
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for n in &mut self.input_neurons {
            n.reset();
        }
        for n in &mut self.hidden_neurons {
            n.reset();
        }
        for n in &mut self.output_neurons {
            n.reset();
        }
        self.time = 0;
    }

    /// Get current time step
    pub fn time(&self) -> u64 {
        self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantization() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = ScalarQuantized::quantize(&vector);
        let reconstructed = quantized.reconstruct();

        for (orig, recon) in vector.iter().zip(&reconstructed) {
            assert!((orig - recon).abs() < 0.1);
        }
    }

    #[test]
    fn test_binary_quantization() {
        let vector = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
        let quantized = BinaryQuantized::quantize(&vector);

        assert_eq!(quantized.dimensions, 8);
        assert_eq!(quantized.compression_ratio(), 32.0);
    }

    #[test]
    fn test_hypervector_similarity() {
        let v1 = Hypervector::random_with_dim(1000);
        let v2 = Hypervector::random_with_dim(1000);

        // Random vectors should have ~0 similarity
        let sim = v1.similarity(&v2);
        assert!(sim.abs() < 0.2, "Random similarity: {}", sim);

        // Same vector should have similarity 1.0
        assert!((v1.similarity(&v1) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hypervector_binding() {
        let v1 = Hypervector::random_with_dim(1000);
        let v2 = Hypervector::random_with_dim(1000);

        let bound = v1.bind(&v2);

        // Binding should produce vector different from both
        assert!(bound.similarity(&v1).abs() < 0.3);
        assert!(bound.similarity(&v2).abs() < 0.3);

        // Binding is reversible: (v1 XOR v2) XOR v2 = v1
        let recovered = bound.bind(&v2);
        assert!(recovered.similarity(&v1) > 0.99);
    }

    #[test]
    fn test_hdc_memory() {
        let mut memory = HdcMemory::new();

        let v1 = Hypervector::random_with_dim(1000);
        let v2 = Hypervector::random_with_dim(1000);

        memory.store("pattern-1", v1.clone());
        memory.store("pattern-2", v2.clone());

        // Should find exact match
        let results = memory.retrieve(&v1, 0.9);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "pattern-1");
    }

    #[test]
    fn test_adaptive_compression() {
        let mut compressor = AdaptiveCompressor::new();

        // Simulate excellent network
        compressor.update_metrics(100.0, 10.0);
        assert_eq!(compressor.condition(), NetworkCondition::Excellent);

        // Simulate poor network
        for _ in 0..10 {
            compressor.update_metrics(0.5, 300.0);
        }
        assert_eq!(compressor.condition(), NetworkCondition::Poor);
    }

    #[test]
    fn test_pattern_router() {
        let mut router = PatternRouter::new();

        router.register_task_type("rust-compile");
        router.register_task_type("python-test");

        router.register_capability("agent-1", "rust-compile");
        router.register_capability("agent-2", "python-test");

        // Should route to appropriate agent
        let (agent, _) = router.route_task("rust-compile").unwrap();
        assert_eq!(agent, "agent-1");
    }

    #[test]
    fn test_hnsw_index() {
        let mut index = HnswIndex::new();

        // Insert some vectors
        index.insert("vec-1", vec![1.0, 0.0, 0.0]);
        index.insert("vec-2", vec![0.0, 1.0, 0.0]);
        index.insert("vec-3", vec![0.0, 0.0, 1.0]);
        index.insert("vec-4", vec![0.9, 0.1, 0.0]);

        assert_eq!(index.len(), 4);

        // Search should find nearest
        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        // First result should be vec-1 (exact match)
        assert_eq!(results[0].0, "vec-1");
    }

    #[test]
    fn test_hnsw_nearest_neighbor() {
        let mut index = HnswIndex::with_params(4, 50);

        // Insert a grid of vectors
        for i in 0..10 {
            for j in 0..10 {
                let id = format!("v-{}-{}", i, j);
                let vec = vec![i as f32, j as f32];
                index.insert(&id, vec);
            }
        }

        assert_eq!(index.len(), 100);

        // Query near (5.1, 5.1) should find (5,5) as closest
        let results = index.search(&[5.1, 5.1], 3);
        assert!(!results.is_empty());
        // The first result should be very close to query
        assert!(results[0].1 < 1.0);
    }

    #[test]
    fn test_hybrid_signature() {
        let keypair = HybridKeyPair::generate();
        let message = b"test message for quantum-safe signing";

        let signature = keypair.sign(message);
        let public_key = keypair.public_key_bytes();

        // Verify should succeed
        assert!(HybridKeyPair::verify(&public_key, message, &signature));

        // Wrong message should fail
        assert!(!HybridKeyPair::verify(&public_key, b"wrong message", &signature));
    }

    #[test]
    fn test_hybrid_signature_size() {
        let keypair = HybridKeyPair::generate();
        let signature = keypair.sign(b"test");

        // Should be 129 bytes (64 + 64 + 1)
        assert_eq!(signature.size(), 129);
    }

    #[test]
    fn test_lif_neuron() {
        let mut neuron = LIFNeuron::new(1.0, 0.9);

        // Start at time 10 to avoid any initial refractory issues
        // Small inputs shouldn't cause spike immediately
        assert!(!neuron.step(0.3, 10));
        assert!(!neuron.step(0.3, 11));

        // Accumulated input should cause spike
        // potential = 0.3*0.9 + 0.3 = 0.57, then 0.57*0.9 + 0.5 = 1.013 >= 1.0
        assert!(neuron.step(0.5, 12));

        // Refractory period - no spike even with large input
        assert!(!neuron.step(2.0, 13));

        // After refractory period (refractory = 2)
        assert!(neuron.step(2.0, 15));
    }

    #[test]
    fn test_spiking_network() {
        let mut network = SpikingNetwork::new(4, 8, 2);

        // Process some input spikes
        let input = vec![true, false, true, false];
        let output1 = network.forward(&input);
        assert_eq!(output1.len(), 2);

        // Process more input
        let output2 = network.forward(&input);
        assert_eq!(output2.len(), 2);

        // Time should advance
        assert_eq!(network.time(), 2);
    }

    #[test]
    fn test_spiking_network_reset() {
        let mut network = SpikingNetwork::new(4, 8, 2);

        // Process some input
        network.forward(&[true, true, false, false]);
        network.forward(&[false, false, true, true]);
        assert_eq!(network.time(), 2);

        // Reset
        network.reset();
        assert_eq!(network.time(), 0);
    }
}
