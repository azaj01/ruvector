//! # RuVector Adversarial Coherence (RAC)
//!
//! **Adversarial Coherence Thesis (circa 2076):**
//!
//! In a browser-scale, adversarial world, the only sustainable definition of "correctness" is:
//! *claims survive continuous challenge, remain traceable, and can be repaired without global resets.*
//!
//! Structural integrity (high min-cut, stable connectivity) is necessary but not sufficient.
//! The core runtime for all large-scale intelligence becomes a second control loop:
//! an adversarial coherence layer that treats disagreement as a first-class signal,
//! keeps an append-only history of what was believed and why, and makes correction
//! a normal operation rather than an exception.
//!
//! ## The 12 Axioms
//!
//! 1. **Connectivity is not truth.** Structural metrics bound failure modes, not correctness.
//! 2. **Everything is an event.** Assertions, challenges, model updates, and decisions are all logged events.
//! 3. **No destructive edits.** Incorrect learning is deprecated, never erased.
//! 4. **Every claim is scoped.** Claims are always tied to a context: task, domain, time window, and authority boundary.
//! 5. **Semantics drift is expected.** Drift is measured and managed, not denied.
//! 6. **Disagreement is signal.** Sustained contradictions increase epistemic temperature and trigger escalation.
//! 7. **Authority is scoped, not global.** Only specific keys can correct specific contexts, ideally thresholded.
//! 8. **Witnesses matter.** Confidence comes from independent, diverse witness paths, not repetition.
//! 9. **Quarantine is mandatory.** Contested claims cannot freely drive downstream decisions.
//! 10. **All decisions are replayable.** A decision must reference the exact events it depended on.
//! 11. **Equivocation is detectable.** The system must make it hard to show different histories to different peers.
//! 12. **Local learning is allowed.** But learning outputs must be attributable, challengeable, and rollbackable via deprecation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    RAC Adversarial Coherence Layer                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
//! │  │ Event Log   │  │  Coherence  │  │  Authority  │  │  Dispute  │  │
//! │  │ (Merkle)    │──│   Engine    │──│   Policy    │──│   Engine  │  │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
//! │  │  Ruvector   │  │  Quarantine │  │   Audit     │                 │
//! │  │  Routing    │  │   Manager   │  │   Proofs    │                 │
//! │  └─────────────┘  └─────────────┘  └─────────────┘                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## References
//!
//! - [FLP Impossibility](https://groups.csail.mit.edu/tds/papers/Lynch/jacm85.pdf) - Distributed consensus limits
//! - [PBFT](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf) - Byzantine fault tolerance
//! - [CRDTs](https://pages.lip6.fr/Marc.Shapiro/papers/RR-7687.pdf) - Conflict-free replicated data types
//! - [RFC 6962](https://www.rfc-editor.org/rfc/rfc6962.html) - Certificate Transparency (Merkle logs)

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::RwLock;

// ============================================================================
// Core Types (from Adversarial Coherence Thesis)
// ============================================================================

/// 32-byte context identifier
pub type ContextId = [u8; 32];

/// 32-byte event identifier (hash of event bytes)
pub type EventId = [u8; 32];

/// 32-byte public key bytes
pub type PublicKeyBytes = [u8; 32];

/// 64-byte signature bytes (Ed25519) - using Vec for serde compatibility
pub type SignatureBytes = Vec<u8>;

/// RuVector embedding for semantic routing and clustering
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ruvector {
    /// Vector dimensions (quantized for efficiency)
    pub dims: Vec<f32>,
}

impl Ruvector {
    /// Create a new RuVector
    pub fn new(dims: Vec<f32>) -> Self {
        Self { dims }
    }

    /// Calculate cosine similarity to another RuVector
    pub fn similarity(&self, other: &Ruvector) -> f64 {
        if self.dims.len() != other.dims.len() {
            return 0.0;
        }

        let dot: f32 = self.dims.iter().zip(&other.dims).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.dims.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.dims.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)) as f64
    }

    /// Compute semantic drift from a baseline
    pub fn drift_from(&self, baseline: &Ruvector) -> f64 {
        1.0 - self.similarity(baseline)
    }
}

/// Evidence reference for claims
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Kind of evidence: "url", "hash", "sensor", "dataset", "log"
    pub kind: String,
    /// Pointer bytes (hash/uri/etc)
    pub pointer: Vec<u8>,
}

impl EvidenceRef {
    /// Create a hash evidence reference
    pub fn hash(hash: &[u8]) -> Self {
        Self {
            kind: "hash".to_string(),
            pointer: hash.to_vec(),
        }
    }

    /// Create a URL evidence reference
    pub fn url(url: &str) -> Self {
        Self {
            kind: "url".to_string(),
            pointer: url.as_bytes().to_vec(),
        }
    }
}

// ============================================================================
// Event Types (Axiom 2: Everything is an event)
// ============================================================================

/// Assertion event - a claim being made
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssertEvent {
    /// Proposition bytes (CBOR/JSON/protobuf)
    pub proposition: Vec<u8>,
    /// Evidence supporting the claim
    pub evidence: Vec<EvidenceRef>,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Expiration timestamp (optional)
    pub expires_at_unix_ms: Option<u64>,
}

/// Challenge event - opening a dispute
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChallengeEvent {
    /// Conflict identifier
    pub conflict_id: [u8; 32],
    /// Claim IDs involved in the conflict
    pub claim_ids: Vec<EventId>,
    /// Reason for the challenge
    pub reason: String,
    /// Requested proof types
    pub requested_proofs: Vec<String>,
}

/// Support event - providing evidence for a disputed claim
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupportEvent {
    /// Conflict being supported
    pub conflict_id: [u8; 32],
    /// Claim being supported
    pub claim_id: EventId,
    /// Supporting evidence
    pub evidence: Vec<EvidenceRef>,
    /// Cost/stake/work score
    pub cost: u64,
}

/// Resolution event - concluding a dispute
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolutionEvent {
    /// Conflict being resolved
    pub conflict_id: [u8; 32],
    /// Accepted claim IDs
    pub accepted: Vec<EventId>,
    /// Deprecated claim IDs
    pub deprecated: Vec<EventId>,
    /// Rationale references
    pub rationale: Vec<EvidenceRef>,
    /// Authority signatures
    pub authority_sigs: Vec<SignatureBytes>,
}

/// Deprecation event (Axiom 3: No destructive edits)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeprecateEvent {
    /// Claim being deprecated
    pub claim_id: EventId,
    /// Resolution that triggered deprecation
    pub by_resolution: [u8; 32],
    /// Superseding claim (if any)
    pub superseded_by: Option<EventId>,
}

/// Event kind enumeration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventKind {
    Assert(AssertEvent),
    Challenge(ChallengeEvent),
    Support(SupportEvent),
    Resolution(ResolutionEvent),
    Deprecate(DeprecateEvent),
}

/// A signed, logged event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    /// Event ID (hash of content)
    pub id: EventId,
    /// Previous event in chain (optional)
    pub prev: Option<EventId>,
    /// Timestamp (ms since epoch)
    pub ts_unix_ms: u64,
    /// Author's public key
    pub author: PublicKeyBytes,
    /// Context binding (Axiom 4: Every claim is scoped)
    pub context: ContextId,
    /// Semantic embedding for routing
    pub ruvector: Ruvector,
    /// Event payload
    pub kind: EventKind,
    /// Author's signature
    pub sig: SignatureBytes,
}

// ============================================================================
// Merkle Event Log (Axiom 2, Axiom 3: Append-only, tamper-evident)
// ============================================================================

/// Append-only Merkle log for audit
#[wasm_bindgen]
pub struct EventLog {
    /// Events in order
    events: RwLock<Vec<Event>>,
    /// Current Merkle root
    root: RwLock<[u8; 32]>,
}

#[wasm_bindgen]
impl EventLog {
    /// Create a new event log
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
            root: RwLock::new([0u8; 32]),
        }
    }

    /// Get current event count
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// Check if log is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.events.read().unwrap().is_empty()
    }

    /// Get current Merkle root as hex string
    #[wasm_bindgen(js_name = getRoot)]
    pub fn get_root(&self) -> String {
        let root = self.root.read().unwrap();
        hex::encode(&*root)
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLog {
    /// Append an event to the log
    pub fn append(&self, event: Event) -> EventId {
        let mut events = self.events.write().unwrap();
        let id = event.id;
        events.push(event);

        // Update Merkle root (simplified - real impl would use proper tree)
        let mut root = self.root.write().unwrap();
        *root = self.compute_root(&events);

        id
    }

    /// Get event by ID
    pub fn get(&self, id: &EventId) -> Option<Event> {
        let events = self.events.read().unwrap();
        events.iter().find(|e| &e.id == id).cloned()
    }

    /// Get events since a timestamp
    pub fn since(&self, timestamp: u64) -> Vec<Event> {
        let events = self.events.read().unwrap();
        events.iter()
            .filter(|e| e.ts_unix_ms >= timestamp)
            .cloned()
            .collect()
    }

    /// Get events for a context
    pub fn for_context(&self, context: &ContextId) -> Vec<Event> {
        let events = self.events.read().unwrap();
        events.iter()
            .filter(|e| &e.context == context)
            .cloned()
            .collect()
    }

    /// Compute Merkle root (simplified hash chain)
    fn compute_root(&self, events: &[Event]) -> [u8; 32] {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        for event in events {
            hasher.update(&event.id);
        }
        let result = hasher.finalize();
        let mut root = [0u8; 32];
        root.copy_from_slice(&result);
        root
    }

    /// Generate inclusion proof for an event
    pub fn prove_inclusion(&self, event_id: &EventId) -> Option<InclusionProof> {
        let events = self.events.read().unwrap();
        let index = events.iter().position(|e| &e.id == event_id)?;
        let root = *self.root.read().unwrap();

        Some(InclusionProof {
            event_id: *event_id,
            index,
            root,
            // Simplified - real impl would include Merkle path
            path: Vec::new(),
        })
    }
}

/// Proof of event inclusion in log
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InclusionProof {
    pub event_id: EventId,
    pub index: usize,
    pub root: [u8; 32],
    pub path: Vec<[u8; 32]>,
}

// ============================================================================
// Conflict Detection (Axiom 6: Disagreement is signal)
// ============================================================================

/// A detected conflict between claims
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conflict {
    /// Conflict identifier
    pub id: [u8; 32],
    /// Context where conflict occurs
    pub context: ContextId,
    /// Conflicting claim IDs
    pub claim_ids: Vec<EventId>,
    /// Detected timestamp
    pub detected_at: u64,
    /// Current status
    pub status: ConflictStatus,
    /// Epistemic temperature (how heated the dispute is)
    pub temperature: f32,
}

/// Status of a conflict
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConflictStatus {
    /// Conflict detected, awaiting challenge
    Detected,
    /// Challenge opened, collecting evidence
    Challenged,
    /// Resolution proposed
    Resolving,
    /// Conflict resolved
    Resolved,
    /// Escalated to higher authority
    Escalated,
}

// ============================================================================
// Quarantine Manager (Axiom 9: Quarantine is mandatory)
// ============================================================================

/// Quarantine levels for contested claims
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuarantineLevel {
    /// Claim can be used normally
    None = 0,
    /// Claim can be used with conservative bounds
    Conservative = 1,
    /// Claim requires multiple independent confirmations
    RequiresWitness = 2,
    /// Claim cannot be used in decisions
    Blocked = 3,
}

/// Manages quarantine status of contested claims
#[wasm_bindgen]
pub struct QuarantineManager {
    /// Quarantine levels by claim ID
    levels: RwLock<HashMap<String, QuarantineLevel>>,
    /// Active conflicts by context
    conflicts: RwLock<HashMap<String, Vec<Conflict>>>,
}

#[wasm_bindgen]
impl QuarantineManager {
    /// Create a new quarantine manager
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            levels: RwLock::new(HashMap::new()),
            conflicts: RwLock::new(HashMap::new()),
        }
    }

    /// Check quarantine level for a claim
    #[wasm_bindgen(js_name = getLevel)]
    pub fn get_level(&self, claim_id: &str) -> u8 {
        let levels = self.levels.read().unwrap();
        levels.get(claim_id)
            .map(|&l| l as u8)
            .unwrap_or(0)
    }

    /// Set quarantine level
    #[wasm_bindgen(js_name = setLevel)]
    pub fn set_level(&self, claim_id: &str, level: u8) {
        let quarantine_level = match level {
            0 => QuarantineLevel::None,
            1 => QuarantineLevel::Conservative,
            2 => QuarantineLevel::RequiresWitness,
            _ => QuarantineLevel::Blocked,
        };
        self.levels.write().unwrap().insert(claim_id.to_string(), quarantine_level);
    }

    /// Check if claim can be used in decisions
    #[wasm_bindgen(js_name = canUse)]
    pub fn can_use(&self, claim_id: &str) -> bool {
        self.get_level(claim_id) < QuarantineLevel::Blocked as u8
    }

    /// Get number of quarantined claims
    #[wasm_bindgen(js_name = quarantinedCount)]
    pub fn quarantined_count(&self) -> usize {
        let levels = self.levels.read().unwrap();
        levels.values().filter(|&&l| l != QuarantineLevel::None).count()
    }
}

impl Default for QuarantineManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Authority Policy (Axiom 7: Authority is scoped, not global)
// ============================================================================

/// Authority policy for a context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScopedAuthority {
    /// Context this policy applies to
    pub context: ContextId,
    /// Authorized keys
    pub authorized_keys: Vec<PublicKeyBytes>,
    /// Threshold (k-of-n)
    pub threshold: usize,
    /// Allowed evidence types
    pub allowed_evidence: Vec<String>,
}

/// Trait for authority policy verification
pub trait AuthorityPolicy: Send + Sync {
    /// Check if a resolution is authorized for this context
    fn authorized(&self, context: &ContextId, resolution: &ResolutionEvent) -> bool;

    /// Get quarantine level for a conflict
    fn quarantine_level(&self, context: &ContextId, conflict_id: &[u8; 32]) -> QuarantineLevel;
}

/// Trait for semantic verification
pub trait Verifier: Send + Sync {
    /// Check if two assertions are incompatible
    fn incompatible(&self, context: &ContextId, a: &AssertEvent, b: &AssertEvent) -> bool;
}

// ============================================================================
// Coherence Engine (The Core Loop)
// ============================================================================

/// Statistics from the coherence engine
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoherenceStats {
    pub events_processed: usize,
    pub conflicts_detected: usize,
    pub conflicts_resolved: usize,
    pub claims_deprecated: usize,
    pub quarantined_claims: usize,
}

/// The main coherence engine running the RAC protocol
#[wasm_bindgen]
pub struct CoherenceEngine {
    /// Event log
    log: EventLog,
    /// Quarantine manager
    quarantine: QuarantineManager,
    /// Statistics
    stats: RwLock<CoherenceStats>,
    /// Active conflicts by context
    conflicts: RwLock<HashMap<String, Vec<Conflict>>>,
    /// Semantic clusters for conflict detection
    clusters: RwLock<HashMap<String, Vec<EventId>>>,
}

#[wasm_bindgen]
impl CoherenceEngine {
    /// Create a new coherence engine
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            log: EventLog::new(),
            quarantine: QuarantineManager::new(),
            stats: RwLock::new(CoherenceStats::default()),
            conflicts: RwLock::new(HashMap::new()),
            clusters: RwLock::new(HashMap::new()),
        }
    }

    /// Get event log length
    #[wasm_bindgen(js_name = eventCount)]
    pub fn event_count(&self) -> usize {
        self.log.len()
    }

    /// Get current Merkle root
    #[wasm_bindgen(js_name = getMerkleRoot)]
    pub fn get_merkle_root(&self) -> String {
        self.log.get_root()
    }

    /// Get quarantined claim count
    #[wasm_bindgen(js_name = quarantinedCount)]
    pub fn quarantined_count(&self) -> usize {
        self.quarantine.quarantined_count()
    }

    /// Get conflict count
    #[wasm_bindgen(js_name = conflictCount)]
    pub fn conflict_count(&self) -> usize {
        self.conflicts.read().unwrap().values().map(|v| v.len()).sum()
    }

    /// Get statistics as JSON
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> String {
        let stats = self.stats.read().unwrap();
        serde_json::to_string(&*stats).unwrap_or_else(|_| "{}".to_string())
    }

    /// Check quarantine level for a claim
    #[wasm_bindgen(js_name = getQuarantineLevel)]
    pub fn get_quarantine_level(&self, claim_id: &str) -> u8 {
        self.quarantine.get_level(claim_id)
    }

    /// Check if a claim can be used in decisions
    #[wasm_bindgen(js_name = canUseClaim)]
    pub fn can_use_claim(&self, claim_id: &str) -> bool {
        self.quarantine.can_use(claim_id)
    }
}

impl Default for CoherenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CoherenceEngine {
    /// Ingest an event into the coherence engine
    pub fn ingest(&mut self, event: Event) {
        // 1. Append to log
        let event_id = self.log.append(event.clone());

        // 2. Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.events_processed += 1;

        // 3. Handle based on event type
        match &event.kind {
            EventKind::Assert(_) => {
                // Add to semantic cluster for conflict detection
                let context_key = hex::encode(&event.context);
                let mut clusters = self.clusters.write().unwrap();
                clusters.entry(context_key).or_default().push(event_id);
            }
            EventKind::Challenge(challenge) => {
                // Record conflict
                let context_key = hex::encode(&event.context);
                let conflict = Conflict {
                    id: challenge.conflict_id,
                    context: event.context,
                    claim_ids: challenge.claim_ids.clone(),
                    detected_at: event.ts_unix_ms,
                    status: ConflictStatus::Challenged,
                    temperature: 0.5,
                };

                let mut conflicts = self.conflicts.write().unwrap();
                conflicts.entry(context_key).or_default().push(conflict);

                // Quarantine disputed claims
                for claim_id in &challenge.claim_ids {
                    self.quarantine.set_level(&hex::encode(claim_id), 2);
                }

                stats.conflicts_detected += 1;
            }
            EventKind::Resolution(resolution) => {
                // Apply resolution
                for claim_id in &resolution.deprecated {
                    self.quarantine.set_level(&hex::encode(claim_id), 3);
                    stats.claims_deprecated += 1;
                }

                // Remove quarantine from accepted claims
                for claim_id in &resolution.accepted {
                    self.quarantine.set_level(&hex::encode(claim_id), 0);
                }

                stats.conflicts_resolved += 1;
            }
            EventKind::Deprecate(deprecate) => {
                self.quarantine.set_level(&hex::encode(&deprecate.claim_id), 3);
                stats.claims_deprecated += 1;
            }
            EventKind::Support(_) => {
                // Support events don't change state directly
            }
        }

        stats.quarantined_claims = self.quarantine.quarantined_count();
    }

    /// Detect conflicts in a context
    pub fn detect_conflicts<V: Verifier>(
        &self,
        context: &ContextId,
        verifier: &V,
    ) -> Vec<Conflict> {
        let context_key = hex::encode(context);
        let clusters = self.clusters.read().unwrap();

        let Some(event_ids) = clusters.get(&context_key) else {
            return Vec::new();
        };

        let mut conflicts = Vec::new();

        // Check all pairs for incompatibility
        for (i, id_a) in event_ids.iter().enumerate() {
            let Some(event_a) = self.log.get(id_a) else { continue };
            let EventKind::Assert(assert_a) = &event_a.kind else { continue };

            for id_b in event_ids.iter().skip(i + 1) {
                let Some(event_b) = self.log.get(id_b) else { continue };
                let EventKind::Assert(assert_b) = &event_b.kind else { continue };

                if verifier.incompatible(context, assert_a, assert_b) {
                    let mut conflict_id = [0u8; 32];
                    // Generate conflict ID from claim IDs
                    for (i, b) in id_a.iter().enumerate() {
                        conflict_id[i % 32] ^= b ^ id_b[i % 32];
                    }

                    conflicts.push(Conflict {
                        id: conflict_id,
                        context: *context,
                        claim_ids: vec![*id_a, *id_b],
                        detected_at: js_sys::Date::now() as u64,
                        status: ConflictStatus::Detected,
                        temperature: 0.3,
                    });
                }
            }
        }

        conflicts
    }

    /// Get audit proof for event inclusion
    pub fn prove_inclusion(&self, event_id: &EventId) -> Option<InclusionProof> {
        self.log.prove_inclusion(event_id)
    }
}

// ============================================================================
// Decision Trace (Axiom 10: All decisions are replayable)
// ============================================================================

/// A replayable decision trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Decision ID
    pub id: [u8; 32],
    /// Events this decision depends on
    pub dependencies: Vec<EventId>,
    /// Decision timestamp
    pub timestamp: u64,
    /// Whether any dependencies are disputed
    pub has_disputed: bool,
    /// Quarantine policy used
    pub quarantine_policy: String,
    /// Decision outcome
    pub outcome: Vec<u8>,
}

impl DecisionTrace {
    /// Create a new decision trace
    pub fn new(dependencies: Vec<EventId>, outcome: Vec<u8>) -> Self {
        use sha2::{Sha256, Digest};

        // Generate decision ID from dependencies
        let mut hasher = Sha256::new();
        for dep in &dependencies {
            hasher.update(dep);
        }
        hasher.update(&outcome);
        let result = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&result);

        Self {
            id,
            dependencies,
            timestamp: js_sys::Date::now() as u64,
            has_disputed: false,
            quarantine_policy: "default".to_string(),
            outcome,
        }
    }

    /// Check if decision can be replayed given current state
    pub fn can_replay(&self, engine: &CoherenceEngine) -> bool {
        // All dependencies must exist and be usable
        for dep in &self.dependencies {
            let dep_hex = hex::encode(dep);
            if !engine.can_use_claim(&dep_hex) {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ruvector_similarity() {
        let v1 = Ruvector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Ruvector::new(vec![1.0, 0.0, 0.0]);
        let v3 = Ruvector::new(vec![0.0, 1.0, 0.0]);

        assert!((v1.similarity(&v2) - 1.0).abs() < 0.001);
        assert!((v1.similarity(&v3) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ruvector_drift() {
        let baseline = Ruvector::new(vec![1.0, 0.0, 0.0]);
        let drifted = Ruvector::new(vec![0.707, 0.707, 0.0]);

        let drift = drifted.drift_from(&baseline);
        assert!(drift > 0.2 && drift < 0.4);
    }

    #[test]
    fn test_event_log() {
        let log = EventLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_quarantine_manager() {
        let manager = QuarantineManager::new();

        assert!(manager.can_use("claim-1"));
        assert_eq!(manager.get_level("claim-1"), 0);

        manager.set_level("claim-1", 3);
        assert!(!manager.can_use("claim-1"));
        assert_eq!(manager.get_level("claim-1"), 3);

        assert_eq!(manager.quarantined_count(), 1);
    }

    #[test]
    fn test_coherence_engine() {
        let engine = CoherenceEngine::new();

        assert_eq!(engine.event_count(), 0);
        assert_eq!(engine.conflict_count(), 0);
        assert_eq!(engine.quarantined_count(), 0);
    }

    #[test]
    fn test_evidence_ref() {
        let hash_evidence = EvidenceRef::hash(&[1, 2, 3]);
        assert_eq!(hash_evidence.kind, "hash");

        let url_evidence = EvidenceRef::url("https://example.com");
        assert_eq!(url_evidence.kind, "url");
    }

    #[test]
    fn test_conflict_status() {
        let status = ConflictStatus::Detected;
        assert_eq!(status, ConflictStatus::Detected);
    }
}
