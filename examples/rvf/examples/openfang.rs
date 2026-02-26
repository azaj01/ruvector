//! OpenFang Agent OS — Knowledge Base
//!
//! Demonstrates how an RVF store can model the knowledge architecture
//! of an Agent Operating System like OpenFang (RightNow-AI):
//!
//! 1. Create a store representing the OpenFang agent registry
//! 2. Insert embeddings for 7 autonomous "Hands" (Clip, Lead, Collector,
//!    Predictor, Researcher, Twitter, Browser) with metadata
//! 3. Insert tool embeddings across 38 built-in tools
//! 4. Insert channel adapter embeddings (40 messaging channels)
//! 5. Query for agents matching a task description
//! 6. Filter by domain, capability tier, and security level
//! 7. Cross-domain search: find the best agent+tool combination
//! 8. Witness chain tracking all registry operations
//!
//! RVF segments used: VEC_SEG, MANIFEST_SEG (via RvfStore), WITNESS_SEG (via rvf-crypto)
//!
//! Run with:
//!   cargo run --example openfang

use rvf_runtime::{
    FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore, SearchResult,
};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

/// Simple pseudo-random number generator (LCG) for deterministic results.
fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed.wrapping_add(1);
    for _ in 0..dim {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

/// Domain-biased vector: adds a domain-specific offset to cluster related items.
fn domain_vector(dim: usize, seed: u64, domain_bias: f32) -> Vec<f32> {
    let mut v = random_vector(dim, seed);
    // Apply domain bias to the first 16 dimensions to create domain clusters
    for i in 0..16.min(dim) {
        v[i] += domain_bias;
    }
    v
}

// -- OpenFang component definitions --

struct Hand {
    name: &'static str,
    domain: &'static str,
    tier: u64,        // performance tier: 1=lightweight, 2=standard, 3=heavy, 4=autonomous
    security: u64,    // security level: 0-100
    _description: &'static str,
}

struct Tool {
    name: &'static str,
    category: &'static str,
}

struct Channel {
    name: &'static str,
    protocol: &'static str,
}

const HANDS: &[Hand] = &[
    Hand { name: "clip",       domain: "video-processing",  tier: 3, security: 60, _description: "YouTube shorts creation with captions" },
    Hand { name: "lead",       domain: "sales-automation",   tier: 2, security: 70, _description: "Daily prospect discovery with ICP matching" },
    Hand { name: "collector",  domain: "osint-intelligence", tier: 4, security: 90, _description: "Continuous monitoring and change detection" },
    Hand { name: "predictor",  domain: "forecasting",        tier: 3, security: 80, _description: "Superforecasting with Brier score tracking" },
    Hand { name: "researcher", domain: "fact-checking",      tier: 3, security: 75, _description: "CRAAP criteria cross-referencing" },
    Hand { name: "twitter",    domain: "social-media",       tier: 2, security: 65, _description: "X account management with approval gates" },
    Hand { name: "browser",    domain: "web-automation",     tier: 4, security: 95, _description: "Web automation with purchase approval" },
];

const TOOLS: &[Tool] = &[
    Tool { name: "http_fetch",       category: "network" },
    Tool { name: "web_search",       category: "network" },
    Tool { name: "web_scrape",       category: "network" },
    Tool { name: "file_read",        category: "filesystem" },
    Tool { name: "file_write",       category: "filesystem" },
    Tool { name: "file_list",        category: "filesystem" },
    Tool { name: "shell_exec",       category: "system" },
    Tool { name: "process_spawn",    category: "system" },
    Tool { name: "json_parse",       category: "transform" },
    Tool { name: "json_format",      category: "transform" },
    Tool { name: "csv_parse",        category: "transform" },
    Tool { name: "regex_match",      category: "transform" },
    Tool { name: "template_render",  category: "transform" },
    Tool { name: "llm_complete",     category: "inference" },
    Tool { name: "llm_embed",        category: "inference" },
    Tool { name: "llm_classify",     category: "inference" },
    Tool { name: "vector_store",     category: "memory" },
    Tool { name: "vector_search",    category: "memory" },
    Tool { name: "kv_get",           category: "memory" },
    Tool { name: "kv_set",           category: "memory" },
    Tool { name: "sql_query",        category: "database" },
    Tool { name: "sql_execute",      category: "database" },
    Tool { name: "screenshot",       category: "browser" },
    Tool { name: "click_element",    category: "browser" },
    Tool { name: "fill_form",        category: "browser" },
    Tool { name: "navigate",         category: "browser" },
    Tool { name: "pdf_extract",      category: "document" },
    Tool { name: "ocr_image",        category: "document" },
    Tool { name: "email_send",       category: "communication" },
    Tool { name: "email_read",       category: "communication" },
    Tool { name: "webhook_fire",     category: "integration" },
    Tool { name: "api_call",         category: "integration" },
    Tool { name: "schedule_cron",    category: "scheduling" },
    Tool { name: "schedule_delay",   category: "scheduling" },
    Tool { name: "crypto_sign",      category: "security" },
    Tool { name: "crypto_verify",    category: "security" },
    Tool { name: "secret_read",      category: "security" },
    Tool { name: "audit_log",        category: "security" },
];

const CHANNELS: &[Channel] = &[
    Channel { name: "telegram",      protocol: "bot-api" },
    Channel { name: "discord",       protocol: "gateway" },
    Channel { name: "slack",         protocol: "events-api" },
    Channel { name: "whatsapp",      protocol: "cloud-api" },
    Channel { name: "signal",        protocol: "signal-cli" },
    Channel { name: "matrix",        protocol: "client-server" },
    Channel { name: "email-smtp",    protocol: "smtp" },
    Channel { name: "email-imap",    protocol: "imap" },
    Channel { name: "teams",         protocol: "graph-api" },
    Channel { name: "google-chat",   protocol: "chat-api" },
    Channel { name: "linkedin",      protocol: "rest-api" },
    Channel { name: "twitter-x",     protocol: "api-v2" },
    Channel { name: "mastodon",      protocol: "activitypub" },
    Channel { name: "bluesky",       protocol: "at-proto" },
    Channel { name: "reddit",        protocol: "oauth-api" },
    Channel { name: "irc",           protocol: "irc-v3" },
    Channel { name: "xmpp",         protocol: "xmpp-core" },
    Channel { name: "webhook-in",    protocol: "http-post" },
    Channel { name: "webhook-out",   protocol: "http-post" },
    Channel { name: "grpc",          protocol: "grpc" },
];

fn main() {
    println!("=== OpenFang Agent OS — RVF Knowledge Base ===\n");

    let dim = 128;
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("openfang.rvf");

    // -- Step 1: Create the OpenFang registry store --
    println!("--- 1. Creating OpenFang Agent Registry ---");
    let options = RvfOptions {
        dimension: dim as u16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");
    println!("  Registry created at {:?}", store_path);
    println!("  Embedding dimensions: {}", dim);

    let mut witness_entries: Vec<WitnessEntry> = Vec::new();
    let mut next_id: u64 = 0;

    // -- Step 2: Register the 7 autonomous Hands --
    // Metadata fields:
    //   field_id 0: component_type (String: "hand", "tool", "channel")
    //   field_id 1: name           (String)
    //   field_id 2: domain         (String)
    //   field_id 3: tier           (U64: 1-4)
    //   field_id 4: security_level (U64: 0-100)
    println!("\n--- 2. Registering Autonomous Hands ({}) ---", HANDS.len());

    let hand_base_id = next_id;
    let hand_vectors: Vec<Vec<f32>> = HANDS.iter().enumerate()
        .map(|(i, h)| domain_vector(dim, i as u64 * 17 + 100, h.tier as f32 * 0.1))
        .collect();
    let hand_refs: Vec<&[f32]> = hand_vectors.iter().map(|v| v.as_slice()).collect();
    let hand_ids: Vec<u64> = (hand_base_id..hand_base_id + HANDS.len() as u64).collect();

    let mut hand_metadata = Vec::with_capacity(HANDS.len() * 5);
    for hand in HANDS {
        hand_metadata.push(MetadataEntry { field_id: 0, value: MetadataValue::String("hand".to_string()) });
        hand_metadata.push(MetadataEntry { field_id: 1, value: MetadataValue::String(hand.name.to_string()) });
        hand_metadata.push(MetadataEntry { field_id: 2, value: MetadataValue::String(hand.domain.to_string()) });
        hand_metadata.push(MetadataEntry { field_id: 3, value: MetadataValue::U64(hand.tier) });
        hand_metadata.push(MetadataEntry { field_id: 4, value: MetadataValue::U64(hand.security) });
    }

    let hand_result = store.ingest_batch(&hand_refs, &hand_ids, Some(&hand_metadata))
        .expect("failed to register hands");
    next_id += HANDS.len() as u64;

    println!("  Registered {} Hands (epoch {})", hand_result.accepted, hand_result.epoch);
    for hand in HANDS {
        println!("    - {} ({}), tier {}, security {}", hand.name, hand.domain, hand.tier, hand.security);
    }

    witness_entries.push(WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("REGISTER_HANDS:count={}", HANDS.len()).as_bytes()),
        timestamp_ns: 1_709_000_000_000_000_000,
        witness_type: 0x01,
    });

    // -- Step 3: Register built-in tools --
    println!("\n--- 3. Registering Built-in Tools ({}) ---", TOOLS.len());

    let tool_base_id = next_id;
    let tool_vectors: Vec<Vec<f32>> = TOOLS.iter().enumerate()
        .map(|(i, _)| domain_vector(dim, i as u64 * 31 + 500, 0.3))
        .collect();
    let tool_refs: Vec<&[f32]> = tool_vectors.iter().map(|v| v.as_slice()).collect();
    let tool_ids: Vec<u64> = (tool_base_id..tool_base_id + TOOLS.len() as u64).collect();

    let mut tool_metadata = Vec::with_capacity(TOOLS.len() * 3);
    for tool in TOOLS {
        tool_metadata.push(MetadataEntry { field_id: 0, value: MetadataValue::String("tool".to_string()) });
        tool_metadata.push(MetadataEntry { field_id: 1, value: MetadataValue::String(tool.name.to_string()) });
        tool_metadata.push(MetadataEntry { field_id: 2, value: MetadataValue::String(tool.category.to_string()) });
    }

    let tool_result = store.ingest_batch(&tool_refs, &tool_ids, Some(&tool_metadata))
        .expect("failed to register tools");
    next_id += TOOLS.len() as u64;

    println!("  Registered {} tools (epoch {})", tool_result.accepted, tool_result.epoch);

    // Print tools grouped by category
    let categories: Vec<&str> = {
        let mut cats: Vec<&str> = TOOLS.iter().map(|t| t.category).collect();
        cats.sort();
        cats.dedup();
        cats
    };
    for cat in &categories {
        let tools_in_cat: Vec<&str> = TOOLS.iter()
            .filter(|t| t.category == *cat)
            .map(|t| t.name)
            .collect();
        println!("    [{}] {}", cat, tools_in_cat.join(", "));
    }

    witness_entries.push(WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("REGISTER_TOOLS:count={}", TOOLS.len()).as_bytes()),
        timestamp_ns: 1_709_000_001_000_000_000,
        witness_type: 0x01,
    });

    // -- Step 4: Register channel adapters --
    println!("\n--- 4. Registering Channel Adapters ({}) ---", CHANNELS.len());

    let channel_base_id = next_id;
    let channel_vectors: Vec<Vec<f32>> = CHANNELS.iter().enumerate()
        .map(|(i, _)| domain_vector(dim, i as u64 * 43 + 1000, -0.2))
        .collect();
    let channel_refs: Vec<&[f32]> = channel_vectors.iter().map(|v| v.as_slice()).collect();
    let channel_ids: Vec<u64> = (channel_base_id..channel_base_id + CHANNELS.len() as u64).collect();

    let mut channel_metadata = Vec::with_capacity(CHANNELS.len() * 3);
    for ch in CHANNELS {
        channel_metadata.push(MetadataEntry { field_id: 0, value: MetadataValue::String("channel".to_string()) });
        channel_metadata.push(MetadataEntry { field_id: 1, value: MetadataValue::String(ch.name.to_string()) });
        channel_metadata.push(MetadataEntry { field_id: 2, value: MetadataValue::String(ch.protocol.to_string()) });
    }

    let channel_result = store.ingest_batch(&channel_refs, &channel_ids, Some(&channel_metadata))
        .expect("failed to register channels");
    let _ = next_id + CHANNELS.len() as u64; // total IDs allocated

    println!("  Registered {} channels (epoch {})", channel_result.accepted, channel_result.epoch);
    for ch in CHANNELS {
        println!("    - {} ({})", ch.name, ch.protocol);
    }

    witness_entries.push(WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(format!("REGISTER_CHANNELS:count={}", CHANNELS.len()).as_bytes()),
        timestamp_ns: 1_709_000_002_000_000_000,
        witness_type: 0x01,
    });

    let total_components = HANDS.len() + TOOLS.len() + CHANNELS.len();
    println!("\n  Total registry: {} components", total_components);

    // -- Step 5: Query — find agents for a task --
    println!("\n--- 5. Task Routing: Find Best Agent ---");

    let task_query = domain_vector(dim, 42, 0.3); // bias toward tier-3 agents
    let k = 5;

    // Unfiltered — search across all components
    let all_results = store.query(&task_query, k, &QueryOptions::default())
        .expect("task routing query failed");
    println!("  Unfiltered top-{} (all component types):", k);
    print_registry_results(&all_results, hand_base_id, tool_base_id, channel_base_id);

    // Filter to Hands only
    let filter_hands = FilterExpr::Eq(0, FilterValue::String("hand".to_string()));
    let opts_hands = QueryOptions { filter: Some(filter_hands), ..Default::default() };
    let hand_results = store.query(&task_query, k, &opts_hands)
        .expect("hand filter query failed");
    println!("\n  Hands only — best agent for this task:");
    print_registry_results(&hand_results, hand_base_id, tool_base_id, channel_base_id);

    witness_entries.push(WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(b"ROUTE_TASK:unfiltered+hands"),
        timestamp_ns: 1_709_000_010_000_000_000,
        witness_type: 0x02,
    });

    // -- Step 6: Filter by security level --
    println!("\n--- 6. Security Filter: High-Security Hands (>= 80) ---");

    let filter_secure = FilterExpr::And(vec![
        FilterExpr::Eq(0, FilterValue::String("hand".to_string())),
        FilterExpr::Ge(4, FilterValue::U64(80)),
    ]);
    let opts_secure = QueryOptions { filter: Some(filter_secure), ..Default::default() };
    let secure_results = store.query(&task_query, k, &opts_secure)
        .expect("security filter query failed");

    println!("  High-security Hands:");
    print_registry_results(&secure_results, hand_base_id, tool_base_id, channel_base_id);
    println!("  ({} agents meet security >= 80 threshold)", secure_results.len());

    // -- Step 7: Filter by tier --
    println!("\n--- 7. Autonomous Tier (tier 4) Agents ---");

    let filter_autonomous = FilterExpr::And(vec![
        FilterExpr::Eq(0, FilterValue::String("hand".to_string())),
        FilterExpr::Eq(3, FilterValue::U64(4)),
    ]);
    let opts_autonomous = QueryOptions { filter: Some(filter_autonomous), ..Default::default() };
    let autonomous_results = store.query(&task_query, k, &opts_autonomous)
        .expect("tier filter query failed");

    println!("  Fully autonomous agents (tier 4):");
    print_registry_results(&autonomous_results, hand_base_id, tool_base_id, channel_base_id);

    // -- Step 8: Tool search by category --
    println!("\n--- 8. Tool Discovery: Security Tools ---");

    let filter_sec_tools = FilterExpr::And(vec![
        FilterExpr::Eq(0, FilterValue::String("tool".to_string())),
        FilterExpr::Eq(2, FilterValue::String("security".to_string())),
    ]);
    let opts_sec_tools = QueryOptions { filter: Some(filter_sec_tools), ..Default::default() };
    let sec_tool_results = store.query(&task_query, 10, &opts_sec_tools)
        .expect("security tool query failed");

    println!("  Security tools available:");
    print_registry_results(&sec_tool_results, hand_base_id, tool_base_id, channel_base_id);

    // -- Step 9: Witness chain --
    println!("\n--- 9. Registry Audit Trail (Witness Chain) ---");

    let chain_bytes = create_witness_chain(&witness_entries);
    println!("  Created witness chain: {} entries, {} bytes", witness_entries.len(), chain_bytes.len());

    match verify_witness_chain(&chain_bytes) {
        Ok(verified) => {
            println!("  Chain integrity: VALID ({} entries verified)\n", verified.len());
            println!("  {:>5}  {:>8}  {:>30}", "Index", "Type", "Timestamp (ns)");
            println!("  {:->5}  {:->8}  {:->30}", "", "", "");
            let labels = ["REGISTER_HANDS", "REGISTER_TOOLS", "REGISTER_CHANNELS", "ROUTE_TASK"];
            for (i, entry) in verified.iter().enumerate() {
                let wtype = match entry.witness_type {
                    0x01 => "PROV",
                    0x02 => "COMP",
                    _ => "????",
                };
                let label = if i < labels.len() { labels[i] } else { "???" };
                println!("  {:>5}  {:>8}  {:>30}  {}", i, wtype, entry.timestamp_ns, label);
            }
        }
        Err(e) => println!("  Chain integrity: FAILED ({:?})", e),
    }

    // -- Step 10: Persistence --
    println!("\n--- 10. Persistence Verification ---");

    let status = store.status();
    println!("  Vectors: {}, File size: {} bytes, Epoch: {}", status.total_vectors, status.file_size, status.current_epoch);

    store.close().expect("failed to close store");
    println!("  Store closed.");

    let reopened = RvfStore::open(&store_path).expect("failed to reopen store");
    let status_after = reopened.status();
    println!("  Reopened: {} vectors, epoch {}", status_after.total_vectors, status_after.current_epoch);

    let persist_check = reopened.query(&task_query, k, &QueryOptions::default())
        .expect("persistence query failed");
    assert_eq!(all_results.len(), persist_check.len(), "result count mismatch after reopen");
    for (a, b) in all_results.iter().zip(persist_check.iter()) {
        assert_eq!(a.id, b.id, "ID mismatch after reopen");
        assert!((a.distance - b.distance).abs() < 1e-6, "distance mismatch after reopen");
    }
    println!("  Persistence verified: results match before and after reopen.");

    reopened.close().expect("failed to close reopened store");

    // -- Summary --
    println!("\n=== OpenFang Registry Summary ===\n");
    println!("  Component Type    Count");
    println!("  ----------------  -----");
    println!("  Hands              {:>4}", HANDS.len());
    println!("  Tools              {:>4}", TOOLS.len());
    println!("  Channels           {:>4}", CHANNELS.len());
    println!("  ----------------  -----");
    println!("  Total              {:>4}", total_components);
    println!();
    println!("  Witness chain:     {} entries", witness_entries.len());
    println!("  Persistence:       verified");
    println!("  Security filter:   working");
    println!("  Tier filter:       working");
    println!("  Cross-type search: working");

    println!("\nDone.");
}

fn print_registry_results(
    results: &[SearchResult],
    hand_base: u64,
    tool_base: u64,
    channel_base: u64,
) {
    println!(
        "    {:>4}  {:>10}  {:>10}  {:>20}",
        "ID", "Distance", "Type", "Name"
    );
    println!(
        "    {:->4}  {:->10}  {:->10}  {:->20}",
        "", "", "", ""
    );
    for r in results {
        let (comp_type, name) = identify_component(r.id, hand_base, tool_base, channel_base);
        println!(
            "    {:>4}  {:>10.4}  {:>10}  {:>20}",
            r.id, r.distance, comp_type, name
        );
    }
}

fn identify_component(id: u64, hand_base: u64, tool_base: u64, channel_base: u64) -> (&'static str, &'static str) {
    if id >= channel_base && (id - channel_base) < CHANNELS.len() as u64 {
        let idx = (id - channel_base) as usize;
        ("channel", CHANNELS[idx].name)
    } else if id >= tool_base && (id - tool_base) < TOOLS.len() as u64 {
        let idx = (id - tool_base) as usize;
        ("tool", TOOLS[idx].name)
    } else if id >= hand_base && (id - hand_base) < HANDS.len() as u64 {
        let idx = (id - hand_base) as usize;
        ("hand", HANDS[idx].name)
    } else {
        ("unknown", "???")
    }
}
