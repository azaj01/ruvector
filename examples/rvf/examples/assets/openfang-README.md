# OpenFang Agent OS — RVF Example

An RVF (RuVector Format) knowledge base modeling the architecture of [OpenFang](https://github.com/RightNow-AI/openfang), a production-grade Agent Operating System built in Rust.

## What It Does

This example creates an RVF vector store that serves as a **component registry** for an agent OS, demonstrating how to model and query a multi-type agent ecosystem:

| Component | Count | Description |
|-----------|-------|-------------|
| **Hands** | 7 | Autonomous agents (Clip, Lead, Collector, Predictor, Researcher, Twitter, Browser) |
| **Tools** | 38 | Built-in capabilities across 12 categories |
| **Channels** | 20 | Messaging adapters (Telegram, Discord, Slack, etc.) |
| **Total** | 65 | Searchable components in a single 35KB RVF file |

## Capabilities Demonstrated

1. **Multi-type registry** — Hands, Tools, and Channels stored in one vector space
2. **Rich metadata** — component type, name, domain, tier, security level
3. **Task routing** — nearest-neighbor search to find the best agent for a task
4. **Security filtering** — query only agents meeting a security threshold (>= 80)
5. **Tier filtering** — isolate autonomous (tier 4) agents
6. **Category search** — find tools by category (e.g., all security tools)
7. **Combined filters** — AND/OR/NOT filter expressions
8. **Witness chain** — cryptographic audit trail of all registry operations
9. **Persistence** — verified round-trip: create, close, reopen, query

## Run

```bash
cd examples/rvf
cargo run --example openfang
```

## Metadata Schema

| Field ID | Name | Type | Applies To |
|----------|------|------|------------|
| 0 | `component_type` | String | All (`"hand"`, `"tool"`, `"channel"`) |
| 1 | `name` | String | All |
| 2 | `domain` / `category` / `protocol` | String | All |
| 3 | `tier` | U64 (1-4) | Hands only |
| 4 | `security_level` | U64 (0-100) | Hands only |

## OpenFang Hands

| Hand | Domain | Tier | Security |
|------|--------|------|----------|
| clip | video-processing | 3 | 60 |
| lead | sales-automation | 2 | 70 |
| collector | osint-intelligence | 4 | 90 |
| predictor | forecasting | 3 | 80 |
| researcher | fact-checking | 3 | 75 |
| twitter | social-media | 2 | 65 |
| browser | web-automation | 4 | 95 |

## Tool Categories

`browser`, `communication`, `database`, `document`, `filesystem`, `inference`, `integration`, `memory`, `network`, `scheduling`, `security`, `system`, `transform`

## Channel Adapters

Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email (SMTP/IMAP), Teams, Google Chat, LinkedIn, Twitter/X, Mastodon, Bluesky, Reddit, IRC, XMPP, Webhooks, gRPC

## About OpenFang

[OpenFang](https://openfang.sh) by RightNow AI is a Rust-based Agent Operating System — 137K lines of code across 14 crates, compiling to a single ~32MB binary. It runs autonomous agents 24/7 with 16 security systems, 27 LLM providers, and 40 channel adapters.

- GitHub: [RightNow-AI/openfang](https://github.com/RightNow-AI/openfang)
- License: MIT / Apache 2.0
