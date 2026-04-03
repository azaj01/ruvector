# Claude Code v2.0.77 (2.0 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 2.0.77 |
| Series | 2.0 |
| Bundle size | 10.5MB |
| RVF size | 2918.0KB |
| Vectors | 5712 |
| RVF File ID | `91f1209a7339...` |
| Classes | 1612 |
| Functions | 20395 |
| Modules | 17 |
| Coverage | 100.5% |
| Extracted | 2026-04-03T03:17:13.618Z |

## Source Metrics

| Metric | Value |
|--------|-------|
| Lines | 5189 |
| Async functions | 1038 |
| Arrow functions | 20920 |
| Class extensions | 861 |
| const declarations | 37 |
| let declarations | 21228 |
| var declarations | 17663 |
| imports | 105 |
| exports | 58 |
| requires | 13 |
| await expressions | 3320 |
| try blocks | 2357 |

## Modules

| Module | Fragments | Size | Confidence |
|--------|-----------|------|------------|
| model-provider | 130 | 261.4KB | 0.267 |
| config | 337 | 777.6KB | 0.692 |
| tool-dispatch | 257 | 630.8KB | 0.528 |
| permission-system | 278 | 614.7KB | 0.571 |
| session | 256 | 516.0KB | 0.525 |
| telemetry | 435 | 1091.7KB | 0.893 |
| commands | 57 | 255.8KB | 0.117 |
| streaming-handler | 15 | 28.7KB | 0.031 |
| context-manager | 19 | 71.7KB | 0.039 |
| agent-loop | 38 | 72.0KB | 0.078 |
| mcp-client | 32 | 61.1KB | 0.066 |
| telemetry-events | 479 | 15.1KB | 0.5 |
| command-defs | 45 | 3.3KB | 0.5 |
| class-hierarchy | 1440 | 22.1KB | 0.5 |
| env-vars | 92 | 2.7KB | 0.5 |
| api-endpoints | 34 | 0.6KB | 0.5 |
| uncategorized | 3018 | 6382.8KB | 0.1 |

## Directory Structure

```
v2.0.x/
  source/               # Source code only (no .rvf files)
    core/               # agent-loop, context-manager, streaming-handler, session
    tools/              # tool-dispatch
    tools/mcp/          # mcp-client
    permissions/        # permission-system
    ui/                 # commands, command-defs
    config/             # config, env-vars, model-provider
    telemetry/          # telemetry, telemetry-events
    types/              # class-hierarchy, api-endpoints
    uncategorized/      # remaining bundle code
    witness.json        # SHA-256 witness chain

  rvf/                  # RVF containers only (no .js files)
    master.rvf          # All vectors combined
    core.rvf            # Core modules only
    tools.rvf           # Tool modules only
    permissions.rvf     # Permission modules only
    config.rvf          # Configuration modules only
    telemetry.rvf       # Telemetry modules only
    ...

  metrics.json          # Overall metrics
```

## RVF Container Details

Each `.rvf` file is a binary container with:

- **128-dimensional fingerprint vectors** for each code fragment
- **HNSW index** (M=16, ef_construction=200) for fast similarity search
- **Cosine distance** metric
- **Witness chain** for provenance verification

```typescript
import { RvfDatabase } from '@ruvector/rvf';

const db = await RvfDatabase.openReadonly('./rvf/master.rvf');
const results = await db.query(queryVector, 10);
await db.close();
```
