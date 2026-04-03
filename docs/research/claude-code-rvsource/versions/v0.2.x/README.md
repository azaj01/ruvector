# Claude Code v0.2.126 (0.2 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 0.2.126 |
| Series | 0.2 |
| Bundle size | 6.9MB |
| RVF size | 1731.3KB |
| Vectors | 3375 |
| RVF File ID | `91f11e1ac2d0...` |
| Classes | 1049 |
| Functions | 13869 |
| Modules | 17 |
| Coverage | 100.3% |
| Extracted | 2026-04-03T03:17:05.079Z |

## Source Metrics

| Metric | Value |
|--------|-------|
| Lines | 2354 |
| Async functions | 350 |
| Arrow functions | 13308 |
| Class extensions | 571 |
| const declarations | 33 |
| let declarations | 10413 |
| var declarations | 11872 |
| imports | 78 |
| exports | 30 |
| requires | 5 |
| await expressions | 1416 |
| try blocks | 1176 |

## Modules

| Module | Fragments | Size | Confidence |
|--------|-----------|------|------------|
| model-provider | 95 | 187.8KB | 0.314 |
| telemetry | 266 | 758.9KB | 0.879 |
| permission-system | 91 | 228.6KB | 0.301 |
| session | 149 | 300.3KB | 0.493 |
| config | 156 | 429.0KB | 0.516 |
| commands | 34 | 204.8KB | 0.112 |
| tool-dispatch | 101 | 314.2KB | 0.334 |
| streaming-handler | 14 | 26.6KB | 0.046 |
| context-manager | 10 | 47.1KB | 0.033 |
| agent-loop | 15 | 28.9KB | 0.05 |
| mcp-client | 8 | 14.9KB | 0.026 |
| telemetry-events | 161 | 4.7KB | 0.5 |
| command-defs | 21 | 1.7KB | 0.5 |
| class-hierarchy | 719 | 10.7KB | 0.5 |
| env-vars | 16 | 0.5KB | 0.5 |
| api-endpoints | 20 | 0.4KB | 0.5 |
| uncategorized | 2086 | 4535.2KB | 0.1 |

## Directory Structure

```
v0.2.x/
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
