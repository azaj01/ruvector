# Claude Code v1.0.128 (1.0 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 1.0.128 |
| Series | 1.0 |
| Bundle size | 8.9MB |
| RVF size | 2388.4KB |
| Vectors | 4669 |
| RVF File ID | `91f11f8d2829...` |
| Classes | 1390 |
| Functions | 16593 |
| Modules | 17 |
| Coverage | 100.4% |
| Extracted | 2026-04-03T03:17:09.126Z |

## Source Metrics

| Metric | Value |
|--------|-------|
| Lines | 3783 |
| Async functions | 623 |
| Arrow functions | 19160 |
| Class extensions | 879 |
| const declarations | 33 |
| let declarations | 15619 |
| var declarations | 15269 |
| imports | 85 |
| exports | 45 |
| requires | 4 |
| await expressions | 2726 |
| try blocks | 1714 |

## Modules

| Module | Fragments | Size | Confidence |
|--------|-----------|------|------------|
| model-provider | 131 | 258.4KB | 0.316 |
| session | 257 | 497.9KB | 0.62 |
| commands | 58 | 263.1KB | 0.14 |
| config | 314 | 729.5KB | 0.758 |
| streaming-handler | 19 | 35.8KB | 0.046 |
| telemetry | 321 | 861.6KB | 0.775 |
| context-manager | 12 | 57.3KB | 0.029 |
| permission-system | 160 | 396.5KB | 0.386 |
| tool-dispatch | 165 | 428.0KB | 0.398 |
| mcp-client | 18 | 32.4KB | 0.043 |
| agent-loop | 32 | 64.1KB | 0.077 |
| telemetry-events | 319 | 10.0KB | 0.5 |
| command-defs | 32 | 2.4KB | 0.5 |
| class-hierarchy | 810 | 12.4KB | 0.5 |
| env-vars | 49 | 1.5KB | 0.5 |
| api-endpoints | 24 | 0.4KB | 0.5 |
| uncategorized | 2655 | 5525.0KB | 0.1 |

## Directory Structure

```
v1.0.x/
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
