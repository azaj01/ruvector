# Claude Code v2.1.91 (2.1 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 2.1.91 |
| Series | 2.1 |
| Bundle size | 12.6MB |
| RVF size | 4617.2KB |
| Vectors | 9058 |
| RVF File ID | `91f120e4b8c0...` |
| Classes | 1632 |
| Functions | 19906 |
| Modules | 17 |
| Coverage | 100.6% |
| Extracted | 2026-04-03T03:17:17.277Z |

## Source Metrics

| Metric | Value |
|--------|-------|
| Lines | 16967 |
| Async functions | 1903 |
| Arrow functions | 25893 |
| Class extensions | 852 |
| const declarations | 230 |
| let declarations | 33496 |
| var declarations | 14837 |
| imports | 290 |
| exports | 50 |
| requires | 8 |
| await expressions | 6541 |
| try blocks | 2790 |

## Modules

| Module | Fragments | Size | Confidence |
|--------|-----------|------|------------|
| model-provider | 165 | 331.2KB | 0.275 |
| tool-dispatch | 531 | 1304.7KB | 0.885 |
| session | 361 | 723.5KB | 0.602 |
| agent-loop | 77 | 143.7KB | 0.128 |
| permission-system | 500 | 1072.8KB | 0.834 |
| context-manager | 49 | 124.9KB | 0.082 |
| config | 473 | 1039.8KB | 0.789 |
| streaming-handler | 24 | 48.0KB | 0.04 |
| telemetry | 524 | 1068.6KB | 0.874 |
| mcp-client | 51 | 107.6KB | 0.085 |
| commands | 80 | 170.3KB | 0.133 |
| telemetry-events | 861 | 26.2KB | 0.5 |
| command-defs | 93 | 8.1KB | 0.5 |
| class-hierarchy | 1467 | 22.5KB | 0.5 |
| env-vars | 223 | 6.5KB | 0.5 |
| api-endpoints | 52 | 0.9KB | 0.5 |
| uncategorized | 3162 | 6731.4KB | 0.1 |

## Directory Structure

```
v2.1.x/
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
