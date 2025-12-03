# @ruvector/postgres-cli

Command-line interface for the RuVector PostgreSQL extension - an advanced AI vector database.

## Installation

```bash
npm install -g @ruvector/postgres-cli
```

## Quick Start

```bash
# Connect to your PostgreSQL database with RuVector extension
ruvector-pg -c "postgresql://user:pass@localhost:5432/mydb" info

# Install the extension
ruvector-pg install

# Create a vector table
ruvector-pg vector create embeddings --dim 384 --index hnsw

# Search vectors
ruvector-pg vector search embeddings --text "hello world" --top-k 10
```

## Commands

### Vector Operations

```bash
# Create vector table with HNSW index
ruvector-pg vector create <name> --dim <dimensions> --index <hnsw|ivfflat>

# Insert vectors from JSON file
ruvector-pg vector insert <table> --file vectors.json

# Search for similar vectors
ruvector-pg vector search <table> --query "[0.1, 0.2, ...]" --top-k 10 --metric cosine
```

### Attention Mechanisms

```bash
# Compute attention
ruvector-pg attention compute --query "[...]" --keys "[[...]]" --values "[[...]]" --type scaled_dot

# List available attention types
ruvector-pg attention list-types
```

### Graph Neural Networks

```bash
# Create GNN layer
ruvector-pg gnn create my_layer --type gcn --input-dim 384 --output-dim 128

# Forward pass
ruvector-pg gnn forward my_layer --features features.json --edges edges.json
```

### Graph & Cypher

```bash
# Execute Cypher query
ruvector-pg graph query "MATCH (n:Person) RETURN n"

# Create node
ruvector-pg graph create-node --labels "Person,Developer" --properties '{"name": "Alice"}'

# Traverse graph
ruvector-pg graph traverse --start node123 --depth 3 --type bfs
```

### Self-Learning

```bash
# Train from trajectories
ruvector-pg learning train --file trajectories.json --epochs 10

# Make prediction
ruvector-pg learning predict --input "[0.1, 0.2, ...]"
```

### Benchmarking

```bash
# Run benchmarks
ruvector-pg bench run --type all --size 10000 --dim 384

# Generate report
ruvector-pg bench report --format table
```

## Global Options

- `-c, --connection <string>` - PostgreSQL connection string (default: `postgresql://localhost:5432/ruvector`)
- `-v, --verbose` - Enable verbose output

## Features

- **Vector Search**: HNSW and IVFFlat indexes with cosine, L2, and inner product metrics
- **39 Attention Mechanisms**: Scaled dot-product, multi-head, flash, sparse, and more
- **Graph Neural Networks**: GCN, GraphSAGE, GAT, GIN layers
- **Graph Operations**: Cypher queries, BFS/DFS traversal
- **Self-Learning**: ReasoningBank-based trajectory learning
- **Hyperbolic Embeddings**: Poincaré and Lorentz models
- **Sparse Vectors**: BM25 and SPLADE for hybrid search

## License

MIT
