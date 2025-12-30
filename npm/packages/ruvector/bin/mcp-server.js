#!/usr/bin/env node

/**
 * RuVector MCP Server
 *
 * Model Context Protocol server for RuVector hooks
 * Provides self-learning intelligence tools for Claude Code
 *
 * Usage:
 *   npx ruvector mcp start
 *   claude mcp add ruvector npx ruvector mcp start
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} = require('@modelcontextprotocol/sdk/types.js');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

// Intelligence class (simplified from cli.js)
class Intelligence {
  constructor() {
    this.intelPath = this.getIntelPath();
    this.data = this.load();
  }

  getIntelPath() {
    const projectPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    const homePath = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');
    if (fs.existsSync(path.dirname(projectPath))) return projectPath;
    if (fs.existsSync(path.join(process.cwd(), '.claude'))) return projectPath;
    if (fs.existsSync(homePath)) return homePath;
    return projectPath;
  }

  load() {
    try {
      if (fs.existsSync(this.intelPath)) {
        return JSON.parse(fs.readFileSync(this.intelPath, 'utf-8'));
      }
    } catch {}
    return { patterns: {}, memories: [], trajectories: [], errors: {}, agents: {}, edges: [] };
  }

  save() {
    const dir = path.dirname(this.intelPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.intelPath, JSON.stringify(this.data, null, 2));
  }

  stats() {
    return {
      total_patterns: Object.keys(this.data.patterns || {}).length,
      total_memories: (this.data.memories || []).length,
      total_trajectories: (this.data.trajectories || []).length,
      total_errors: Object.keys(this.data.errors || {}).length
    };
  }

  embed(text) {
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
    return embedding;
  }

  similarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  remember(content, type = 'general') {
    this.data.memories = this.data.memories || [];
    this.data.memories.push({
      content,
      type,
      created: new Date().toISOString(),
      embedding: this.embed(content)
    });
    this.save();
    return { stored: true, total: this.data.memories.length };
  }

  recall(query, topK = 5) {
    const queryEmbed = this.embed(query);
    const scored = (this.data.memories || []).map((m, i) => ({
      ...m,
      index: i,
      score: this.similarity(queryEmbed, m.embedding)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  route(task, file = null) {
    const ext = file ? path.extname(file) : '';
    const state = `edit:${ext || 'unknown'}`;
    const actions = this.data.patterns[state] || {};

    // Default agent mapping
    const defaults = {
      '.rs': 'rust-developer',
      '.ts': 'typescript-developer',
      '.tsx': 'react-developer',
      '.js': 'javascript-developer',
      '.py': 'python-developer',
      '.go': 'go-developer',
      '.sql': 'database-specialist'
    };

    let bestAgent = defaults[ext] || 'coder';
    let bestScore = 0.5;

    for (const [agent, score] of Object.entries(actions)) {
      if (score > bestScore) {
        bestAgent = agent;
        bestScore = score;
      }
    }

    return {
      agent: bestAgent,
      confidence: Math.min(bestScore, 1.0),
      reason: Object.keys(actions).length > 0 ? 'learned from patterns' : 'default mapping'
    };
  }
}

// Create MCP server
const server = new Server(
  {
    name: 'ruvector',
    version: '0.1.51',
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

const intel = new Intelligence();

// Define tools
const TOOLS = [
  {
    name: 'hooks_stats',
    description: 'Get RuVector intelligence statistics including learned patterns, memories, and trajectories',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },
  {
    name: 'hooks_route',
    description: 'Route a task to the best agent based on learned patterns',
    inputSchema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'Task description' },
        file: { type: 'string', description: 'File path (optional)' }
      },
      required: ['task']
    }
  },
  {
    name: 'hooks_remember',
    description: 'Store context in vector memory for later recall',
    inputSchema: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Content to remember' },
        type: { type: 'string', description: 'Memory type (project, code, decision, context)', default: 'general' }
      },
      required: ['content']
    }
  },
  {
    name: 'hooks_recall',
    description: 'Search vector memory for relevant context',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        top_k: { type: 'number', description: 'Number of results', default: 5 }
      },
      required: ['query']
    }
  },
  {
    name: 'hooks_init',
    description: 'Initialize RuVector hooks in the current project',
    inputSchema: {
      type: 'object',
      properties: {
        pretrain: { type: 'boolean', description: 'Run pretrain after init', default: false },
        build_agents: { type: 'string', description: 'Focus for agent generation (quality, speed, security, testing, fullstack)' },
        force: { type: 'boolean', description: 'Force overwrite existing settings', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_pretrain',
    description: 'Pretrain intelligence by analyzing the repository structure and git history',
    inputSchema: {
      type: 'object',
      properties: {
        depth: { type: 'number', description: 'Git history depth to analyze', default: 100 },
        skip_git: { type: 'boolean', description: 'Skip git history analysis', default: false },
        verbose: { type: 'boolean', description: 'Show detailed progress', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_build_agents',
    description: 'Generate optimized agent configurations based on repository analysis',
    inputSchema: {
      type: 'object',
      properties: {
        focus: {
          type: 'string',
          description: 'Focus type for agent generation',
          enum: ['quality', 'speed', 'security', 'testing', 'fullstack'],
          default: 'quality'
        },
        include_prompts: { type: 'boolean', description: 'Include system prompts in agent configs', default: true }
      },
      required: []
    }
  },
  {
    name: 'hooks_verify',
    description: 'Verify that hooks are configured correctly',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },
  {
    name: 'hooks_doctor',
    description: 'Diagnose and optionally fix setup issues',
    inputSchema: {
      type: 'object',
      properties: {
        fix: { type: 'boolean', description: 'Automatically fix issues', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_export',
    description: 'Export intelligence data for backup',
    inputSchema: {
      type: 'object',
      properties: {
        include_all: { type: 'boolean', description: 'Include all data (patterns, memories, trajectories)', default: false }
      },
      required: []
    }
  }
];

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'hooks_stats': {
        const stats = intel.stats();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              stats,
              intel_path: intel.intelPath
            }, null, 2)
          }]
        };
      }

      case 'hooks_route': {
        const result = intel.route(args.task, args.file);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              task: args.task,
              file: args.file,
              ...result
            }, null, 2)
          }]
        };
      }

      case 'hooks_remember': {
        const result = intel.remember(args.content, args.type || 'general');
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              ...result
            }, null, 2)
          }]
        };
      }

      case 'hooks_recall': {
        const results = intel.recall(args.query, args.top_k || 5);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              query: args.query,
              results: results.map(r => ({
                content: r.content,
                type: r.type,
                score: r.score.toFixed(3),
                created: r.created
              }))
            }, null, 2)
          }]
        };
      }

      case 'hooks_init': {
        let cmd = 'npx ruvector hooks init';
        if (args.force) cmd += ' --force';
        if (args.pretrain) cmd += ' --pretrain';
        if (args.build_agents) cmd += ` --build-agents ${args.build_agents}`;

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 60000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_pretrain': {
        let cmd = 'npx ruvector hooks pretrain';
        if (args.depth) cmd += ` --depth ${args.depth}`;
        if (args.skip_git) cmd += ' --skip-git';
        if (args.verbose) cmd += ' --verbose';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 120000 });
          // Reload intelligence after pretrain
          intel.data = intel.load();
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({
                success: true,
                output,
                new_stats: intel.stats()
              }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_build_agents': {
        let cmd = 'npx ruvector hooks build-agents';
        if (args.focus) cmd += ` --focus ${args.focus}`;
        if (args.include_prompts) cmd += ' --include-prompts';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 30000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_verify': {
        try {
          const output = execSync('npx ruvector hooks verify', { encoding: 'utf-8', timeout: 15000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message, output: e.stdout }, null, 2)
            }]
          };
        }
      }

      case 'hooks_doctor': {
        let cmd = 'npx ruvector hooks doctor';
        if (args.fix) cmd += ' --fix';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 15000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_export': {
        const exportData = {
          version: '1.0',
          exported_at: new Date().toISOString(),
          patterns: intel.data.patterns || {},
          memories: args.include_all ? (intel.data.memories || []) : [],
          trajectories: args.include_all ? (intel.data.trajectories || []) : [],
          errors: intel.data.errors || {},
          stats: intel.stats()
        };
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, data: exportData }, null, 2)
          }]
        };
      }

      default:
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: false, error: `Unknown tool: ${name}` }, null, 2)
          }],
          isError: true
        };
    }
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ success: false, error: error.message }, null, 2)
      }],
      isError: true
    };
  }
});

// Resources - expose intelligence data
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: 'ruvector://intelligence/stats',
        name: 'Intelligence Stats',
        description: 'Current RuVector intelligence statistics',
        mimeType: 'application/json'
      },
      {
        uri: 'ruvector://intelligence/patterns',
        name: 'Learned Patterns',
        description: 'Q-learning patterns for agent routing',
        mimeType: 'application/json'
      },
      {
        uri: 'ruvector://intelligence/memories',
        name: 'Vector Memories',
        description: 'Stored context memories',
        mimeType: 'application/json'
      }
    ]
  };
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  switch (uri) {
    case 'ruvector://intelligence/stats':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(intel.stats(), null, 2)
        }]
      };

    case 'ruvector://intelligence/patterns':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(intel.data.patterns || {}, null, 2)
        }]
      };

    case 'ruvector://intelligence/memories':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify((intel.data.memories || []).map(m => ({
            content: m.content,
            type: m.type,
            created: m.created
          })), null, 2)
        }]
      };

    default:
      throw new Error(`Unknown resource: ${uri}`);
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('RuVector MCP server running on stdio');
}

main().catch(console.error);
