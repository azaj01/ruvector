#!/usr/bin/env node
/**
 * module-splitter.mjs - Split a Claude Code CLI bundle into logical modules.
 *
 * Given a path to cli.js / cli.mjs, extracts recognizable subsystems
 * (tools, MCP, permissions, streaming, agent-loop, compaction, telemetry)
 * and writes individual .js files plus a metrics.json manifest.
 *
 * Usage:
 *   node scripts/lib/module-splitter.mjs <cli-bundle> <output-dir>
 */

import { readFileSync, writeFileSync, mkdirSync, statSync } from 'fs';
import { join, basename } from 'path';

// Module extraction: keyword -> module name.
// A line containing the keyword is assigned to that module.
// Order matters: first match wins for each line.
const MODULE_KEYWORDS = {
  'tool-dispatch': [
    'BashTool', 'FileReadTool', 'FileEditTool', 'FileWriteTool',
    'AgentOutputTool', 'WebFetch', 'WebSearch', 'TodoWrite',
    'NotebookEdit', 'GlobTool', 'GrepTool', 'ListFilesTool',
    'SearchTool', 'ReadTool', 'EditTool', 'WriteTool',
    'tool_use', 'tool_result', 'ToolUse', 'ToolResult',
    'toolDefinition', 'toolSchema', 'inputSchema',
  ],
  'permission-system': [
    'canUseTool', 'alwaysAllowRules', 'denyWrite',
    'Permission', 'permission', 'allowedTools',
    'permissionMode', 'sandbox', 'allowList', 'denyList',
    'isAllowed', 'checkPermission', 'grantPermission',
  ],
  'mcp-client': [
    'mcp__', 'McpClient', 'McpServer', 'McpError',
    'callTool', 'listTools', 'McpTransport',
    'StdioTransport', 'SseTransport', 'StreamableHttp',
    'mcp_server', 'mcp_client', 'mcpConnection',
  ],
  'streaming-handler': [
    'content_block_delta', 'message_start', 'message_stop',
    'message_delta', 'content_block_start', 'content_block_stop',
    'stream_event', 'text_delta', 'input_json_delta',
    'StreamEvent', 'onStream', 'streamHandler',
  ],
  'context-manager': [
    'tengu_compact', 'microcompact', 'auto_compact',
    'compact_boundary', 'preCompactTokenCount',
    'postCompactTokenCount', 'compaction',
    'tokenCount', 'contextWindow', 'maxTokens',
    'promptCache', 'cacheControl',
  ],
  'agent-loop': [
    'agentLoop', 'mainLoop', 'querySource',
    'toolUseContext', 'systemPrompt',
    'conversationTurn', 'assistantMessage',
    'userMessage', 'messageHistory',
  ],
  'commands': [
    'slashCommand', 'registerCommand', 'commandHandler',
    'parseCommand', '/help', '/clear', '/compact',
    '/bug', '/init', '/login', '/logout',
    '/doctor', '/config', '/cost', '/memory',
  ],
  'telemetry': [
    'telemetry', 'Telemetry', 'opentelemetry', 'otel',
    'datadog', 'perfetto', 'tracing', 'span',
    'metric_', 'counter_', 'histogram_',
    'tengu_', 'sentry',
  ],
  'config': [
    'settings', 'Settings', 'configuration',
    'CLAUDE_', 'environment', 'envVar',
    'dotenv', 'loadConfig', 'parseConfig',
  ],
  'session': [
    'session', 'Session', 'conversationId',
    'checkpoint', 'resume', 'restore',
    'sessionState', 'persistSession',
  ],
  'model-provider': [
    'anthropic', 'Anthropic', 'claude-', 'claude_',
    'bedrock', 'vertex', 'openai', 'provider',
    'apiKey', 'modelId', 'modelName',
  ],
};

// Simple global regex patterns for small, fast extractions.
const SIMPLE_PATTERNS = {
  'telemetry-events': /"tengu_[^"]*"/g,
  'command-defs': /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
  'env-vars': /CLAUDE_[A-Z_]+/g,
  'api-endpoints': /\/v\d+\/[a-z][-a-z/]*/g,
};

/**
 * Split source into statements (semicolon-delimited chunks).
 * For minified bundles, this gives us logical units.
 */
function splitStatements(source) {
  // Split on semicolons that are not inside strings.
  // For minified JS, simple semicolon split works well enough.
  // Limit chunk size to ~2KB for vector embedding granularity.
  const MAX_CHUNK = 2048;
  const raw = source.split(';');
  const chunks = [];
  let buffer = '';

  for (const part of raw) {
    if (buffer.length + part.length > MAX_CHUNK && buffer.length > 0) {
      chunks.push(buffer);
      buffer = part;
    } else {
      buffer += (buffer ? ';' : '') + part;
    }
  }
  if (buffer.length > 0) chunks.push(buffer);
  return chunks;
}

/**
 * Assign statements to modules based on keyword matching.
 * Unmatched statements go to _unclassified for 100% coverage.
 */
function classifyStatements(statements) {
  const modules = {};
  const unclassified = [];

  for (const stmt of statements) {
    if (stmt.length < 10) continue;

    let matched = false;
    for (const [modName, keywords] of Object.entries(MODULE_KEYWORDS)) {
      if (keywords.some((kw) => stmt.includes(kw))) {
        if (!modules[modName]) modules[modName] = [];
        modules[modName].push(stmt.trim());
        matched = true;
        break; // first-match wins
      }
    }

    if (!matched) {
      unclassified.push(stmt.trim());
    }
  }

  if (unclassified.length > 0) {
    modules['uncategorized'] = unclassified;
  }

  return modules;
}

/**
 * Extract simple pattern matches (telemetry events, commands, classes).
 */
function extractSimplePatterns(source) {
  const results = {};

  for (const [modName, pattern] of Object.entries(SIMPLE_PATTERNS)) {
    pattern.lastIndex = 0;
    const matches = new Set();
    let m;
    while ((m = pattern.exec(source)) !== null) {
      const frag = m[0].trim();
      if (frag.length > 3) matches.add(frag);
    }
    if (matches.size > 0) {
      results[modName] = [...matches];
    }
  }

  return results;
}

/**
 * Compute basic metrics about the CLI bundle.
 */
function computeMetrics(source, filePath) {
  const sizeBytes = statSync(filePath).size;
  const versionMatch = source.match(/VERSION[=:]"?(\d+\.\d+\.\d+)/);
  const version = versionMatch ? versionMatch[1] : 'unknown';

  return {
    version,
    sizeBytes,
    lines: source.split('\n').length,
    functions: (source.match(/function\s*\w*\s*\(/g) || []).length,
    asyncFunctions: (source.match(/async\s+function/g) || []).length,
    arrowFunctions: (source.match(/=>/g) || []).length,
    classes: (source.match(/class \w+/g) || []).length,
    extends: (source.match(/extends \w+/g) || []).length,
  };
}

/**
 * Main entry point.
 */
function main() {
  const [bundlePath, outputDir] = process.argv.slice(2);
  if (!bundlePath || !outputDir) {
    console.error('Usage: node module-splitter.mjs <cli-bundle> <output-dir>');
    process.exit(1);
  }

  mkdirSync(outputDir, { recursive: true });

  console.log(`Reading bundle: ${bundlePath}`);
  const source = readFileSync(bundlePath, 'utf-8');
  const metrics = computeMetrics(source, bundlePath);
  console.log(`  Size: ${(metrics.sizeBytes / 1024 / 1024).toFixed(1)} MB, ` +
    `${metrics.classes} classes, ${metrics.functions} functions`);

  // Phase 1: statement-based classification (fast, O(n) per keyword set)
  console.log('  Splitting into statements...');
  const statements = splitStatements(source);
  console.log(`  ${statements.length} statements`);

  const classified = classifyStatements(statements);
  const moduleResults = {};

  for (const [modName, fragments] of Object.entries(classified)) {
    const outFile = join(outputDir, `${modName}.js`);
    writeFileSync(outFile, fragments.join('\n\n'), 'utf-8');
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: Buffer.byteLength(fragments.join('\n\n')),
    };
    console.log(`  Module "${modName}": ${fragments.length} fragments`);
  }

  // Phase 2: simple pattern extractions (telemetry, commands, classes)
  console.log('  Extracting simple patterns...');
  const simple = extractSimplePatterns(source);

  for (const [modName, fragments] of Object.entries(simple)) {
    const outFile = join(outputDir, `${modName}.js`);
    writeFileSync(outFile, fragments.join('\n'), 'utf-8');
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: Buffer.byteLength(fragments.join('\n')),
    };
    console.log(`  Module "${modName}": ${fragments.length} fragments`);
  }

  // Write metrics manifest
  const manifest = {
    ...metrics,
    sourceFile: basename(bundlePath),
    extractedAt: new Date().toISOString(),
    modules: moduleResults,
  };
  writeFileSync(
    join(outputDir, 'metrics.json'),
    JSON.stringify(manifest, null, 2)
  );

  // Output JSON summary to stdout for the caller script
  console.log(JSON.stringify(manifest));
}

main();
