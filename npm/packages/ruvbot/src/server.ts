/**
 * RuvBot HTTP Server - Cloud Run Entry Point
 *
 * Provides REST API endpoints for RuvBot including:
 * - Health checks (required for Cloud Run)
 * - Chat API
 * - Session management
 * - Agent management
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { URL } from 'node:url';
import { randomUUID } from 'node:crypto';
import pino from 'pino';
import { RuvBot, createRuvBot } from './RuvBot.js';
import { createAIDefenceGuard, type AIDefenceConfig } from './security/AIDefenceGuard.js';
import type { AgentConfig } from './core/types.js';

// ============================================================================
// Configuration
// ============================================================================

const PORT = parseInt(process.env.PORT || '8080', 10);
const HOST = process.env.HOST || '0.0.0.0';
const NODE_ENV = process.env.NODE_ENV || 'development';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: NODE_ENV !== 'production'
    ? { target: 'pino-pretty', options: { colorize: true } }
    : undefined,
});

// ============================================================================
// Types
// ============================================================================

interface RequestContext {
  req: IncomingMessage;
  res: ServerResponse;
  url: URL;
  body: Record<string, unknown> | null;
}

type RouteHandler = (ctx: RequestContext) => Promise<void>;

interface Route {
  method: string;
  pattern: RegExp;
  handler: RouteHandler;
}

// ============================================================================
// Server State
// ============================================================================

let bot: RuvBot | null = null;
let aiDefence: ReturnType<typeof createAIDefenceGuard> | null = null;
const startTime = Date.now();

// ============================================================================
// Utility Functions
// ============================================================================

async function parseBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      if (chunks.length === 0) {
        resolve(null);
        return;
      }
      const rawBody = Buffer.concat(chunks).toString('utf-8');
      try {
        const body = JSON.parse(rawBody);
        resolve(body);
      } catch (parseError) {
        logger.error({
          rawBody: rawBody.substring(0, 500),
          contentType: req.headers['content-type'],
          contentLength: req.headers['content-length'],
          err: parseError
        }, 'JSON parse error');
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', (err) => {
      logger.error({ err }, 'Request body read error');
      reject(err);
    });
  });
}

function sendJSON(res: ServerResponse, statusCode: number, data: unknown): void {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(JSON.stringify(data));
}

function sendError(res: ServerResponse, statusCode: number, message: string, code?: string): void {
  sendJSON(res, statusCode, { error: message, code: code || 'ERROR' });
}

// ============================================================================
// Route Handlers
// ============================================================================

async function handleHealth(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  sendJSON(res, 200, {
    status: 'healthy',
    version: '0.1.0',
    uptime: Math.floor((Date.now() - startTime) / 1000),
    timestamp: new Date().toISOString(),
  });
}

async function handleReady(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  if (bot?.getStatus().isRunning) {
    sendJSON(res, 200, { status: 'ready' });
  } else {
    sendError(res, 503, 'Service not ready', 'NOT_READY');
  }
}

async function handleStatus(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }
  sendJSON(res, 200, bot.getStatus());
}

async function handleModels(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  sendJSON(res, 200, {
    models: [
      // Gemini 2.x (recommended)
      { id: 'google/gemini-2.5-pro-preview-05-06', name: 'Gemini 2.5 Pro Preview', provider: 'openrouter' },
      { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash', provider: 'openrouter' },
      { id: 'google/gemini-2.0-flash-thinking-exp:free', name: 'Gemini 2.0 Flash Thinking (Free)', provider: 'openrouter' },
      // Anthropic Claude
      { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'openrouter' },
      { id: 'anthropic/claude-3-opus', name: 'Claude 3 Opus', provider: 'openrouter' },
      { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet (Direct)', provider: 'anthropic' },
      // OpenAI
      { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'openrouter' },
      { id: 'openai/o1-preview', name: 'O1 Preview (Reasoning)', provider: 'openrouter' },
      // Qwen
      { id: 'qwen/qwq-32b', name: 'Qwen QwQ 32B (Reasoning)', provider: 'openrouter' },
      { id: 'qwen/qwq-32b:free', name: 'Qwen QwQ 32B (Free)', provider: 'openrouter' },
      // DeepSeek
      { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1 (Reasoning)', provider: 'openrouter' },
      // Meta
      { id: 'meta-llama/llama-3.1-405b-instruct', name: 'Llama 3.1 405B', provider: 'openrouter' },
    ],
    default: 'google/gemini-2.5-pro-preview-05-06',
  });
}

async function handleCreateAgent(ctx: RequestContext): Promise<void> {
  const { res, body } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }
  if (!body || typeof body.name !== 'string') {
    sendError(res, 400, 'Agent name is required', 'INVALID_REQUEST');
    return;
  }

  const config: AgentConfig = {
    id: (body.id as string) || randomUUID(),
    name: body.name as string,
    model: (body.model as string) || 'claude-3-haiku-20240307',
    systemPrompt: body.systemPrompt as string,
    temperature: body.temperature as number,
    maxTokens: body.maxTokens as number,
  };

  const agent = await bot.spawnAgent(config);
  sendJSON(res, 201, agent);
}

async function handleListAgents(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }
  sendJSON(res, 200, { agents: bot.listAgents() });
}

async function handleCreateSession(ctx: RequestContext): Promise<void> {
  const { res, body } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }
  if (!body || typeof body.agentId !== 'string') {
    sendError(res, 400, 'Agent ID is required', 'INVALID_REQUEST');
    return;
  }

  try {
    const session = await bot.createSession(body.agentId as string, {
      userId: body.userId as string,
      channelId: body.channelId as string,
      platform: body.platform as 'slack' | 'discord' | 'api',
      metadata: body.metadata as Record<string, unknown>,
    });
    sendJSON(res, 201, session);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    sendError(res, 400, message, 'SESSION_ERROR');
  }
}

async function handleListSessions(ctx: RequestContext): Promise<void> {
  const { res } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }
  sendJSON(res, 200, { sessions: bot.listSessions() });
}

async function handleChat(ctx: RequestContext): Promise<void> {
  const { res, body, url } = ctx;
  if (!bot) {
    sendError(res, 503, 'Bot not initialized', 'NOT_INITIALIZED');
    return;
  }

  const sessionId = url.pathname.split('/')[3]; // /api/sessions/:id/chat
  if (!sessionId) {
    sendError(res, 400, 'Session ID is required', 'INVALID_REQUEST');
    return;
  }
  if (!body || typeof body.message !== 'string') {
    sendError(res, 400, 'Message is required', 'INVALID_REQUEST');
    return;
  }

  // Validate input with AIDefence if enabled
  let messageContent = body.message as string;
  let inputBlocked = false;
  if (aiDefence) {
    const analysisResult = await aiDefence.analyze(messageContent);
    if (!analysisResult.safe) {
      logger.warn({
        threats: analysisResult.threats,
        threatLevel: analysisResult.threatLevel,
      }, 'Threats detected in message');

      // Block critical threats entirely
      if (analysisResult.threatLevel === 'critical') {
        sendError(res, 400, 'Message blocked due to security concerns', 'SECURITY_BLOCKED');
        inputBlocked = true;
        return;
      }

      // Use sanitized input for lower-level threats
      if (analysisResult.sanitizedInput) {
        messageContent = analysisResult.sanitizedInput;
      }
    }
  }

  if (inputBlocked) return;

  try {
    logger.debug({ sessionId, messageLength: messageContent.length }, 'Processing chat request');

    const response = await bot.chat(sessionId, messageContent, {
      userId: body.userId as string,
      metadata: body.metadata as Record<string, unknown>,
    });

    logger.debug({ sessionId, responseId: response.id }, 'Chat response generated');

    // Validate output with AIDefence if enabled
    if (aiDefence && response.content) {
      try {
        const outputResult = await aiDefence.validateResponse(response.content, messageContent);
        if (!outputResult.safe) {
          logger.warn({
            threats: outputResult.threats,
          }, 'Threats detected in response');
        }
      } catch (defenceError) {
        // Log but don't fail the request if AIDefence validation fails
        logger.warn({ err: defenceError }, 'AIDefence output validation failed');
      }
    }

    sendJSON(res, 200, response);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error({ err: error, sessionId, errorMessage: message }, 'Chat request failed');
    sendError(res, 400, message, 'CHAT_ERROR');
  }
}

// ============================================================================
// Router
// ============================================================================

const routes: Route[] = [
  { method: 'GET', pattern: /^\/health$/, handler: handleHealth },
  { method: 'GET', pattern: /^\/ready$/, handler: handleReady },
  { method: 'GET', pattern: /^\/api\/status$/, handler: handleStatus },
  { method: 'GET', pattern: /^\/api\/models$/, handler: handleModels },
  { method: 'POST', pattern: /^\/api\/agents$/, handler: handleCreateAgent },
  { method: 'GET', pattern: /^\/api\/agents$/, handler: handleListAgents },
  { method: 'POST', pattern: /^\/api\/sessions$/, handler: handleCreateSession },
  { method: 'GET', pattern: /^\/api\/sessions$/, handler: handleListSessions },
  { method: 'POST', pattern: /^\/api\/sessions\/[^/]+\/chat$/, handler: handleChat },
];

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  const method = req.method || 'GET';

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Find matching route
  for (const route of routes) {
    if (route.method === method && route.pattern.test(url.pathname)) {
      try {
        const body = method !== 'GET' && method !== 'HEAD'
          ? await parseBody(req)
          : null;
        await route.handler({ req, res, url, body });
        return;
      } catch (error) {
        // Use 'err' key for proper pino error serialization
        const errorMessage = error instanceof Error ? error.message : String(error);
        logger.error({ err: error, path: url.pathname, errorMessage }, 'Request handler error');
        sendError(res, 500, 'Internal server error', 'INTERNAL_ERROR');
        return;
      }
    }
  }

  // 404 Not Found
  sendError(res, 404, 'Not found', 'NOT_FOUND');
}

// ============================================================================
// Server Initialization
// ============================================================================

async function initializeBot(): Promise<void> {
  logger.info('Initializing RuvBot...');

  bot = createRuvBot({
    config: {
      name: process.env.BOT_NAME || 'RuvBot',
      api: {
        enabled: false, // We're handling API ourselves
        port: PORT,
        host: HOST,
        cors: true,
        rateLimit: { max: 100, timeWindow: 60000 },
        auth: { enabled: false, type: 'bearer' },
      },
      llm: {
        provider: 'anthropic',
        apiKey: process.env.ANTHROPIC_API_KEY || '',
        model: process.env.DEFAULT_MODEL || 'claude-3-haiku-20240307',
        temperature: 0.7,
        maxTokens: 4096,
        streaming: true,
      },
      slack: {
        enabled: !!process.env.SLACK_BOT_TOKEN,
        botToken: process.env.SLACK_BOT_TOKEN,
        appToken: process.env.SLACK_APP_TOKEN,
        signingSecret: process.env.SLACK_SIGNING_SECRET,
        socketMode: true,
      },
      discord: {
        enabled: !!process.env.DISCORD_TOKEN,
        token: process.env.DISCORD_TOKEN,
        clientId: process.env.DISCORD_CLIENT_ID,
        guildId: process.env.DISCORD_GUILD_ID,
      },
      memory: {
        dimensions: 384,
        maxVectors: 100000,
        indexType: 'hnsw',
        efConstruction: 200,
        efSearch: 50,
        m: 16,
      },
      logging: {
        level: (process.env.LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error') || 'info',
        pretty: NODE_ENV !== 'production',
      },
    },
  });

  await bot.start();

  // Initialize AIDefence if not in development
  if (NODE_ENV === 'production') {
    const aiDefenceConfig: Partial<AIDefenceConfig> = {
      detectPromptInjection: true,
      detectJailbreak: true,
      detectPII: true,
      blockThreshold: 'medium',
      enableAuditLog: true,
    };
    aiDefence = createAIDefenceGuard(aiDefenceConfig);
    logger.info('AIDefence security layer enabled');
  }

  // Create default agent
  await bot.spawnAgent({
    id: 'default-agent',
    name: 'default-agent',
    model: process.env.DEFAULT_MODEL || 'claude-3-haiku-20240307',
    systemPrompt: process.env.SYSTEM_PROMPT || 'You are RuvBot, a helpful AI assistant.',
  });

  logger.info('RuvBot initialized successfully');
}

async function startServer(): Promise<void> {
  // Initialize bot first
  await initializeBot();

  // Create HTTP server
  const server = createServer((req, res) => {
    handleRequest(req, res).catch((error) => {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error({ err: error, errorMessage }, 'Unhandled request error');
      if (!res.headersSent) {
        sendError(res, 500, 'Internal server error', 'INTERNAL_ERROR');
      }
    });
  });

  // Graceful shutdown
  const shutdown = async (signal: string): Promise<void> => {
    logger.info({ signal }, 'Received shutdown signal');

    server.close(async () => {
      logger.info('HTTP server closed');

      if (bot) {
        await bot.stop();
        logger.info('RuvBot stopped');
      }

      process.exit(0);
    });

    // Force exit after timeout
    setTimeout(() => {
      logger.error('Forced shutdown due to timeout');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Start listening
  server.listen(PORT, HOST, () => {
    logger.info({ port: PORT, host: HOST, env: NODE_ENV }, 'RuvBot server started');
  });
}

// ============================================================================
// Main Entry Point
// ============================================================================

startServer().catch((error) => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  logger.error({ err: error, errorMessage }, 'Failed to start server');
  process.exit(1);
});
