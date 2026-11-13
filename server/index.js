import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client as McpClient } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json({ limit: '4mb' }));

// 课程问答助手的模型服务地址（硅基流动 SiliconFlow，OpenAI 兼容接口；强制固定，不读取环境变量，避免被 .env.local 覆盖）
const FORCED_LLM_BASE_URL = 'https://api.siliconflow.cn/v1';
// 硅基流动上的 Qwen3 模型 ID 必须带命名空间前缀
const DEFAULT_LLM_MODEL = 'Qwen/Qwen3-8B';

const config = {
  // API Key 从 .env.local 的 SILICONFLOW_API_KEY 读取
  apiKey: process.env.SILICONFLOW_API_KEY,
  baseUrl: FORCED_LLM_BASE_URL.replace(/\/$/, ''),
  model: process.env.SILICONFLOW_MODEL || DEFAULT_LLM_MODEL,
  port: Number(process.env.SERVER_PORT || 8787),
  // Qwen3 默认开启思考模式（enable_thinking=true）。思考模式下：
  // 1) 非流式请求容易只返回思考链、正文为空；2) 与 function calling 配合不稳定且明显变慢。
  // 因此默认关闭思考模式，需要思考链时可设 SILICONFLOW_ENABLE_THINKING=1。
  enableThinking: process.env.SILICONFLOW_ENABLE_THINKING === '1',
  // 主模型触发 TPM 限流时自动降级用的备用模型（免费额度更宽松），设空可关闭降级
  fallbackModel: process.env.SILICONFLOW_FALLBACK_MODEL === 'off'
    ? ''
    : process.env.SILICONFLOW_FALLBACK_MODEL || 'Qwen/Qwen2.5-7B-Instruct'
};

// 只有 Qwen3 系列支持 enable_thinking 参数，其它模型传了可能被上游拒掉
function thinkingParams(model = config.model) {
  if (!/qwen3/i.test(model)) return {};
  return { enable_thinking: config.enableThinking };
}

// 课程问答助手的系统提示词
const SYSTEM_PROMPT = [
  '你是一个灵活的高校教务智能助手，当前服务对象为25级。你的职责是基于真实课表数据解答问题。请严格按照以下逻辑执行：',
  '1. 【身份识别】：',
  '   - 如果用户是【辅导员】（问题中带有班级，如“25级软件3班”），直接提取班级信息并调用工具查询。',
  '   - 如果用户是【学生】（用“我”、“我的”提问，且未提班级），你必须先反问：“请问你是25级软件3班还是软件技术4班的同学呢？”等待用户回答后，利用上下文记忆，再调用工具。',
  '2. 【下节课推算】：',
  '   - 如果用户问“下节课在哪上”或“今天还有什么课”。',
  '   - 首先，看系统时间里的“星期几”。调用工具查询该班级“今天”的所有课程。',
  '   - 其次，根据系统时间里的“钟点”（结合学校作息时间表：1-2节 08:30-10:00，3-4节 10:20-11:50，5-6节 14:30-16:00，7-8节 16:20-17:50，9-10节 19:00-20:30），推算出当前时间之后的“下一节课”。',
  '   - 最后，告诉学生下一节课的课程名称、教室和时间。如果今天的课已经全部结束，则回答“今天没课了”。',
  '3. 【纪律】：',
  '   - 回答必须简洁、准确，绝不能编造数据。如果工具返回没课，就回答没课。'
].join('\n');

// 小模型（如降级后的 Qwen2.5-7B）容易“凭日期推断没课”而不查表，运行期追加一条纪律提醒
const TOOL_DISCIPLINE_REMINDER =
  '【工具调用纪律（最高优先级）】凡是要给出课程名称、授课老师、教室或上课时间的答案，必须先调用 query_courses 查询真实课表，再根据返回结果作答。' +
  '即使系统时间显示今天是周末、或看起来已经下课，也不能直接推断“今天没课”，必须先查询当天课表。';

const WEEKDAY_LABELS = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];
const WEEKDAY_SHORT = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];

function pad2(value) {
  return String(value).padStart(2, '0');
}

// 可选：通过 QWEN_FAKE_NOW=2026-11-09T10:30 模拟系统时间，方便在周末验证“下节课”推算逻辑
function resolveNow() {
  const forced = process.env.QWEN_FAKE_NOW;
  if (forced) {
    const parsed = new Date(forced);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed;
    }
  }
  return new Date();
}

// 生成注入到 System Prompt 的当前时间上下文
function buildTimeContext(date = resolveNow()) {
  const label = `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日 ${WEEKDAY_LABELS[date.getDay()]} ${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
  return `【当前时间：${label}】\n【今日星期标识：${WEEKDAY_SHORT[date.getDay()]}（课表 day 字段使用「周一」~「周日」格式）】`;
}

function createAppError(code, message, details = '', status = 500) {
  const error = new Error(message);
  error.code = code;
  error.details = details;
  error.status = status;
  return error;
}

function getErrorPayload(error, fallbackMessage) {
  if (error && typeof error === 'object' && 'message' in error) {
    return {
      code: error.code || 'UNKNOWN_ERROR',
      message: error.message || fallbackMessage,
      details: error.details || '',
      status: error.status || 500
    };
  }

  return {
    code: 'UNKNOWN_ERROR',
    message: fallbackMessage,
    details: '',
    status: 500
  };
}

// 硅基流动免费模型（如 Qwen/Qwen3-8B）有较低的 TPM 限额：一轮对话会连续发起
// 「工具决策请求 + 流式回答请求」多次调用，很容易撞上 TPM。这里对 429/503 做退避重试。
const RETRY_DELAYS_MS = [1500, 3500];

function wait(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function llmFetch(endpoint, body, stream = false) {
  if (!config.apiKey) {
    throw createAppError(
      'MISSING_API_KEY',
      '缺少 SiliconFlow API Key',
      '请检查服务端 `.env.local` 中的 `SILICONFLOW_API_KEY` 配置（在 https://cloud.siliconflow.cn 控制台获取）。',
      500
    );
  }

  let lastError = null;

  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    let response;
    try {
      response = await fetch(`${config.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${config.apiKey}`
        },
        body: JSON.stringify(body)
      });
    } catch {
      throw createAppError(
        'NETWORK_UNREACHABLE',
        '无法连接到 SiliconFlow 服务',
        '当前运行环境访问 api.siliconflow.cn 失败。请检查网络、代理、VPN 或防火墙设置。',
        502
      );
    }

    if (response.ok) {
      return stream ? response : response.json();
    }

    const text = await response.text().catch(() => '');

    if (response.status === 401) {
      throw createAppError('INVALID_API_KEY', 'SiliconFlow API Key 无效或已过期', text || '请检查 `SILICONFLOW_API_KEY` 是否正确。', 401);
    }

    if (response.status === 403) {
      throw createAppError('PERMISSION_DENIED', 'SiliconFlow 拒绝访问该模型', text || '请确认账号已开通该模型（如 Qwen/Qwen3-8B）的调用权限。', 403);
    }

    // 429 / 503 以及 TPM（code 50602）属于可恢复的限流，退避后重试
    const retryable = response.status === 429 || response.status === 503 || text.includes('50602');
    if (retryable) {
      lastError = text;
      if (attempt < RETRY_DELAYS_MS.length) {
        const delay = RETRY_DELAYS_MS[attempt];
        console.warn(`[rate-limit] ${response.status} 等待 ${delay}ms 后重试（第 ${attempt + 1} 次）`);
        await wait(delay);
        continue;
      }

      throw createAppError(
        'RATE_LIMITED',
        'SiliconFlow 请求过于频繁（TPM 限额）',
        `${text || 'SiliconFlow 返回限流'}｜免费模型 Qwen/Qwen3-8B 的每分钟 token 额度较低，稍等 30~60 秒再问一次，或在控制台提额/换用额度更高的模型。`,
        429
      );
    }

    if (response.status === 429) {
      throw createAppError('RATE_LIMITED', 'SiliconFlow 请求过于频繁', text || '请稍后重试，或检查账户配额是否充足。', 429);
    }

    throw createAppError('LLM_HTTP_ERROR', `SiliconFlow 请求失败（${response.status}）`, text || '上游模型服务返回异常响应。', 502);
  }

  throw createAppError('RATE_LIMITED', 'SiliconFlow 请求过于频繁', lastError || '请稍后重试。', 429);
}

// 主模型被限流后的冷却期：冷却期内直接走备用模型，避免每轮都白等重试
let primaryBlockedUntil = 0;
const PRIMARY_COOLDOWN_MS = 60 * 1000;

async function chatFetch(body, stream = false) {
  const primaryBlocked = Date.now() < primaryBlockedUntil;

  if (!primaryBlocked) {
    try {
      const result = await llmFetch('/chat/completions', body, stream);
      primaryBlockedUntil = 0;
      return result;
    } catch (error) {
      if (error?.code !== 'RATE_LIMITED' || !config.fallbackModel) {
        throw error;
      }
      primaryBlockedUntil = Date.now() + PRIMARY_COOLDOWN_MS;
      console.warn(`[fallback] ${config.model} 触发限流，本轮改用 ${config.fallbackModel}`);
    }
  }

  if (!config.fallbackModel) {
    throw createAppError('RATE_LIMITED', 'SiliconFlow 请求过于频繁', '请稍后重试。', 429);
  }

  const { enable_thinking: _ignored, ...rest } = body;
  return llmFetch('/chat/completions', { ...rest, model: config.fallbackModel, ...thinkingParams(config.fallbackModel) }, stream);
}

function sendSse(res, event, data) {  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

let mcpSessionPromise = null;

async function createMcpSession() {
  if (mcpSessionPromise) {
    return mcpSessionPromise;
  }

  mcpSessionPromise = (async () => {
    const client = new McpClient({
      name: 'yuan-agent-chat-orchestrator',
      version: '1.0.0'
    });

    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [path.resolve(__dirname, './mcp-server.js')],
      cwd: path.resolve(__dirname, '..'),
      env: {
        ...process.env,
        NODE_ENV: process.env.NODE_ENV || 'development'
      },
      stderr: 'pipe'
    });

    if (transport.stderr) {
      transport.stderr.on('data', (chunk) => {
        const message = chunk.toString().trim();
        if (message) {
          console.error(`[mcp-server] ${message}`);
        }
      });
    }

    await client.connect(transport);

    const session = {
      client,
      transport,
      async close() {
        mcpSessionPromise = null;
        await client.close().catch(() => {});
        await transport.close().catch(() => {});
      }
    };

    return session;
  })().catch((error) => {
    mcpSessionPromise = null;
    throw error;
  });

  return mcpSessionPromise;
}

function normalizeStructuredContent(result) {
  if (result && typeof result === 'object' && 'structuredContent' in result && result.structuredContent) {
    return result.structuredContent;
  }
  if (result && typeof result === 'object' && 'toolResult' in result) {
    return result.toolResult;
  }
  return {};
}

function contentToText(result) {
  if (!result || typeof result !== 'object' || !('content' in result) || !Array.isArray(result.content)) {
    return '';
  }

  return result.content
    .filter((item) => item && typeof item === 'object' && item.type === 'text')
    .map((item) => item.text || '')
    .join('\n')
    .trim();
}

// 暴露给模型的工具白名单（知识库导入/删除等操作走 REST 接口，模型无需调用）
const EXPOSED_TOOL_NAMES = new Set([
  'query_courses',
  'get_current_time',
  'retrieve_knowledge',
  'list_knowledge_documents'
]);

function buildToolDefinitions(tools) {
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.inputSchema || {
        type: 'object',
        properties: {}
      }
    }
  }));
}

async function executeToolCallWithMcp(client, toolCall) {
  const name = toolCall.function?.name || 'unknown';
  const args = toolCall.function?.arguments ? JSON.parse(toolCall.function.arguments) : {};
  const result = await client.callTool({
    name,
    arguments: args
  });

  const structured = normalizeStructuredContent(result);
  const text = contentToText(result);
  const isError = !!(result && typeof result === 'object' && 'isError' in result && result.isError);

  return {
    args,
    isError,
    resultText: text || JSON.stringify(structured, null, 2),
    resultPayload: structured,
    citations: Array.isArray(structured?.citations) ? structured.citations : []
  };
}

app.get('/api/health', async (_, res) => {
  try {
    const session = await createMcpSession();
    const result = await session.client.callTool({
      name: 'list_knowledge_documents',
      arguments: {}
    });

    const structured = normalizeStructuredContent(result);
    res.json({
      ok: true,
      documents: Array.isArray(structured.documents) ? structured.documents.length : 0,
      mcp: true
    });
  } catch (error) {
    const payload = getErrorPayload(error, 'MCP 健康检查失败');
    res.status(payload.status).json({
      ok: false,
      code: payload.code,
      error: payload.message,
      details: payload.details
    });
  }
});

app.get('/api/knowledge', async (_, res) => {
  try {
    const session = await createMcpSession();
    const result = await session.client.callTool({
      name: 'list_knowledge_documents',
      arguments: {}
    });

    const structured = normalizeStructuredContent(result);
    res.json({
      documents: structured.documents || []
    });
  } catch (error) {
    const payload = getErrorPayload(error, '加载知识库失败');
    res.status(payload.status).json({
      error: payload.message,
      code: payload.code,
      details: payload.details
    });
  }
});

app.post('/api/knowledge/upload', upload.array('files'), async (req, res) => {
  try {
    const files = Array.isArray(req.files) ? req.files : [];
    const session = await createMcpSession();
    const result = await session.client.callTool({
      name: 'ingest_knowledge_documents',
      arguments: {
        documents: files.map((file) => ({
          name: file.originalname,
          content: file.buffer.toString('utf-8')
        }))
      }
    });

    const structured = normalizeStructuredContent(result);
    if (result.isError) {
      throw createAppError(structured.code || 'MCP_TOOL_ERROR', structured.message || '知识库导入失败', structured.details || '', 400);
    }

    res.json({
      documents: structured.documents || [],
      message: structured.message || '上传成功'
    });
  } catch (error) {
    const payload = getErrorPayload(error, '知识库导入失败');
    res.status(payload.status).json({
      error: payload.message,
      code: payload.code,
      details: payload.details
    });
  }
});

app.delete('/api/knowledge/:id', async (req, res) => {
  try {
    const session = await createMcpSession();
    const result = await session.client.callTool({
      name: 'delete_knowledge_document',
      arguments: { id: req.params.id }
    });

    const structured = normalizeStructuredContent(result);
    if (result.isError) {
      throw createAppError(structured.code || 'MCP_TOOL_ERROR', structured.message || '删除失败', structured.details || '', 400);
    }

    res.json({ ok: true });
  } catch (error) {
    const payload = getErrorPayload(error, '删除失败');
    res.status(payload.status).json({
      error: payload.message,
      code: payload.code,
      details: payload.details
    });
  }
});

app.delete('/api/knowledge', async (_, res) => {
  try {
    const session = await createMcpSession();
    const result = await session.client.callTool({
      name: 'clear_knowledge_documents',
      arguments: {}
    });

    const structured = normalizeStructuredContent(result);
    if (result.isError) {
      throw createAppError(structured.code || 'MCP_TOOL_ERROR', structured.message || '清空失败', structured.details || '', 400);
    }

    res.json({ ok: true });
  } catch (error) {
    const payload = getErrorPayload(error, '清空失败');
    res.status(payload.status).json({
      error: payload.message,
      code: payload.code,
      details: payload.details
    });
  }
});

app.post('/api/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');

  try {
    const session = await createMcpSession();
    const availableTools = await session.client.listTools();
    const incomingMessages = Array.isArray(req.body?.messages) ? req.body.messages : [];
    const messages = [
      {
        role: 'system',
        content: `${SYSTEM_PROMPT}\n\n${buildTimeContext()}\n\n${TOOL_DISCIPLINE_REMINDER}`
      },
      ...incomingMessages
    ];

    const gatheredCitations = [];
    const gatheredTools = [];
    const toolDefinitions = buildToolDefinitions(
      // 只暴露当前场景用得到的工具：tool schema 会按调用次数计入输入 token，
      // 免费模型 TPM 很小，工具越多越容易触发限流
      (availableTools.tools || []).filter((tool) => EXPOSED_TOOL_NAMES.has(tool.name))
    );

    for (let round = 0; round < 4; round += 1) {
      const completion = await chatFetch({
        model: config.model,
        stream: false,
        temperature: 0.4,
        max_tokens: 512,
        ...thinkingParams(),
        messages,
        tools: toolDefinitions,
        tool_choice: 'auto'
      });

      const choice = completion.choices?.[0]?.message;
      const toolCalls = choice?.tool_calls || [];

      if (process.env.QWEN_DEBUG === '1') {
        console.error('[usage] tool-round:', JSON.stringify(completion.usage || {}));
      }

      if (!toolCalls.length) {
        // 注意：这里不要把本轮的完整回答再塞回 messages。
        // 否则最后一条消息会变成 assistant，部分 Qwen 兼容网关会把它当成续写预填充，
        // 导致后面的流式请求直接返回空内容（前端表现为“只有工具卡片、没有文字”）。
        break;
      }

      // 关键：一条 assistant 消息里带多个 tool_calls 时，部分 Qwen 兼容网关会丢失工具结果，
      // 模型随后会凭空编造数据。因此这里把每个工具调用拆成独立的「assistant + tool」消息对。
      for (const toolCall of toolCalls) {
        let previewArgs = {};
        try {
          previewArgs = toolCall.function?.arguments ? JSON.parse(toolCall.function.arguments) : {};
        } catch {
          previewArgs = {};
        }

        messages.push({
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: toolCall.id,
              type: 'function',
              function: {
                name: toolCall.function?.name,
                arguments: toolCall.function?.arguments || '{}'
              }
            }
          ]
        });

        sendSse(res, 'tool', {
          id: toolCall.id,
          name: toolCall.function?.name,
          args: previewArgs,
          status: 'running'
        });

        const executed = await executeToolCallWithMcp(session.client, toolCall);

        gatheredCitations.push(...executed.citations);
        gatheredTools.push({
          id: toolCall.id,
          name: toolCall.function?.name,
          args: executed.args,
          status: executed.isError ? 'error' : 'success',
          result: executed.resultText
        });

        sendSse(res, 'tool', {
          id: toolCall.id,
          name: toolCall.function?.name,
          args: executed.args,
          status: executed.isError ? 'error' : 'success',
          result: executed.resultPayload
        });

        if (executed.citations.length) {
          sendSse(res, 'citations', { citations: executed.citations });
        }

        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(executed.resultPayload)
        });
      }
    }

    const streamResponse = await chatFetch({
      model: config.model,
      stream: true,
      temperature: 0.4,
      max_tokens: 1024,
      ...thinkingParams(),
      messages
    }, true);

    if (process.env.QWEN_DEBUG === '1') {
      console.error('[debug] messages before stream:', JSON.stringify(messages).slice(0, 4000));
    }

    const reader = streamResponse.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';

      for (const event of events) {
        const lines = event.split('\n').map((line) => line.trim()).filter(Boolean);
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (raw === '[DONE]') {
            sendSse(res, 'done', { citations: gatheredCitations, tools: gatheredTools });
            res.end();
            return;
          }

          try {
            const json = JSON.parse(raw);

            // SiliconFlow 在流式过程中可能推送错误对象，这里透传给前端，避免出现“空白回答”
            if (json.error) {
              sendSse(res, 'error', {
                code: 'UPSTREAM_STREAM_ERROR',
                message: json.error.message || 'SiliconFlow 流式返回异常',
                details: JSON.stringify(json.error)
              });
              res.end();
              return;
            }

            // 只取正文；若开启思考模式，思考链在 delta.reasoning_content 中，此处默认不展示
            const token = json.choices?.[0]?.delta?.content;
            if (token) {
              sendSse(res, 'token', { token });
            }
          } catch {
            // ignore invalid chunks
          }
        }
      }
    }

    sendSse(res, 'done', { citations: gatheredCitations, tools: gatheredTools });
    res.end();
  } catch (error) {
    const payload = getErrorPayload(error, '服务异常');
    sendSse(res, 'error', {
      code: payload.code,
      message: payload.message,
      details: payload.details
    });
    res.end();
  }
});

app.use(express.static(path.resolve(__dirname, '../dist')));
app.get('*', (_, res) => {
  res.sendFile(path.resolve(__dirname, '../dist/index.html'));
});

app.listen(process.env.PORT || config.port, '0.0.0.0', () => {
  console.log(`Server running at http://127.0.0.1:${config.port}`);
  console.log(`[model] ${config.model} @ ${config.baseUrl} (thinking: ${config.enableThinking ? 'on' : 'off'})`);
  console.log(`[fallback] ${config.fallbackModel || '(已关闭)'}`);
  console.log(
    config.apiKey
      ? `[api-key] SILICONFLOW_API_KEY 已加载（${config.apiKey.slice(0, 6)}***${config.apiKey.slice(-4)}）`
      : '[api-key] 未检测到 SILICONFLOW_API_KEY，请在 .env.local 中填写后再试'
  );
  console.log(`[time-context] ${buildTimeContext().replace(/\n/g, ' ')}`);
});
