import dotenv from 'dotenv';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const COURSES_FILE = path.resolve(__dirname, './courses.json');

const state = {
  documents: [],
  chunks: []
};

const config = {
  apiKey: process.env.QWEN_API_KEY,
  baseUrl: (process.env.QWEN_BASE_URL || 'https://dashscope.aliyuncs.com/compatible-mode/v1').replace(/\/$/, ''),
  embeddingModel: process.env.QWEN_EMBEDDING_MODEL || 'text-embedding-v3'
};

function uid(prefix) {
  return `${prefix}-${crypto.randomUUID()}`;
}

function createAppError(code, message, details = '', status = 500) {
  const error = new Error(message);
  error.code = code;
  error.details = details;
  error.status = status;
  return error;
}

function tokenize(input) {
  return input
    .toLowerCase()
    .replace(/[`*_>#\-\[\]\(\)]/g, ' ')
    .split(/[\s，。；：！？、,.!?;:\/\\|]+/)
    .filter((token) => token.length > 1);
}

function chunkText(text, chunkSize = 900, overlap = 160) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(text.length, start + chunkSize);
    const value = text.slice(start, end).trim();
    if (value) chunks.push(value);
    start += chunkSize - overlap;
  }
  return chunks;
}

function cosineSimilarity(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < len; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] * a[index];
    normB += b[index] * b[index];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function qwenFetch(endpoint, body) {
  if (!config.apiKey) {
    throw createAppError('MISSING_API_KEY', '缺少 Qwen API Key', '请检查服务端 `.env.local` 中的 `QWEN_API_KEY` 配置。', 500);
  }

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
      '无法连接到 Qwen 服务',
      '当前运行环境访问 DashScope 失败。请检查网络、代理、VPN 或防火墙设置。',
      502
    );
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    if (response.status === 401) {
      throw createAppError('INVALID_API_KEY', 'Qwen API Key 无效或已过期', text || '请检查 `QWEN_API_KEY` 是否正确。', 401);
    }

    if (response.status === 429) {
      throw createAppError('RATE_LIMITED', 'Qwen 请求过于频繁', text || '请稍后重试，或检查账户配额是否充足。', 429);
    }

    throw createAppError('QWEN_HTTP_ERROR', `Qwen 请求失败（${response.status}）`, text || '上游模型服务返回异常响应。', 502);
  }

  return response.json();
}

async function createEmbedding(text) {
  const result = await qwenFetch('/embeddings', {
    model: config.embeddingModel,
    input: text.slice(0, 6000)
  });

  const vector = result.data?.[0]?.embedding;
  if (!vector) {
    throw createAppError('EMBEDDING_EMPTY', 'Embedding 生成失败', '模型返回为空，无法建立向量索引。', 502);
  }

  return vector;
}

function buildCitation(chunk, score) {
  return {
    id: chunk.id,
    title: chunk.documentName,
    snippet: chunk.text,
    source: `向量知识库 / ${chunk.documentName}`,
    score: Number(score.toFixed(4))
  };
}

async function searchKnowledge(query, topK = 4) {
  if (!state.chunks.length) {
    return [];
  }

  const queryEmbedding = await createEmbedding(query);
  return state.chunks
    .map((chunk) => ({
      chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding)
    }))
    .filter((item) => item.score > 0.15)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ chunk, score }) => buildCitation(chunk, score));
}

async function ingestDocuments(documents) {
  const supported = documents.filter((document) => /\.(txt|md|markdown|json)$/i.test(document.name));

  if (!supported.length) {
    throw createAppError('UNSUPPORTED_FILES', '没有可导入的知识文件', '仅支持 `.md`、`.markdown`、`.txt`、`.json` 文件。', 400);
  }

  const inserted = [];
  const nextDocuments = [];
  const nextChunks = [];

  for (const source of supported) {
    const content = source.content.trim();
    if (!content) continue;

    const document = {
      id: uid('doc'),
      name: source.name,
      content,
      createdAt: Date.now()
    };

    nextDocuments.push(document);
    inserted.push({ id: document.id, name: document.name, createdAt: document.createdAt });

    const parts = chunkText(content);
    for (const part of parts) {
      const embedding = await createEmbedding(part);
      nextChunks.push({
        id: uid('chunk'),
        documentId: document.id,
        documentName: document.name,
        text: part,
        tokens: tokenize(part),
        embedding
      });
    }
  }

  if (!inserted.length) {
    throw createAppError('EMPTY_FILES', '上传的文件内容为空', '请确认文件不是空文件，且编码为 UTF-8。', 400);
  }

  state.documents.push(...nextDocuments);
  state.chunks.push(...nextChunks);

  return inserted;
}

function listDocuments() {
  return state.documents.map((document) => ({
    id: document.id,
    name: document.name,
    createdAt: document.createdAt
  }));
}

function deleteDocument(id) {
  const before = state.documents.length;
  state.documents = state.documents.filter((document) => document.id !== id);
  state.chunks = state.chunks.filter((chunk) => chunk.documentId !== id);

  if (before === state.documents.length) {
    throw createAppError('DOCUMENT_NOT_FOUND', '知识文件不存在', '请确认传入的文档 ID 是否正确。', 404);
  }

  return { ok: true, id };
}

function clearDocuments() {
  state.documents = [];
  state.chunks = [];
  return { ok: true };
}

const DAY_ORDER = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];

const DAY_ALIAS_MAP = {
  一: '周一',
  二: '周二',
  三: '周三',
  四: '周四',
  五: '周五',
  六: '周六',
  日: '周日',
  天: '周日',
  1: '周一',
  2: '周二',
  3: '周三',
  4: '周四',
  5: '周五',
  6: '周六',
  7: '周日'
};

// 常规作息时间表（用于“下节课”推算的参考信息，随工具结果一并返回）
// 单节课作息：第1节 08:30-09:10，第2节 09:20-10:00，第3节 10:20-11:00，第4节 11:10-11:50，
// 第5节 14:30-15:10，第6节 15:20-16:00，第7节 16:20-17:00，第8节 17:10-17:50，
// 第9节 19:00-19:40，第10节 19:50-20:30（课程表按两节连排的大节记录）
const TIME_SLOTS = {
  '1-2节': { start: '08:30', end: '10:00', startMinutes: 510 },
  '3-4节': { start: '10:20', end: '11:50', startMinutes: 620 },
  '5-6节': { start: '14:30', end: '16:00', startMinutes: 870 },
  '7-8节': { start: '16:20', end: '17:50', startMinutes: 980 },
  '9-10节': { start: '19:00', end: '20:30', startMinutes: 1140 }
};

function normalizeText(input) {
  return String(input ?? '')
    .toLowerCase()
    .replace(/[\s，。；：！？、,.!?;:：\-_/\\|"'“”‘’()（）\[\]【】]/g, '');
}

function normalizeClass(input) {
  // “25级软件技术4班” 与 “25级软件4班” 视为同一个班，便于口语化查询
  return normalizeText(input).replace(/技术/g, '');
}

function normalizeDay(input) {
  const matched = String(input ?? '').match(/(?:周|星期|礼拜)\s*([一二三四五六日天1-7])/);
  if (matched) {
    return DAY_ALIAS_MAP[matched[1]] || '';
  }
  return '';
}

function normalizeTime(input) {
  const text = String(input ?? '')
    .replace(/[第\s]/g, '')
    .replace(/[～~—–－]/g, '-')
    .replace(/节/g, '');
  const matched = text.match(/(\d+)\s*-\s*(\d+)/);
  return matched ? `${matched[1]}-${matched[2]}节` : text.trim();
}

function isSubsequence(needle, haystack) {
  if (!needle) return false;
  let index = 0;
  for (const char of haystack) {
    if (char === needle[index]) {
      index += 1;
      if (index === needle.length) return true;
    }
  }
  return false;
}

function fieldHit(value, query, { subsequence = false } = {}) {
  const left = normalizeText(value);
  const right = normalizeText(query);
  if (!left || !right) return false;
  if (left === right || left.includes(right)) return true;
  if (right.includes(left) && left.length >= 2) return true;
  return subsequence ? isSubsequence(right, left) : false;
}

function dayHit(value, query) {
  const target = normalizeDay(query) || normalizeText(query);
  const current = normalizeDay(value) || normalizeText(value);
  return current === target;
}

function timeHit(value, query) {
  const target = normalizeTime(query);
  const current = normalizeTime(value);
  if (!target || !current) return false;
  return current === target || current.includes(target) || target.includes(current);
}

const WEEKDAY_LABELS = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];

// 与 server/index.js 保持一致：支持 QWEN_FAKE_NOW 模拟系统时间
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

function loadCourses() {
  let raw;
  try {
    raw = fs.readFileSync(COURSES_FILE, 'utf-8');
  } catch {
    throw createAppError('COURSES_FILE_MISSING', '课程数据文件读取失败', '请确认 server/courses.json 是否存在并可读。', 500);
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw createAppError('COURSES_FILE_INVALID', '课程数据文件格式错误', 'courses.json 必须是合法的 JSON 数组。', 500);
  }

  return (Array.isArray(parsed) ? parsed : []).map((course) => ({
    className: String(course.className ?? ''),
    day: String(course.day ?? ''),
    time: String(course.time ?? ''),
    courseName: String(course.courseName ?? ''),
    teacher: String(course.teacher ?? ''),
    location: String(course.location ?? '')
  }));
}

function sortCourses(courses) {
  return [...courses].sort((a, b) => {
    const dayDiff = (DAY_ORDER.indexOf(a.day) + 1 || 99) - (DAY_ORDER.indexOf(b.day) + 1 || 99);
    if (dayDiff !== 0) return dayDiff;
    const timeA = TIME_SLOTS[normalizeTime(a.time)]?.startMinutes ?? 9999;
    const timeB = TIME_SLOTS[normalizeTime(b.time)]?.startMinutes ?? 9999;
    return timeA - timeB;
  });
}

function queryCourses(filters = {}) {
  const { className, day, time, courseName, teacher, location, keyword, listAll = false } = filters;
  const courses = loadCourses();
  const availableClasses = [...new Set(courses.map((course) => course.className))];

  const conditions = [];

  if (className) {
    conditions.push((course) => {
      const target = normalizeClass(className);
      const current = normalizeClass(course.className);
      return current === target || current.includes(target) || target.includes(current);
    });
  }

  if (day) {
    conditions.push((course) => dayHit(course.day, day));
  }

  if (time) {
    conditions.push((course) => timeHit(course.time, time));
  }

  if (courseName) {
    conditions.push((course) => fieldHit(course.courseName, courseName, { subsequence: true }));
  }

  if (teacher) {
    conditions.push((course) => fieldHit(course.teacher, teacher));
  }

  if (location) {
    conditions.push((course) => fieldHit(course.location, location));
  }

  if (keyword) {
    conditions.push(
      (course) =>
        fieldHit(course.courseName, keyword, { subsequence: true }) ||
        fieldHit(course.teacher, keyword) ||
        fieldHit(course.location, keyword) ||
        fieldHit(course.className, keyword, { subsequence: true }) ||
        dayHit(course.day, keyword) ||
        timeHit(course.time, keyword)
    );
  }

  const filtered = listAll || !conditions.length
    ? courses
    : courses.filter((course) => conditions.every((condition) => condition(course)));

  const results = sortCourses(filtered).map((course) => ({
    ...course,
    timeRange: TIME_SLOTS[normalizeTime(course.time)] || null
  }));

  const message = results.length
    ? `共查询到 ${results.length} 门课程。`
    : `未查询到符合条件的课程${className ? `（班级：${className}）` : ''}${day ? `（星期：${day}）` : ''}。`;

  return {
    filters: { className, day, time, courseName, teacher, location, keyword, listAll },
    count: results.length,
    courses: results,
    availableClasses,
    total: courses.length,
    timeSlots: TIME_SLOTS,
    message
  };
}


function toTextContent(value) {
  return {
    content: [
      {
        type: 'text',
        text: JSON.stringify(value, null, 2)
      }
    ],
    structuredContent: value
  };
}

function toErrorContent(error) {
  return {
    content: [
      {
        type: 'text',
        text: error.details ? `${error.message}\n${error.details}` : error.message
      }
    ],
    structuredContent: {
      code: error.code || 'UNKNOWN_ERROR',
      message: error.message,
      details: error.details || ''
    },
    isError: true
  };
}

const server = new McpServer({
  name: 'yuan-agent-mcp-server',
  version: '1.0.0'
});

server.registerTool(
  'retrieve_knowledge',
  {
    description: '从后端向量知识库检索与用户问题最相关的文档片段',
    inputSchema: z.object({
      query: z.string().min(1, 'query 不能为空'),
      topK: z.number().int().min(1).max(10).optional()
    })
  },
  async ({ query, topK = 4 }) => {
    try {
      const citations = await searchKnowledge(query, topK);
      return toTextContent({ citations, count: citations.length });
    } catch (error) {
      return toErrorContent(error);
    }
  }
);

server.registerTool(
  'list_knowledge_documents',
  {
    description: '列出当前后端知识库中的文档',
    inputSchema: z.object({})
  },
  async () => toTextContent({ documents: listDocuments() })
);

server.registerTool(
  'ingest_knowledge_documents',
  {
    description: '导入知识文档到向量知识库中并建立向量索引',
    inputSchema: z.object({
      documents: z.array(
        z.object({
          name: z.string().min(1, 'name 不能为空'),
          content: z.string().min(1, 'content 不能为空')
        })
      ).min(1, '至少导入一份文档')
    })
  },
  async ({ documents }) => {
    try {
      const inserted = await ingestDocuments(documents);
      return toTextContent({ documents: inserted, message: `已成功导入 ${inserted.length} 份知识文件。` });
    } catch (error) {
      return toErrorContent(error);
    }
  }
);

server.registerTool(
  'delete_knowledge_document',
  {
    description: '删除指定的知识文档以及对应的向量索引',
    inputSchema: z.object({
      id: z.string().min(1, 'id 不能为空')
    })
  },
  async ({ id }) => {
    try {
      return toTextContent(deleteDocument(id));
    } catch (error) {
      return toErrorContent(error);
    }
  }
);

server.registerTool(
  'clear_knowledge_documents',
  {
    description: '清空知识库中的所有文档与向量索引',
    inputSchema: z.object({})
  },
  async () => toTextContent(clearDocuments())
);

server.registerTool(
  'get_current_time',
  {
    description: '获取当前系统时间',
    inputSchema: z.object({})
  },
  async () => {
    const now = resolveNow();
    return toTextContent({
      iso: now.toISOString(),
      locale: now.toLocaleString('zh-CN', { hour12: false }),
      weekday: WEEKDAY_LABELS[now.getDay()]
    });
  }
);

server.registerTool(
  'query_courses',
  {
    description:
      '查询真实课表。数据字段：班级（className）、星期（day）、节次（time）、课程名（courseName）、老师（teacher）、教室（location）。' +
      '按班级查某天上的课：传 className + day。按时间查某节上什么课：传 className + day + time。' +
      '查某门课在哪上：传 className + courseName（课程名支持缩写，如「JS」「TS」）。查某位老师：传 teacher。查某个教室：传 location。' +
      'className 必须使用「25级软件3班」或「25级软件技术4班」；day 使用「周一」到「周日」；time 使用「1-2节」「3-4节」「5-6节」「7-8节」。',
    inputSchema: z.object({
      className: z.string().optional().describe('班级名称，如「25级软件3班」「25级软件技术4班」'),
      day: z.string().optional().describe('星期，如「周一」「星期二」'),
      time: z.string().optional().describe('节次，如「1-2节」「3-4节」「5-6节」「7-8节」'),
      courseName: z.string().optional().describe('课程名称，支持缩写，如「JavaScript」「JS」「数据结构」'),
      teacher: z.string().optional().describe('老师姓名，如「黄月」「张杰」'),
      location: z.string().optional().describe('教室，如「10-211」「操场」'),
      keyword: z.string().optional().describe('任意关键词，会在课程名/老师/教室/班级/星期/节次中一起匹配'),
      listAll: z.boolean().optional().describe('是否返回全部课表数据（调试用，正常问答不要使用）')
    })
  },
  async (args = {}) => {
    try {
      return toTextContent(queryCourses(args));
    } catch (error) {
      return toErrorContent(error);
    }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
