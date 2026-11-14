<template>
  <article class="tool-card" :class="{ 'tool-card-schedule': isSchedule }">
    <div class="tool-topline">
      <strong>{{ title }}</strong>
      <span :class="['tool-status', `tool-${tool.status}`]">{{ statusLabel }}</span>
    </div>

    <template v-if="isSchedule">
      <div class="schedule-meta">
        <span v-if="filterDay" class="schedule-chip">查询：{{ filterDay }}</span>
        <span v-if="filterClass" class="schedule-chip soft">{{ filterClass }}</span>
        <span class="schedule-chip soft">{{ courses.length }} 门课</span>
        <button type="button" class="schedule-toggle" @click="showRaw = !showRaw">
          {{ showRaw ? '收起原始数据' : '原始数据' }}
        </button>
      </div>

      <div v-if="courses.length" class="schedule-list">
        <div v-for="(course, index) in courses" :key="`${course.courseName}-${index}`" class="schedule-item">
          <div class="schedule-time">
            <strong>{{ course.time }}</strong>
            <span>{{ periodText(course) }}</span>
          </div>
          <div class="schedule-body">
            <strong class="schedule-name">{{ course.courseName }}</strong>
            <p class="schedule-sub">
              <span v-if="course.teacher">教师 {{ course.teacher }}</span>
              <span v-if="course.location">教室 {{ course.location }}</span>
            </p>
          </div>
          <div class="schedule-tags">
            <span v-if="course.day" class="schedule-tag">{{ course.day }}</span>
            <span v-if="course.className" class="schedule-tag">{{ shortClass(course.className) }}</span>
          </div>
        </div>
      </div>
      <p v-else class="schedule-empty">{{ emptyText }}</p>

      <pre v-if="showRaw" class="schedule-raw">{{ rawText }}</pre>
    </template>

    <template v-else>
      <pre>{{ argsText }}</pre>
      <p v-if="plainResult">{{ plainResult }}</p>
    </template>
  </article>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ToolInvocation } from '@/types/chat';

interface CourseRecord {
  courseName?: string;
  teacher?: string;
  location?: string;
  className?: string;
  day?: string;
  time?: string;
  timeRange?: { start?: string; end?: string } | null;
}

interface CoursePayload {
  courses?: CourseRecord[];
  count?: number;
  message?: string;
  filters?: { className?: string; day?: string; keyword?: string };
  error?: string;
  details?: string;
}

const props = defineProps<{
  tool: ToolInvocation;
}>();

const showRaw = ref(false);

// 单节作息（与后端 SYSTEM_PROMPT、TIME_SLOTS 保持一致）
const PERIOD_TIMES: Record<string, string> = {
  '1-2节': '08:30-10:00',
  '3-4节': '10:20-11:50',
  '5-6节': '14:30-16:00',
  '7-8节': '16:20-17:50',
  '9-10节': '19:00-20:30'
};

const payload = computed<CoursePayload | null>(() => {
  const raw = props.tool.result as unknown;
  if (!raw) return null;
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw) as CoursePayload;
    } catch {
      return null;
    }
  }
  return raw as CoursePayload;
});

const isSchedule = computed(() => props.tool.name === 'query_courses' && Array.isArray(payload.value?.courses));

const courses = computed(() => payload.value?.courses ?? []);
const filterDay = computed(() => payload.value?.filters?.day || '');
const filterClass = computed(() => payload.value?.filters?.className || '');

const title = computed(() => (props.tool.name === 'query_courses' ? '课表查询' : props.tool.name || '工具调用'));

const statusLabel = computed(() => {
  switch (props.tool.status) {
    case 'running':
      return '查询中';
    case 'success':
      return '已完成';
    case 'error':
      return '查询失败';
    default:
      return '等待中';
  }
});

const emptyText = computed(() => payload.value?.message || '没有查询到符合条件的课程');

const rawText = computed(() => JSON.stringify(payload.value ?? props.tool.result, null, 2));

const argsText = computed(() => JSON.stringify(props.tool.args ?? {}, null, 2));

const plainResult = computed(() => {
  const raw = props.tool.result;
  if (!raw) return '';
  return typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
});

function periodText(course: CourseRecord): string {
  const range = course.timeRange;
  if (range && range.start && range.end) {
    return `${range.start}-${range.end}`;
  }
  const key = (course.time || '').replace(/\s/g, '');
  return PERIOD_TIMES[key] || '';
}

function shortClass(className: string): string {
  return className.replace(/^25级/, '');
}
</script>
