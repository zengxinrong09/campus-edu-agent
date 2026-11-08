# 校园教务智能体：前后端同进程部署
# 前端由 vite 构建到 dist/，由 Express 通过 express.static 托管，只需暴露一个端口
FROM node:20-alpine

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY package*.json ./
RUN npm ci

# 复制源码并构建前端（构建需要 devDependencies，所以 NODE_ENV 不能提前设为 production）
COPY . .
RUN npm run build

# 运行期环境
ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "server/index.js"]
