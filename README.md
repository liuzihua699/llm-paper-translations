# 大语言模型论文中文译文站

这是一个可直接启动和部署的 npm Web 项目。根路径 `/` 是中文译文索引，索引中的 78 篇完整文章、Markdown 文档和原页图片均可访问；原始 PDF 与未完成草稿不会进入生产构建。

## Linux 一键部署（推荐）

服务器需要安装 Docker Engine 与 Docker Compose 插件。进入项目目录后执行：

```bash
bash deploy-linux.sh
```

默认发布到 `0.0.0.0:8080`，访问地址为：

```text
http://服务器公网IP:8080
```

如需直接使用 80 端口：

```bash
SITE_PORT=80 bash deploy-linux.sh
```

容器使用 `restart: unless-stopped`，服务器重启后会自动恢复。常用运维命令：

```bash
docker compose ps
docker compose logs -f
bash stop-linux.sh
```

如果公网无法访问，需要在云服务安全组和 Linux 防火墙中放行对应 TCP 端口。例如 Ubuntu 使用：

```bash
sudo ufw allow 8080/tcp
```

## Linux 纯 npm 启动

不使用 Docker 时，需要 Node.js 22 或更高版本：

```bash
npm install
```

开发预览（仅监听本机地址）：

```bash
npm run dev
```

直接启动站点：

```bash
npm start
```

`npm start` 默认监听 `0.0.0.0:4173`。如需改端口：

```bash
PORT=8080 npm start
```

## 临时公网分享

```bash
npm run share
```

命令会启动本地站点，并显示一个 `https://*.trycloudflare.com` 临时公网地址。该地址不要求 Cloudflare 账号，进程停止后即失效，适合临时验收和分享。

## 长期部署

项目使用 Vite 生成生产目录，并通过 Cloudflare Workers Static Assets 发布。首次发布前只需登录一次：

```bash
npm run cf:login
```

之后每次一条命令发布：

```bash
npm run deploy
```

部署完成后会得到稳定的 `workers.dev` 公网地址，也可以在 Cloudflare 控制台绑定自己的域名。Wrangler 会按内容哈希增量上传，后续发布不会重复上传未变化的图片。

## 内容维护

重新生成索引与首页：

```bash
python3 ../tmp/build_translation_index.py
npm run content:organizations
```

`译文索引.html` 与首页中的发布主体由 `scripts/article-organizations.mjs` 统一维护；重新生成索引后，上述命令会把它们同步到全部文章卡片。

索引日期变更或重新生成文章 HTML 后，将首页中的论文首次公开日期同步到所有文章标题下方：

```bash
npm run content:dates
```

发布前校验：

```bash
npm run check
npm run build
```

`npm run build` 只从首页读取已发布的 78 篇文章，并将它们及其资源加入 `dist/`。本地构建优先使用硬链接，因此不会再额外占用约 1.09 GB 磁盘空间；跨文件系统或不支持硬链接时会自动改为复制。
