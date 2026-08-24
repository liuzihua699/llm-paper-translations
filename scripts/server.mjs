import { createReadStream, promises as fs, readFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";

import {
  projectRoot,
  publishedArticlePaths,
  resolveInside,
} from "./site-paths.mjs";

const host = process.env.HOST || "0.0.0.0";
const port = Number(process.env.PORT || 4173);
const siteRoot = process.env.SITE_ROOT
  ? path.resolve(projectRoot, process.env.SITE_ROOT)
  : projectRoot;
const publicRootFiles = new Set(["index.html", "译文索引.html", "译文索引.md"]);
const publishedDirectories = new Set(
  publishedArticlePaths(readFileSync(path.join(siteRoot, "index.html"), "utf8"))
    .map((articlePath) => `${path.dirname(articlePath).split(path.sep).join("/")}/`),
);
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

function sendText(response, status, message) {
  response.writeHead(status, {
    "Content-Type": "text/plain; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(message);
}

const server = createServer(async (request, response) => {
  try {
    if (!request.url || !["GET", "HEAD"].includes(request.method || "")) {
      sendText(response, 405, "Method Not Allowed");
      return;
    }

    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    } catch {
      sendText(response, 400, "Bad Request");
      return;
    }
    if (pathname.includes("\\") || pathname.includes("\0")) {
      sendText(response, 400, "Bad Request");
      return;
    }

    const relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
    if (siteRoot === projectRoot) {
      const normalized = relativePath.split(path.sep).join("/");
      const articleIsPublished = [...publishedDirectories].some((directory) =>
        normalized.startsWith(directory),
      );
      if (!publicRootFiles.has(normalized) && !articleIsPublished) {
        sendText(response, 404, "Not Found");
        return;
      }
    }

    let target;
    try {
      target = resolveInside(siteRoot, relativePath);
    } catch {
      sendText(response, 403, "Forbidden");
      return;
    }
    const stats = await fs.stat(target).catch(() => null);
    if (!stats?.isFile()) {
      sendText(response, 404, "Not Found");
      return;
    }

    const extension = path.extname(target).toLowerCase();
    const cacheControl = extension === ".png" || extension === ".webp"
      ? "public, max-age=604800"
      : "no-cache";
    response.writeHead(200, {
      "Cache-Control": cacheControl,
      "Content-Length": stats.size,
      "Content-Type": contentTypes[extension] || "application/octet-stream",
      "Last-Modified": stats.mtime.toUTCString(),
      "X-Content-Type-Options": "nosniff",
    });
    if (request.method === "HEAD") {
      response.end();
    } else {
      createReadStream(target).pipe(response);
    }
  } catch (error) {
    console.error(error);
    if (!response.headersSent) sendText(response, 500, "Internal Server Error");
    else response.destroy();
  }
});

server.listen(port, host, () => {
  const displayHost = host === "0.0.0.0" ? "localhost" : host;
  console.log(`译文站点已启动：http://${displayHost}:${port}`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
