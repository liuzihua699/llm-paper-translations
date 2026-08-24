import { fileURLToPath } from "node:url";
import path from "node:path";

export const projectRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
export const indexPath = path.join(projectRoot, "index.html");
export const distPath = path.join(projectRoot, "dist");

export function publishedArticles(indexHtml) {
  return [...indexHtml.matchAll(/<article class="paper">([\s\S]*?)<\/article>/g)].map(
    ([, card]) => {
      const date = card.match(/<time datetime="(\d{4}-\d{2}-\d{2})">/)?.[1];
      const encodedPath = card.match(/<a href="([^"]+\.html)">阅读网页版<\/a>/)?.[1];
      if (!date || !encodedPath) {
        throw new Error("首页文章缺少发布日期或网页版链接");
      }
      const organization = card.match(
        /<p class="organization"><span>发布主体<\/span>([^<]+)<\/p>/,
      )?.[1];
      return { date, organization, path: decodeURIComponent(encodedPath) };
    },
  );
}

export function publishedArticlePaths(indexHtml) {
  return [...new Set(publishedArticles(indexHtml).map((article) => article.path))];
}

export function resolveInside(root, relativePath) {
  const resolved = path.resolve(root, relativePath);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error(`路径越界：${relativePath}`);
  }
  return resolved;
}
