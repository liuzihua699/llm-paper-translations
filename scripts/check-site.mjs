import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  projectRoot,
  publishedArticles,
  publishedArticlePaths,
  resolveInside,
} from "./site-paths.mjs";

const requestedRoot = process.argv[2]
  ? path.resolve(projectRoot, process.argv[2])
  : projectRoot;
const indexFile = path.join(requestedRoot, "index.html");

function fail(message) {
  console.error(`检查失败：${message}`);
  process.exitCode = 1;
}

function findPdf(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      const nested = findPdf(target);
      if (nested) return nested;
    } else if (entry.isFile() && path.extname(entry.name).toLowerCase() === ".pdf") {
      return target;
    }
  }
  return null;
}

if (!existsSync(indexFile)) {
  fail(`${indexFile} 不存在`);
} else {
  const indexHtml = readFileSync(indexFile, "utf8");
  const articles = publishedArticles(indexHtml);
  const articlePaths = publishedArticlePaths(indexHtml);
  const markdownLinks = [
    ...indexHtml.matchAll(/<a href="([^"]+\.md)">Markdown<\/a>/g),
  ].map((match) => decodeURIComponent(match[1]));
  const sourceLinks = [
    ...indexHtml.matchAll(
      /<a href="(https?:\/\/[^"]+)" target="_blank" rel="noopener noreferrer">阅读原文<\/a>/g,
    ),
  ].map((match) => match[1]);

  if (articlePaths.length !== 78) fail(`网页版链接应为 78 个，实际为 ${articlePaths.length} 个`);
  const organizations = articles.filter((article) => article.organization?.trim());
  if (organizations.length !== 78) {
    fail(`发布主体标注应为 78 个，实际为 ${organizations.length} 个`);
  }
  if (markdownLinks.length !== 78) fail(`Markdown 链接应为 78 个，实际为 ${markdownLinks.length} 个`);
  if (sourceLinks.length !== 78) fail(`原文链接应为 78 个，实际为 ${sourceLinks.length} 个`);
  if (new Set(sourceLinks).size !== sourceLinks.length) fail("原文链接存在重复");

  for (const relativePath of [...articlePaths, ...markdownLinks]) {
    const target = resolveInside(requestedRoot, relativePath);
    if (!existsSync(target)) fail(`本地链接不存在：${relativePath}`);
  }

  for (const article of articles) {
    const articleFile = resolveInside(requestedRoot, article.path);
    if (!existsSync(articleFile)) continue;
    const articleHtml = readFileSync(articleFile, "utf8");
    const publicationDate = articleHtml.match(
      /<p class="publication-date"><span class="publication-date-label">论文发布日期<\/span><time datetime="(\d{4}-\d{2}-\d{2})">([^<]+)<\/time><\/p>/,
    );
    if (!publicationDate) {
      fail(`${article.path} 缺少标题下方的论文发布日期`);
    } else if (publicationDate[1] !== article.date) {
      fail(`${article.path} 的发布日期与首页不一致`);
    } else {
      const [year, month, day] = article.date.split("-").map(Number);
      if (publicationDate[2] !== `${year} 年 ${month} 月 ${day} 日`) {
        fail(`${article.path} 的发布日期显示格式不正确`);
      }
    }
    for (const match of articleHtml.matchAll(/\bsrc="([^"]+)"/g)) {
      const asset = match[1];
      if (/^(?:https?:|data:)/.test(asset)) continue;
      const assetPath = resolveInside(path.dirname(articleFile), decodeURIComponent(asset));
      if (!existsSync(assetPath)) fail(`${article.path} 缺少资源：${asset}`);
    }
  }

  if (indexHtml.toLowerCase().includes(".pdf")) fail("首页不应链接原始 PDF");
  const pdfRoots = requestedRoot === projectRoot
    ? [...new Set(articlePaths.map((articlePath) => path.dirname(resolveInside(requestedRoot, articlePath))))]
    : [requestedRoot];
  for (const pdfRoot of pdfRoots) {
    const pdf = findPdf(pdfRoot);
    if (pdf) {
      fail(`站点目录不应包含 PDF：${path.relative(requestedRoot, pdf)}`);
      break;
    }
  }

  if (!process.exitCode) {
    console.log(`检查通过：${path.relative(projectRoot, requestedRoot) || "."} 中的 78 篇文章及资源完整`);
  }
}
