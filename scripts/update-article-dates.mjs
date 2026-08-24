import { promises as fs } from "node:fs";

import {
  indexPath,
  projectRoot,
  publishedArticles,
  resolveInside,
} from "./site-paths.mjs";

const dateStyles = `    article .publication-date {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 30px;
      color: #77746d;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
      font-size: 13.5px;
      line-height: 1.5;
    }
    article .publication-date-label { padding-right: 10px; color: #4e514d; font-weight: 680; border-right: 1px solid #d8d5ce; }
    article .publication-date time { font-variant-numeric: tabular-nums; }
`;

function formatDate(date) {
  const [year, month, day] = date.split("-").map(Number);
  return `${year} 年 ${month} 月 ${day} 日`;
}

function dateMarkup(date) {
  return `<p class="publication-date"><span class="publication-date-label">论文发布日期</span><time datetime="${date}">${formatDate(date)}</time></p>`;
}

const indexHtml = await fs.readFile(indexPath, "utf8");
const articles = publishedArticles(indexHtml);
if (articles.length !== 78) {
  throw new Error(`应更新 78 篇文章，实际从首页识别到 ${articles.length} 篇`);
}

let updatedCount = 0;
for (const article of articles) {
  const articleFile = resolveInside(projectRoot, article.path);
  let articleHtml = await fs.readFile(articleFile, "utf8");

  articleHtml = articleHtml.replace(
    "article h1 { margin: 0 0 27px;",
    "article h1 { margin: 0 0 14px;",
  );
  if (!articleHtml.includes("article .publication-date {")) {
    articleHtml = articleHtml.replace(
      /(    article h1 \{[^\n]+\}\n)/,
      `$1${dateStyles}`,
    );
  }

  const markup = dateMarkup(article.date);
  if (articleHtml.includes('<p class="publication-date">')) {
    articleHtml = articleHtml.replace(
      /<p class="publication-date">[\s\S]*?<\/p>/,
      markup,
    );
  } else {
    articleHtml = articleHtml.replace(
      /(<main><article><h1\b[^>]*>[\s\S]*?<\/h1>)/,
      `$1\n${markup}`,
    );
  }

  const dateCount = [...articleHtml.matchAll(/<p class="publication-date">/g)].length;
  if (dateCount !== 1 || !articleHtml.includes(dateStyles.trim())) {
    throw new Error(`无法为文章写入发布日期：${article.path}`);
  }

  await fs.writeFile(articleFile, articleHtml, "utf8");
  updatedCount += 1;
}

console.log(`已同步 ${updatedCount} 篇文章的论文发布日期`);
