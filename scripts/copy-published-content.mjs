import { promises as fs } from "node:fs";
import path from "node:path";

import {
  distPath,
  indexPath,
  projectRoot,
  publishedArticlePaths,
  resolveInside,
} from "./site-paths.mjs";

let fileCount = 0;
let byteCount = 0;

async function linkOrCopy(source, target) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  try {
    await fs.link(source, target);
  } catch (error) {
    if (!["EXDEV", "EPERM", "EACCES", "ENOTSUP"].includes(error.code)) {
      throw error;
    }
    await fs.copyFile(source, target);
  }
  const stats = await fs.stat(source);
  fileCount += 1;
  byteCount += stats.size;
}

async function copyTree(sourceDir, targetDir) {
  for (const entry of await fs.readdir(sourceDir, { withFileTypes: true })) {
    const source = path.join(sourceDir, entry.name);
    const target = path.join(targetDir, entry.name);
    if (entry.isDirectory()) {
      await copyTree(source, target);
    } else if (entry.isFile()) {
      if (path.extname(entry.name).toLowerCase() === ".pdf") continue;
      await linkOrCopy(source, target);
    }
  }
}

const indexHtml = await fs.readFile(indexPath, "utf8");
const articlePaths = publishedArticlePaths(indexHtml);
if (articlePaths.length !== 78) {
  throw new Error(`应发布 78 篇文章，实际从首页识别到 ${articlePaths.length} 篇`);
}

for (const fileName of ["译文索引.html", "译文索引.md"]) {
  await linkOrCopy(path.join(projectRoot, fileName), path.join(distPath, fileName));
}

for (const articlePath of articlePaths) {
  const articleDir = path.dirname(resolveInside(projectRoot, articlePath));
  const relativeDir = path.relative(projectRoot, articleDir);
  await copyTree(articleDir, path.join(distPath, relativeDir));
}

console.log(
  `已加入 ${articlePaths.length} 篇文章、${fileCount.toLocaleString("zh-CN")} 个内容文件（${(
    byteCount /
    1024 /
    1024
  ).toFixed(2)} MiB）`,
);
