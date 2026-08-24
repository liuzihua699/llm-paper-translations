import { promises as fs } from "node:fs";
import path from "node:path";

import { articleOrganizations } from "./article-organizations.mjs";
import { projectRoot } from "./site-paths.mjs";

const indexFiles = ["index.html", "译文索引.html"];
const organizationStyles = `    .paper .organization { display: flex; align-items: baseline; gap: 8px; max-width: none; margin: -2px 0 11px; color: #69707a; font-size: 11.5px; line-height: 1.5; text-align: left; }
    .organization span { flex: 0 0 auto; color: #9ca3af; font-weight: 650; }
`;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function updateIndex(source, fileName) {
  const seenTitles = new Set();
  let cardCount = 0;
  let html = source.replace(
    /<article class="paper">[\s\S]*?<\/article>/g,
    (card) => {
      cardCount += 1;
      const title = card.match(/<h3><a\b[^>]*>([^<]+)<\/a><\/h3>/)?.[1];
      if (!title) throw new Error(`${fileName} 中有文章卡片缺少标题`);
      const organization = articleOrganizations.get(title);
      if (!organization) throw new Error(`缺少《${title}》的发布主体`);
      seenTitles.add(title);

      const cleanCard = card.replace(
        /\n\s*<p class="organization"><span>发布主体<\/span>[^<]*<\/p>/,
        "",
      );
      return cleanCard.replace(
        /(<h3><a\b[^>]*>[^<]+<\/a><\/h3>)/,
        `$1\n            <p class="organization"><span>发布主体</span>${escapeHtml(organization)}</p>`,
      );
    },
  );

  if (cardCount !== 78) {
    throw new Error(`${fileName} 应有 78 篇文章，实际为 ${cardCount} 篇`);
  }
  const unusedTitles = [...articleOrganizations.keys()].filter((title) => !seenTitles.has(title));
  if (unusedTitles.length) {
    throw new Error(`未匹配的主体数据：${unusedTitles.join("、")}`);
  }

  if (!html.includes(".paper .organization {")) {
    html = html.replace(
      /(    \.paper p \{[^\n]+\}\n)/,
      `$1${organizationStyles}`,
    );
  }
  if (!html.includes(organizationStyles.trim())) {
    throw new Error(`无法为 ${fileName} 写入发布主体样式`);
  }
  return html;
}

for (const fileName of indexFiles) {
  const filePath = path.join(projectRoot, fileName);
  const source = await fs.readFile(filePath, "utf8");
  await fs.writeFile(filePath, updateIndex(source, fileName), "utf8");
}

console.log(`已向 ${indexFiles.length} 个索引页同步 78 篇文章的发布主体`);
