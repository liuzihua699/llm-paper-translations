import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createServer } from "node:net";
import path from "node:path";

import { bin as cloudflaredBin, install } from "cloudflared";

import { projectRoot } from "./site-paths.mjs";

const port = Number(process.env.PORT || 4173);
const origin = `http://127.0.0.1:${port}`;
let tunnel;
await new Promise((resolve, reject) => {
  const probe = createServer();
  probe.once("error", () => reject(new Error(`端口 ${port} 已被占用，请先停止已有站点或设置 PORT`)));
  probe.listen(port, "127.0.0.1", () => probe.close(resolve));
}).catch((error) => {
  console.error(error.message);
  process.exit(1);
});

const server = spawn(process.execPath, [path.join(projectRoot, "scripts", "server.mjs")], {
  cwd: projectRoot,
  env: { ...process.env, HOST: "127.0.0.1", PORT: String(port) },
  stdio: "inherit",
});

function stop() {
  if (tunnel && !tunnel.killed) tunnel.kill("SIGTERM");
  if (!server.killed) server.kill("SIGTERM");
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    stop();
    process.exit(0);
  });
}

async function waitForServer() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (server.exitCode !== null) {
      throw new Error(`本地站点启动失败，退出码：${server.exitCode}`);
    }
    try {
      const response = await fetch(origin);
      if (response.ok) return;
    } catch {
      // The server may still be starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("本地站点启动超时");
}

try {
  await waitForServer();
  if (process.env.SHARE_DRY_RUN === "1") {
    console.log("公网分享预检通过");
    stop();
  } else {
    if (!existsSync(cloudflaredBin)) {
      console.log("首次运行：正在下载 Cloudflare Tunnel 客户端…");
      await install(cloudflaredBin);
    }
    console.log("正在创建临时公网地址；按 Ctrl+C 停止分享。\n");
    tunnel = spawn(
      cloudflaredBin,
      ["tunnel", "--no-autoupdate", "--url", origin],
      { cwd: projectRoot, stdio: "inherit" },
    );
    tunnel.on("exit", (code) => {
      stop();
      process.exit(code ?? 0);
    });
  }
} catch (error) {
  stop();
  console.error(error.message);
  process.exit(1);
}
