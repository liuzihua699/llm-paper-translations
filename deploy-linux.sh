#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未安装 Docker。请先安装 Docker Engine 与 Compose 插件。" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "错误：当前 Docker 缺少 Compose 插件。" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "错误：无法连接 Docker daemon，请启动 Docker 或检查当前用户权限。" >&2
  exit 1
fi

SITE_PORT="${SITE_PORT:-8080}"
export SITE_PORT
if ! [[ "$SITE_PORT" =~ ^[0-9]+$ ]] || ((SITE_PORT < 1 || SITE_PORT > 65535)); then
  echo "错误：SITE_PORT 必须是 1 到 65535 之间的端口号。" >&2
  exit 1
fi

echo "正在构建并启动译文站点（端口 ${SITE_PORT}）……"
docker compose up -d --build --remove-orphans

CONTAINER_ID="$(docker compose ps -q site)"
for _ in $(seq 1 30); do
  STATUS="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER_ID")"
  if [[ "$STATUS" == "healthy" ]]; then
    break
  fi
  if [[ "$STATUS" == "unhealthy" || "$STATUS" == "exited" || "$STATUS" == "dead" ]]; then
    docker compose logs --tail=50 site >&2
    echo "错误：站点容器状态为 ${STATUS}。" >&2
    exit 1
  fi
  sleep 1
done

if [[ "${STATUS:-unknown}" != "healthy" ]]; then
  docker compose logs --tail=50 site >&2
  echo "错误：站点未在 30 秒内通过健康检查。" >&2
  exit 1
fi

echo
docker compose ps
echo
echo "部署完成："
echo "  本机访问：http://127.0.0.1:${SITE_PORT}"
echo "  公网访问：http://<服务器公网 IP>:${SITE_PORT}"
echo
echo "如果公网无法打开，请在云平台安全组和 Linux 防火墙中放行 TCP ${SITE_PORT}。"
