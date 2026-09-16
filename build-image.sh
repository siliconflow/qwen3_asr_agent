#!/bin/bash
# Qwen3-ASR Agent (proxy sidecar) 本地构建脚本
# 强制规则：每次构建前必须递增 VERSION 文件，否则构建失败
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="$SCRIPT_DIR/VERSION"

REGISTRY="${REGISTRY:-hub.6scloud.com}"
NAMESPACE="${NAMESPACE:-clxuaivn500083i7ncuxpw8cf}"
PROJECT_NAME="qwen3-asr-proxy"
PLATFORM="${PLATFORM:-linux/amd64}"

# 读取版本号
if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: VERSION file not found at $VERSION_FILE"
    echo "Please create it with the next version number (e.g., 1.0.0)"
    exit 1
fi
VERSION=$(tr -d '[:space:]' < "$VERSION_FILE")
DATE_TAG=$(date +%Y%m%d)
TAG="${DATE_TAG}-v${VERSION}"
FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${TAG}"

echo "========================================"
echo "Qwen3-ASR Agent Build"
echo "Version : $VERSION"
echo "Tag     : $TAG"
echo "Registry: $REGISTRY/$NAMESPACE/$PROJECT_NAME"
echo "========================================"
echo ""

# 检查本地/远程是否已有相同 tag 的镜像
check_existing() {
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${FULL_IMAGE}$"; then
        echo "ERROR: Image already exists locally: $FULL_IMAGE"
        echo "Rule: Every build MUST have a unique tag."
        echo "Fix: Edit $VERSION_FILE and bump the version number before rebuilding."
        exit 1
    fi
    if command -v skopeo &>/dev/null; then
        if skopeo inspect "docker://${FULL_IMAGE}" &>/dev/null; then
            echo "ERROR: Image already exists in registry: $FULL_IMAGE"
            echo "Fix: Edit $VERSION_FILE and bump the version number."
            exit 1
        fi
    fi
}

echo "Checking for existing tags..."
check_existing
echo "OK — tag is unique."
echo ""

echo ">>> Building proxy image..."
docker buildx build \
    --platform "$PLATFORM" \
    -t "$FULL_IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$SCRIPT_DIR"

echo ">>> Pushing $FULL_IMAGE"
docker push "$FULL_IMAGE"
echo ""

echo "========================================"
echo "Build completed successfully!"
echo ""
echo "Image: $FULL_IMAGE"
echo ""
echo "Next steps:"
echo "  1. 把 siliconflow/ready-to-use/qwen3-asr/template.jinja 中 proxy-api sidecar 的 image 更新为:"
echo "     $FULL_IMAGE"
echo "  2. 递增 $VERSION_FILE 为下次构建做准备"
echo "  3. git commit -am \"Bump proxy VERSION to $VERSION\" && git push"
echo "========================================"
