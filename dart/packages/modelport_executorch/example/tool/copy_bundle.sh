#!/usr/bin/env bash
# Copy the MobileNetV3 bundle made by the Python CLI into this app's assets,
# and download a sample photo:
#
#   cd python
#   uv run modelport export torchvision:mobilenet_v3_small --target onnx,executorch
#   ../dart/packages/modelport_executorch/example/tool/copy_bundle.sh
set -euo pipefail
app="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$app/../../../.." && pwd)"
src="${1:-$repo/dist/mobilenet_v3_small}"
dst="$app/assets/models/mobilenet_v3_small"

if [ ! -f "$src/executorch-xnnpack-fp32/model.pte" ]; then
  echo "No ExecuTorch variant in $src. Export it first (see the top of this script)." >&2
  exit 1
fi
mkdir -p "$dst/executorch-xnnpack-fp32" "$dst/golden"
cp "$src/modelport.json" "$src/labels.txt" "$dst/"
cp "$src/executorch-xnnpack-fp32/model.pte" "$dst/executorch-xnnpack-fp32/"
cp "$src/golden/"*.bin "$dst/golden/"

mkdir -p "$app/assets/images"
if [ ! -f "$app/assets/images/dog.jpg" ]; then
  curl -sSL -o "$app/assets/images/dog.jpg" https://github.com/pytorch/hub/raw/master/images/dog.jpg
fi
echo "Copied $src to $dst"
