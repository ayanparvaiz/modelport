#!/usr/bin/env bash
# Copy the MobileNetV3 bundle made by the Python CLI into this app's assets,
# and download a sample photo. Run from anywhere:
#
#   cd python
#   uv run modelport export torchvision:mobilenet_v3_small
#   uv run modelport quantize ../dist/mobilenet_v3_small --fp16 --int8
#   ../dart/packages/modelport_onnx/example/tool/copy_bundle.sh
set -euo pipefail
app="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$app/../../../.." && pwd)"
src="${1:-$repo/dist/mobilenet_v3_small}"
dst="$app/assets/models/mobilenet_v3_small"

if [ ! -f "$src/modelport.json" ]; then
  echo "No bundle at $src. Export it first (see the comment at the top of this script)." >&2
  exit 1
fi
mkdir -p "$dst"
cp "$src/modelport.json" "$src/labels.txt" "$dst/"
for dir in onnx-fp32 onnx-fp16 onnx-int8 golden; do
  if [ -d "$src/$dir" ]; then
    mkdir -p "$dst/$dir"
    cp "$src/$dir/"* "$dst/$dir/"
  fi
done
find "$dst" -name '*.sha256' -delete

mkdir -p "$app/assets/images"
if [ ! -f "$app/assets/images/dog.jpg" ]; then
  curl -sSL -o "$app/assets/images/dog.jpg" https://github.com/pytorch/hub/raw/master/images/dog.jpg
fi
echo "Copied $src to $dst"
