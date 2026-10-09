#!/usr/bin/env bash
# Copy the files made by spikes/python/export_mobilenet.py into this app's assets.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/../python/out/mobilenet"
for f in model.onnx model.pte pixel_values.bin logits.bin; do
  cp "$src/$f" "$here/assets/$f"
done
echo "copied to $here/assets"
