#!/usr/bin/env bash
set -euo pipefail

echo "Building WASM package..."
wasm-pack build --target web --out-dir pkg

echo "Done. Serve with:"
echo "  python3 -m http.server 8080"
echo "Then open http://localhost:8080"
