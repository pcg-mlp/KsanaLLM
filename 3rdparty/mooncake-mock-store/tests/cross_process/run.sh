#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/../../build}"
SHARED_DIR="/tmp/mooncake_mock_test_$$"

rm -rf "$SHARED_DIR"
mkdir -p "$SHARED_DIR"

echo "=== Cross-process test (shared dir: $SHARED_DIR) ==="

# Run writer, then reader as separate processes
"$BUILD_DIR/writer" "$SHARED_DIR"
"$BUILD_DIR/reader" "$SHARED_DIR"

rm -rf "$SHARED_DIR"
echo "=== DONE ==="
