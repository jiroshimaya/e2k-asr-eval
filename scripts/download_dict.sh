#!/bin/bash
set -e

# スクリプト自身の絶対パスからディレクトリ部分を取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$SCRIPT_DIR/../local"
DEST_FILE="$DEST_DIR/bep-eng.dic"

SRC_URL="https://fastapi.metacpan.org/source/MASH/Lingua-JA-Yomi-0.01/lib/Lingua/JA/bep-eng.dic"

mkdir -p "$DEST_DIR"

echo "Downloading bep-eng.dic from $SRC_URL ..."
curl -L "$SRC_URL" -o "$DEST_FILE"

echo "Saved to: $DEST_FILE"