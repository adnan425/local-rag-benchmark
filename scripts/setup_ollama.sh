#!/usr/bin/env bash
# Pulls the candidate small LLMs and embedding models used by configs/benchmark.yaml.
# Edit this list if you add/remove candidates in the config.
set -euo pipefail

echo "Checking Ollama is installed..."
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Install it from https://ollama.com/download and re-run this script."
  exit 1
fi

MODELS=(
  "llama3.2:3b"
  "gemma4:e4b"
  "phi4-mini"
  "phi4:14b"
  "qwen3.5:4b"
  "qwen3.5:9b"
  "ministral-3:3b"
)

EMBED_MODELS=(
  "nomic-embed-text"
  "qwen3-embedding:0.6b"
)

for m in "${MODELS[@]}"; do
  echo "Pulling $m ..."
  ollama pull "$m"
done

for m in "${EMBED_MODELS[@]}"; do
  echo "Pulling $m ..."
  ollama pull "$m"
done

echo "Done. Run 'ollama list' to confirm."
