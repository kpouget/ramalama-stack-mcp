cd ..
export INFERENCE_MODEL=qwen
uv run llama stack run test_mcp/run.yaml --image-type venv
