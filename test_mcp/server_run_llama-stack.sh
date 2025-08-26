export INFERENCE_MODEL="$(cat model.txt)"
cd ..
uv run llama stack run test_mcp/run.yaml --image-type venv
