Read this doc:

https://cookbook.openai.com/articles/gpt-oss/run-vllm


1. Startup prime-intellect instance
2.
```
sudo snap install astral-uv
# Step 1: Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 2: Add uv to PATH
export PATH="/root/.local/bin:$PATH"

# Step 3: Install vLLM (regular version worked!)
uv pip install --system vllm torch torchvision

# Step 4: Start the server
vllm serve openai/gpt-oss-20b
```

3. 
```
ssh -L 8000:localhost:8000 root@194.68.245.47 -p 22182
```