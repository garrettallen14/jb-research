from openai import OpenAI

# Connect to your tunneled endpoint
client = OpenAI(
    base_url="https://l0rm96tmeqicmo-8000.proxy.runpod.net/v1",
    api_key="dummy"  # vLLM doesn't require real API key
)

# Test with thinking tokens and logprobs
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Solve this step by step: What is 15 * 23?"}
    ],
    logprobs=True,
    top_logprobs=10,
    max_tokens=500,
    temperature=0.7
)

print("Response:", response.choices[0].message.content)
print("Logprobs available:", bool(response.choices[0].logprobs))

# For thinking token prefilling, you'd also install:
# pip install openai-harmony gpt-oss
# and use the harmony encoding as shown earlier