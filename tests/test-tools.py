import uuid
import json
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import AgentConfig

LLAMA_STACK_PORT = 8321
INFERENCE_MODEL = "qwen2.5"

# ---------------------- Tool Definitions ----------------------
available_tools = [
    {
        "tool_name": "favorite_color_tool",
        "description": "Returns the favorite color for a person given their city and country.",
        "parameters": {
            "city": {"param_type": "string", "required": True},
            "country": {"param_type": "string", "required": True},
        },
    },
    {
        "tool_name": "favorite_hockey_tool",
        "description": "Returns the favorite hockey team for a person given their city and country.",
        "parameters": {
            "city": {"param_type": "string", "required": True},
            "country": {"param_type": "string", "required": True},
        },
    },
]

# ---------------------- Tool Logic ----------------------
def get_favorite_color(args):
    city = args.get("city")
    country = args.get("country")
    if city == "Ottawa" and country == "Canada":
        return "Favorite color for Ottawa, Canada is black."
    elif city == "Montreal" and country == "Canada":
        return "Favorite color for Montreal, Canada is red."
    return "City or country not recognized. Assistant, please ask the user again."

def get_favorite_hockey_team(args):
    city = args.get("city")
    country = args.get("country")
    if city == "Ottawa" and country == "Canada":
        return "Favorite hockey team for Ottawa, Canada is The Ottawa Senators."
    elif city == "Montreal" and country == "Canada":
        return "Favorite hockey team for Montreal, Canada is The Montreal Canadiens."
    return "City or country not recognized. Assistant, please ask the user again."

funcs = {
    "favorite_color_tool": get_favorite_color,
    "favorite_hockey_tool": get_favorite_hockey_team,
}

# ---------------------- Client ----------------------
def create_http_client():
    return LlamaStackClient(base_url=f"http://localhost:{LLAMA_STACK_PORT}", timeout=2000.0)

client = create_http_client()

# ---------------------- Response Handler ----------------------
def handle_response(messages, response):
    messages.append(response.completion_message)

    if response.completion_message.tool_calls:
        for tool in response.completion_message.tool_calls:
            print(f"[TOOL CALL] {tool.tool_name} with args: {tool.arguments}")
            tool_fn = funcs.get(tool.tool_name)

            if tool_fn:
                result = tool_fn(tool.arguments)
            else:
                result = "Invalid tool called."

            messages.append({
                "role": "tool",
                "content": result,
                "call_id": tool.call_id,
                "tool_name": tool.tool_name,
            })

        # Follow-up call to model after tool execution
        follow_up = client.inference.chat_completion(
            messages=messages,
            model_id=INFERENCE_MODEL,
            tools=available_tools,
        )
        return handle_response(messages, follow_up)

    return response.completion_message.content

# ---------------------- Questions ----------------------
questions = [
    "What is my favorite color?",
    "My city is Ottawa",
    "My country is Canada",
    "I moved to Montreal. What is my favorite color now?",
    "My city is Montreal and my country is Canada",
    "What is the fastest car in the world?",
    "My city is Ottawa and my country is Canada, what is my favorite color?",
    "What is my favorite hockey team?",
    "My city is Montreal and my country is Canada",
    "Who was the first president of the United States?",
]

# ---------------------- Chat Loop ----------------------
messages = []
for i, question in enumerate(questions):
    print(f"\n[QUESTION {i+1}] {question}")
    messages.append({"role": "user", "content": question})

    response = client.inference.chat_completion(
        messages=messages,
        model_id=INFERENCE_MODEL,
        tools=available_tools,
    )

    final_response = handle_response(messages, response)
    print(f"[RESPONSE] {final_response}")
