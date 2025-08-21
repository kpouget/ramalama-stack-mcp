import logging
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
import json

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INFERENCE_MODEL = "llama3.2"
LLAMA_STACK_PORT = 8321

# ---------------------------
# Create LlamaStack HTTP client
# ---------------------------
def create_http_client():
    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}",
        timeout=2000.0,
    )

client = create_http_client()

# ---------------------------
# Cleanup previously registered MCP toolgroups
# ---------------------------

for toolgroup in client.toolgroups.list():
    print(f"Unregistering MCP toolgroup: {toolgroup.identifier}")
    client.toolgroups.unregister(toolgroup_id=toolgroup.identifier)

# ---------------------------
# Register your test MCP server
# ---------------------------
client.toolgroups.register(
    toolgroup_id="mcp::demo",
    provider_id="model-context-protocol",
    mcp_endpoint={"uri": "http://localhost:8000/sse"},
)
# ---------------------------
# Cleanup previous agents
# ---------------------------
for agent_info in client.agents.list().data:
    agent_id = agent_info["agent_id"]
    print(f"Unregistering agent: {agent_id}")
    client.agents.delete(agent_id=agent_id)

# ---------------------------
# Create a new agent with your MCP tool
# ---------------------------
agent = Agent(
    client=client,
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant with access to MCP tools.",
    enable_session_persistence=False,
    tools=["mcp::demo"],  # just the toolgroup identifier
    sampling_params={"max_tokens": 2048},
)
print("\nCurrent Agent ID: ", agent.agent_id)

# ---------------------------
# Start a new session
# ---------------------------
session_id = agent.create_session(session_name="mcp_test_session")
print("\nStarted Agent Session: ", session_id)

# ---------------------------
# Run a test turn
# ---------------------------
turn_response = agent.create_turn(
    session_id=session_id,
    messages=[{"role": "user", "content": "Call the greet tool with name Brian"}],
    stream=True,
)

# ---------------------------
# Print streamed output
# ---------------------------
for log in AgentEventLogger().log(turn_response):
    log.print()
