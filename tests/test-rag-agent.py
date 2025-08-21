import uuid
import logging
from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INFERENCE_MODEL = "llama3.2"
LLAMA_STACK_PORT = 8321

def create_http_client():
    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}",
        timeout=2000.0,
    )

client = create_http_client()

# Clean up old vector DBs
for vector_db_id in client.vector_dbs.list():
    print(f"Unregistering vector database: {vector_db_id.identifier}")
    client.vector_dbs.unregister(vector_db_id=vector_db_id.identifier)

# Register vector DB
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="milvus",
)

# Ingest doc
source = "https://github.com/containers/podman-desktop-extension-ai-lab/blob/main/README.md"
print("rag_tool> Ingesting document:", source)
document = RAGDocument(
    document_id="document_1",
    content=source,
    mime_type="text/html",
    metadata={},
)

print("inserting...")
client.tool_runtime.rag_tool.insert(
    documents=[document], vector_db_id=vector_db_id, chunk_size_in_tokens=200,
)

print("\nCreated Vectordb ", vector_db_id)

# Cleanup Agents
response = client.agents.list()
for agent in response.data:
    agent_id = agent["agent_id"]
    print(f"Unregistering agent: {agent_id}")
    client.agents.delete(agent_id=agent_id)

agent = Agent(
    client=client,
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant.",
    enable_session_persistence=False,
    tools=[{ 
        "name": "builtin::rag/knowledge_search",
        "args": {"vector_db_ids": [vector_db_id]}
    }],
    sampling_params={
        "max_tokens": 2048,
    },
)
print("\nCurrent Agent: ")
print(agent.agent_id)

# Start session
session_id = agent.create_session(session_name="rag")

print("\nStarted Agent Session: ", session_id)

# Run turn
turn_response = agent.create_turn(
    session_id=session_id,
    messages=[{"role": "user", "content": "What is podman desktop ai lab"}],
    stream=True,
)

# Print response
for log in AgentEventLogger().log(turn_response):
    log.print()

# curl -N -X POST http://localhost:8321/v1/agents/a7b66240-0fc2-416b-b803-0f652814da83/session/f44c0d82-8048-429f-81f6-74fb43858648/turn \
#   -H "Content-Type: application/json" \
#   -d '{
#     "messages": [
#       {
#         "role": "user",
#         "content": "What is podman desktop ai lab and what is it used for"
#       }
#     ],
#     "stream": true
#   }'

# We need to change inline milvus to remote milvus and spin up a seperate container. Maybe then we can use hybrid search?
# vector_io:
#   - provider_id: milvus
#     provider_type: inline::milvus
#     config:
#       db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/ramalama/milvus.db}
#       kvstore:
#         type: sqlite
#         namespace: null
#         db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/ramalama/milvus_registry.db}
