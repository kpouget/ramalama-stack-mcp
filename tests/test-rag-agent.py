import uuid
import logging
from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient
from llama_stack_client.types.tool_definition import ToolDefinition

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INFERENCE_MODEL = "qwen2.5"
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
source = "https://www.paulgraham.com/do.html"
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

# Tool definition (this is the critical fix)
tools = [
    ToolDefinition(
        name="builtin::rag/knowledge_search",
        parameters={"vector_db_ids": [vector_db_id]},
    )
]

# Create agent
agent = Agent(
    client=client,
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant.",
    enable_session_persistence=False,
    tools=tools,
    sampling_params={"max_tokens": 2048},
)

# Start session
session_id = agent.create_session(session_name="rag")

# Run turn
turn_response = agent.create_turn(
    session_id=session_id,
    messages=[{"role": "user", "content": "Tell me about llama models"}],
    stream=False,
)

# Print response
for log in AgentEventLogger().log(turn_response):
    log.print()
