import uuid
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient
import json

import logging
logging.basicConfig(
    level=logging.DEBUG,  # Show everything from DEBUG and up
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INFERENCE_MODEL = "llama3.2"
LLAMA_STACK_PORT = 8321


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}",  # Your Llama Stack Server URL
        timeout=2000.0,
    )


client = create_http_client()

for vector_db_id in client.vector_dbs.list():
    print(f"Unregistering vector database: {vector_db_id.identifier}")
    client.vector_dbs.unregister(vector_db_id=vector_db_id.identifier)

vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="milvus",
)

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

# Create a session
session_id = agent.create_session(session_name="rag")

# Create a turn with streaming response
turn_response = agent.create_turn(
    session_id=session_id,
    messages=[{"role": "user", "content": "Tell me about Llama models"}],
    stream=False,
)

print(turn_response)

# Log and process the response
for log in AgentEventLogger().log(turn_response):
    log.print()