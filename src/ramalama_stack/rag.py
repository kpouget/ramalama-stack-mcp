import uuid
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client import Agent, AgentEventLogger, RAGDocument, LlamaStackClient

INFERENCE_MODEL = "llama3.2"
LLAMA_STACK_PORT = 8321


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}"  # Your Llama Stack Server URL
    )


client = create_http_client()

vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="milvus",
)

source = "https://www.paulgraham.com/greatwork.html"
print("rag_tool> Ingesting document:", source)
document = RAGDocument(
    document_id="document_1",
    content=source,
    mime_type="text/html",
    metadata={},
)

print("inserting...")
client.tool_runtime.rag_tool.insert(
    documents=[document], vector_db_id=vector_db_id, chunk_size_in_tokens=1024,
)

agent = Agent(
    client,
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant",
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        }
    ],
)

prompt = "How do you do great work?"
print("prompt>", prompt)

response = agent.create_turn(
    messages=[{"role": "user", "content": prompt}],
    session_id=agent.create_session("rag_session"),
    stream=True,
)

for item in response:
    print(item)

# for chunk in response:
#     print(chunk)


# for log in AgentEventLogger().log(response):
#     log.print()
