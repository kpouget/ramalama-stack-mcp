from fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool(description="Greets a person by name")
def greet(name: str, session_id: str | None = None) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()


## start mcp service

# uv run npx -y supergateway --port 8000 --stdio 'python mcp-test-server.py'
