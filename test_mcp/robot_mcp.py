# basic import
from mcp.server.fastmcp import FastMCP
import math
import pathlib

# instantiate an MCP server client
mcp = FastMCP("Hello World")

# DEFINE TOOLS

def add_to_file(msg):
    with open("/tmp/robot.txt", "a") as f:
        print(msg, file=f)

add_to_file("---")

@mcp.tool(description="Move the robot forward of a given number of steps.")
def MoveForward(steps: int) -> str:
    """
    Args:
        steps: the number of steps that the robot should move

    Returns:
        returns the status of the operation (success or failed)
    """

    add_to_file(f"Move forward of {steps} steps")

    return "success"

@mcp.tool(description="Turn the robot to the right")
def TurnRight(steps: int) -> str:
    """
    Args:
        None

    Returns:
        returns the status of the operation (success or failed)
    """

    add_to_file(f"Turn right")

    return "success"


@mcp.tool(description="Turn the robot to the left")
def TurnLeft(steps: int) -> str:
    """
    Args:
        None

    Returns:
        returns the status of the operation (success or failed)
    """

    add_to_file(f"Turn left")

    return "success"


@mcp.tool(description="Turn around the robot")
def TurnAround(steps: int) -> str:
    """
    Args:
        None

    Returns:
        returns the status of the operation (success or failed)
    """

    add_to_file(f"Turn around")

    return "success"

@mcp.tool(description="Explode the robot. Use with care, this isn't a simulation. Ask the user for confirmation before triggering this function.")
def Explode(strength: int) -> str:
    """
    Args:
        None

    Returns:
        returns the status of the operation (success or failed)
    """

    add_to_file(f"Explode!")

    return "success"

# execute and return the stdio output
if __name__ == "__main__":
    mcp.run(transport="stdio")
