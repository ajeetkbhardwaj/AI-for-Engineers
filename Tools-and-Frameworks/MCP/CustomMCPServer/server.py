from mcp.server.fastmcp import FastMCP

import logging
import inspect
 


# setup mcp server intialization
mcp = FastMCP(name="SimpleCalculator")

class SimpleCalculator:
    def __init__(self):
        pass
    # let's build a simple calculator tools
    @mcp.tool(name="add", description="Adds two integers")
    def add(a:int, b: int) -> int:
       return a + b
    @mcp.tool( name="subtract", description="Subtracts second integer from first")
    def subtract(a:int, b: int) -> int:
       return a - b
    @mcp.tool( name="multiply", description="Multiplies two integers")
    def multiply(a:int, b: int) -> int:
       return a * b
    @mcp.tool( name="divide", description="Divides first float by second float")
    def divide(a: float, b: float) -> float:
        if b == 0:
           raise ValueError("Cannot divide by zero")
        return a / b
    @mcp.tool( name="power", description="Raises first integer to the power of second integer")
    def power(a: int, b: int) -> int:
       return a ** b 
    @mcp.tool( name="square_root", description="Calculates the square root of a float")
    def square_root(a: float) -> float:
        if a < 0:
          raise ValueError("Cannot compute square root of negative number")
        return a ** 0.5
    @mcp.tool( name="factorial", description="Calculates the factorial of a non-negative integer")
    def factorial(n: int) -> int:
       if n < 0:
            raise ValueError("Cannot compute factorial of negative number")
       if n == 0 or n == 1:
          return 1
       result = 1
       for i in range(2, n+1):
              result *= i
       return result
    

tools = SimpleCalculator()
# Register the tools with the MCP server
for _,methods in inspect.getmembers(tools, predicate=callable):
    if getattr(methods, 'is_tool', False):
        mcp.add_tool(methods)

if __name__ == "__main__":
    # Start the MCP server
    #mcp.run(transport="stdio")
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting SimpleCalculator MCP server...")
    mcp.run(transport="streamable-http")
    