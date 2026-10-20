
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER = Path(__file__).resolve().parent / "mcp_server.py"

REQUESTS = [
    ("get_annual_sales", {"year": 2025}),
    ("compare_sales", {"year1": 2024, "year2": 2025}),
    ("get_latest_sales", {}),
    ("get_forecast_evaluation", {}),
    ("get_sales_forecast", {}),
]


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            available = await session.list_tools()
            tool_names = {tool.name for tool in available.tools}

            print("Available MCP tools:", sorted(tool_names))

            results = {}

            for name, arguments in REQUESTS:
                if name not in tool_names:
                    raise RuntimeError(f"MCP tool not registered: {name}")

                response = await session.call_tool(
                    name,
                    arguments=arguments,
                )

                if response.isError:
                    raise RuntimeError(f"MCP tool failed: {name}")

                if not response.content:
                    raise RuntimeError(f"MCP tool returned no data: {name}")

                values = [
                    item.text
                    for item in response.content
                    if item.type == "text"
                ]

                if not values:
                    raise RuntimeError(f"No text response from {name}")

                result = json.loads(values[0])

                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(f"{name}: {result['error']}")

                if isinstance(result, list) and any(
                    isinstance(row, dict) and "error" in row
                    for row in result
                ):
                    raise RuntimeError(f"{name}: Invalid result")

                results[name] = result
                print(f"PASSED: {name}")

            output = (
                Path(__file__).resolve().parents[1]
                / "outputs"
                / "mcp_protocol_results.json"
            )

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )

            print(f"MCP protocol test passed: {output}")


if __name__ == "__main__":
    asyncio.run(main())
