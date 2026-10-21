
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "mcp_server.py"
OUTPUT = ROOT / "outputs" / "mcp_test_results.json"


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}

            expected = {
                "calculate",
                "get_document_list",
                "search_documents",
            }

            assert expected.issubset(tool_names), (
                f"Missing tools: {expected - tool_names}"
            )

            test_cases = [
                {
                    "name": "Calculator",
                    "tool": "calculate",
                    "arguments": {"expression": "150 + 250"},
                    "expected": {"result": 400},
                },
                {
                    "name": "Document discovery",
                    "tool": "get_document_list",
                    "arguments": {},
                    "expected_document": "remote_work_policy.md",
                },
                {
                    "name": "Document retrieval",
                    "tool": "search_documents",
                    "arguments": {
                        "question": "How many days can employees work remotely?"
                    },
                    "expected_text": "three days per week",
                },
                {
                    "name": "Unsafe expression rejection",
                    "tool": "calculate",
                    "arguments": {
                        "expression": "__import__('os').system('echo test')"
                    },
                    "expected_error": True,
                },
            ]

            results = []

            for case in test_cases:
                response = await session.call_tool(
                    case["tool"],
                    arguments=case["arguments"],
                )

                if response.isError:
                    raise AssertionError(
                        f"MCP tool failed: {case['name']}"
                    )

                if response.structuredContent is not None:
                    data = response.structuredContent
                else:
                    text_items = [
                        item.text
                        for item in response.content
                        if item.type == "text"
                    ]
                    if not text_items:
                        raise AssertionError(
                            f"No content returned by {case['tool']}"
                        )
                    data = json.loads(text_items[0])

                if "result" in case.get("expected", {}):
                    passed = (
                        data.get("result")
                        == case["expected"]["result"]
                    )

                elif "expected_document" in case:
                    passed = (
                        case["expected_document"]
                        in data.get("documents", [])
                    )

                elif "expected_text" in case:
                    passages = " ".join(
                        item["text"]
                        for item in data.get("matches", [])
                    ).lower()

                    passed = case["expected_text"] in passages

                else:
                    passed = "error" in data

                results.append({
                    "test": case["name"],
                    "tool": case["tool"],
                    "passed": passed,
                    "response": data,
                })

                print(
                    f"{'PASS' if passed else 'FAIL'}: {case['name']}"
                )

            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )

            if not all(item["passed"] for item in results):
                raise AssertionError("One or more MCP tests failed.")

            print(f"All MCP tests passed. Report: {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
