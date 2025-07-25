import asyncio
import os
import sys
import json
from typing import Optional, List
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyACCLfAy2hdeEjo7TaGY0LZNITBDrOYvoQ"
GEMINI_MODEL   = "models/gemini-1.5-pro-latest"

def clean_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    schema.pop("title", None)
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            props[k] = clean_schema(v)
    return schema

def convert_mcp_tools_to_gemini(mcp_tools) -> List[Tool]:
    gemini_tools: List[Tool] = []
    for tool in mcp_tools:
        params = clean_schema(tool.inputSchema)
        decl = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=params
        )
        gemini_tools.append(Tool(function_declarations=[decl]))
    return gemini_tools

class MCPClient:
    def __init__(self):
        self.genai = genai.Client(api_key=GEMINI_API_KEY)
        self.exit   = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.funcs:   List[Tool]           = []

    async def connect_to_server(self, script_path: str):
        cmd = "python" if script_path.endswith(".py") else "node"
        params = StdioServerParameters(command=cmd, args=[script_path])
        transport = await self.exit.enter_async_context(stdio_client(params))
        self.stdio, self.write = transport

        self.session = await self.exit.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()
        resp = await self.session.list_tools()
        tools = resp.tools
        print("Connected with tools:", [t.name for t in tools])
        self.funcs = convert_mcp_tools_to_gemini(tools)

    async def process_query(self, query: str) -> str:
        instruction = """
You have three tools at your disposal:

1. rag_search(query: str, top_k: int = 5)
   • Searches indexed document chunks in Qdrant for relevant passages.

2. kg_expand(entity: str)
   • Retrieves one-hop triples (subject – predicate → object) from Neo4j.

3. cot_retrieve(triplet: str, top_k: int = 1)
   • Fetches the chain-of-thought explanation for a given triple from Qdrant.

Notes:
- Knowledge Graph (KG) facts are structured, highly efficient data; plain text is unstructured.
- Chain-of-Thought (COT) reveals the step-by-step reasoning used to arrive at an answer.

Tool‐selection strategy:
- If the question asks “who,” “when,” “where,” or “what” about document content, try **rag_search** first.
- If it asks about relationships or properties of a named entity, try **kg_expand** first.
- If it asks “why,” “how,” or for an explanation, try **cot_retrieve** first.
- Always inspect the tool’s output. If it’s empty or irrelevant:
  - For **rag_search** failures: fallback to **kg_expand**, then **cot_retrieve**.
  - For **kg_expand** failures: fallback to **rag_search**, then **cot_retrieve**.
  - For **cot_retrieve** failures: fallback to **rag_search**, then **kg_expand**.
- If multiple tools return useful data, merge their outputs into a single, concise answer, citing KG facts for precision and COT for clarity.
- If none yield meaningful information, respond: “I’m sorry, I don’t know.”

Examples:
“Who chairs the Audit Committee?”
→ rag_search(query="chairs audit committee")

“Who sits on the Corporate Laws Committee?”
→ kg_expand(entity="Corporate Laws Committee")

“Why does the Audit Committee meet quarterly?”
→ cot_retrieve(triplet="Audit Committee – meets -> Quarterly")

"""
        user_inst = types.Content(role="user", parts=[types.Part.from_text(text=instruction)])
        user_q    = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        contents  = [user_inst, user_q]

        final: List[str] = []
        while True:
            resp = self.genai.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=GenerateContentConfig(tools=self.funcs)
            )
            cand = resp.candidates[0]
            parts = cand.content.parts or []
            func = next((p for p in parts if p.function_call), None)

            if func:
                name = func.function_call.name
                raw  = func.function_call.args
                args = json.loads(raw) if isinstance(raw, str) else raw
                print(f"→ calling {name} with {args}")
                try:
                    out = await self.session.call_tool(name, args)
                    data = out.content
                except Exception as e:
                    data = {"error": str(e)}

                # Fallback logic
                if name == "kg_expand" and isinstance(data, list) and not data and query.lower().startswith(("why ", "how ", "explain ")):
                    print("→ kg_expand empty, falling back to cot_retrieve")
                    try:
                        cot = await self.session.call_tool("cot_retrieve", {"triplet": query, "top_k": 1})
                        name, data = "cot_retrieve", cot.content
                    except Exception as e:
                        data = {"error": str(e)}

                if name == "cot_retrieve" and (not data or (isinstance(data, str) and not data.strip())):
                    print("→ cot_retrieve empty, falling back to rag_search")
                    try:
                        rag = await self.session.call_tool("rag_search", {"query": query, "top_k": 5})
                        name, data = "rag_search", rag.content
                    except Exception as e:
                        data = {"error": str(e)}
                    if isinstance(data, list) and not data:
                        print("→ rag_search empty, falling back to kg_expand")
                        try:
                            kg = await self.session.call_tool("kg_expand", {"entity": query})
                            name, data = "kg_expand", kg.content
                        except Exception as e:
                            data = {"error": str(e)}

                result   = data if isinstance(data, dict) else {"result": data}
                part_resp = types.Part.from_function_response(name=name, response=result)
                contents.append(types.Content(role="assistant", parts=[func]))
                contents.append(types.Content(role="tool", parts=[part_resp]))
                continue

            # No function call → gather text
            text_list = [p.text for p in parts if p.text]
            final     = [t for t in text_list if t]
            break

        cleaned = [s for s in final if isinstance(s, str) and s.strip()]
        return "\n".join(cleaned) if cleaned else "I’m sorry, I don’t know."

    async def chat_loop(self):
        print("\nType 'quit' to exit.")
        while True:
            q = input("Query: ").strip()
            if not q:
                continue
            if q.lower() == "quit":
                break
            ans = await self.process_query(q)
            print("\n" + ans + "\n")

    async def cleanup(self):
        await self.exit.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python bot_with_mcp.py <server_script.py>")
        sys.exit(1)
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
