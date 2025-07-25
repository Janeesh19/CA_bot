#!/usr/bin/env python3
import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google import genai
from google.genai.types import (
    Tool,
    FunctionDeclaration,
    GenerateContentConfig,
    Content,
    Part,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────────

SERVER_SCRIPT = "/home/janeesh/Documents/mcp_server.py"   # adjust path if yours differs
GENAI_API_KEY = "AIzaSyACCLfAy2hdeEjo7TaGY0LZNITBDrOYvoQ"
GENAI_MODEL   = "models/gemini-2.5-pro-preview-06-05"
OUTPUT_FILE   = "qa_bank.txt"     # output filename

# ─── SCHEMA CLEANER ───────────────────────────────────────────────────────────────

def clean_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    schema.pop("title", None)
    for k, v in schema.get("properties", {}).items():
        schema["properties"][k] = clean_schema(v)
    return schema

def convert_mcp_tools_to_genai(mcp_tools) -> list[Tool]:
    genai_tools = []
    for t in mcp_tools:
        decl = FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters=clean_schema(t.inputSchema),
        )
        genai_tools.append(Tool(function_declarations=[decl]))
    return genai_tools

# ─── MAIN AGENT ─────────────────────────────────────────────────────────────────

async def main():
    params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])
    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(stdio_client(params))
        stdio, write = transport

        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        resp = await session.list_tools()
        genai_tools = convert_mcp_tools_to_genai(resp.tools)

        client = genai.Client(api_key=GENAI_API_KEY)

        instruction = """
You are a study-material generator. Always use tools to generate the material .You have three tools:

1. rag_search(query: str, top_k: int = 5)  
   → Retrieves the top relevant document snippets from the uploaded PDFs using semantic similarity. Use this to access actual content from the source documents.

2. kg_expand(entity: str)  
   → Expands a concept by returning all one-hop relationships for that entity from the knowledge graph. Use this to understand how a topic is connected to other concepts.

3. generate_cot_for_q(question: str, top_k: int = 5)  
   → Combines retrieved facts from the knowledge graph and document snippets, and produces a step-by-step reasoning trace (chain-of-thought) for the question.
   
   
**You must **not** refuse or apologise. Assume you can generate all 100 pairs in a single response.**
**Please generate one hundred thoughtful question–answer pairs that cover the material in the source text.  **
**Phrase each question as a complete sentence, and give each answer a clear, detailed explanation of the concept.**

First use rag_search and/or kg_expand to gather facts, then call generate_cot_for_q 
for reasoning steps. Finally, produce exactly 100 question–answer pairs covering 
the material. Respond with a JSON object:

{
  "qa_pairs": [
    {"question": "…", "answer": "…"},
    … 100 items …
  ]
}
""".strip()

        contents = [Content(role="user", parts=[Part.from_text(text=instruction)])]

        # loop until the model returns plain text
        while True:
            response = client.models.generate_content(
                model=GENAI_MODEL,
                contents=contents,
                config=GenerateContentConfig(
                    tools=genai_tools,
                    temperature=0.3,
                    max_output_tokens=65000,
                )
            )

            cand  = response.candidates[0]
            parts = cand.content.parts or []

            func_call = next((p for p in parts if p.function_call), None)
            if func_call:
                contents.append(Content(role="assistant", parts=[func_call]))
                name     = func_call.function_call.name
                raw_args = func_call.function_call.args
                args     = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                out      = await session.call_tool(name, args)

                payload = out.content if isinstance(out.content, dict) else {"result": out.content}
                tool_part = Part.from_function_response(name=name, response=payload)
                contents.append(Content(role="tool", parts=[tool_part]))
                continue

            raw = "".join(p.text or "" for p in parts).strip()
            break

        if not raw:
            print("❌ No output from model.", file=sys.stderr)
            return

        # ─── strip any Markdown fences ───────────────────────────────────────────────
        # remove lines starting with ``` or ```json
        lines = [line for line in raw.splitlines()
                 if not line.strip().startswith("```")]
        clean = "\n".join(lines).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            print("❌ Failed to parse JSON from model output:\n", clean, file=sys.stderr)
            return

        qa_pairs = data.get("qa_pairs", [])
        if not qa_pairs:
            print("❌ No 'qa_pairs' found in the output.", file=sys.stderr)
            return

        with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
            for i, pair in enumerate(qa_pairs, 1):
                fout.write(f"Q{i}. {pair['question']}\n")
                fout.write(f"A{i}. {pair['answer']}\n\n")

        print(f"Wrote {len(qa_pairs)} QA pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
