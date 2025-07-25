import os
import re
from typing import List

from mcp.server.fastmcp import FastMCP
import google.generativeai as genai

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from neo4j import GraphDatabase

from wordllama import WordLlama
from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig

# ─── 1) CONFIGURATION ───────────────────────────────────────────────────────────

QDRANT_URL        = os.getenv(
    "QDRANT_URL",
    "https://2bb626d0-8e3b-4aa5-87ba-03803e523506.eu-west-2-0.aws.cloud.qdrant.io:6333"
)
QDRANT_API_KEY    = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZYF3sjRoh_oJE49OhXZji6f1yQqBQAccxpsz83TFPt4"
)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "schedule_chunks")

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j+s://6d414d36.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "BixI6q3apS6SMUaV3WovDx8oKkT02dZaUIL_JDMfX-o")

GENAI_API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyACCLfAy2hdeEjo7TaGY0LZNITBDrOYvoQ")
GENAI_MODEL   = os.getenv("GENAI_MODEL", "models/gemini-1.5-pro-latest")

EMBED_DIM = 64

# ─── 2) INITIALISE EXTERNAL CLIENTS ──────────────────────────────────────────────

wl = WordLlama.load(trunc_dim=EMBED_DIM)

qdrant_kwargs = {"url": QDRANT_URL}
if QDRANT_API_KEY:
    qdrant_kwargs["api_key"] = QDRANT_API_KEY
else:
    print("WARNING: QDRANT_API_KEY is empty; rag_search may fail.")
qdrant_client = QdrantClient(**qdrant_kwargs)

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

model = None
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel(model_name=GENAI_MODEL)
else:
    print("WARNING: GENAI_API_KEY is empty; generate_cot_for_q & answer_question will fail.")

mcp = FastMCP(
    name="FastMCPServer",
    stateless_http=True,
    json_response=True,
    dependencies=[
        "qdrant-client>=1.8.1",
        "neo4j>=5.10.0",
        "google-generativeai>=0.8.5",
        "wordllama>=0.1.0",
    ],
)

# HELPER FUNCTIONS 

def _embed_query(text: str) -> List[float]:
    """
    Use WordLlama to encode the user’s query into a 64-dim vector.
    """
    vecs = wl.embed([text])
    return vecs[0].tolist()


async def _extract_entity_via_llm(question: str) -> str:
    """
    Use Gemini to pick out exactly one “focus entity” from the question.
    """
    if model is None:
        raise RuntimeError("Generative model not configured; cannot extract entity.")

    short_prompt = (
        "Extract the single focus entity (person, organisation, or product) from the user's question.\n"
        "Return *only* that one name, nothing else.\n\n"
        f"Question: \"{question}\""
    )

    response = model.generate_content(
        contents=short_prompt,
        config=GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=20
        ),
        stream=False
    )

    text = response.text.strip()
    return text.strip('"').strip("'").strip()

# TOOL DEFINITIONS 

@mcp.tool()
async def rag_search(query: str, top_k: int = 5) -> List[str]:
    """
    Return up to top_k text chunks from Qdrant, using WordLlama embeddings.
    """
    query_vector = _embed_query(query)
    hits: List[ScoredPoint] = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )

    chunks: List[str] = []
    for pt in hits:
        if pt.payload:
            chunks.append(pt.payload.get("content", ""))
        else:
            chunks.append("")
    return chunks

@mcp.tool()
async def kg_expand(entity: str) -> List[str]:
    """
    Case-insensitive, one-hop Neo4j lookup for the given entity name.
    Returns a deduplicated list of "Subject - Predicate -> Object".
    """
    triples: List[str] = []

    cypher_out = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
MATCH (e)-[r]->(x:Entity)
RETURN e.name AS subject, type(r) AS predicate, x.name AS object
"""
    cypher_in = """
MATCH (x:Entity)-[r]->(e:Entity)
WHERE toLower(e.name) = toLower($name)
RETURN x.name AS subject, type(r) AS predicate, e.name AS object
"""

    with neo4j_driver.session() as session:
        outgoing = list(session.run(cypher_out, {"name": entity}))
        incoming = list(session.run(cypher_in, {"name": entity}))

        print(
            f"DEBUG kg_expand for '{entity}': "
            f"{len(outgoing)} outgoing, {len(incoming)} incoming"
        )

        for rec in outgoing:
            triples.append(f"{rec['subject']} - {rec['predicate']} -> {rec['object']}")
        for rec in incoming:
            triples.append(f"{rec['subject']} - {rec['predicate']} -> {rec['object']}")

    return list(set(triples))

@mcp.tool()
async def generate_cot_for_q(
    question: str,
    top_k: int = 5
) -> str:
    """
    1) Extract focus entity via LLM
    2) Fetch KG facts and RAG snippets
    3) Build a fused CoT prompt
    4) Stream Gemini’s chain-of-thought
    """
    raw_entity = await _extract_entity_via_llm(question)
    lookup_entity = raw_entity.lower()

    kg_triplets = await kg_expand(lookup_entity)
    rag_chunks  = await rag_search(question, top_k=top_k)

    lines: List[str] = [
        f"You are an expert reasoning assistant. The user’s question is:\n\"{question}\"",
        f"Extracted focus entity: \"{raw_entity}\"",
        "",
    ]

    if kg_triplets:
        lines.append("One-hop facts from the knowledge graph:")
        for t in kg_triplets:
            lines.append(f"- {t}")
        lines.append("")
    else:
        lines.append("One-hop facts from the knowledge graph: <none found>\n")

    if rag_chunks:
        lines.append("Top document snippets retrieved from RAG search:")
        for snippet in rag_chunks:
            clean = snippet.replace("\n", " ")
            if len(clean) > 200:
                clean = clean[:200] + "…"
            lines.append(f"- {clean}")
        lines.append("")
    else:
        lines.append("Top document snippets retrieved from RAG search: <none found>\n")

    lines.append(
        "Using only the KG facts above and the document snippets above, "
        "please think through the user’s question step by step, showing your reasoning. "
        "Do not provide a final summary—just output your full chain of thought."
    )

    fused_prompt = "\n".join(lines)

    if model is None:
        raise RuntimeError("Generative model not configured; set GENAI_API_KEY.")

    response_stream = model.generate_content(
        contents=fused_prompt,
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=512,
        stream=True
    )

    cot_parts: List[str] = []
    async for part in response_stream:
        if part.text:
            cot_parts.append(part.text)

    return "".join(cot_parts)

# ─── 6) START THE SERVER ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
