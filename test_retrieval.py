import sys
from app.config import get_settings
from app.rag.embeddings import embed_query  # agar import error aaye to us file ka asli path likhein
from qdrant_client import QdrantClient

s = get_settings()
c = QdrantClient(url=s.vector_db_url, api_key=s.vector_db_api_key)
v = embed_query(sys.argv[1])
res = c.query_points(s.collection_name, query=v, limit=5, with_payload=True).points
for r in res:
    print(round(r.score, 3), r.payload["document_name"], "p", r.payload["page_number"], "|", r.payload["text"][:80].replace("\n", " "))