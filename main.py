# main.py

"""
 Vector DB Lab - Enhanced
- Uses Qdrant (v1.9.1) as a vector database
- Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
- Demonstrates advanced capabilities:
    - Multi-vector search
    - Filtered search with metadata
    - Dataset scaling
    - Optional evaluation metrics
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Recreate collection with HNSW vector index
client.recreate_collection(
    collection_name="movies",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Option 1: Load from predefined sample
movie_data = [
    ("A young wizard attends a magic school.", "Fantasy"),
    ("A hobbit sets out on a journey to destroy a powerful ring.", "Fantasy"),
    ("A group of superheroes team up to save the world.", "Action"),
    ("A space crew travels through a wormhole.", "Sci-Fi"),
    ("A detective solves crimes with the help of a genius roommate.", "Mystery"),
]

# Option 2: Load larger dataset
# df = pd.read_csv("./data/movie_plots.csv")
# movie_data = list(zip(df["plot"], df["genre"]))

plots = [plot for plot, genre in movie_data]
genres = [genre for plot, genre in movie_data]
vectors = model.encode(plots)

# Insert data into Qdrant
client.upsert(
    collection_name="movies",
    points=[
        PointStruct(
            id=i, vector=vec.tolist(), payload={"plot": plots[i], "genre": genres[i]}
        )
        for i, vec in enumerate(vectors)
    ],
)

print("✅ Inserted movie data into Qdrant")

# Basic similarity search
query_text = "magic and school"
query_vector = model.encode(query_text)

results = client.search(
    collection_name="movies", query_vector=query_vector.tolist(), limit=3
)

print(f"\n🔍 Top results for query: '{query_text}'\n")
for hit in results:
    print("Plot:", hit.payload["plot"])
    print("Genre:", hit.payload["genre"])
    print("Score:", hit.score)
    print("---")

# Filtered search (only Fantasy genre)
print("\n🎯 Filtered Search (Genre = Fantasy)")
filtered = client.search(
    collection_name="movies",
    query_vector=query_vector.tolist(),
    query_filter=Filter(
        must=[FieldCondition(key="genre", match=MatchValue(value="Fantasy"))]
    ),
    limit=3,
)
for hit in filtered:
    print("Plot:", hit.payload["plot"])
    print("Score:", hit.score)

# Multi-vector search (average of multiple queries)
query_texts = ["magic", "teamwork"]
query_vectors = model.encode(query_texts)
combined_query = np.mean(query_vectors, axis=0)

multi_results = client.search(
    collection_name="movies",
    query_vector=combined_query.tolist(),
    limit=3,
    with_payload=True,
)

print("\n🔗 Multi-vector Query Result for: 'magic' + 'teamwork'")
for hit in multi_results:
    print("Plot:", hit.payload["plot"])
    print("Score:", hit.score)


# function to print collection
def show_points(title):
    points = client.scroll(collection_name="movies", limit=10)[0]
    print(f"\n📋 {title}")
    for p in points:
        print(f"ID: {p.id}, Payload: {p.payload}")


# function call to display collection information before update and delete
show_points("Movies BEFORE update & delete")

# Update metadata
client.set_payload(collection_name="movies", payload={"updated": True}, points=[0])
print("\n🔧 Updated metadata for movie ID 0")

# Delete example
# client.delete(collection_name="movies", points_selector={"points": [4]})
client.delete(collection_name="movies", points_selector=PointIdsList(points=[4]))

print("🗑️ Deleted movie ID 4")

# function call to display collection information after update and delete
show_points("Movies AFTER update & delete")


# only show highly relevant results
threshold = 0.50
print(f"\n🧮 Filtering results with similarity > {threshold}")
for r in results:
    if r.score > threshold:
        print("Plot:", r.payload["plot"], "(Score:", r.score, ")")
