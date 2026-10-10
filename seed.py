import os, uuid
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
client = QdrantClient(url=os.environ["QDRANT_URL"],
                        api_key=os.environ["QDRANT_API_KEY"], timeout=30)
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")  # 384 dimensions

for name in ["shelters", "incidents"]:
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("created collection:", name)

# name, area, capacity, type, facilities
SHELTERS = [
    ("Government High School Relief Camp", "Malakpet", 300, "school", "medical aid, drinking water"),
    ("Community Hall Shelter", "Dilsukhnagar", 150, "community hall", "food, blankets"),
    ("Charminar Function Hall Camp", "Charminar", 200, "function hall", "food, drinking water, first aid"),
    ("Osmania General Hospital", "Afzal Gunj", 100, "hospital", "emergency ward, ambulance, burn care"),
    ("GHMC Ward Office Shelter", "Mehdipatnam", 120, "government office", "drinking water, blankets"),
    ("Indoor Stadium Relief Centre", "LB Nagar", 500, "stadium", "food, medical aid, toilets, power backup"),
    ("Secunderabad Railway Community Centre", "Secunderabad", 250, "community hall", "food, drinking water"),
    ("Gandhi Hospital", "Musheerabad", 150, "hospital", "emergency ward, ambulance, ICU"),
    ("Kukatpally Municipal School Camp", "Kukatpally", 350, "school", "food, drinking water, toilets"),
    ("Uppal Stadium Relief Centre", "Uppal", 600, "stadium", "food, medical aid, power backup"),
    ("Begumpet Fire Station Assembly Point", "Begumpet", 80, "fire station", "fire crew, rescue equipment, first aid"),
    ("Tolichowki Function Hall Camp", "Tolichowki", 220, "function hall", "food, blankets, boats for flood rescue"),
    ("Ameerpet Metro Station Upper Level", "Ameerpet", 180, "metro station", "elevated flood-safe area, drinking water"),
    ("Hitec City Convention Centre", "Madhapur", 700, "convention centre", "food, medical aid, power backup, toilets"),
    ("Falaknuma Government College Camp", "Falaknuma", 280, "college", "food, drinking water, first aid"),
]

texts = [f"{n} in {a}, Hyderabad. Capacity {c}. Type: {t}. Facilities: {f}."
        for n, a, c, t, f in SHELTERS]
vectors = list(embedder.embed(texts))

points = [
    PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, n)),
        vector=vec.tolist(),
        payload={"name": n, "area": a, "capacity": c, "type": t,
                "facilities": f, "text": text},
    )
    for (n, a, c, t, f), vec, text in zip(SHELTERS, vectors, texts)
]
client.upsert("shelters", points=points)
print("shelters stored:", client.count("shelters").count)