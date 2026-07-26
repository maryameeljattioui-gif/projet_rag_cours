from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# 1. Configuration & Connexion à MongoDB Atlas
# --------------------------------------------------------------
# ⚠️ Remplacez VOTRE_MOT_DE_PASSE par votre mot de passe MongoDB
MONGO_URI = "mongodb+srv://maryam49oujda_db_user:masterGLCC@cluster0.eo7eh0l.mongodb.net/?appName=Cluster0"
print("🔄 Connexion à MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client["university_rag"]        # Base de données
collection = db["course_chunks"]     # Collection

# --------------------------------------------------------------
# 2. Chargement du même modèle d'embedding
# --------------------------------------------------------------
print("🔄 Chargement du modèle...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------------
# 3. Fonction de recherche vectorielle
# --------------------------------------------------------------
def poser_une_question(question, top_k=3):
    print(f"\n❓ Question : '{question}'")
    
    # Conversion de la question en vecteur
    query_vector = embed_model.encode(question).tolist()
    
    # Pipeline de recherche vectorielle Atlas
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",      # Nom exact de votre index Atlas
                "path": "embedding",         # Champ vectoriel dans la BDD
                "queryVector": query_vector,  # Vecteur de la question
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "source_file": 1,
                "page_number": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    print(f"\n🎯 Top {len(results)} des passages les plus pertinents trouvés dans vos cours :\n" + "-"*60)
    
    for idx, doc in enumerate(results, start=1):
        score = round(doc.get("score", 0) * 100, 2)
        print(f"📌 [Résultat {idx}] — Pertinence : {score}% (Page {doc['page_number']} de '{doc['source_file']}')")
        print(f"{doc['text'][:300]}...")
        print("-" * 60)

# --------------------------------------------------------------
# 4. Testez avec vos propres questions !
# --------------------------------------------------------------
if __name__ == "__main__":
    # Remplacez cette question par une question en rapport avec votre cours de Machine Learning
    ma_question = "Qu'est-ce que l'apprentissage supervisé ?" 
    poser_une_question(ma_question)