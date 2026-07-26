from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# 1. CONFIGURATIONS
# --------------------------------------------------------------
MONGO_URI = "mongodb+srv://maryam49oujda_db_user:masterGLCC@cluster0.eo7eh0l.mongodb.net/?appName=Cluster0"

print("🔄 Connexion à MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client["university_rag"]
collection = db["course_chunks"]

print("🔄 Chargement du modèle d'embedding...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------------
# 2. FONCTION DE RECHERCHE
# --------------------------------------------------------------
def rechercher_dans_les_cours(question):
    # Transformation de la question en vecteur
    query_vector = embed_model.encode(question).tolist()
    
    # Pipeline de recherche vectorielle MongoDB
    pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_vector,
                    "path": "embedding",
                    "k": 3
                }
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if not results:
        print("\n❌ Aucun résultat trouvé pour cette question.")
        return

    print(f"\n✅ {len(results)} résultats trouvés :")
    for i, doc in enumerate(results, 1):
        # Récupération des métadonnées (avec valeurs par défaut si absentes)
        source = doc.get('source', doc.get('filename', doc.get('file_name', 'Document inconnu')))
        page = doc.get('page', doc.get('page_number', 'Non spécifiée'))
        texte = doc.get('text', 'Pas de texte')

        print(f"\n--- Résultat {i} ---")
        print(f"📄 Source : {source}")
        print(f"📌 Page   : {page}")
        print(f"📝 Texte  :\n{texte}")

# --------------------------------------------------------------
# 3. BOUCLE INTERACTIVE (Le Menu)
# --------------------------------------------------------------
def main():
    print("\n==========================================")
    print("🎓 Bienvenue dans votre assistant de cours")
    print("   Tapez 'quitter' pour fermer.")
    print("==========================================")
    
    while True:
        question = input("\n👉 Quelle est votre question ? : ")
        
        if question.lower().strip() == 'quitter':
            print("\n👋 Au revoir ! À bientôt.")
            break
            
        if question.strip() == "":
            continue
            
        rechercher_dans_les_cours(question)

if __name__ == "__main__":
    main()