import os
from pypdf import PdfReader
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# 1. Configuration & Connexion à MongoDB Atlas
# --------------------------------------------------------------
# --------------------------------------------------------------
# 1. Configuration & Connexion à MongoDB Atlas
# --------------------------------------------------------------
# Remplacez "VOTRE_MOT_DE_PASSE" par le mot de passe réel (sans < >)
MONGO_URI = "mongodb+srv://maryam49oujda_db_user:masterGLCC@cluster0.eo7eh0l.mongodb.net/?appName=Cluster0"

print("🔄 Connexion à MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client["university_rag"]        # Base de données
collection = db["course_chunks"]     # Collection

# --------------------------------------------------------------
# 2. Chargement du modèle d'embedding (gratuit et rapide)
# --------------------------------------------------------------
print("🔄 Chargement du modèle d'embedding (SentenceTransformer)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------------
# 3. Traitement des fichiers PDF du dossier 'data/'
# --------------------------------------------------------------
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"⚠️ Le dossier '{DATA_DIR}' a été créé. Veuillez y placer vos fichiers PDF et relancer le script.")
    exit()

pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

if not pdf_files:
    print(f"⚠️ Aucun fichier PDF trouvé dans le dossier '{DATA_DIR}'.")
    exit()

documents_to_insert = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    print(f"📄 Lecture du fichier : {pdf_file}...")
    
    reader = PdfReader(pdf_path)
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue  # Sauter les pages vides ou images sans texte
        
        # Génération du vecteur d'embedding pour le texte de la page
        embedding = embed_model.encode(text).tolist()
        
        # Structure du document NoSQL à insérer
        doc = {
            "text": text.strip(),
            "source_file": pdf_file,
            "page_number": page_num,
            "embedding": embedding
        }
        documents_to_insert.append(doc)

# --------------------------------------------------------------
# 4. Enregistrement dans MongoDB Atlas
# --------------------------------------------------------------
# --------------------------------------------------------------
# 4. Enregistrement dans MongoDB Atlas
# --------------------------------------------------------------
if documents_to_insert:
    try:
        # Essai de nettoyage de la collection
        collection.delete_many({})
    except Exception as e:
        print(f"⚠️ Note : Impossible de vider la collection ({e}), passage à l'insertion.")
    
    # Insertion des documents
    result = collection.insert_many(documents_to_insert)
    print(f"✅ SUCCÈS : {len(result.inserted_ids)} pages/chunks ont été insérés dans MongoDB Atlas !")
else:
    print("❌ Aucun texte n'a pu être extrait des PDFs.")