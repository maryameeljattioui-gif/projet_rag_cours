import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pypdf

# --------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# --------------------------------------------------------------
st.set_page_config(
    page_title="Assistant RAG - Cours Universitaires",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .meta-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .page-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# 2. CACHED RESOURCE LOADING
# --------------------------------------------------------------
MONGO_URI = "mongodb+srv://maryam49oujda_db_user:masterGLCC@cluster0.eo7eh0l.mongodb.net/?appName=Cluster0"

@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client["university_rag"]
    return db["course_chunks"]

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("🔄 Connexion à MongoDB Atlas et chargement du modèle d'embedding..."):
    collection = get_mongo_collection()
    embed_model = get_embedding_model()

# --------------------------------------------------------------
# 3. HELPER FUNCTIONS (Ingestion & Search)
# --------------------------------------------------------------
def process_and_store_pdf(uploaded_file, chunk_size=500, overlap=50):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    documents_to_insert = []
    
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
            
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if len(chunk_text.strip()) > 30:
                vector = embed_model.encode(chunk_text).tolist()
                documents_to_insert.append({
                    "text": chunk_text,
                    "embedding": vector,
                    "source": uploaded_file.name,
                    "filename": uploaded_file.name,
                    "page": page_num
                })
                
    if documents_to_insert:
        collection.insert_many(documents_to_insert)
        return len(documents_to_insert)
    return 0

def search_courses(query, k=3):
    query_vector = embed_model.encode(query).tolist()
    pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_vector,
                    "path": "embedding",
                    "k": k
                }
            }
        }
    ]
    return list(collection.aggregate(pipeline))

# --------------------------------------------------------------
# 4. USER INTERFACE
# --------------------------------------------------------------
st.markdown('<div class="main-header">🎓 Assistant RAG - Recherche dans les Cours</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interrogez et enrichissez la base de connaissances de vos cours grâce à la recherche vectorielle.</div>', unsafe_allow_html=True)

# --- SIDEBAR : Import PDF & Settings ---
st.sidebar.header("📤 Ajouter un nouveau cours")
uploaded_file = st.sidebar.file_uploader("Téléverser un document PDF", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("⚙️ Traiter et enregistrer dans la Base", type="primary"):
        with st.spinner("Extraction, vectorisation et stockage dans MongoDB..."):
            try:
                total_chunks = process_and_store_pdf(uploaded_file)
                if total_chunks > 0:
                    st.sidebar.success(f"✅ Reçu ! {total_chunks} extraits ont été ajoutés à MongoDB.")
                else:
                    st.sidebar.warning("⚠️ Aucun texte n'a pu être extrait du PDF.")
            except Exception as e:
                st.sidebar.error(f"❌ Erreur : {e}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Paramètres de Recherche")
top_k = st.sidebar.slider("Nombre de résultats à afficher (k)", min_value=1, max_value=10, value=3)

# --- MAIN PAGE : Search ---
query = st.text_input("🔍 Poser une question ou entrer une notion de cours :", placeholder="Ex: Comment fonctionne la régression polynomiale ?")

if st.button("Rechercher dans les cours") or query:
    if not query.strip():
        st.warning("⚠️ Veuillez entrer un mot-clé ou une question.")
    else:
        with st.spinner("🔎 Recherche des meilleurs passages dans la base de cours..."):
            results = search_courses(query, k=top_k)
            
        if not results:
            st.error("❌ Aucun résultat trouvé pour cette recherche.")
        else:
            st.success(f"✅ {len(results)} extrait(s) pertinent(s) trouvé(s) pour : **{query}**")
            st.markdown("---")
            
            for idx, doc in enumerate(results, 1):
                source = doc.get('source') or doc.get('filename') or doc.get('file_name') or 'Document inconnu'
                page = doc.get('page', doc.get('page_number', 'N/A'))
                text = doc.get('text', 'Pas de texte disponible')
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="margin-bottom: 12px;">
                        <span class="meta-badge">📄 Document : {source}</span>
                        <span class="page-badge">📌 Page : {page}</span>
                    </div>
                    <div style="color: #1F2937; font-size: 1.05rem; line-height: 1.6; white-space: pre-line;">
                        {text}
                    </div>
                </div>
                """, unsafe_allow_html=True)