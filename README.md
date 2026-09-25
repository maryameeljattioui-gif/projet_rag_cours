# Assistant RAG - Recherche Vectorielle dans les Cours Universitaires

Ce projet est une application web basée sur l'architecture **RAG (Retrieval-Augmented Generation)** permettant la recherche sémantique intelligente dans des supports de cours au format PDF.

##  Fonctionnalités
-  **Ingestion dynamique :** Ajout de cours au format PDF directement depuis l'interface web.
-  **Chunking & Vectorisation :** Extraction automatique du texte et génération d'embeddings vectoriels avec `all-MiniLM-L6-v2`.
-  **Recherche Vectorielle k-NN :** Indexation et recherche sémantique ultra-rapide hébergée sur **MongoDB Atlas**.
-  **Interface Web Interactive :** Développée avec **Streamlit** pour une expérience fluide.

##  Technologies utilisées
- **Langage :** Python 3.10+
- **Frontend :** Streamlit
- **Embeddings :** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Base de Données Vectorielle :** MongoDB Atlas Vector Search
- **PDF Processing :** PyPDF

##  Installation et Exécution

1. **Cloner le projet :**
   ```bash
   git clone <LIEN_DE_VOTRE_DEPOT_GITHUB>
   cd projet_rag_cours
