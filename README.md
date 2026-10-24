# Hybrid RAG Studio

Hybrid RAG Studio is a Streamlit application that combines local Chroma vector
search with a Neo4j knowledge graph and OpenAI models.

## Run the app in Docker

The Docker Compose setup starts only the Streamlit app. Configure `.env` with
your `OPENAI_API_KEY` and connection settings for an already-running Neo4j
instance (`NEO4J_URI`, `NEO4J_USER` or `NEO4J_USERNAME`, and
`NEO4J_PASSWORD`). Neo4j is not started by Compose.

Build and start the app:

```powershell
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501). Chroma data is persisted
in the local `chroma_db_store` directory.

View app logs or stop it with:

```powershell
docker compose logs -f app
docker compose down
```
