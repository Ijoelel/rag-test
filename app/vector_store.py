from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

model = SentenceTransformer(
    "nvidia/llama-nemotron-embed-1b-v2",
    trust_remote_code=True
)

def create_vector_store(chunks):
    texts = [doc.page_content for doc in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    text_embeddings = list(zip(texts, embeddings))

    return FAISS.from_embeddings(text_embeddings, embeddings)