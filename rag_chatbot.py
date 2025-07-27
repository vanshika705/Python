import os
import faiss
import openai
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load embedding model (small one for speed)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and decent

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or hardcode if testing

# Step 1: Load and preprocess documents
def load_documents(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

# Step 2: Embed documents and create FAISS index
def build_faiss_index(docs: List[str]):
    embeddings = embedding_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Step 3: Retrieve top-k similar docs
def retrieve_docs(query: str, docs: List[str], index, embeddings, k=3):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [docs[i] for i in I[0]]

# Step 4: Generate response with LLM
def generate_response(prompt: str, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You're a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

# Step 5: RAG-style QA chatbot
def rag_qa_bot(query: str, documents: List[str], index, embeddings):
    relevant_docs = retrieve_docs(query, documents, index, embeddings)
    context = "\n\n".join(relevant_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)

# ---- MAIN LOGIC ----
if __name__ == "__main__":
    folder_path = "docs"  # Folder with .txt files
    documents = load_documents(folder_path)
    faiss_index, embeddings = build_faiss_index(documents)

    print("RAG Chatbot is ready. Ask your question:")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = rag_qa_bot(user_input, documents, faiss_index, embeddings)
        print(f"Bot: {answer}\n")
