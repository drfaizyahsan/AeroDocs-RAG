
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


# configurations

CHROMA_PATH = "./chromadb"
COLLECTION_NAME = "aircraft_manuals"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# component initializations

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

# make sure ollama has the mentioned model and is running
# ollama pull model_name
# ollama serve
# after usage
# ollama rm model_name
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# system prompt
SYSTEM_PROMPT = """
You are an expert aerospace maintenance assistant.

Use ONLY the retrieved maintenance documentation provided
in the context to answer the technician's question.

Rules:
1. Never use outside knowledge.
2. If the answer cannot be verified from the provided context,
   explicitly state:
   "I do not know based on the provided maintenance documentation."
3. Always cite the exact source document and system ID
   from the metadata.
4. If multiple sources are used, cite all of them.
5. Maintain professional aerospace maintenance terminology.
6. Do not fabricate procedures, torque values,
   part numbers, limitations, or maintenance steps.

Retrieved Context:
{context}
"""


# similarity search
def retrieve_context(
        question: str,
        top_k: int = TOP_K
) -> List[Dict]:
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    contexts = []

    for doc, score in zip(results["documents"][0],
                          results["distances"][0]):
        contexts.append(
            {
                "text": doc,
                "score": score
            }
        )
    return contexts


# build context string
def build_context(
        retrieved_docs: List[Dict]
) -> str:
    context = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        context.append(
            f"""
            DOCUMENT {idx}
            
            CONTENT:
            {doc["text"]}
            """
        )

    return "\n".join(context)


# ask answer
def generate_answer(
        question: str,
        context: str,
) -> str:
    print("create prompt")
    prompt = SYSTEM_PROMPT.format(
        context=context
    )
    print("prompt:", prompt)

    response = llm.invoke(
        [
            ("system", prompt),
            ("human", question)
        ]
    )

    return response.content


# main query pipeline
def ask(question: str):

    print("\nRetrieving relevant documents...\n")

    retrieved_docs = retrieve_context(question)

    print("Retrieved Sources:")
    for doc in retrieved_docs:
        print(doc)

    context = build_context(retrieved_docs)

    print("context:", context)

    answer = generate_answer(
        question, context
    )
    return answer


if __name__ == "__main__":

    query = """
    What procedure should be followed
    when inspecting the bleed air valve
    in the pneumatic system?
    """

    print("query:", query)

    response = ask(query)

    print("\n===================")
    print("Technical Response")
    print("=====================\n")
    print(response)
