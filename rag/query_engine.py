from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


class RAGEngine:

    def __init__(self):

        # configurations
        CHROMA_PATH = "./chromadb"
        COLLECTION_NAME = "aircraft_manuals"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

        # component initializations

        # Embedding Model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.chroma_client.get_collection(
            COLLECTION_NAME
        )

        # make sure ollama has the mentioned model and is running
        # ollama pull model_name
        # ollama serve
        # after usage
        # ollama rm model_name
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0
        )

        # system prompt
        self.SYSTEM_PROMPT = """
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
    def retrieve(
            self,
            question: str,
            top_k: int = 5
    ) -> List[Dict]:
        """
        does a similarity search
        :param question:
        :param top_k:
        :return:
        """

        query_embedding = self.embedding_model.encode(question).tolist()
        results = self.collection.query(
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
            self,
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
    def query(
            self,
            question: str,
    ) -> str:
        print("create prompt")
        docs = self.retrieve(question)
        context = self.build_context(docs)
        prompt = self.SYSTEM_PROMPT.format(
            context=context
        )
        print("prompt:", prompt)

        response = self.llm.invoke(
            [
                ("system", prompt),
                ("human", question)
            ]
        )

        return {"answer": response.content}
