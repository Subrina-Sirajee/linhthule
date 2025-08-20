from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import uuid


class VectorSearchBase:
    """
    A semi-concrete base class for vector database operations.
    Designed for minimal method overriding when extended.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.embeddings_model = OpenAIEmbeddings(chunk_size=100)
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     separators=["\n\n", "\n", ".", " "]
        # )
        self.agency_data: List[Dict[str, Any]] = []
        self.index = None  # Expected to be set by connect()
        self.index_name = None  # Optional for tracking
        self.backend_type = "generic"

    def connect(self):
        """
        Default placeholder. Should set self.index in subclass if needed.
        """
        raise NotImplementedError("Override 'connect' in your backend-specific subclass.")

    def prepare(self, agency: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Converts agency dict into text chunks and metadata for embedding.

        Args:
            agency: Dict containing agency details.

        Returns:
            Tuple of (text chunks, metadata list).
        """
        text = (
            f"{agency['name']}. {agency['tagline']}. {agency['description']}. "
            f"Tags: {', '.join(agency['tags'])}. Rating: {agency['rating']}. "
            f"Reviews: {agency['reviews']}. Projects: {agency['projects']}. "
            f"Location: {agency['location']}. Rate: {agency['rate']}. "
            f"Budget: {agency['budget']}. Industry: {agency['industry']}. "
            f"Expertise: {agency['expertise']}"
        )

        # Return the text as a single chunk and the metadata
        metadata = [{
            "id": agency['id'],
            **agency  # Include all agency metadata
        }]
    
        return [text], metadata  # Return as a tuple of lists (texts, metadata)


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of text chunks.
        """
        return self.embeddings_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate a vector embedding for a search query.
        """
        return self.embeddings_model.embed_query(query)

    def format_upsert_items(self, metadata: List[Dict[str, Any]], vectors: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Combine metadata and vectors into a backend-compatible upsert format.
        """
        return [
            {
                "id": str(meta["id"]),
                "values": vector,
                "metadata": {k: v for k, v in meta.items()}
            }
            for meta, vector in zip(metadata, vectors)
        ]

    def upsert(self, items: List[Dict[str, Any]]):
        """
        Perform upsert to vector DB. To be overridden by subclasses.
        """
        raise NotImplementedError("Override 'upsert' in your backend-specific subclass.")

    def delete_vectors(self, ids: List[str]):
        """
        Delete vectors by ID. To be overridden by subclass.
        """
        raise NotImplementedError("Override 'delete_vectors' in your backend-specific subclass.")

    def search_index(self, vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector DB. To be overridden by subclass.
        """
        raise NotImplementedError("Override 'search_index' in your backend-specific subclass.")

    def build(self, data: List[Dict[str, Any]]):
        """
        Build the index from a full list of agencies.

        Args:
            data: List of agency dicts.
        """
        self.agency_data = data
        all_texts, all_metadata = [], []

        for agency in data:
            texts, meta = self.prepare(agency)
            all_texts.extend(texts)
            all_metadata.extend(meta)

        vectors = self.embed_texts(all_texts)
        upsert_items = self.format_upsert_items(all_metadata, vectors)
        self.upsert(upsert_items)

    def insert(self, agency: Dict[str, Any]) -> int:
        """
        Insert a new agency into in-memory and vector index.

        Args:
            agency: New agency data.

        Returns:
            The assigned ID.
        """
        new_id = max([a["id"] for a in self.agency_data], default=0) + 1
        agency["id"] = new_id
        agency["logo"] = agency.get("logo", f"https://picsum.photos/seed/agency-{new_id}/120/120")
        self.agency_data.append(agency)

        texts, metadata = self.prepare(agency)
        vectors = self.embed_texts(texts)
        upsert_items = self.format_upsert_items(metadata, vectors)
        self.upsert(upsert_items)
        return new_id

    def update(self, item_id: int, new_data: Dict[str, Any]) -> bool:
        """
        Update agency in both in-memory and vector index.

        Args:
            item_id: ID to update.
            new_data: New fields.

        Returns:
            True if successful, False if not found.
        """
        for agency in self.agency_data:
            if agency["id"] == item_id:
                agency.update(new_data)
                texts, metadata = self.prepare(agency)
                vectors = self.embed_texts(texts)
                upsert_items = self.format_upsert_items(metadata, vectors)
                self.upsert(upsert_items)
                return True
        return False

    
    # def delete(self, item_id: int) -> bool:
    #     """
    #     Delete agency from in-memory and vector DB by its ID.

    #     Args:
    #         item_id: The ID of the agency to delete.

    #     Returns:
    #         True if deletion was successful, False otherwise.
    #     """
    #     # Iterate through agency data to find the agency with the given item_id
    #     for i, agency in enumerate(self.agency_data):
    #         if agency["id"] == item_id:
    #             # Remove the agency from in-memory data
    #             self.agency_data.pop(i)

    #             # Delete the vector associated with the given item_id from Pinecone
    #             self.index.delete(ids=[str(item_id)])  # Deleting the vector by ID (single ID, not chunked)
    #             print(f"🗑️ Deleted vector with ID {item_id} from Pinecone.")
    #             return True

    #     return False

    def delete(self, item_id: int) -> bool:
        """
        Delete agency from in-memory and vector DB by its ID.

        Args:
            item_id: The ID of the agency to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        for i, agency in enumerate(self.agency_data):
            if agency["id"] == item_id:
                self.agency_data.pop(i)
                self.delete_vectors([str(item_id)])  # Call backend-specific delete_vectors
                print(f"🗑️ Deleted agency with ID {item_id} from in-memory and vector DB.")
                return True
        print(f"❌ Agency with ID {item_id} not found.")
        return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search on indexed data.

        Args:
            query: Search string.
            top_k: Number of results.

        Returns:
            List of match metadata.
        """
        query_vector = self.embed_query(query)
        # Debugging: Print the query vector to ensure it's correct
        # print(f"🔍 Query vector (embedding): {query_vector}")
        return self.search_index(query_vector, top_k=top_k)
