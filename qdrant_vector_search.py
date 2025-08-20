from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointIdsList
from vector_search_base import VectorSearchBase
import os
from dotenv import load_dotenv

load_dotenv()

class QdrantVectorSearch(VectorSearchBase):
    """
    Qdrant-based implementation of VectorSearchBase.
    Handles backend-specific logic for connect, upsert, delete, and search.
    """

    def __init__(self, index_name: str = "ai-agency-chatbot"):
        super().__init__()
        self.index_name = index_name
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://10.10.13.23:6333"),
            api_key=os.environ.get("QDRANT_API_KEY", None)
        )
        self.index = None
        self.backend_type = "qdrant"

    def connect(self):
        """
        Connects to Qdrant and creates collection if it doesn't exist.
        """
        collections = [col.name for col in self.client.get_collections().collections]
        if self.index_name not in collections:
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=VectorParams(
                    size=1536,  # Dimension of embeddings (matches OpenAI's text-embedding-ada-002)
                    distance=Distance.COSINE  # Metric for similarity search
                )
            )
            print(f"🆕 Created Qdrant collection: {self.index_name}")

        self.index = self.client.get_collection(self.index_name)
        print(f"✅ Connected to Qdrant collection: {self.index_name}")

    def upsert(self, items: List[Dict[str, Any]]):
        """
        Upserts vectors to Qdrant.

        Args:
            items: List of dicts with 'id', 'values', and 'metadata'.
        """
        if not self.index:
            raise RuntimeError("Index not connected. Call `connect()` first.")

        # Prepare data for upsert into Qdrant
        upsert_data = [
            {
                "id": int(item["id"]),  # Use integer ID
                "vector": item["values"],
                "payload": item["metadata"]
            }
            for item in items
        ]

        self.client.upsert(collection_name=self.index_name, points=upsert_data)
        print(f"📦 Upserted {len(items)} vectors to Qdrant.")

    def delete_vectors(self, ids: List[str]):
        """
        Deletes vectors from Qdrant by IDs.

        Args:
            ids: List of vector IDs to delete.
        """
        if not self.index:
            raise RuntimeError("Index not connected. Call `connect()` first.")

        # Convert string IDs to integers to match upsert
        int_ids = [int(id_str) for id_str in ids]
        self.client.delete(collection_name=self.index_name, points_selector=PointIdsList(points=int_ids))
        print(f"🗑️ Deleted {len(ids)} vectors from Qdrant.")

    def search_index(self, vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs similarity search using Qdrant.

        Args:
            vector: Query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of matched metadata.
        """
        if not self.index:
            raise RuntimeError("Index not connected. Call `connect()` first.")

        results = self.client.search(
            collection_name=self.index_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )
        # print(f"🔍 Found {len(results)} matches for the query.")
        return [{"metadata": result.payload, "score": result.score} for result in results]

# Sample agency data for testing
sample_agencies = [
    {
        "id": 1,
        "name": "Quantum AI Solutions",
        "logo": "https://picsum.photos/seed/quantum-ai/120/120",
        "rating": 4.9,
        "reviews": 127,
        "tagline": "Transforming NLP into business value",
        "description": "Experts in chatbot and NLP-driven customer engagement platforms tailored for support automation.",
        "tags": ["Chatbots", "NLP", "Customer Service"],
        "projects": "Deployed over 100 multilingual chatbots",
        "location": "United States",
        "rate": "$60-90",
        "budget": "mid",
        "industry": "technology",
        "expertise": "machine-learning"
    },
    {
        "id": 2,
        "name": "ConnectRight",
        "logo": "https://picsum.photos/seed/connect-right/120/120",
        "rating": 4.8,
        "reviews": 89,
        "tagline": "Data-driven insights that empower decisions",
        "description": "Specialists in finance and healthcare AI analytics, predictive modeling, and ETL pipelines.",
        "tags": ["Predictive Analytics", "Finance", "Healthcare"],
        "projects": "Built 30+ predictive finance dashboards",
        "location": "United States",
        "rate": "$70-100",
        "budget": "high",
        "industry": "healthcare",
        "expertise": "computer-vision"
    },
    {
        "id": 3,
        "name": "ProviderLink",
        "logo": "https://picsum.photos/seed/provider-link/120/120",
        "rating": 4.7,
        "reviews": 156,
        "tagline": "Seeing is believing—AI in every frame",
        "description": "Computer vision specialists with real-time retail monitoring and security applications.",
        "tags": ["Computer Vision", "Retail", "Security AI"],
        "projects": "Developed 50+ surveillance systems",
        "location": "United States",
        "rate": "$55-80",
        "budget": "low",
        "industry": "retail",
        "expertise": "nlp"
    },
    {
        "id": 4,
        "name": "MatchPoint Agency",
        "logo": "https://picsum.photos/seed/matchpoint/120/120",
        "rating": 4.6,
        "reviews": 203,
        "tagline": "AI pipelines for enterprise data",
        "description": "Data engineers focused on large-scale data ingestion, transformation, and machine learning ops.",
        "tags": ["Data Science", "Big Data", "ETL"],
        "projects": "Optimized 120+ data pipelines",
        "location": "United States",
        "rate": "$75-110",
        "budget": "mid",
        "industry": "manufacturing",
        "expertise": "data-science"
    },
    {
        "id": 5,
        "name": "Direct Match",
        "logo": "https://picsum.photos/seed/direct-match/120/120",
        "rating": 4.5,
        "reviews": 174,
        "tagline": "AI Agents that think, act, and interact",
        "description": "Experts in AI agent development for education and virtual assistants.",
        "tags": ["AI Agents", "Automation", "Conversational AI"],
        "projects": "Created 20+ AI agent-based LMS platforms",
        "location": "United States",
        "rate": "$50-70",
        "budget": "high",
        "industry": "education",
        "expertise": "ai-agents"
    },
    {
        "id": 6,
        "name": "Align Partners",
        "logo": "https://picsum.photos/seed/align-partners/120/120",
        "rating": 4.8,
        "reviews": 91,
        "tagline": "Intelligent machines in action",
        "description": "Advanced robotics and embedded AI projects for automation in manufacturing.",
        "tags": ["Robotics", "Embedded AI", "IoT"],
        "projects": "Delivered 15 robotic automation projects",
        "location": "United States",
        "rate": "$85-120",
        "budget": "mid",
        "industry": "technology",
        "expertise": "machine-learning"
    },
    {
        "id": 7,
        "name": "LinkMakers",
        "logo": "https://picsum.photos/seed/linkmakers/120/120",
        "rating": 4.9,
        "reviews": 112,
        "tagline": "Smarter product discovery through AI",
        "description": "Built scalable recommender systems for e-commerce platforms using deep learning.",
        "tags": ["Recommendation Systems", "E-commerce", "Deep Learning"],
        "projects": "Powered recommendations for 10M+ users",
        "location": "United States",
        "rate": "$65-100",
        "budget": "low",
        "industry": "healthcare",
        "expertise": "computer-vision"
    },
    {
        "id": 8,
        "name": "CoreConnect",
        "logo": "https://picsum.photos/seed/core-connect/120/120",
        "rating": 4.6,
        "reviews": 145,
        "tagline": "Pioneers in transformer-based applications",
        "description": "Fine-tuned BERT/GPT models for various NLP tasks including summarization and QA.",
        "tags": ["Language Models", "Text Generation", "BERT"],
        "projects": "Fine-tuned 80+ language models",
        "location": "United States",
        "rate": "$90-130",
        "budget": "high",
        "industry": "retail",
        "expertise": "nlp"
    },
    {
        "id": 9,
        "name": "Synergy Connect",
        "logo": "https://picsum.photos/seed/synergy-connect/120/120",
        "rating": 4.7,
        "reviews": 128,
        "tagline": "Voice-first AI experiences",
        "description": "Speech-to-text solutions and smart voice assistants for mobile and desktop apps.",
        "tags": ["Speech Recognition", "Voice Assistants", "ASR"],
        "projects": "Implemented 40+ voice interfaces",
        "location": "United States",
        "rate": "$60-85",
        "budget": "mid",
        "industry": "manufacturing",
        "expertise": "data-science"
    },
    {
        "id": 10,
        "name": "MatchBridge",
        "logo": "https://picsum.photos/seed/match-bridge/120/120",
        "rating": 4.8,
        "reviews": 167,
        "tagline": "Privacy-first machine learning",
        "description": "Synthetic data solutions for training privacy-preserving models.",
        "tags": ["Synthetic Data", "Data Augmentation", "Privacy AI"],
        "projects": "Generated datasets for 500k+ samples",
        "location": "United States",
        "rate": "$70-95",
        "budget": "low",
        "industry": "education",
        "expertise": "ai-agents"
    },
    {
        "id": 11,
        "name": "Provider Nexus",
        "logo": "https://picsum.photos/seed/provider-nexus/120/120",
        "rating": 4.5,
        "reviews": 134,
        "tagline": "Transforming NLP into business value",
        "description": "Experts in chatbot and NLP-driven customer engagement platforms tailored for support automation.",
        "tags": ["Chatbots", "NLP", "Customer Service"],
        "projects": "Deployed over 100 multilingual chatbots",
        "location": "United States",
        "rate": "$60-90",
        "budget": "high",
        "industry": "technology",
        "expertise": "machine-learning"
    },
    {
        "id": 12,
        "name": "ConnectQuest",
        "logo": "https://picsum.photos/seed/connect-quest/120/120",
        "rating": 4.9,
        "reviews": 198,
        "tagline": "Data-driven insights that empower decisions",
        "description": "Specialists in finance and healthcare AI analytics, predictive modeling, and ETL pipelines.",
        "tags": ["Predictive Analytics", "Finance", "Healthcare"],
        "projects": "Built 30+ predictive finance dashboards",
        "location": "United States",
        "rate": "$70-100",
        "budget": "mid",
        "industry": "healthcare",
        "expertise": "computer-vision"
    }
]

if __name__ == "__main__":
    print("🚀 Starting test of QdrantVectorSearch...")

    # Step 1: Initialize and connect
    searcher = QdrantVectorSearch()
    searcher.connect()

    # Step 2: Index with initial data
    print("\n🔧 Building index...")
    searcher.build(sample_agencies)


    # Step 3: Insert a new agency
    print("\n➕ Inserting a new agency...")
    new_agency = {
        "name": "SmartML Agency",
        "tagline": "Transforming Data into Decisions",
        "description": "We offer custom ML pipelines and data analytics tools.",
        "tags": ["ML", "Data Science", "Cloud AI"],
        "rating": 4.9,
        "reviews": 88,
        "projects": "Designed 25+ machine learning pipelines",
        "location": "Canada",
        "rate": "$80-120",
        "budget": "high",
        "industry": "finance",
        "expertise": "data-science"
    }
    new_id = searcher.insert(new_agency)
    print("new_id", new_id)

    # # Step 5: Verify if the new agency is inserted
    # print("\n✅ Verifying if new data is inserted...")
    # verify_insert = searcher.search("SmartML Agency", top_k=1)
    # print("Search result after insert:", verify_insert)

    # # Step 6: Update the inserted agency
    print("\n✏️ Updating the inserted agency...")
    update_result = searcher.update(new_id, {
        "description": "Updated description with more insights.",
        "rating": 4.95,
        "tags": ["ML", "DataOps", "Cloud"]
    })
    print("Update status:", "✅ Success" if update_result else "❌ Failed")

    # Step 7: Verify if the data is updated
    print("\n✅ Verifying if data is updated...")
    verify_update = searcher.search("Updated description with more insights", top_k=1)
    print("Search result after update:", verify_update)

    # # Step 8: Search the index
    # print("\n🔍 Performing a search...")
    # search_results = searcher.search("machine learning pipelines for business", top_k=3)
    # for i, match in enumerate(search_results):
    #     print(f"{i+1}. {match['metadata']['name']} - Score: {match['score']:.4f}")

    # Step 9: Delete the new agency
    print("\n🗑️ Deleting the inserted agency...")
    delete_result = searcher.delete(new_id)
    print("Delete status:", "✅ Success" if delete_result else "❌ Failed")

    # # Step 10: Verify if the agency is deleted
    # print("\n✅ Verifying if the data is deleted...")
    # verify_delete = searcher.search("SmartML Agency", top_k=1)
    # print("Search result after deletion:", verify_delete)

    print("\n✅ Test complete!")