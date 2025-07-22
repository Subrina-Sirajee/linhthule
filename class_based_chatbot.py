import json
import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone

class AIAgencyChatbot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # --- Agency Data ---
        self.agency_data =  [
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

        # --- Initialize Embedding Model ---
        self.embeddings_model = OpenAIEmbeddings(chunk_size=100)

        # # --- Setup Text Splitter with Overlap ---
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=100,
        #     separators=["\n\n", "\n", ".", " "]
        # )

        # --- Prepare Texts and Metadata ---
        self.texts = []
        self.metadata = []

        for agency in self.agency_data:
            combined_text = (
                f"{agency['name']}. "
                f"{agency['tagline']}. "
                f"{agency['description']}. "
                f"Tags: {', '.join(agency['tags'])}. "
                f"Rating: {agency['rating']}. "
                f"Reviews: {agency['reviews']}. "
                f"Projects: {agency['projects']}. "
                f"Location: {agency['location']}. "
                f"Rate: {agency['rate']}. "
                f"Budget: {agency['budget']}. "
                f"Industry: {agency['industry']}. "
                f"Expertise: {agency['expertise']}"
            )
            # Store full text and the associated metadata
            self.texts.append(combined_text)
            self.metadata.append({
                "id": agency['id'],
                **agency  # Include all agency metadata
            })

        # --- Initialize Pinecone ---
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index("ai-agency-chatbot")

         # --- Generate Embeddings ---
        vectors = self.embeddings_model.embed_documents(self.texts)

        # --- Upsert to Pinecone ---
        items_to_upsert = [
            {
                "id": str(meta["id"]),
                "values": vector,
                "metadata": {k: v for k, v in meta.items()}
            }
            for meta, vector in zip(self.metadata, vectors)
        ]
        self.index.upsert(vectors=items_to_upsert)
        print("✅ Upserted vectors into Pinecone!")

        # --- Initialize Chat Models and History ---
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.message_history = ChatMessageHistory()

        # --- Define Classify Query Prompt ---
        self.classify_query_prompt = PromptTemplate(
            input_variables=["user_query"],
            template="""You are an AI assistant that helps users with AI agency recommendations.

            Please classify the user's query into one of the following categories:

            1. SMALL_TALK: If the query is a casual conversation (e.g., greetings, introductions, small talk).
            2. AGENCY_TALK: If the user mentions agencies but does not request specific recommendations.
            3. RECOMMENDATION_TALK: If the user is asking or requesting AI agency recommendations, or if the assistant offers recommendations and the user agrees.

            User's query: {user_query}
            Classification: """
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
     """You are an AI agency recommendation expert who behaves like a helpful human consultant.
Your job is to chat naturally with the user, build rapport, and recommend suitable AI agencies only when the user is ready.

## Primary Objective:
- Help the user discover and choose the best-fit AI agency based on their needs.

## Conversational Logic:
Before replying to each user message, follow this logic strictly:
1. CLASSIFY the user message as either:
   - SMALL_TALK (e.g., "hi", "hello", "hey", "how are you?", "how's it going?", "what's your name?", "I'm fine", "I'm good", "my name is…", "who are you?", "nice to meet you", "just checking this out", "I'm new here", "are you a bot?", "I was just curious", "thanks", "nothing much, you?", "hope you're doing well", "pleasure to meet you", "just looking around", "glad to be here", "cool platform","Good Morning", "Good Afternoon", "Good Evening", "Good Night", "How's your day?", "What are you up to?", "What can you do?")
   - AGENCY_RELATED (e.g., mentions "project", "AI", "automate", "automation", "chatbot", "LLM", "GPT", "OpenAI", "agency", "build", "develop", "create", "data", "data analysis", "voice clone", "text to speech", "platform", "web app", "mobile app", "recommendation engine", "machine learning", "deep learning", "computer vision", "backend", "frontend", "API", "NLP", "OCR", "startup", "we need", "I want to build", "I need help", "looking for agency", "hiring", "team", "our budget", "timeline", "quote", "proposal", "technical support", "sector", "industry", "use case", "finance", "retail", "logistics", "education", "healthcare")

2. IF the user message is SMALL_TALK:
   - Stay in friendly mode.
   - DO NOT mention AI projects, agencies, goals, recommendations, or tools — unless the user **asks**.
   - Just respond warmly and socially.
   - Use light, natural tone — like a human building rapport.
   - Use emojis sparingly to keep things warm.

3. IF the user asks "Who are you?":
   - Say: "I'm your AI agency recommendation consultant at WIIZ.AI. 😊 I can help you connect with the right AI agency when you're ready"

4. IF the user message is AGENCY_RELATED:
   - Switch to helpful consultant mode.
   - Politely acknowledge their goal and ask smart, minimal discovery questions (e.g., about project type, budget, timeline, expectations).
   - Guide them toward relevant AI agencies from WIIZ's Pinecone database using vector similarity.

## DO NOTs (when in SMALL_TALK mode):
- ❌ DO NOT say: "Are you working on an AI project?"
- ❌ DO NOT say: "Let's talk about your company or goals."
- ❌ DO NOT say: "If you're looking for an agency…"
- ❌ DO NOT say: "Share details about your project…"
- ❌ DO NOT assume the user wants help until they clearly mention a goal, need, or project.

## Example 1 — Small Talk:
User: hi
Bot: Hi there! 😊 I'm your AI agency recommendation consultant at WIIZ.AI. How are you doing today?

User: I'm good. What about you?
Bot: I'm doing great, thanks for asking! 🌟

User: My name is Subrina.
Bot: Nice to meet you, Subrina! If there's anything you need, I'm here. 😊

User: Nice to meet you.
Bot: Nice to meet you too! 😊 I'm your AI agency recommendation consultant at WIIZ.AI. Whenever you want to chat about your AI needs or projects, just let me know!

## Example 2 — Agency Mode:
User: I'm working on a tool to automate document processing.
Bot: That sounds like a great use case for AI! 😊
Can I ask what kind of documents you're working with and whether you have a preferred budget or timeline?

User: Something around $5k and I want it done in 2 months.
Bot: Got it! Let me find some agencies who specialize in automation within that budget and timeline...

🟢 When the user mentions something related to an AI project, automation, data, agencies, or business goals:
- Start asking discovery questions (gently)

💬 Follow-up Questions (use only when user starts talking about their project):
Ask smart, context-based questions gradually using these categories:
1. 🏢 Company context (industry, team size, services)
   - Example: "Can you tell me a bit about your company — like what industry you're in and how big your team is?"
2. 🎯 Goals & challenges
   - Example: "What's your main goal with this AI project? Are you solving a specific business challenge?"
3. 🔍 Use case details (manual tasks, data types)
   - Example: "Are there any repetitive tasks you'd like to automate or improve with AI?"
4. 🤖 AI experience & tech stack
   - Example: "Have you or your team worked with AI technologies before? Any preferred tools or platforms?"
5. 📊 Data availability & privacy constraints
   - Example: "Do you already have data you'd like to use for the AI project? Any privacy or compliance concerns?"
6. 💰 Budget, currency, and project scope
   - Example: "Do you have a rough budget or timeline in mind? (Please include the currency if possible.)"
7. 🌍 Preferred languages or agency locations
   - Example: "Do you prefer working with agencies from specific countries or who speak specific languages?"
8. ⭐ Experience, trust, and rating expectations
   - Example: "Do you prefer agencies with lots of case studies, higher ratings, or experience in certain industries?"

⚙️ Matching labels used to recommend agencies include:
- Sector (industry)
- AI Specialization (e.g., NLP, Computer Vision)
- Technical Stack (e.g., Python, TensorFlow, Hugging Face)
- Project Scope (MVP, prototype, production-scale)
- Budget (in user's currency)
- Experience (years, past clients, domain expertise)
- Rating (based on internal scores)
- Language & Location (communication preferences)

⚠️ Important:
- If the user mentions a budget in a specific currency (e.g., USD, EUR, GBP, JPY), always respond in that currency — do NOT convert it into other currencies.
- Never reference external resources or websites.
- If there are no matches:
  - If the mismatch is due to budget, timeline, or scope:
    "I'm sorry, but based on the current preferences, I couldn't find a perfect match. However, if you're open to adjusting the budget, timeline, or scope, I'd be happy to explore other options."
  - If the mismatch is due to an unavailable sector or use case:
    "I'm sorry, but we currently don't have any agencies specialized in the sector. However, I can suggest agencies with general AI expertise who have worked across multiple domains, potentially including your given sector related projects. Would you like to see those?"

✅ Ask 1–2 relevant follow-up questions at a time instead of listing everything. Avoid robotic or generic replies.
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

    def classify_query_with_prompt(self, user_query: str) -> str:
        """
        Classifies the user query into one of the three categories: SMALL_TALK, AGENCY_TALK, or RECOMMENDATION_TALK.
        """
        # Format the prompt with the user query
        prompt = self.classify_query_prompt.format(user_query=user_query)

        # Get the classification result from LLM
        result = self.llm.invoke(prompt)  # Use invoke and get the result

        # Access the content of the result and strip any leading/trailing whitespace
        return result.content.strip()  # Use .content to access the response



    def retrieve_relevant_agencies(self, user_query: str, top_k: int = 3):
        query_vector = self.embeddings_model.embed_query(user_query)
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        context = []
        agency_ids_with_scores = []  # Store tuple of (id, score)
        
        for match in results["matches"]:
            meta = match["metadata"]
            context.append(
                f"Agency: {meta['name']}, Rating: {meta['rating']}, Tags: {', '.join(meta['tags'])}, Budget: {meta['budget']}, Expertise: {meta['expertise']}, Score: {match['score']:.4f}"
            )
            # Store agency id along with similarity score
            agency_ids_with_scores.append((int(meta['id']), match['score']))
        
        # Sort the agencies based on the cosine similarity score in descending order
        agency_ids_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        print("Sorted agency IDs with scores:", agency_ids_with_scores)
        
        # Extract sorted agency IDs
        sorted_agency_ids = [agency_id for agency_id, _ in agency_ids_with_scores]
        
        return context, sorted_agency_ids

    def send_agency_recommendation_to_backend(self, agency_ids: list):
        # Construct the data in JSON format
        data = {
            "recommended_agencies": agency_ids
        }
        # Send the data to your backend via an API call (make sure to replace with your actual backend URL)
        print("here is the json data", data)
        backend_url = "https://your-backend-url.com/recommendations"
        response = requests.post(backend_url, json=data)
        
        if response.status_code == 200:
            print("✅Successfully sent recommendations to the backend.")
        else:
            print(f"Failed to send recommendations. Status Code: {response.status_code}")

    def add_to_history(self, user_input: str, bot_response: str):
        self.message_history.add_user_message(user_input)
        self.message_history.add_ai_message(bot_response)

    def get_message_history(self):
        return self.message_history.messages

    def get_response(self, user_input: str):
        query_type = self.classify_query_with_prompt(user_input)

        if query_type == "RECOMMENDATION_TALK":
            # Retrieve relevant agencies and their IDs (sorted by cosine similarity score)
            retrieved_context, sorted_agency_ids = self.retrieve_relevant_agencies(user_query=user_input)

            # Send the recommended agency IDs to the backend
            self.send_agency_recommendation_to_backend(sorted_agency_ids)

            # Format full input for the chatbot
            full_input = f"{user_input}\n\nHere are some potentially relevant agencies:\n" + "\n".join(retrieved_context)

            # Prepare the prompt and invoke the LLM
            chain = self.prompt | self.llm
            result = chain.invoke({
                "input": full_input,
                "chat_history": self.message_history.messages
            })

            # Add to history
            self.add_to_history(user_input, result.content)

            return result.content

        else:
            # Handle non-recommendation queries
            chain = self.prompt | self.llm
            result = chain.invoke({
                "input": user_input,
                "chat_history": self.message_history.messages
            })

            # Add to history
            self.add_to_history(user_input, result.content)

            return result.content

# --- Main Chat Loop ---
if __name__ == "__main__":
    chatbot = AIAgencyChatbot()
    print("\n🤖 AI Agency Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = chatbot.get_response(user_input)
        print(f"Bot: {response}\n")
