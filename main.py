
# --- Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# --- Load Environment Variables ---
load_dotenv()

# --- Agency Data ---
agency_data = [
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
embeddings_model = OpenAIEmbeddings(chunk_size=100)

# --- Setup Text Splitter with Overlap ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# --- Prepare Texts and Metadata ---
texts = []
metadata = []

for agency in agency_data:
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
    chunks = splitter.split_text(combined_text)
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "id": f"{agency['id']}_{i}",
            **agency,
            "chunk_index": i
        })

# --- Generate Embeddings ---
vectors = embeddings_model.embed_documents(texts)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "ai-agency-chatbot" not in pc.list_indexes().names():
    pc.create_index(
        name="ai-agency-chatbot",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index("ai-agency-chatbot")

# --- Upsert to Pinecone ---
items_to_upsert = [
    {
        "id": str(meta["id"]),
        "values": vector,
        "metadata": {k: v for k, v in meta.items() if k != "id"}
    }
    for meta, vector in zip(metadata, vectors)
]
index.upsert(vectors=items_to_upsert)
print("✅ Upserted overlapping chunked vectors into Pinecone!")

# --- Define Retrieval Tool ---
def retrieve_relevant_agencies(user_query: str, top_k: int = 3) -> str:
    query_vector = embeddings_model.embed_query(user_query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    context = []
    for match in results["matches"]:
        meta = match["metadata"]
        context.append(
            f"Agency: {meta['name']}, Rating: {meta['rating']}, Tags: {', '.join(meta['tags'])}, Budget: {meta['budget']}, Expertise: {meta['expertise']}, Score: {match['score']:.4f}"
        )
    return "\n".join(context)

# --- Custom Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an AI agency recommendation expert who behaves like a helpful human consultant.
Your job is to chat naturally with the user, build rapport, and recommend suitable AI agencies only when the user is ready.

## Primary Objective:
- Help the user discover and choose the best-fit AI agency based on their needs.

## Conversational Logic:
Before replying to each user message, follow this logic strictly:
1. CLASSIFY the user message as either:
   - SMALL_TALK (e.g., "hi", "hello", "hey", "how are you?", "how's it going?", "what's your name?", "I'm fine", "I'm good", "my name is…", "who are you?", "nice to meet you", "just checking this out", "I'm new here", "are you a bot?", "I was just curious", "thanks", "nothing much, you?", "hope you're doing well", "pleasure to meet you", "just looking around", "glad to be here", "cool platform")
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
Bot: Hi there! 😊 How are you doing today?

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

# --- Initialize LLM and Message History ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
message_history = ChatMessageHistory()

# --- Chat Loop ---
print("\n🤖 AI Agency Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant agencies
    retrieved_context = retrieve_relevant_agencies(user_query=user_input)

    # Format full input
    full_input = f"{user_input}\n\nHere are some potentially relevant agencies:\n{retrieved_context}"

    # Prepare prompt
    chain = prompt | llm
    result = chain.invoke({
        "input": full_input,
        "chat_history": message_history.messages
    })

    # Display and store result
    print(f"Bot: {result.content}\n")
    message_history.add_user_message(user_input)
    message_history.add_ai_message(result.content)
