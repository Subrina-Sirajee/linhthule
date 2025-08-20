
import os
import json
import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages.human import HumanMessage
import langdetect

from .qdrant_vector_search import VectorSearchBase
from .qdrant_vector_search import QdrantVectorSearch


class AgencySuggestionChatbot:
    # Class variables
    index_name: str = "ai-agency-chatbot"
    vector_search: VectorSearchBase = None
    message_history: ChatMessageHistory = None
    embeddings_model: OpenAIEmbeddings = None
    llm: ChatOpenAI = None

    supported_languages = {
        'en': 'English',
        'bn': 'Bangla',
        'fr': 'French',
        'nl': 'Dutch',
        'de': 'German',
        'es': 'Spanish'
    }

    def get_message_history(self):
        if self.message_history is None:
            self.message_history = ChatMessageHistory()
        return self.message_history

    @classmethod
    def get_llm(cls):
        if cls.llm is None:
            cls.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        return cls.llm

    @classmethod
    def get_vector_search(cls):
        if cls.vector_search is None:
            cls.vector_search = QdrantVectorSearch(
                index_name=cls.index_name
            )
            # --- Agency Data ---
            agency_data = cls.get_agency_data()
            cls.vector_search.connect()

            # Check if Collection Has Data
            collection_info = cls.vector_search.client.get_collection(cls.index_name)
            if collection_info.points_count == 0:
                print(f"Collection {cls.index_name} is empty. Building index with agency data...")
                cls.vector_search.build(agency_data)
                print("✅ Initialized Qdrant vector search and upserted agency data!")
            else:
                print(f"✅ Connected to existing Qdrant collection {cls.index_name} with {collection_info.points_count} points.")
                # Populate in-memory agency_data for ID assignment and other operations
                cls.vector_search.agency_data = agency_data
        return cls.vector_search

    @classmethod
    def get_embeddings_model(cls):
        if cls.embeddings_model is None:
            cls.embeddings_model = OpenAIEmbeddings(chunk_size=100)
        return cls.embeddings_model

    def __init__(self, index_name: str = None):
        if index_name is not None:
            self.__class__.index_name = index_name

        # Initialize class methods - this ensures they're called at least once
        self.__class__.get_vector_search()
        self.__class__.get_llm()
        self.__class__.get_embeddings_model()

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
     """You are a multilingual AI agency recommendation expert who behaves like a helpful human consultant.
Your job is to chat naturally with the user in THEIR LANGUAGE, build rapport, and recommend suitable AI agencies only when the user is ready.

## CRITICAL LANGUAGE RULE:
- ALWAYS respond in the SAME LANGUAGE as the user's input
- If user writes in Spanish, respond in Spanish
- If user writes in French, respond in French  
- If user writes in German, respond in German
- If user writes in Dutch, respond in Dutch
- If user writes in Bangla, respond in Bangla
- And so on for all languages
- If you don't know how to respond in a specific language, politely ask the user if they can communicate in English

## Primary Objective:
- Help the user discover and choose the best-fit AI agency based on their needs IN THEIR LANGUAGE.

## Conversational Logic:
Before replying to each user message, follow this logic strictly:
1. DETECT the user's language and respond in that SAME language
2. CLASSIFY the user message as either:
   - SMALL_TALK (e.g., "hi", "hello", "hey", "how are you?", "how's it going?", "what's your name?", "I'm fine", "I'm good", "my name is…", "who are you?", "nice to meet you", "just checking this out", "I'm new here", "are you a bot?", "I was just curious", "thanks", "nothing much, you?", "hope you're doing well", "pleasure to meet you", "just looking around", "glad to be here", "cool platform","Good Morning", "Good Afternoon", "Good Evening", "Good Night", "How's your day?", "What are you up to?", "What can you do?")
   - AGENCY_RELATED (e.g., mentions "project", "AI", "automate", "automation", "chatbot", "LLM", "GPT", "OpenAI", "agency", "build", "develop", "create", "data", "data analysis", "voice clone", "text to speech", "platform", "web app", "mobile app", "recommendation engine", "machine learning", "deep learning", "computer vision", "backend", "frontend", "API", "NLP", "OCR", "startup", "we need", "I want to build", "I need help", "looking for agency", "hiring", "team", "our budget", "timeline", "quote", "proposal", "technical support", "sector", "industry", "use case", "finance", "retail", "logistics", "education", "healthcare")

3. IF the user message is SMALL_TALK:
   - Stay in friendly mode IN THEIR LANGUAGE.
   - DO NOT mention AI projects, agencies, goals, recommendations, or tools — unless the user **asks**.
   - Just respond warmly and socially in their language.
   - Use light, natural tone — like a human building rapport.
   - Use emojis sparingly to keep things warm.
   - Use appropriate cultural context for their language.

4. IF the user asks "Who are you?" (in any language):
   - Respond in their language: "I'm your AI agency recommendation consultant at WIIZ.AI. I can help you connect with the right AI agency when you're ready"

5. IF the user message is AGENCY_RELATED:
   - Switch to helpful consultant mode IN THEIR LANGUAGE.
   - Politely acknowledge their goal and ask smart, minimal discovery questions (e.g., about project type, budget, timeline, expectations).
   - Guide them toward relevant AI agencies from WIIZ's Qdrant database using vector similarity.

## DO NOTs (when in SMALL_TALK mode):
- ❌ DO NOT say: "Are you working on an AI project?" (in any language)
- ❌ DO NOT say: "Let's talk about your company or goals." (in any language)
- ❌ DO NOT say: "If you're looking for an agency…" (in any language)
- ❌ DO NOT say: "Share details about your project…" (in any language)
- ❌ DO NOT assume the user wants help until they clearly mention a goal, need, or project.

## Language-Specific Examples:

### English:
User: hi
Bot: Hi there! I'm your AI agency recommendation consultant at WIIZ.AI. How are you doing today?

User: I'm good. What about you?
Bot: I'm doing great, thanks for asking! 

### Spanish:
User: Hola, ¿cómo estás?
Bot: ¡Hola!  Soy tu consultor de recomendaciones de agencias de IA en WIIZ.AI. ¿Cómo estás hoy?

User: Necesito una agencia para automatizar documentos
Bot: ¡Eso suena como un gran caso de uso para IA! 
¿Puedes contarme qué tipo de documentos manejas y si tienes un presupuesto o cronograma preferido?

### French:
User: Bonjour, comment allez-vous?
Bot: Bonjour ! Je suis votre consultant en recommandations d'agences IA chez WIIZ.AI. Comment allez-vous aujourd'hui ?

User: Je cherche une agence pour l'analyse de données
Bot: C'est un excellent projet ! 
Pouvez-vous me parler de votre secteur d'activité et du type de données que vous souhaitez analyser ?

### German:
User: Hallo, wie geht es dir?
Bot: Hallo! Ich bin Ihr KI-Agentur-Empfehlungsberater bei WIIZ.AI. Wie geht es Ihnen heute?

User: Ich brauche eine Agentur für Chatbot-Entwicklung
Bot: Das klingt nach einem großartigen KI-Projekt!
Können Sie mir mehr über Ihr Unternehmen und Ihre spezifischen Chatbot-Anforderungen erzählen?

### Dutch:
User: Hallo, hoe gaat het?
Bot: Hallo! Ik ben uw AI-bureau aanbevelingsconsultant bij WIIZ.AI. Hoe gaat het vandaag met u?

User: Ik zoek een bureau voor machine learning
Bot: Dat klinkt als een geweldig AI-project! 
Kunt u me vertellen over uw bedrijf en welke specifieke machine learning behoeften u heeft?

### Bangla:
User: হ্যালো, কেমন আছেন?
Bot: হ্যালো! আমি WIIZ.AI-এর আপনার AI এজেন্সি সুপারিশ পরামর্শদাতা। আজ আপনার কেমন কাটছে?

User: আমার ডেটা বিশ্লেষণের জন্য একটি এজেন্সি দরকার
Bot: এটি AI-এর জন্য একটি দুর্দান্ত ব্যবহারের ক্ষেত্র! 
আপনি কী ধরনের ডেটা নিয়ে কাজ করেন এবং আপনার বাজেট বা সময়সীমা সম্পর্কে বলতে পারেন?

## Agency Discovery Mode:
 When the user mentions something related to an AI project, automation, data, agencies, or business goals:
- Start asking discovery questions (gently) IN THEIR LANGUAGE

💬 Follow-up Questions (use only when user starts talking about their project):
Ask smart, context-based questions gradually using these categories (translate to user's language):
1. 🏢 Company context (industry, team size, services)
English: "Can you tell me a bit about your company — like what industry you're in and how big your team is?"
Spanish: "¿Puedes contarme un poco sobre tu empresa, como en qué industria están y qué tan grande es tu equipo?"
French: "Pouvez-vous me parler un peu de votre entreprise — dans quel secteur vous travaillez et quelle est la taille de votre équipe ?"
German: "Können Sie mir etwas über Ihr Unternehmen erzählen — in welcher Branche Sie tätig sind und wie groß Ihr Team ist?"
Dutch: "Kunt u me iets vertellen over uw bedrijf — in welke sector u actief bent en hoe groot uw team is?"
2. 🎯 Goals & challenges
English: "What's your main goal with this AI project? Are you solving a specific business challenge?"
Spanish: "¿Cuál es tu objetivo principal con este proyecto de IA? ¿Están resolviendo un desafío empresarial específico?"
French: "Quel est votre objectif principal avec ce projet d'IA ? Résolvez-vous un défi commercial spécifique ?"
German: "Was ist Ihr Hauptziel mit diesem KI-Projekt? Lösen Sie eine spezifische Geschäftsherausforderung?"
Dutch: "Wat is uw hoofddoel met dit AI-project? Lost u een specifieke bedrijfsuitdaging op?"
3. 🔍 Use case details (manual tasks, data types)
English: "Are there any repetitive tasks you'd like to automate or improve with AI?"
Spanish: "¿Hay tareas repetitivas que les gustaría automatizar o mejorar con IA?"
French: "Y a-t-il des tâches répétitives que vous aimeriez automatiser ou améliorer avec l'IA ?"
German: "Gibt es repetitive Aufgaben, die Sie mit KI automatisieren oder verbessern möchten?"
Dutch: "Zijn er repetitieve taken die u zou willen automatiseren of verbeteren met AI?"
4. 🤖 AI experience & tech stack
English: "Have you or your team worked with AI technologies before? Any preferred tools or platforms?"
Spanish: "¿Tú o tu equipo han trabajado con tecnologías de IA antes? ¿Alguna herramienta o plataforma preferida?"
French: "Avez-vous ou votre équipe déjà travaillé avec des technologies d'IA ? Des outils ou plateformes préférés ?"
German: "Haben Sie oder Ihr Team bereits mit KI-Technologien gearbeitet? Bevorzugte Tools oder Plattformen?"
Dutch: "Heeft u of uw team eerder gewerkt met AI-technologieën? Eventuele voorkeur voor tools of platforms?"
5. 📊 Data availability & privacy constraints
English: "Do you already have data you'd like to use for the AI project? Any privacy or compliance concerns?"
Spanish: "¿Ya tienen datos que les gustaría usar para el proyecto de IA? ¿Alguna preocupación de privacidad o cumplimiento?"
French: "Avez-vous déjà des données que vous aimeriez utiliser pour le projet d'IA ? Des préoccupations de confidentialité ou de conformité ?"
German: "Haben Sie bereits Daten, die Sie für das KI-Projekt verwenden möchten? Datenschutz- oder Compliance-Bedenken?"
Dutch: "Heeft u al gegevens die u zou willen gebruiken voor het AI-project? Eventuele privacy- of compliance-zorgen?"
6. 💰 Budget, currency, and project scope
English: "Do you have a rough budget or timeline in mind? (Please include the currency if possible.)"
Spanish: "¿Tienen un presupuesto aproximado o cronograma en mente? (Por favor incluyan la moneda si es posible.)"
French: "Avez-vous un budget approximatif ou un calendrier à l'esprit ? (Veuillez inclure la devise si possible.)"
German: "Haben Sie ein ungefähres Budget oder einen Zeitplan im Kopf? (Bitte geben Sie die Währung an, wenn möglich.)"
Dutch: "Heeft u een globaal budget of tijdschema in gedachten? (Gelieve de valuta te vermelden indien mogelijk.)"
7. 🌍 Preferred languages or agency locations
English: "Do you prefer working with agencies from specific countries or who speak specific languages?"
Spanish: "¿Prefieren trabajar con agencias de países específicos o que hablen idiomas específicos?"
French: "Préférez-vous travailler avec des agences de pays spécifiques ou parlant des langues spécifiques ?"
German: "Bevorzugen Sie die Zusammenarbeit mit Agenturen aus bestimmten Ländern oder die bestimmte Sprachen sprechen?"
Dutch: "Werkt u liever samen met bureaus uit specifieke landen of die specifieke talen spreken?"
8. ⭐ Experience, trust, and rating expectations
English: "Do you prefer agencies with lots of case studies, higher ratings, or experience in certain industries?"
Spanish: "¿Prefieren agencias con muchos casos de estudio, calificaciones más altas, o experiencia en ciertas industrias?"
French: "Préférez-vous des agences avec de nombreuses études de cas, des notes élevées, ou de l'expérience dans certaines industries ?"
German: "Bevorzugen Sie Agenturen mit vielen Fallstudien, höheren Bewertungen oder Erfahrung in bestimmten Branchen?"
Dutch: "Geeft u de voorkeur aan bureaus met veel casestudies, hogere ratings, of ervaring in bepaalde industrieën?""

⚙️ Matching labels used to recommend agencies include:
- Sector (industry)
- AI Specialization (e.g., NLP, Computer Vision)
- Technical Stack (e.g., Python, TensorFlow, Hugging Face)
- Project Scope (MVP, prototype, production-scale)
- Budget (in user's currency)
- Experience (years, past clients, domain expertise)
- Rating (based on internal scores)
- Language & Location (communication preferences)

⚠️ Important Rules:
- If the user mentions a budget in a specific currency (e.g., USD, EUR, GBP, JPY), always respond in that currency — do NOT convert it into other currencies.
- Never reference external resources or websites.
- NEVER translate agency names, but translate all descriptions, tags, and recommendations into the user's language.
- Always maintain natural, culturally appropriate conversation in the user's language.
- When recommending agencies, present the information in the user's language but keep agency names in English.
- Ask follow-up questions that are culturally sensitive and appropriate for the user's language context.

## No Match Scenarios:
- If the mismatch is due to budget, timeline, or scope (translate to user's language):
  "I'm sorry, but based on the current preferences, I couldn't find a perfect match. However, if you're open to adjusting the budget, timeline, or scope, I'd be happy to explore other options."
- If the mismatch is due to an unavailable sector or use case (translate to user's language):
  "I'm sorry, but we currently don't have any agencies specialized in the sector. However, I can suggest agencies with general AI expertise who have worked across multiple domains, potentially including your given sector related projects. Would you like to see those?"

✅ Ask 1–2 relevant follow-up questions at a time instead of listing everything. Avoid robotic or generic replies. Always maintain the warmth and natural flow of conversation in the user's native language.
"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
    def detect_language(self, text: str) -> str:
        """
        Detects the language of the input text.
        Returns language code (e.g., 'es', 'fr', 'de', etc.)
        """
        try:
            detected = langdetect.detect(text)
            # Store for error handling
            self._last_detected_language = detected if detected in self.supported_languages else 'en'
            return self._last_detected_language
        except:
            # Default to English if detection fails
            self._last_detected_language = 'en'
            return 'en'

    def translate_text(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """
        Translates text using OpenAI instead of Google Translate.
        More reliable and no dependency conflicts.
        """
        try:
            if target_language == 'en' and source_language == 'en':
                return text
            
            # Language mapping for clearer instructions
            language_names = {
                'en': 'English',
                'bn': 'Bengali/Bangla', 
                'fr': 'French',
                'nl': 'Dutch',
                'de': 'German',
                'es': 'Spanish'
            }
            
            target_lang_name = language_names.get(target_language, target_language)
            
            if target_language == 'en':
                prompt = f"Translate the following text to English. Only return the translation, nothing else:\n\n{text}"
            else:
                prompt = f"Translate the following text to {target_lang_name}. Only return the translation, nothing else:\n\n{text}"
            
            result = self.translation_llm.invoke(prompt)
            return result.content.strip()
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails

    def translate_query_to_english(self, user_query: str, detected_language: str) -> str:
        """
        Translates user query to English for database search if needed.
        """
        if detected_language == 'en':
            return user_query
        
        return self.translate_text(user_query, 'en', detected_language)

    @classmethod
    def get_agency_data(self):
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
        return agency_data

    def classify_query_with_prompt(self, user_query: str, language: str) -> str:
        """
        Classifies the user query into one of the three categories: SMALL_TALK, AGENCY_TALK, or RECOMMENDATION_TALK.
        """
        # Format the prompt with the user query
        prompt = self.classify_query_prompt.format(user_query=user_query)

        # Get the classification result from LLM
        result = self.llm.invoke(prompt)

        # Access the content of the result and strip any leading/trailing whitespace
        return result.content.strip()

    def retrieve_relevant_agencies(self, user_query: str, detected_language: str, top_k: int = 3):
        """
        Retrieves relevant agencies from Qdrant based on the user query.
        Translates query to English for database search.
        """
        # Translate query to English for vector search
        english_query = self.translate_query_to_english(user_query, detected_language)
        
        # Perform vector search with English query
        query_vector = self.embeddings_model.embed_query(english_query)
        results = self.vector_search.search_index(vector=query_vector, top_k=top_k)

        context = []
        agency_ids_with_scores = []

        for match in results:
            meta = match["metadata"]
            
            # Translate key information to user's language
            translated_tags = ', '.join([
                self.translate_text(tag, detected_language, 'en') for tag in meta['tags']
            ])
            
            translated_description = self.translate_text(meta['description'], detected_language, 'en')
            
            context.append(
                f"Agency: {meta['name']}, Rating: {meta['rating']}, Tags: {translated_tags}, "
                f"Budget: {meta['budget']}, Expertise: {meta['expertise']}, Score: {match['score']:.4f}, "
                f"Description: {translated_description}"
            )
            
            agency_ids_with_scores.append((int(meta['id']), match['score']))

        # Sort by similarity score
        agency_ids_with_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_agency_ids = [agency_id for agency_id, _ in agency_ids_with_scores]

        return context, sorted_agency_ids

    def send_agency_recommendation_to_backend(self, agency_ids: list, user_language: str):
        """
        Sends the recommended agency IDs and user language to the backend via an API call.
        """
        data = {
            "recommended_agencies": agency_ids,
            "user_language": user_language
        }
        
        print("Recommendation data:", data)

    def add_to_history(self, user_input: str, bot_response: str):
        """
        Adds user input and bot response to the chat history.
        """
        message_history = self.get_message_history()
        message_history.add_user_message(user_input)
        message_history.add_ai_message(bot_response)

    def get_title(self, user_input: str) -> str:
        """
        Generates a chat title based on user messages, optimized for AI agency recommendation scenarios.

        Returns:
            str: A concise title representing the main topic of the conversation
        """

        # Get chat history
        messages = self.get_message_history()

        combined_context = ""
        user_message_count = 1

        for msg in messages.messages:
            if isinstance(msg, HumanMessage):
                combined_context += f"{msg.content}\n"
                user_message_count += 1
        combined_context += user_input
        print(f"Combined context: {combined_context}")

        # If no messages or empty context, return default
        if user_message_count == 1 or not combined_context.strip():
            print("Generated Title: New Chat")
            return "New Chat"

        # Detect language from the conversation
        detected_language = self.detect_language(messages)
        language_name = self.supported_languages.get(detected_language, 'English')

        # Predefined titles for common casual conversation patterns
        casual_patterns = {
            'en': {
                'greetings': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'status': ['how are you', 'how\'s it going', 'what\'s up', 'fine', 'good', 'great'],
                'introductions': ['my name is', 'i\'m', 'nice to meet you', 'who are you'],
                'titles': {
                    'greetings': 'Casual Chat',
                    'status': 'General Conversation',
                    'introductions': 'Introduction Chat'
                }
            },
            # Other languages as needed...
        }

        # Check for casual conversation patterns
        context_lower = combined_context.lower().strip()
        patterns = casual_patterns.get(detected_language, casual_patterns['en'])

        # Check for agency-related content
        agency_keywords = ['project', 'ai', 'agency', 'automate', 'build', 'develop', 'chatbot', 
                        'machine learning', 'data', 'recommendation', 'budget', 'timeline']
        has_agency_content = any(keyword in context_lower for keyword in agency_keywords)

        if not has_agency_content:
            # Check which casual pattern matches
            for pattern_type, keywords in patterns.items():
                if pattern_type != 'titles':
                    for keyword in keywords:
                        if keyword in context_lower:
                            print(f"Generated Title: {patterns['titles'][pattern_type]}")
                            return patterns['titles'].get(pattern_type, patterns['titles']['greetings'])
            # Default casual title
            print(f"Generated Title: {patterns['titles']['greetings']}")
            return patterns['titles']['greetings']

        # For conversations with AI/agency content, generate dynamic title
        title_prompt = f"""Generate a concise title (2-4 words, max 35 characters) in {language_name} for an AI agency recommendation chat based on the user messages below. 

        TITLE RULES:
        1. Focus on the specific business need, industry, or AI solution type (e.g., 'Chatbot', 'Analytics', 'Automation').
        2. Use precise keywords that reflect the user's goal or domain (e.g., 'Finance', 'E-commerce', 'Healthcare').
        3. Avoid vague or generic terms like 'chat', 'conversation', 'help', 'question', 'AI project'.
        4. If the input is vague or lacks clear intent, default to a broad but relevant title like 'AI Solution'.
        5. Ensure the title is actionable and reflects what the user wants to build or solve.
        6. Examples of good titles:
        - English: 'Finance AI Analytics', 'E-commerce Chatbot', 'Healthcare Automation'
        - Spanish: 'Análisis IA Finanzas', 'Chatbot E-commerce', 'Automatización Salud'
        - French: 'Analyse IA Finance', 'Chatbot E-commerce', 'Automatisation Santé'
        - German: 'KI Finanz-Analyse', 'E-commerce Chatbot', 'Gesundheits-Automatisierung'
        - Dutch: 'AI Financiële Analyse', 'E-commerce Chatbot', 'Zorgautomatisering'
        - Bangla: 'ফিন্যান্স AI বিশ্লেষণ', 'ই-কমার্স চ্যাটবট', 'স্বাস্থ্যসেবা অটোমেশন'

        USER MESSAGES:
        {combined_context}

        OUTPUT FORMAT:
        Return ONLY the title (no quotes, explanations, or extra text). If no clear topic is identified, use 'AI Solution'. """

        try:
            result = self.llm.invoke(title_prompt)
            title = result.content.strip().replace('"', '').replace("'", "").replace('Title:', '').strip()
            if not title or len(title) > 35:
                title = 'AI Solution'
            print(f"Generated title: {title}")
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            return "AI Solution"



    def get_response(self, user_input: str):
        """
        Processes the user input and generates a response based on query classification and language detection.
        """
        self.get_title(user_input)
        try:
            # Detect user's language
            detected_language = self.detect_language(user_input)
            # print(f"Detected language: {detected_language}")


            
            # Classify the query
            query_type = self.classify_query_with_prompt(user_input, detected_language)
            # print(f"Query type: {query_type}")

            msg_history = self.get_message_history()

            if query_type == "RECOMMENDATION_TALK":
                # Retrieve relevant agencies with translations
                retrieved_context, sorted_agency_ids = self.retrieve_relevant_agencies(
                    user_input, detected_language
                )

                # Send recommendations to backend with language info
                self.send_agency_recommendation_to_backend(sorted_agency_ids, detected_language)

                # Format full input for the chatbot (in user's language)
                full_input = f"{user_input}\n\nHere are some potentially relevant agencies:\n" + "\n".join(retrieved_context)

                # Prepare the prompt and invoke the LLM with timeout handling
                chain = self.prompt | self.llm
                
                # print("Generating response... (this may take a few seconds)")
                result = chain.invoke({
                    "input": full_input,
                    "chat_history": msg_history.messages
                })

                # Add to history
                self.add_to_history(user_input, result.content)
                return result.content

            else:
                # Handle non-recommendation queries (small talk, agency questions)
                chain = self.prompt | self.llm
                
                # print("Generating response... (this may take a few seconds)")
                result = chain.invoke({
                    "input": user_input,
                    "chat_history": msg_history.messages
                })

                # Add to history
                self.add_to_history(user_input, result.content)
                return result.content
            
            
        except Exception as e:
            # traceback.print_exception(e)
            raise RuntimeError(f"------------{e}-------")
            print(f"❌ Error generating response: {e}")
            # Return error message in user's language
            error_messages = {
                'es': "Lo siento, hubo un error al generar la respuesta. Por favor, inténtalo de nuevo.",
                'fr': "Désolé, il y a eu une erreur lors de la génération de la réponse. Veuillez réessayer.",
                'de': "Entschuldigung, es gab einen Fehler bei der Antwortgenerierung. Bitte versuchen Sie es erneut.",
                'nl': "Sorry, er was een fout bij het genereren van het antwoord. Probeer het opnieuw.",
                'bn': "দুঃখিত, উত্তর তৈরি করতে একটি ত্রুটি হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
                'en': "Sorry, there was an error generating the response. Please try again."
            }
            
            detected_language = getattr(self, '_last_detected_language', 'en')
            return error_messages.get(detected_language, error_messages['en'])

import traceback

