# ALMA — Multilingual AI Chatbot for YouTube Video Q&A

**ALMA** is a multimodal, multilingual AI assistant designed to provide warm, emotionally intelligent answers to questions about health and wellness. Built using GPT-4, LangChain, OpenAI embeddings, and Pinecone vector stores, ALMA retrieves relevant information from YouTube videos and can even recommend helpful video segments in both English and Spanish.

---

## 🌐 Features
- **Multilingual**: Supports English and Spanish
- **Multimodal Input**: Accepts both voice and text input
- **Voice Output**: Speaks responses using Microsoft Edge TTS
- **Video Integration**: Suggests helpful YouTube clips based on similarity
- **Health Tools**: Built-in BMI, TDEE, and Macronutrient calculators
- **Short-Term Memory**: Remembers last 3 user exchanges for natural flow
- **Powered by GPT-4** and OpenAI embeddings
- **Retrieval-Augmented Generation (RAG)** pipeline with Pinecone
- **Fully integrated with LangChain and LangSmith for evaluation and tracing**

---

## 🔄 How It Works

### Data Pipeline
1. **Extract YouTube Transcripts** using `youtube_transcript_api`
2. **Chunk and Tag** transcript content for semantic retrieval
3. **Manually Curate Video Segments** with metadata and timestamps
4. **Embed** all data using OpenAI embeddings
5. **Store in Pinecone** vector databases: `alma-index` (transcripts) & `alma-video-index` (segments)

### Conversational Agent (LangChain)
- Uses a **Zero-Shot ReAct Agent**
- Retrieves context from vector DB
- Builds prompt with short-term memory (last 3 turns)
- Applies emotionally aware prompt templates (EN/ES)
- GPT-4 generates final response

### Tools
- `alma_calculator` tool handles BMI, TDEE, and macro calculations
- Agent chooses whether to use tool or GPT-4 directly
- Responses are wrapped in ALMA’s caring tone

---

## 🎤 Voice Interaction
- **Input**: Google Speech Recognition via `speech_recognition`
- **Output**: Edge TTS with natural voices (e.g., JennyNeural, ElviraNeural)
- Option to choose between text or voice at runtime

---

## 🔍 Evaluation with LangSmith
- Tracks all runs and responses
- Logs:
  - Input question
  - Tool selection
  - Retrieved documents
  - GPT-4 prompt & final output
- Helps trace bugs, misfires, and optimize prompt performance

---

## 🌟 Technologies Used
- Python
- LangChain
- OpenAI GPT-4
- Pinecone
- Streamlit / CLI Interface
- Edge TTS
- Google STT
- LangSmith

---

## 💼 Setup & Running Locally

1. Clone this repo:
git clone https://github.com/your-username/alma-chatbot.git
cd alma-chatbot
2. pip install -r requirements.txt
3. Add a .env file:
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key
FFMPEG_PATH=/your/path/to/ffmpeg
4. Run the app:
streamlit run ALMA_app.py
