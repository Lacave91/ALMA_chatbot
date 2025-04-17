# ALMA Streamlit App with Voice Support (Public-Friendly Version)
import os
import time
import tempfile
import asyncio
import re
import json
import edge_tts
from datetime import datetime

from langsmith import Client
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.context import tracing_v2_enabled
from pinecone import Pinecone

import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr

# Load .env
load_dotenv(r"C:\Users\alvar\OneDrive\文件\Iron Hack\ALMA-Chatbot\.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "").strip()

# Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\\ffmpeg\\...\\bin")
os.environ["PATH"] += os.pathsep + FFMPEG_PATH
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ALMA-Assistant"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("alma-index")
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name="alma-index", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
video_vectorstore = PineconeVectorStore(index_name="alma-video-index", embedding=embeddings)
llm = ChatOpenAI(model="gpt-4")

# Streamlit Setup
st.set_page_config(page_title="ALMA - AI Assistant", layout="centered")
st.title("🌿 Welcome to ALMA / Bienvenido a ALMA 🌿")

lang = st.radio("Choose your language / Elige tu idioma", ["English", "Español"])
input_mode = st.radio("Input mode / Modo de entrada", ["Text", "Voice"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "shown_video_ids" not in st.session_state:
    st.session_state.shown_video_ids = set()

if st.button("🗑️ Clear chat"):
    st.session_state.chat_history = []
    st.session_state.shown_video_ids = set()
    st.experimental_rerun()

# Welcome
if lang == "Español":
    st.markdown("""
Hola, yo soy **ALMA** — tu consultora de IA para una vida mejor. 🌿
Estoy aquí para ayudarte con el sueño, nutrición, estado de ánimo y bienestar — con base científica y adaptado a ti.
Puedes preguntarme lo que quieras, o pedirme calcular tu IMC/TDEE. También puedo sugerir videos relevantes. 🎥
""")
else:
    st.markdown("""
Hi, I'm **ALMA** — your AI companion for better living. 🌿
I’m here to help you improve sleep, nutrition, mood, and long-term health — grounded in science and personalized.
Ask me anything, or try calculating your BMI/TDEE. I’ll also suggest helpful videos when relevant. 🎥
""")

for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Voice input handler
def transcribe_audio(uploaded_file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    with sr.AudioFile(temp_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="es-ES" if lang == "Español" else "en-US")
    except:
        return ""

# Chat input
user_input = ""
if input_mode == "Text":
    user_input = st.chat_input("Type your message" if lang == "English" else "Escribe tu mensaje")
elif input_mode == "Voice":
    uploaded_audio = st.file_uploader("🎙️ Record and upload your voice (WAV only)", type=["wav"])
    if uploaded_audio:
        user_input = transcribe_audio(uploaded_audio)
        st.markdown(f"**🎧 You said:** {user_input}")

if user_input:
    history = st.session_state.chat_history
    full_question = "Conversation so far:\n" + "\n".join([
        f"User: {q}\nALMA: {a}" for q, a in history[-3:]
    ]) + f"\nUser: {user_input}" if history else user_input

    docs = retriever.invoke(full_question)
    context_parts, total_chars, max_chars = [], 0, 3000
    for doc in docs:
        text = doc.page_content.strip()
        if total_chars + len(text) <= max_chars:
            context_parts.append(text)
            total_chars += len(text)
        else:
            break
    context = "\n\n".join(context_parts)

    # Prompt Setup
    prompt_template = ChatPromptTemplate.from_template("""
You are ALMA — a warm, intelligent, and caring AI assistant specialized in health, nutrition, sleep, mental wellness, and healthy aging.
You speak to the user like a thoughtful health coach: clear, knowledgeable, emotionally intelligent, and supportive.

When responding:
- If the user input is a **question**, begin with one of these:
    - "You’ve brought up something really meaningful."
    - "I'm really glad you asked that."
    - "That’s such an important question."
- If the user input is a **statement** or emotion, begin with one of these:
    - "Let’s take a closer look together."
    - "Here’s something that might help you."
    - "Thanks for sharing that with me."

Give a clear, grounded, and caring answer first — with insights the user can act on.
If you have more helpful info, end with a warm, relevant follow-up question — explain why you're asking it.
Use only the info in the context. If unsure, say:
> "I'm not sure based on what I know, but I can help you explore something else."

If a related video enhances your answer, suggest it:
> "Would you like to watch a video that explains this further? I found one that seems really helpful."
---
{context}
---
Question:
{question}
""")
    if lang == "Español":
        prompt_template = ChatPromptTemplate.from_template("""
Eres ALMA — una IA cálida, inteligente y empática en salud, nutrición, sueño y bienestar.
Hablas como una coach de salud reflexiva, emocionalmente inteligente y comprensiva.

Al responder:
- Si el usuario hace una **pregunta**, empieza con:
    - "Has planteado algo realmente importante."
    - "Me alegra mucho que me hayas preguntado eso."
    - "Esa es una pregunta muy valiosa."
- Si es una afirmación o emoción, empieza con:
    - "Echemos un vistazo más profundo juntas/os."
    - "Esto podría ayudarte."
    - "Gracias por compartirlo conmigo."

Da una respuesta clara y empática, con ideas prácticas.
Si tienes más info útil, termina con una pregunta cálida y relevante.
Responde solo con el contexto. Si no estás segura:
> "No estoy segura basándome en lo que sé, pero puedo ayudarte a explorar otras opciones."

Si hay un video útil, sugiérelo así:
> "¿Te gustaría ver un video que explique esto con más detalle? Encontré uno que podría ayudarte."
---
{context}
---
Pregunta:
{question}
""")

    prompt = prompt_template.format(context=context, question=user_input)
    response = llm.invoke(prompt).content

    # Video suggestion
    video_results = video_vectorstore.similarity_search_with_score(user_input, k=2)
    if video_results:
        top_doc, score = video_results[0]
        if score >= 0.8:
            meta = top_doc.metadata
            vid_id = meta.get("video_id")
            if vid_id not in st.session_state.shown_video_ids:
                snippet = top_doc.page_content[:300]
                note = f"\n🎥 **{meta.get('video_title')}**\n{meta.get('video_url')}\n_{', '.join(meta.get('tags', []))}_"
                follow_up = (
                    "\n\nWould you like to watch a video that explains this further? I found one that seems really helpful."
                    if lang == "English" else
                    "\n\n¿Te gustaría ver un video que explique esto con más detalle? Encontré uno que podría ayudarte."
                )
                response += f"{follow_up}{note}"
                st.session_state.shown_video_ids.add(vid_id)

    # Voice response (edge_tts)
    async def speak(text):
        voice = "en-US-JennyNeural" if lang == "English" else "es-ES-ElviraNeural"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            audio_path = f.name
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+10%")
        await communicate.save(audio_path)
        st.audio(audio_path, format="audio/mp3")

    # Show and speak
    st.session_state.chat_history.append((user_input, response))
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
        asyncio.run(speak(response))
