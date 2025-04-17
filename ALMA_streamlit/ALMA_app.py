import os
import time
import tempfile
import asyncio
import numpy as np
import wave
import av

import edge_tts
import streamlit as st
from dotenv import load_dotenv
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode


from langsmith import Client
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.context import tracing_v2_enabled
from pinecone import Pinecone

# AudioProcessor for webrtc
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.audio_frames.append(audio.astype(np.int16))
        return frame

# Load env vars
load_dotenv(dotenv_path="/workspaces/ALMA_chatbot/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "").strip()

# API keys & config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\\ffmpeg\\...\\bin")
os.environ["PATH"] += os.pathsep + FFMPEG_PATH
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ALMA-Assistant"

# LangChain Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("alma-index")
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name="alma-index", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

video_vectorstore = PineconeVectorStore(index_name="alma-video-index", embedding=embeddings)
llm = ChatOpenAI(model="gpt-4")

# App Config
st.set_page_config(page_title="ALMA - AI Assistant", layout="centered")
st.title("🌿 Welcome to ALMA / Bienvenido a ALMA 🌿")

lang = st.radio("Choose your language / Elige tu idioma", ["English", "Español"])
input_mode = st.radio("Input mode / Modo de entrada", ["Text", "Voice"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "shown_video_ids" not in st.session_state:
    st.session_state.shown_video_ids = set()

if st.button("🗑️  Clear chat"):
    st.session_state.chat_history = []
    st.session_state.shown_video_ids = set()
    st.experimental_rerun()

# Welcome message
if lang == "Español":
    st.markdown("""
Hola, yo soy **ALMA** — tu consultora de IA para una vida mejor. 🌿
Estoy aquí para ayudarte a mejorar tu sueño, nutrición, estado de ánimo, energía y bienestar a largo plazo — con base científica y adaptado a ti. 💖
Puedes escribirme cualquier pregunta o compartir cómo te sientes. También puedo ayudarte a calcular tu IMC o TDEE si lo necesitas. 📊
Cuando un tema lo amerite, también puedo sugerirte **videos útiles** que complementen la conversación. 🎥
¡Estoy aquí para ti!
""")
else:
    st.markdown("""
Hi, I'm **ALMA** — your AI companion for better living. 🌿
I'm here to help you improve your sleep, nutrition, mood, energy, and long-term well-being — grounded in science and tailored to you. 💖
Feel free to ask me questions or share how you're feeling. I can also help calculate your BMI or TDEE if needed. 📊
When relevant, I’ll even suggest **helpful videos** to enrich what we talk about. 🎥
I'm here for you!
""")

# Show history
for msg in st.session_state.chat_history:
    user_msg, alma_msg = msg
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(alma_msg)

# Get input
user_input = ""
if input_mode == "Text":
    user_input = st.chat_input("Type your message" if lang == "English" else "Escribe tu mensaje")
elif input_mode == "Voice":
    st.subheader("🎙️ Voice Input")
    ctx = webrtc_streamer(
    key="voice_input",
    mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        audio_processor_factory=AudioRecorder,
        async_processing=True,
    )
    if not ctx.state.playing and ctx.audio_processor and ctx.audio_processor.audio_frames:
        audio_data = np.concatenate(ctx.audio_processor.audio_frames)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            with wave.open(f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())
            recognizer = sr.Recognizer()
            with sr.AudioFile(f.name) as source:
                try:
                    audio = recognizer.record(source)
                    user_input = recognizer.recognize_google(audio, language="es-ES" if lang == "Espa\u00f1ol" else "en-US")
                except Exception:
                    st.error("Speech recognition failed.")

# Main logic
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

    formatted_prompt = prompt_template.format(context=context, question=user_input)
    response = llm.invoke(formatted_prompt).content

    # Video suggestion
    video_results = video_vectorstore.similarity_search_with_score(user_input, k=2)
    if video_results:
        top_doc, score = video_results[0]
        if score >= 0.8:
            meta = top_doc.metadata
            vid_id = meta.get("video_id")
            if vid_id not in st.session_state.shown_video_ids:
                response += f"\n\n🎥 **{meta.get('video_title')}**\n{meta.get('video_url')}\n_{', '.join(meta.get('tags', []))}_"
                st.session_state.shown_video_ids.add(vid_id)

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append((user_input, response))

    # Speak
    async def speak(text):
        voice = "en-US-JennyNeural" if lang == "English" else "es-ES-ElviraNeural"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name
        await edge_tts.Communicate(text=text, voice=voice, rate="+10%").save(path)
        st.audio(path, format="audio/mp3")

    asyncio.run(speak(response))
