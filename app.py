import streamlit as st
import openai
import dotenv
import os
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import asyncio

dotenv.load_dotenv()

# Define available OpenAI models
openai_models = [
    "gpt-4", 
    "gpt-3.5-turbo-16k", 
    "gpt-4-turbo", 
    "gpt-4-32k",
]

# Function to stream the response from the LLM asynchronously
async def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model_params["model"],
            messages=st.session_state.messages,
            temperature=model_params["temperature"],
            max_tokens=4096,
            stream=True,  # this enables streaming
        )
        async for chunk in response:
            chunk_text = chunk["choices"][0].get("delta", {}).get("content", "")
            response_message += chunk_text
            yield chunk_text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {"type": "text", "text": response_message}
        ]
    })


# Function to handle full-duplex communication (text and audio)
async def chat_flow():
    # --- Page Config ---
    st.set_page_config(
        page_title="ChatBot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.markdown("<h1 style='text-align: center; color: #6ca395;'>ChatBot 💬</h1>", unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key : ",
            value=default_openai_api_key,
            type="password"
        )

    # --- Main Content ---
    if not openai_api_key or "sk-" not in openai_api_key:
        st.warning("Please enter a valid OpenAI API Key to continue.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Sidebar options
        with st.sidebar:
            st.divider()
            model = st.selectbox("Select a model:", openai_models, index=0)
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
            audio_response = st.checkbox("Enable audio response", value=False)

            if audio_response:
                tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable"])
                tts_model = st.selectbox("Select a TTS model:", ["tts-1", "tts-1-hd"], index=1)

            def reset_conversation():
                st.session_state.messages = []

            st.button("Reset Conversation", on_click=reset_conversation)
            st.divider()

        # Audio input - capturing audio in parallel with chat
        audio_prompt = None
        speech_input = audio_recorder("Press to record audio:", icon_size="3x", neutral_color="#6ca395")

        if speech_input:
            audio_file = BytesIO(speech_input)
            audio_file.name = "audio.wav"
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
            )
            audio_prompt = transcript["text"]

        # Chat input (text input or audio input)
        if prompt := st.chat_input("Ask anything") or audio_prompt:
            st.session_state.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })
            with st.chat_message("user"):
                st.write(prompt)

            # Start streaming the assistant's response asynchronously
            placeholder = st.empty()
            async for chunk_text in stream_llm_response(
                model_params={"model": model, "temperature": model_temp},
                model_type="openai",
                api_key=openai_api_key,
            ):
                placeholder.markdown(chunk_text, unsafe_allow_html=True)

            if audio_response:
                response_text = st.session_state.messages[-1]["content"][0]["text"]
                response = openai.Audio.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=response_text,
                )
                audio_base64 = base64.b64encode(response["audio"]).decode("utf-8")
                st.markdown(
                    f"""
                    <audio controls autoplay>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True,
                )

# Main function to start the chat flow
async def main():
    await chat_flow()

if __name__ == "__main__":
    asyncio.run(main())
