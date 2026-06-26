import streamlit as st
import requests
import os

# URL da API: usa variável de ambiente se definida, senão aponta pra produção
API_URL = os.getenv("API_URL", "https://music-genre-classifier-os8j.onrender.com")

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

st.title("🎵 Music Genre Classifier")
st.write("Envie um arquivo de áudio e descubra o gênero musical predito pelo modelo.")

uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio",
    type=["wav", "mp3", "ogg", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Classificar gênero"):
        with st.spinner("Analisando o áudio... isso pode levar até 1 minuto."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_URL}/predict", files=files, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Gênero predito: **{result['genre'].upper()}**")
                    st.metric("Confiança", f"{result['confidence']:.1%}")
                else:
                    st.error(f"Erro: {response.json().get('detail', 'falha na predição')}")

            except requests.exceptions.Timeout:
                st.error("A API demorou muito para responder. Tente novamente — pode ser o cold start do Render.")
            except requests.exceptions.ConnectionError:
                st.error("Não foi possível conectar à API.")