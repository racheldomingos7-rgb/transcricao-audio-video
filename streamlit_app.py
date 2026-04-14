import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="Transcrever Audio e Video",
    page_icon="🎙️",
    layout="centered",
)


@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")


def format_timestamp(seconds: float) -> str:
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def build_srt(segments) -> str:
    lines = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        lines.append(f"{index}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def transcribe_file(file_path: str):
    model = load_model()
    segments, info = model.transcribe(
        file_path,
        language="pt",
        task="transcribe",
        vad_filter=True,
        beam_size=5,
    )
    segment_list = list(segments)
    transcript_text = " ".join(segment.text.strip() for segment in segment_list if segment.text.strip())
    return transcript_text, segment_list, info


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }
    .hero {
        padding: 1.4rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #f7efe4 0%, #e5f3ee 100%);
        border: 1px solid #d7e4de;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        color: #1f3b35;
        font-size: 2rem;
    }
    .hero p {
        margin: 0;
        color: #38574f;
        font-size: 1rem;
    }
    </style>
    <div class="hero">
        <h1>Transcrever audio e video</h1>
        <p>Envie um arquivo, aguarde a transcricao em portugues e baixe o resultado em TXT e SRT.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Formatos aceitos: `mp3`, `wav`, `m4a`, `mp4`, `mov`")

uploaded_file = st.file_uploader(
    "Escolha seu arquivo",
    type=["mp3", "wav", "m4a", "mp4", "mov"],
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        st.info("Se for video, o audio sera lido automaticamente para a transcricao.")

        with st.spinner("Transcrevendo... Isso pode levar alguns minutos."):
            transcript_text, segments, info = transcribe_file(temp_file_path)

        txt_data = transcript_text if transcript_text else "Nenhum texto encontrado."
        srt_data = build_srt(segments)

        st.success("Transcricao concluida.")

        detected_language = getattr(info, "language", "pt")
        st.caption(f"Idioma detectado: {detected_language}")

        st.subheader("Texto")
        st.text_area(
            "Resultado da transcricao",
            value=txt_data,
            height=320,
        )

        base_name = Path(uploaded_file.name).stem

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Baixar TXT",
                data=txt_data.encode("utf-8"),
                file_name=f"{base_name}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Baixar SRT",
                data=srt_data.encode("utf-8"),
                file_name=f"{base_name}.srt",
                mime="text/plain",
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Ocorreu um erro ao transcrever: {error}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
else:
    st.caption("Depois do upload, a transcricao aparece aqui na tela.")
