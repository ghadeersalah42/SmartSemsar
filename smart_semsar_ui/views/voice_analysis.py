import streamlit as st
from components.cards import page_header
from services.voice_service import analyze_audio_demo


def render_voice_analysis() -> None:
    page_header(
        "Voice intelligence",
        "Analyze Customer Calls",
        "Upload a sample recording to preview the intended workflow. Uploading does not trigger AI processing.",
    )

    with st.container(border=True):
        st.subheader("🎙️ Customer Call")
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "m4a"],
            help="Demo UI only. The file is not sent to a model or external service.",
        )
        if uploaded is not None:
            st.audio(uploaded)
            st.caption(f"Selected file: {uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
            if st.button("Preview Analysis Workflow", type="primary"):
                result = analyze_audio_demo(uploaded.name)
                st.session_state["uploaded_audio_name"] = uploaded.name
                st.session_state["audio_demo_result"] = result
                st.success("Demo workflow preview created. No AI processing was run.")

    st.divider()
    st.subheader("Analysis Pipeline")
    st.caption("These are planned stages; statuses remain illustrative until backend integration.")
    steps = [
        ("1", "Audio Standardization", "Convert audio to a consistent format."),
        ("2", "Voice Activity Detection", "Identify likely speech regions."),
        ("3", "Speech Recognition", "Generate a transcript."),
        ("4", "Speaker Diarization", "Separate speakers where possible."),
        ("5", "Arabic Normalization", "Normalize transcript text."),
        ("6", "Intent Detection", "Extract real-estate requirements."),
        ("7", "Customer Profile", "Structure customer preferences."),
    ]
    for number, title, description in steps:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 7, 2])
            c1.markdown(f"### {number}")
            c2.markdown(f"**{title}**")
            c2.caption(description)
            c3.caption("Not connected")

    result = st.session_state.get("audio_demo_result")
    if result:
        st.info(result["message"])
        with st.expander("View demo result object"):
            st.json(result)
