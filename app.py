import os
import hashlib
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from crisperwhisper import CrisperWhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from audio_analysis import (
    analyze_local_audio,
    assess_pronunciation_with_azure,
    normalized_audio_file,
)
from coaching import analyze_transcript, speech_metrics
from storage import initialize_database, list_reports, save_report


st.set_page_config(page_title="Voice Practice", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #18312c; --green: #176b5b; --coral: #c45a43; }
    .stApp { background: #f5f7f3; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245, 247, 243, 0.94); }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    [data-testid="stWidgetLabel"] p { color: var(--ink) !important; }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input,
    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        border-color: #a5b7ae !important;
    }
    [data-testid="stMetric"] { border-left: 3px solid var(--green); padding-left: 0.8rem; }
    div.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)

DATABASE_PATH = Path(__file__).with_name("reports.sqlite3")
initialize_database(DATABASE_PATH)


@st.cache_resource(show_spinner=False)
def load_transcription_model(device: str, model_path: str) -> CrisperWhisperModel:
    return CrisperWhisperModel(model_path, device=device)


@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(hf_token: str) -> Any:
    if not hf_token.strip():
        raise ValueError(
            "A Hugging Face token is required. Accept the model terms and set HF_TOKEN or enter a token in the sidebar."
        )
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token.strip(),
    )
    pipeline.to(torch.device("cpu"))
    return pipeline


def transcribe_uploaded_audio(uploaded_audio: Any, device: str) -> tuple[Any, str]:
    suffix = Path(uploaded_audio.name).suffix or ".wav"
    input_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(uploaded_audio.getvalue())
            input_path = temp_audio.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_audio:
            wav_path = converted_audio.name
        with st.status("Preparing local transcription...", expanded=True) as status:
            status.update(label="Converting uploaded audio to mono WAV...", state="running")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    input_path,
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            status.update(label="Checking for the speech model in the local cache...", state="running")
            try:
                model_path = snapshot_download(
                    "nyralabs/CrisperWhisper2.0_turbo",
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                status.update(
                    label="Downloading the speech model from Hugging Face. Keep this tab open; an internet connection is required for the first download...",
                    state="running",
                )
                model_path = snapshot_download("nyralabs/CrisperWhisper2.0_turbo")
            status.update(label="Loading the cached speech model...", state="running")
            transcription_model = load_transcription_model(device, model_path)
            status.update(
                label=f"Transcribing on {device.upper()}. This can take several minutes on CPU...",
                state="running",
            )
            result = transcription_model.transcribe(
                wav_path,
                language="en",
                mode="verbatim",
                word_timestamps=True,
            )
            status.update(label="Transcription complete.", state="complete")
        return result, model_path
    finally:
        if input_path:
            Path(input_path).unlink(missing_ok=True)
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


def metric_value(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0))


def clear_transcript_state() -> None:
    for key in (
        "transcript_editor",
        "word_timestamps",
        "audio_duration",
        "audio_name",
        "audio_sha256",
        "transcription_model_path",
        "last_report_id",
    ):
        st.session_state.pop(key, None)


def comparison_value(report: dict, group_name: str, key: str) -> float | None:
    values = (
        report.get("analysis", {}).get("scores", {})
        if group_name == "scores"
        else report.get(group_name, {})
    )
    if group_name == "metrics" and values.get("metrics_available") is False:
        return None
    if (
        group_name == "metrics"
        and key == "long_pauses_per_minute"
        and values.get("timing_available") is False
    ):
        return None
    value = values.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def show_audio_analysis(
    audio_analysis: dict | None,
    audio_coaching: list[dict] | None = None,
) -> None:
    if not audio_analysis:
        return

    st.subheader("Audio analysis")
    if audio_coaching:
        st.markdown("#### What to try next")
        for item in audio_coaching:
            st.markdown(
                f"**{item.get('speaker_id', 'Speaker')} · {item.get('category', 'Delivery practice')}**"
            )
            if item.get("observation"):
                st.write(item["observation"])
            if item.get("evidence"):
                st.caption(f"Based on: {item['evidence']}")
            st.write(f"Try: {item.get('suggestion', '')}")
            st.write(f"Practice: {item.get('practice', '')}")
            if item.get("success_metric"):
                st.caption(f"Next meeting: {item['success_metric']}")

    local = audio_analysis.get("local", {})
    local_status = local.get("status", "unavailable")
    if local_status not in ("complete", "partial"):
        st.info(local.get("reason", "Local audio analysis is unavailable for this report."))
    else:
        st.caption(
            "Speaker labels are anonymous. Pitch and intensity are recording-relative measurements, not quality grades."
        )
        for speaker in local.get("speakers", []):
            st.markdown(f"#### {speaker.get('speaker_id', 'Speaker')}")
            metric_columns = st.columns(3)
            speech_rate = speaker.get("speech_rate_wpm")
            pitch_median = speaker.get("pitch_median_hz")
            pitch_spread = speaker.get("pitch_spread_hz")
            intensity_spread = speaker.get("intensity_spread_db")
            metric_columns[0].metric(
                "Speech rate",
                f"{speech_rate:.1f} words/min" if isinstance(speech_rate, (int, float)) else "N/A",
            )
            metric_columns[1].metric(
                "Median pitch",
                f"{pitch_median:.1f} Hz" if isinstance(pitch_median, (int, float)) else "N/A",
            )
            metric_columns[2].metric(
                "Pitch spread (10th-90th percentile)",
                f"{pitch_spread:.1f} Hz" if isinstance(pitch_spread, (int, float)) else "N/A",
            )
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Intensity spread",
                f"{intensity_spread:.1f} dB"
                if isinstance(intensity_spread, (int, float))
                else "N/A",
            )
            metric_columns[1].metric("Speaking turns", speaker.get("turn_count", 0))
            metric_columns[2].metric(
                "Within-turn pauses",
                speaker.get("within_turn_pause_count", 0),
            )
        for note in local.get("notes", []):
            st.caption(note)

    pronunciation = audio_analysis.get("pronunciation", {})
    pronunciation_status = pronunciation.get("status", "not_requested")
    if pronunciation_status != "not_requested":
        st.markdown("#### Pronunciation assessment")
        if pronunciation_status not in ("complete", "partial"):
            st.info(pronunciation.get("reason", "Cloud pronunciation scoring is unavailable."))
        else:
            st.caption(
                "Azure Speech scores are provider outputs, not universal or accent-neutral measures. The speaker audio was sent to Azure for assessment."
            )
            for speaker in pronunciation.get("speakers", []):
                st.markdown(f"**{speaker.get('speaker_id', 'Speaker')}**")
                score_columns = st.columns(4)
                for column, label, key in zip(
                    score_columns,
                    ("Pronunciation", "Accuracy", "Fluency", "Prosody"),
                    ("pronunciation_score", "accuracy_score", "fluency_score", "prosody_score"),
                ):
                    value = speaker.get(key)
                    column.metric(label, f"{value:.1f}/100" if isinstance(value, (int, float)) else "N/A")
                low_scoring_words = speaker.get("low_scoring_words", [])
                if low_scoring_words:
                    st.dataframe(low_scoring_words, hide_index=True)
            for warning in pronunciation.get("warnings", []):
                st.warning(warning)
            for note in pronunciation.get("notes", []):
                st.caption(note)


def show_report(analysis: dict, audio_analysis: dict | None = None) -> None:
    st.subheader("Coaching report")
    st.write(analysis.get("summary", ""))

    scores = analysis.get("scores", {})
    score_columns = st.columns(4)
    for column, label, key in zip(
        score_columns,
        ("Fluency", "Clarity", "Grammar", "Concision"),
        ("fluency", "clarity", "grammar", "concision"),
    ):
        column.metric(label, f"{int(scores.get(key, 0))}/100")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Strengths")
        for strength in analysis.get("strengths", []):
            st.markdown(f"- {strength}")
    with right:
        st.markdown("#### Patterns to work on")
        for pattern in analysis.get("patterns", []):
            category = pattern.get("category", "Communication")
            priority = pattern.get("priority", "")
            st.markdown(f"**{category}** · {priority}")
            st.write(pattern.get("observation", ""))
            if pattern.get("evidence"):
                st.caption(f"Example: {pattern['evidence']}")
            st.write(f"Try: {pattern.get('suggestion', '')}")

    plan = analysis.get("practice_plan", {})
    st.markdown("#### Next practice")
    st.markdown(f"**Focus:** {plan.get('focus', 'Choose one pattern to practice.')}")
    st.write(plan.get("exercise", ""))
    st.caption(f"Measure: {plan.get('success_metric', 'Review this pattern in your next meeting.')}")
    show_audio_analysis(audio_analysis, analysis.get("audio_coaching", []))


def render_progress() -> None:
    reports = list_reports(DATABASE_PATH)
    st.header("Progress")
    if not reports:
        st.info("Saved reports will appear here.")
        return

    chart_rows = list(reversed(reports))
    chart_metric = st.selectbox(
        "Trend",
        (
            "Filler words per 1,000 words",
            "Specific filler per 1,000 words",
            "Long pauses per minute",
            "Repeated words",
            "Fluency score",
        ),
    )
    metric_keys = {
        "Filler words per 1,000 words": ("filler_per_1000", "metrics"),
        "Long pauses per minute": ("long_pauses_per_minute", "metrics"),
        "Repeated words": ("repeated_words", "metrics"),
        "Fluency score": ("fluency", "scores"),
    }
    if chart_metric == "Specific filler per 1,000 words":
        available_fillers = sorted(
            {
                filler
                for report in chart_rows
                if report["metrics"].get("metrics_available") is not False
                for filler in report["metrics"].get("fillers_by_type", {})
            }
        )
        if available_fillers:
            selected_filler = st.selectbox("Filler", available_fillers)
            chart_label = f"{selected_filler} / 1,000 words"
            chart_values = [
                round(
                    report["metrics"].get("fillers_by_type", {}).get(selected_filler, 0)
                    * 1000
                    / max(report["metrics"].get("word_count", 0), 1),
                    1,
                )
                if report["metrics"].get("metrics_available") is not False
                else None
                for report in chart_rows
            ]
        else:
            chart_label = chart_metric
            chart_values = [0 for _ in chart_rows]
            st.caption("No saved reports contain per-filler counts yet.")
    else:
        metric_key, group = metric_keys[chart_metric]
        chart_label = chart_metric
        chart_values = [
            None
            if item["metrics"].get("metrics_available") is False
            or (
                metric_key == "long_pauses_per_minute"
                and item["metrics"].get("timing_available") is False
            )
            else metric_value(item[group], metric_key)
            for item in chart_rows
        ]
    chart_data = {
        "Date": [item["session_date"] for item in chart_rows],
        chart_label: chart_values,
    }
    st.line_chart(chart_data, x="Date", y=chart_label)

    st.subheader("Compare two reports")
    if len(reports) < 2:
        st.caption("Save another report to compare changes over time.")
    else:
        labels = {
            item["id"]: f"{item['session_date']} · {item['source_name']} · {item['created_at']}"
            for item in reports
        }
        ids = [item["id"] for item in reports]
        earlier_id = st.selectbox("Earlier", ids, index=len(ids) - 1, format_func=labels.get)
        later_options = [report_id for report_id in ids if report_id != earlier_id]
        later_id = st.selectbox("Later", later_options, format_func=labels.get)
        earlier = next(item for item in reports if item["id"] == earlier_id)
        later = next(item for item in reports if item["id"] == later_id)

        comparisons = (
            ("Fillers / 1,000 words", "filler_per_1000", "metrics", True),
            ("Long pauses / minute", "long_pauses_per_minute", "metrics", True),
            ("Repeated words", "repeated_words", "metrics", True),
            ("Fluency score", "fluency", "scores", False),
            ("Clarity score", "clarity", "scores", False),
            ("Grammar score", "grammar", "scores", False),
            ("Concision score", "concision", "scores", False),
        )
        columns = st.columns(4)
        for index, (label, key, group_name, lower_is_better) in enumerate(comparisons):
            old_value = comparison_value(earlier, group_name, key)
            new_value = comparison_value(later, group_name, key)
            if old_value is None or new_value is None:
                columns[index % len(columns)].metric(label, "N/A")
                continue
            delta = new_value - old_value
            columns[index % len(columns)].metric(
                label,
                f"{new_value:.1f}",
                f"{delta:+.1f}",
                delta_color="inverse" if lower_is_better else "normal",
            )

        filler_names = sorted(
            set(earlier["metrics"].get("fillers_by_type", {}))
            | set(later["metrics"].get("fillers_by_type", {}))
        )
        if filler_names:
            filler_comparison = []
            for filler in filler_names:
                earlier_rate = None
                if earlier["metrics"].get("metrics_available") is not False:
                    earlier_rate = (
                        earlier["metrics"].get("fillers_by_type", {}).get(filler, 0)
                        * 1000
                        / max(earlier["metrics"].get("word_count", 0), 1)
                    )
                later_rate = None
                if later["metrics"].get("metrics_available") is not False:
                    later_rate = (
                        later["metrics"].get("fillers_by_type", {}).get(filler, 0)
                        * 1000
                        / max(later["metrics"].get("word_count", 0), 1)
                    )
                filler_comparison.append(
                    {
                        "Filler": filler,
                        "Earlier / 1,000": round(earlier_rate, 1) if earlier_rate is not None else None,
                        "Later / 1,000": round(later_rate, 1) if later_rate is not None else None,
                        "Change / 1,000": (
                            round(later_rate - earlier_rate, 1)
                            if earlier_rate is not None and later_rate is not None
                            else None
                        ),
                    }
                )
            st.markdown("#### Filler-by-filler comparison")
            st.dataframe(filler_comparison, hide_index=True, width="stretch")

    st.subheader("Saved sessions")
    for report in reports:
        title = f"{report['session_date']} · {report['source_name']}"
        with st.expander(title):
            st.write(report["analysis"].get("summary", ""))
            st.caption(f"Participant: {report['target_label'] or 'Whole conversation'}")
            show_report(report["analysis"], report["metrics"].get("audio_analysis"))


st.title("Voice Practice")
st.caption("Meeting speech feedback, saved so progress is visible over time.")

with st.sidebar:
    st.subheader("Transcription settings")
    transcription_devices = ["cpu"]
    if torch.backends.mps.is_available():
        transcription_devices.append("mps")
    selected_device = st.selectbox(
        "Compute device",
        transcription_devices,
        index=transcription_devices.index("mps") if "mps" in transcription_devices else 0,
        key="transcription_device",
        format_func=lambda device: "CPU" if device == "cpu" else "Apple GPU (default, experimental)",
        help="The Apple GPU backend can crash natively with some audio workloads. CPU is slower but avoids that MPS command-buffer failure.",
    )
    if selected_device == "cpu":
        st.caption("CPU mode is the fallback and can take several minutes to transcribe a short recording. Keep this tab open while it runs.")

    st.subheader("AI settings")
    ai_provider = st.radio(
        "LLM provider",
        ("Hosted provider", "Ollama on this Mac"),
        index=1,
        key="llm_provider",
    )
    if ai_provider == "Ollama on this Mac":
        model_name = st.text_input(
            "Local Ollama model",
            value="ollama_chat/llama3.1:8b",
            key="local_model_name",
            help="LiteLLM model ID. Download the Ollama model first, for example: ollama pull llama3.1:8b.",
        )
        api_base = st.text_input(
            "Ollama URL",
            value=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            key="local_api_base",
        ).strip().rstrip("/")
        api_key = ""
        st.caption("Transcript and measured-audio coaching stay on this Mac. Ollama must be installed and running.")
    else:
        model_name = st.text_input(
            "Provider / model",
            value=os.getenv("AI_MODEL", "openai/gpt-4o-mini"),
            key="hosted_model_name",
            help="Use a LiteLLM model identifier, for example openai/gpt-4o-mini or anthropic/claude-3-5-sonnet-20241022.",
        )
        api_key = st.text_input(
            "Provider API key",
            value=os.getenv("AI_API_KEY", ""),
            type="password",
            key="hosted_api_key",
            help="Used only for this session. Leave blank to use the provider's environment variable.",
        )
        api_base = None
    pause_threshold = st.slider("Long-pause threshold", 0.5, 2.0, 0.8, 0.1, format="%.1f s")
    communication_goal = st.text_input(
        "Communication goal",
        value="Clear, engaging meeting contributions",
        help="Audio coaching will tailor its practice suggestion to this goal.",
    )

    st.subheader("Audio analysis")
    speaker_handling = st.selectbox(
        "Speaker handling",
        ("Auto-detect speakers", "Single speaker"),
        help="Choose Single speaker when only one person talks and automatic diarization splits their voice into multiple IDs.",
    )
    local_audio_enabled = st.checkbox(
        "Analyze speaker delivery locally",
        value=True,
        help="Uses pyannote for speaker turns and Praat for pitch and intensity measurements. A Hugging Face token is required to download the diarization model.",
    )
    hf_token = st.text_input(
        "Hugging Face token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        key="audio_hf_token",
        help="Accept the access conditions for pyannote/speaker-diarization-community-1 before using this token.",
    )
    azure_pronunciation_enabled = st.checkbox(
        "Add Azure pronunciation assessment",
        value=False,
        help="Requires local speaker analysis. Speaker audio clips will be sent to Azure Speech; credentials are used for this session only.",
    )
    azure_speech_key = ""
    azure_speech_region = ""
    if azure_pronunciation_enabled:
        azure_speech_key = st.text_input(
            "Azure Speech key",
            value=os.getenv("AZURE_SPEECH_KEY", ""),
            type="password",
            key="azure_speech_key_input",
        )
        azure_speech_region = st.text_input(
            "Azure Speech region",
            value=os.getenv("AZURE_SPEECH_REGION", ""),
            key="azure_speech_region_input",
        )
        st.warning("Audio clips will be sent to Azure Speech for pronunciation assessment.")

new_tab, progress_tab = st.tabs(("New report", "Progress"))

with new_tab:
    input_mode = st.radio(
        "Input type",
        ("Meeting audio", "Transcript"),
        horizontal=True,
        key="input_mode",
        on_change=clear_transcript_state,
    )
    uploaded_audio = None
    uploaded_transcript = None
    if input_mode == "Meeting audio":
        uploaded_audio = st.file_uploader(
            "Meeting audio",
            type=("wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"),
        )
    else:
        uploaded_transcript = st.file_uploader(
            "Transcript file",
            type=("txt", "md", "srt", "vtt", "tsv", "csv", "json"),
        )
        if st.button("Load transcript", disabled=uploaded_transcript is None):
            try:
                transcript_text = uploaded_transcript.getvalue().decode("utf-8-sig")
                if not transcript_text.strip():
                    raise ValueError("The uploaded transcript is empty.")
                st.session_state["transcript_editor"] = transcript_text.strip()
                st.session_state["word_timestamps"] = []
                st.session_state["audio_duration"] = 0
                st.session_state["audio_name"] = uploaded_transcript.name
                st.success("Transcript ready for analysis.")
            except UnicodeDecodeError:
                st.error("Could not read this file as UTF-8 text. Upload the transcript encoded as UTF-8.")
            except ValueError as error:
                st.error(str(error))

    session_date = st.date_input("Meeting date", value=date.today())
    target_label = st.text_input(
        "Speaker to focus on (optional)",
        help="Transcript-based speaker metrics require labels such as Alex: or WebVTT <v Alex> tags. Audio analysis separately detects anonymous speakers and reports delivery for each one.",
    )

    if input_mode == "Meeting audio" and st.button(
        "Transcribe for review",
        disabled=uploaded_audio is None,
        help="Optional: transcribe first if you want to edit the transcript before analysis.",
    ):
        try:
            result, model_path = transcribe_uploaded_audio(uploaded_audio, selected_device)
            st.session_state["transcript_editor"] = result.text
            st.session_state["word_timestamps"] = [
                {"word": word.word, "start": word.start, "end": word.end}
                for word in (result.words or [])
            ]
            st.session_state["audio_duration"] = result.duration
            st.session_state["audio_name"] = uploaded_audio.name
            st.session_state["audio_sha256"] = hashlib.sha256(uploaded_audio.getvalue()).hexdigest()
            st.session_state["transcription_model_path"] = model_path
            st.success("Transcript ready to review.")
        except Exception as error:
            st.error(f"Transcription failed: {error}")

    has_transcript = bool(st.session_state.get("transcript_editor"))
    can_analyze = has_transcript or uploaded_audio is not None or uploaded_transcript is not None
    transcript = (
        st.text_area("Transcript", height=260, key="transcript_editor")
        if has_transcript
        else ""
    )
    if can_analyze:
        if st.button(
            "Analyze and save report",
            type="primary",
            help="Transcribes audio and runs both analyses in one step. Use the optional review buttons only if you want to edit first.",
        ):
            try:
                uploaded_audio_hash = (
                    hashlib.sha256(uploaded_audio.getvalue()).hexdigest()
                    if uploaded_audio is not None
                    else None
                )
                needs_transcription = uploaded_audio is not None and (
                    not transcript.strip()
                    or st.session_state.get("audio_sha256") != uploaded_audio_hash
                )
                if needs_transcription:
                    result, model_path = transcribe_uploaded_audio(uploaded_audio, selected_device)
                    transcript = result.text
                    st.session_state["transcript_editor"] = transcript
                    st.session_state["word_timestamps"] = [
                        {"word": word.word, "start": word.start, "end": word.end}
                        for word in (result.words or [])
                    ]
                    st.session_state["audio_duration"] = result.duration
                    st.session_state["audio_name"] = uploaded_audio.name
                    st.session_state["audio_sha256"] = uploaded_audio_hash
                    st.session_state["transcription_model_path"] = model_path
                elif not transcript.strip() and uploaded_transcript is not None:
                    try:
                        transcript = uploaded_transcript.getvalue().decode("utf-8-sig").strip()
                    except UnicodeDecodeError as error:
                        raise ValueError("Upload the transcript encoded as UTF-8 and try again.") from error
                    st.session_state["transcript_editor"] = transcript
                    st.session_state["word_timestamps"] = []
                    st.session_state["audio_duration"] = 0
                    st.session_state["audio_name"] = uploaded_transcript.name
                if not transcript.strip():
                    raise ValueError("The transcript is empty. Check the audio and try again.")
                metrics = speech_metrics(
                    transcript,
                    st.session_state.get("word_timestamps", []),
                    st.session_state.get("audio_duration", 0),
                    pause_threshold,
                    target_label.strip(),
                )
                audio_analysis = {
                    "local": {
                        "status": "unavailable",
                        "reason": "Audio analysis is available only for meeting audio uploads.",
                        "speakers": [],
                    },
                    "pronunciation": {"status": "not_requested", "speakers": []},
                }
                if uploaded_audio is not None:
                    submitted_audio_hash = hashlib.sha256(uploaded_audio.getvalue()).hexdigest()
                    original_audio_hash = st.session_state.get("audio_sha256")
                    if original_audio_hash != submitted_audio_hash:
                        audio_analysis["local"]["reason"] = (
                            "The uploaded audio changed after transcription. Transcribe this audio before analyzing it."
                        )
                        if azure_pronunciation_enabled:
                            audio_analysis["pronunciation"] = {
                                "status": "unavailable",
                                "reason": "Transcribe the currently uploaded audio before pronunciation assessment.",
                                "speakers": [],
                            }
                    elif not local_audio_enabled:
                        audio_analysis["local"]["reason"] = "Local audio analysis was turned off."
                        if azure_pronunciation_enabled:
                            audio_analysis["pronunciation"] = {
                                "status": "unavailable",
                                "reason": "Local speaker attribution is required before cloud pronunciation scoring.",
                                "speakers": [],
                            }
                    else:
                        try:
                            with st.status("Analyzing speaker audio locally...", expanded=True) as audio_status:
                                audio_status.update(label="Loading local diarization model...", state="running")
                                diarization_pipeline = load_diarization_pipeline(hf_token)
                                model_path = st.session_state.get("transcription_model_path", "")
                                if not model_path:
                                    model_path = snapshot_download(
                                        "nyralabs/CrisperWhisper2.0_turbo",
                                        local_files_only=True,
                                    )
                                transcription_model = load_transcription_model(
                                    selected_device,
                                    model_path,
                                )
                                audio_status.update(label="Aligning and measuring each speaker...", state="running")
                                with normalized_audio_file(
                                    uploaded_audio.getvalue(),
                                    uploaded_audio.name,
                                ) as audio_path:
                                    local_result = analyze_local_audio(
                                        audio_path,
                                        transcript,
                                        transcription_model,
                                        diarization_pipeline,
                                        pause_threshold,
                                        num_speakers=1 if speaker_handling == "Single speaker" else None,
                                    )
                                    aligned_words = local_result.pop("_assigned_words", [])
                                    audio_analysis["local"] = local_result
                                    if azure_pronunciation_enabled:
                                        audio_status.update(
                                            label="Sending speaker clips to Azure Speech for pronunciation assessment...",
                                            state="running",
                                        )
                                        try:
                                            audio_analysis["pronunciation"] = assess_pronunciation_with_azure(
                                                audio_path,
                                                aligned_words,
                                                azure_speech_key,
                                                azure_speech_region,
                                            )
                                        except Exception as error:
                                            audio_analysis["pronunciation"] = {
                                                "status": "unavailable",
                                                "backend": "azure_speech",
                                                "reason": str(error),
                                                "speakers": [],
                                            }
                                audio_status.update(label="Audio analysis complete.", state="complete")
                        except Exception as error:
                            audio_analysis["local"] = {
                                "status": "unavailable",
                                "reason": str(error),
                                "speakers": [],
                            }
                            if azure_pronunciation_enabled:
                                audio_analysis["pronunciation"] = {
                                    "status": "unavailable",
                                    "backend": "azure_speech",
                                    "reason": "Local speaker attribution is required before cloud pronunciation scoring.",
                                    "speakers": [],
                                }
                metrics["audio_analysis"] = audio_analysis
                with st.spinner("Preparing transcript and audio coaching..."):
                    analysis = analyze_transcript(
                        transcript,
                        metrics,
                        target_label.strip(),
                        model_name.strip(),
                        api_key.strip(),
                        api_base=api_base or None,
                        communication_goal=communication_goal.strip(),
                    )
                report_id = save_report(
                    DATABASE_PATH,
                    session_date,
                    st.session_state.get("audio_name", "meeting audio"),
                    target_label.strip(),
                    transcript,
                    metrics,
                    analysis,
                )
                st.session_state["last_report_id"] = report_id
                st.success("Report saved to your local history.")
            except Exception as error:
                st.error(f"Analysis failed: {error}")

    if transcript:
        st.subheader("Speech metrics")
        metrics = speech_metrics(
            transcript,
            st.session_state.get("word_timestamps", []),
            st.session_state.get("audio_duration", 0),
            pause_threshold,
            target_label.strip(),
        )
        timing_available = metrics["timing_available"]
        metrics_available = metrics["metrics_available"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Words", metrics["word_count"] if metrics_available else "N/A")
        metric_columns[1].metric(
            "Fillers / 1,000 words",
            f"{metrics['filler_per_1000']:.1f}" if metrics_available else "N/A",
        )
        metric_columns[2].metric("Repeated words", metrics["repeated_words"] if metrics_available else "N/A")
        metric_columns[3].metric(
            "Long pauses",
            metrics["long_pause_count"] if metrics_available and timing_available else "N/A",
        )
        filler_rows = sorted(
            (
                {"Filler": filler, "Count": count, "Per 1,000 words": round(count * 1000 / max(metrics["word_count"], 1), 1)}
                for filler, count in metrics["fillers_by_type"].items()
                if count and metrics_available
            ),
            key=lambda row: row["Count"],
            reverse=True,
        )
        st.markdown("#### Filler breakdown")
        if not metrics_available:
            st.caption(metrics["metrics_unavailable_reason"])
        elif filler_rows:
            st.dataframe(filler_rows, hide_index=True, width="stretch")
            top_filler = filler_rows[0]
            st.caption(
                f"Most frequent: **{top_filler['Filler']}** ({top_filler['Count']} times). "
                "Practice replacing it with a brief silent pause. Counts are transcript-based; correct any transcription errors above."
            )
        else:
            st.caption("No tracked filler words detected. Counts are transcript-based; correct any transcription errors above.")
        if not metrics_available:
            st.caption("Pause metrics are unavailable because this participant could not be isolated.")
        elif timing_available:
            metric_scope = (
                f"Metrics are for {metrics['speaker_label']}; pause rate uses the meeting timeline."
                if target_label.strip()
                else "Metrics are for the full recording."
            )
            st.caption(
                f"Longest pause: {metrics['longest_pause_seconds']:.1f} s · "
                f"Long pauses per minute: {metrics['long_pauses_per_minute']:.1f} · "
                f"{metric_scope}"
            )
        else:
            st.caption("Pause timing is unavailable for transcript-only input.")

        last_report_id = st.session_state.get("last_report_id")
        if last_report_id:
            saved = next(
                (item for item in list_reports(DATABASE_PATH) if item["id"] == last_report_id),
                None,
            )
            if saved:
                show_report(saved["analysis"], saved["metrics"].get("audio_analysis"))

with progress_tab:
    render_progress()