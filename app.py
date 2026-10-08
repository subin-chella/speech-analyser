import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st
import torch
from crisperwhisper import CrisperWhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

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


def metric_value(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0))


def clear_transcript_state() -> None:
    for key in (
        "transcript_editor",
        "word_timestamps",
        "audio_duration",
        "audio_name",
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


def show_report(analysis: dict) -> None:
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
            st.dataframe(filler_comparison, hide_index=True, use_container_width=True)

    st.subheader("Saved sessions")
    for report in reports:
        title = f"{report['session_date']} · {report['source_name']}"
        with st.expander(title):
            st.write(report["analysis"].get("summary", ""))
            st.caption(f"Participant: {report['target_label'] or 'Whole conversation'}")
            show_report(report["analysis"])


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
        format_func=lambda device: "CPU (recommended)" if device == "cpu" else "Apple GPU (experimental)",
        help="The Apple GPU backend can crash natively with some audio workloads. CPU is slower but avoids that MPS command-buffer failure.",
    )
    if selected_device == "cpu":
        st.caption("CPU mode is stable but can take several minutes to transcribe a short recording. Keep this tab open while it runs.")

    st.subheader("AI settings")
    ai_provider = st.radio("LLM provider", ("Hosted provider", "Ollama on this Mac"))
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
        st.caption("Transcript analysis stays on this Mac. Ollama must be installed and running.")
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
        help="For speaker-specific metrics, the transcript must include speaker labels such as Alex: or WebVTT <v Alex> tags. Name order and unambiguous shortened names are matched. Audio transcription does not identify speakers automatically. Leave blank for whole-conversation metrics.",
    )

    if input_mode == "Meeting audio" and st.button(
        "Transcribe audio",
        type="primary",
        disabled=uploaded_audio is None,
    ):
        suffix = Path(uploaded_audio.name).suffix or ".wav"
        input_path = None
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
                temp_audio.write(uploaded_audio.getvalue())
                input_path = temp_audio.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_audio:
                wav_path = converted_audio.name
            with st.status("Preparing local transcription...", expanded=True) as transcription_status:
                transcription_status.update(
                    label="Converting uploaded audio to mono WAV...",
                    state="running",
                )
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
                transcription_status.update(
                    label="Checking for the speech model in the local cache...",
                    state="running",
                )
                try:
                    model_path = snapshot_download(
                        "nyralabs/CrisperWhisper2.0_turbo",
                        local_files_only=True,
                    )
                except LocalEntryNotFoundError:
                    transcription_status.update(
                        label="Downloading the speech model from Hugging Face. Keep this tab open; an internet connection is required for the first download...",
                        state="running",
                    )
                    model_path = snapshot_download("nyralabs/CrisperWhisper2.0_turbo")
                transcription_status.update(
                    label="Loading the cached speech model...",
                    state="running",
                )
                transcription_model = load_transcription_model(selected_device, model_path)
                transcription_status.update(
                    label=f"Transcribing on {selected_device.upper()}. This can take several minutes on CPU...",
                    state="running",
                )
                result = transcription_model.transcribe(
                    wav_path,
                    language="en",
                    mode="verbatim",
                    word_timestamps=True,
                )
                transcription_status.update(label="Transcription complete.", state="complete")
            st.session_state["transcript_editor"] = result.text
            st.session_state["word_timestamps"] = [
                {"word": word.word, "start": word.start, "end": word.end}
                for word in (result.words or [])
            ]
            st.session_state["audio_duration"] = result.duration
            st.session_state["audio_name"] = uploaded_audio.name
            st.success("Transcription ready.")
        except Exception as error:
            st.error(f"Transcription failed: {error}")
        finally:
            if input_path:
                Path(input_path).unlink(missing_ok=True)
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

    if st.session_state.get("transcript_editor"):
        transcript = st.text_area("Transcript", height=260, key="transcript_editor")
        if st.button("Analyze and save report", type="primary"):
            try:
                metrics = speech_metrics(
                    transcript,
                    st.session_state.get("word_timestamps", []),
                    st.session_state.get("audio_duration", 0),
                    pause_threshold,
                    target_label.strip(),
                )
                with st.spinner("Preparing your coaching report..."):
                    analysis = analyze_transcript(
                        transcript,
                        metrics,
                        target_label.strip(),
                        model_name.strip(),
                        api_key.strip(),
                        api_base=api_base or None,
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
            st.dataframe(filler_rows, hide_index=True, use_container_width=True)
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
                show_report(saved["analysis"])

with progress_tab:
    render_progress()