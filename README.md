# Voice Practice

A local Streamlit app for transcribing meeting audio or uploading an existing transcript, reviewing communication patterns with an AI model, and comparing saved reports over time.

## Run

```bash
brew install ffmpeg
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r requirements-audio.txt
.venv/bin/streamlit run app.py
```

Create the environment first with `uv venv --python 3.12 .venv` if `.venv` does not exist yet.
The audio-analysis dependencies are separate because they add a sizeable PyTorch/pyannote stack; the existing transcription and coaching dependencies remain unchanged.

Choose **Meeting audio** or **Transcript**, then use **Analyze and save report** to run the complete workflow in one action. Audio is transcribed locally before local speaker analysis, transcript coaching, and saving. If you want to edit first, **Transcribe for review** and **Load transcript** remain available as optional steps. Transcript uploads support UTF-8 formats such as `.txt`, `.md`, `.srt`, and `.vtt`. To calculate transcript-based metrics for one participant, the transcript must include speaker labels (for example, `Alex:` or WebVTT `<v Alex>` tags) that match the focus name. If the speaker cannot be isolated, focused metrics are shown as unavailable rather than using whole-conversation values. Pause timing is available from audio word timestamps or timed transcript cues. The first audio transcription may download the CrisperWhisper model. Transcription defaults to the Apple GPU (MPS) when available, with CPU as the fallback. MPS can crash natively with some audio workloads; choose CPU in the sidebar if that occurs. Speaker diarization itself currently runs on CPU. The transcript and calculated speech metrics are sent to the selected AI provider. Reports are stored in `reports.sqlite3` in this folder.

Audio uploads can also produce a separate audio analysis in the same report. Local analysis uses pyannote.audio to identify anonymous speaker turns, CrisperWhisper to align the corrected transcript, and Parselmouth/Praat to measure speaker-level pitch, intensity, pace, and within-turn pauses. Use **Speaker handling → Single speaker** when a solo recording is split into multiple IDs; IDs with no aligned words are omitted automatically. The selected LLM uses those measurements and your communication goal to give concrete vocal-variety, word-stress, and pause tips, each tied to audio evidence and paired with a short practice drill. Existing transcript metrics and coaching remain in the same report. Transcript-only uploads continue to work and show audio analysis as unavailable.

The diarization model is gated on Hugging Face. Accept the conditions for [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), create a read token, and either set `HF_TOKEN` before launching Streamlit or enter the token in the sidebar. The model is downloaded once and diarization runs locally on CPU. The first audio analysis can take additional time while the model is downloaded and loaded. pyannote speaker attribution can be wrong during overlapping speech; the report uses anonymous speaker IDs and avoids a pronunciation grade for local analysis.

**Azure pronunciation assessment is optional and off by default.** Enable it in the sidebar only when you consent to send speaker audio clips to Azure Speech. Set `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` before launch, or enter credentials for the current session. The first implementation requests phoneme-level assessment for `en-US`, chunks speaker turns into short clips, and scores them against the edited transcript. Meeting conversation is less controlled than read-aloud assessment, so treat scores as provider feedback, not accent-neutral or universal judgments. Azure usage may incur charges. If diarization, alignment, credentials, or a service call fails, the app still saves the transcript-based report and records audio analysis as unavailable or partial.

Audio files are processed from temporary files and are not stored in SQLite. The existing transcript and both analysis results are saved in the report. pyannote.audio 4.x requires a recent PyTorch/Torchaudio stack; verify package installation for your Python/macOS architecture. Parselmouth supports macOS and runs locally without a GPU. CrisperWhisper's standard model weights are licensed for non-commercial research; check the model license for your use case.

## AI provider

Ollama on this Mac is selected by default. To keep analysis local, install and start Ollama and download a model as described below. You can explicitly choose a hosted provider instead; hosted analysis defaults to `openai/gpt-4o-mini`, and requires a provider API key or environment variable. `AI_MODEL` and `AI_API_KEY` can be used as defaults.

### Run a local LLM with Ollama on macOS

1. Install Ollama from [ollama.com/download/mac](https://ollama.com/download/mac) and open the Ollama app once. The app runs the local model service in the background.
2. In Terminal, download a model. For an Apple Silicon Mac with at least 16 GB of unified memory, start with:

	```bash
	ollama pull llama3.1:8b
	```

	On a lower-memory Mac, use a smaller model such as `qwen2.5:3b` instead, then enter `ollama_chat/qwen2.5:3b` in the app.
3. Confirm Ollama has the model:

	```bash
	ollama list
	```

4. Start Voice Practice with `.venv/bin/streamlit run app.py`, select **Ollama on this Mac**, and keep the URL at `http://localhost:11434`. The default LiteLLM model ID is `ollama_chat/llama3.1:8b`.

If you prefer to run the service from Terminal instead of the Ollama app, use `ollama serve` in a separate Terminal window. Keep it running while using Voice Practice. The audio transcription and acoustic measurements run locally; with Ollama selected, transcript and audio coaching also stay on the Mac and need no hosted LLM API key. Azure pronunciation scoring, if enabled, sends speaker audio clips to Azure Speech. Ollama may download a model locally the first time you pull it; model files can require several gigabytes of disk space.

For individual participant feedback, prefix transcript lines with speaker labels (for example, `Alex:`) and enter the participant label. Without speaker labels, the report treats the recording as a conversation and avoids attributing observations to one person.

Only upload recordings you are allowed to process. Keep in mind that the transcript is sent to the configured AI provider and the saved report contains the transcript.