# Voice Practice

A local Streamlit app for transcribing meeting audio, reviewing communication patterns with an AI model, and comparing saved reports over time.

## Run

```bash
brew install ffmpeg
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

FFmpeg is used to convert uploaded audio to mono WAV before transcription. The first transcription may download the CrisperWhisper model. Transcription defaults to CPU for stability; the sidebar offers the Apple GPU backend as an experimental option, because some MPS audio workloads can trigger a native macOS crash. Audio is transcribed locally; only the transcript and calculated speech metrics are sent to the selected AI provider. Reports are stored in `reports.sqlite3` in this folder.

## AI provider

Choose either a hosted provider or Ollama on this Mac in the sidebar. Hosted analysis defaults to `openai/gpt-4o-mini`; enter a provider API key or set its environment variable. The hosted model uses LiteLLM identifiers, so supported providers can be selected without changing application code. `AI_MODEL` and `AI_API_KEY` can be used as defaults.

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

If you prefer to run the service from Terminal instead of the Ollama app, use `ollama serve` in a separate Terminal window. Keep it running while using Voice Practice. The audio transcription already runs locally; with Ollama selected, the transcript analysis also stays on the Mac and needs no hosted API key. Ollama may download a model locally the first time you pull it; model files can require several gigabytes of disk space.

For individual participant feedback, prefix transcript lines with speaker labels (for example, `Alex:`) and enter the participant label. Without speaker labels, the report treats the recording as a conversation and avoids attributing observations to one person.

Only upload recordings you are allowed to process. Keep in mind that the transcript is sent to the configured AI provider and the saved report contains the transcript.