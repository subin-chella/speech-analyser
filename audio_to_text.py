from crisperwhisper import CrisperWhisperModel

# Route the workload to the Mac's GPU
model = CrisperWhisperModel("turbo", device="mps")         # nyralabs/CrisperWhisper2.0_large
# or pick a size: CrisperWhisperModel("turbo")  # turbo / medium / small

# Verbatim transcription (default): every filler, repetition, stutter,
# false start, and vocal event
result = model.transcribe("voice.wav", language="en")
print('111111111111111111')
print(result.text)

# Intended: the clean, readable version
clean = model.transcribe("voice.wav", language="en", mode="intended")

# Word-level timestamps
result = model.transcribe("voice.wav", language="en", word_timestamps=True)
for w in result.words:
    print(f"{w.start:6.2f}-{w.end:6.2f}  {w.word}")

# Verbatimize: upgrade an existing clean transcript with the
# disfluencies that are actually in the audio
result = model.verbatimize("voice.wav", "I think we should ship it Friday.")