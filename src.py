from faster_whisper import WhisperModel

MODEL_SIZE = "tiny.en"

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "audio.mp3", 
    beam_size=5, 
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))