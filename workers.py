import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from piper.voice import PiperVoice

WHISPER_MODEL = "base.en"
WHISPER_MODEL_PATH = "./models"
WHISPER_COMPUTE_TYPE = "int8"
PIPER_VOICE = "./voices/en_US-lessac-medium.onnx"
PIPER_VOICE_JSON = "./voices/en_US-lessac-medium.json"

WORKER_POOL_SIZE = 2


def init_child():
    """Load ASR and TTS models in each worker processs."""
    global WH_MODEL, TTS_VOICE
    whisper_model = WhisperModel(
        model_size_or_path=WHISPER_MODEL,
        device="cpu",
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=WHISPER_MODEL_PATH,
    )
    WH_MODEL = BatchedInferencePipeline(model=whisper_model)
    TTS_VOICE = PiperVoice.load(PIPER_VOICE, PIPER_VOICE_JSON)


def asr_worker(pcm_bytes: bytes) -> str:
    """Do STT using Whisper."""
    audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0
    segments, _ = WH_MODEL.transcribe(
        audio=audio,
        language="en",
        beam_size=5,
        batch_size=8,
        condition_on_previous_text=True,
    )
    return " ".join([seg.text for seg in segments])


def tts_worker(text: str) -> list[bytes]:
    """Do TTS using Piper."""
    return list(TTS_VOICE.synthesize_stream_raw(text))
