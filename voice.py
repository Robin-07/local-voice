import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sounddevice as sd
import webrtcvad
from ollama import AsyncClient
from piper.voice import PiperVoice
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Config
AUDIO_BUFFER_TYPE = "int16"
RATE, FRAME_MS = 16_000, 20
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2  # 20 ms of int16 audio
SILENCE_THRESHOLD = 1  # 1s of silence
VAD_MODE = 3
ASR_INPUT_CHUNK_SIZE = RATE * 1 * 2  # 1 s of speech

# Model pipeline config
WHISPER_MODEL = "base.en"
WHISPER_MODEL_PATH = "./models"
WHISPER_COMPUTE_TYPE = "int8"
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "phi3:mini"
SYSTEM_PROMPT = """
You are Local Voice, a helpful AI voice assistant. Answer the user's queries in one sentence.
"""
PIPER_VOICE = "./voices/en_US-lessac-medium.onnx"
PIPER_VOICE_JSON = "./voices/en_US-lessac-medium.json"
TTS_RATE = 22_050

logging.basicConfig(
    stream=sys.stdout,
    level="INFO",
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


# Model Initialization (1x per worker)
def _init_child():
    global WH_MODEL, TTS_VOICE
    logging.info("initializing models...")
    whisper_model = WhisperModel(
        model_size_or_path=WHISPER_MODEL,
        device="cpu",
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=WHISPER_MODEL_PATH,
    )
    WH_MODEL = BatchedInferencePipeline(model=whisper_model)
    TTS_VOICE = PiperVoice.load(PIPER_VOICE, PIPER_VOICE_JSON)


def asr_worker(pcm_bytes: bytes) -> str:
    audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0
    segments, _ = WH_MODEL.transcribe(
        audio=audio,
        language="en",
        beam_size=5,
        batch_size=8,
        condition_on_previous_text=True,
        log_progress=True,
    )
    return " ".join([seg.text for seg in segments])


def tts_worker(text: str) -> list[bytes]:
    return list(TTS_VOICE.synthesize_stream_raw(text))


class LocalVoice:
    def __init__(self):
        self.client = AsyncClient(host=OLLAMA_HOST)
        self.vad = webrtcvad.Vad(VAD_MODE)
        self._ring: list[bytes] = []
        self._silence: float = 0.0
        self._out_q = asyncio.Queue(maxsize=100)
        self.loop = asyncio.get_event_loop()
        self.pool = ProcessPoolExecutor(max_workers=2, initializer=_init_child)
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # boolean flags
        self._stop_llm = False
        self._stop_tts = False
        self._speaking = False

    async def _mic_task(self):
        def cb(indata, frames, t, status):
            self._ring.append(bytes(indata))

        with sd.RawInputStream(
            samplerate=RATE,
            channels=1,
            blocksize=FRAME_BYTES // 2,
            dtype=AUDIO_BUFFER_TYPE,
            callback=cb,
        ):
            buf = bytearray()
            while True:
                if not self._ring:
                    await asyncio.sleep(0.001)
                    continue
                frame = self._ring.pop(0)
                buf.extend(frame)

                if self.vad.is_speech(frame, RATE):
                    self._silence = 0.0
                    if self._speaking and len(buf) > ASR_INPUT_CHUNK_SIZE:
                        logging.info("detected user interruption...")
                        self._handle_interruption()
                else:
                    self._silence += FRAME_MS / 1000

                if (
                    self._silence > SILENCE_THRESHOLD
                    and len(buf) > ASR_INPUT_CHUNK_SIZE
                ):
                    pcm = bytes(buf)
                    buf.clear()
                    self._silence = 0.0
                    logging.debug("[AUDIO] utterance captured (%d bytes)", len(pcm))
                    self._mark_speech_end()
                    asyncio.create_task(self._asr_task(pcm))

    async def _asr_task(self, pcm: bytes):
        text = await self.loop.run_in_executor(self.pool, asr_worker, pcm)
        if text:
            logging.debug("[ASR] %s", text)
            asyncio.create_task(self._llm_task(text))

    async def _llm_task(self, prompt: str):
        logging.info("[USER] %s", prompt)
        self._messages.append({"role": "user", "content": prompt})
        stream = await self.client.chat(
            model=LLM_MODEL, messages=self._messages, stream=True
        )
        buf = []
        async for part in stream:
            if self._stop_llm:
                self._stop_llm = False
                logging.debug("stopping llm task...")
                return
            token = part["message"]["content"]
            buf.append(token)
            if any(token.endswith(p) for p in ".?!") or len(buf) >= 7:
                chunk = "".join(buf)
                buf.clear()
                logging.info("[BOT] %s", chunk.strip())
                await self._tts_task(chunk)
        if buf:
            chunk = "".join(buf)
            logging.info("[BOT] %s", chunk.strip())
            await self._tts_task(chunk)

    async def _tts_task(self, text: str):
        chunks = await self.loop.run_in_executor(self.pool, tts_worker, text)
        for c in chunks:
            if self._stop_tts:
                self._stop_tts = False
                logging.debug("stopping tts task...")
                return
            await self._out_q.put(c)
        logging.debug(
            "[TTS] queued %d chunks (%d bytes)",
            len(chunks),
            sum(len(c) for c in chunks),
        )

    async def _speaker_task(self):
        with sd.RawOutputStream(
            samplerate=TTS_RATE, channels=1, dtype=AUDIO_BUFFER_TYPE, latency="low"
        ) as out:
            first_chunk = True
            while True:
                data = await self._out_q.get()
                if first_chunk:
                    self._mark_playback_start()
                    first_chunk = False
                self._speaking = True
                out.write(data)
                if not self._speaking:
                    try:
                        while True:
                            self._out_q.get_nowait()
                            self._out_q.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    logging.debug("stopping speaker task...")
                    continue
                self._speaking = False
                logging.debug("[SPEAK] wrote %d bytes", len(data))

    def _handle_interruption(self):
        self._speaking = False
        self._stop_llm = True
        self._stop_tts = True

    def _mark_speech_end(self):
        self._speech_end_ts = time.monotonic()

    def _mark_playback_start(self):
        now = time.monotonic()
        ms = (now - self._speech_end_ts) * 1000
        logging.info(f"Round-trip latency: {ms:.0f} ms")

    def run(self):
        try:
            self.loop.run_until_complete(
                asyncio.gather(self._mic_task(), self._speaker_task())
            )
        except KeyboardInterrupt:
            pass
        finally:
            self.pool.shutdown(wait=False)


if __name__ == "__main__":
    logging.info("starting LocalVoice...")
    LocalVoice().run()
