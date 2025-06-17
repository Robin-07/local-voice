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
from pywhispercpp.model import Model
from pywhispercpp.utils import download_model

# Config
AUDIO_PRECISION = "int16"
RATE, FRAME_MS = 16_000, 20
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2  # 20 ms of int16 mono
SILENCE_THRESHOLD = 0.25  # 250 ms of silence
VAD_MODE = 2
ASR_INPUT_CHUNK_SIZE = RATE * 0.25 * 2  # 250 ms of speech
LLM_MODEL = "llama3:8b"
WHISPER_MODEL = "base.en"
PIPER_VOICE = "./voices/en_US-lessac-medium.onnx"
PIPER_VOICE_JSON = "./voices/en_US-lessac-medium.json"
TTS_RATE = 22_050
OLLAMA_HOST = "http://localhost:11434"
LOGLEVEL = os.getenv("LOCALVOICE_LOGLEVEL", "INFO").upper()

logging.basicConfig(
    stream=sys.stdout,
    level=LOGLEVEL,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


# Model Initialization (1x per worker)
def _init_child():
    global WH_MODEL, TTS_VOICE
    WH_MODEL = Model(
        model=WHISPER_MODEL,
        print_progress=False,
        print_realtime=False,
        single_segment=True,
        no_context=True,
        n_threads=max(1, os.cpu_count() // 2),
    )
    TTS_VOICE = PiperVoice.load(PIPER_VOICE, PIPER_VOICE_JSON)


def asr_worker(pcm_bytes: bytes) -> str:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments = WH_MODEL.transcribe(audio.reshape(-1, 1))
    return " ".join(seg.text for seg in segments)


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
            dtype=AUDIO_PRECISION,
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
                    # self._handle_interruption(buf)

                else:
                    self._silence += FRAME_MS / 1000

                if (
                    self._silence > SILENCE_THRESHOLD
                    and len(buf) > ASR_INPUT_CHUNK_SIZE
                ):
                    pcm = bytes(buf)
                    buf.clear()
                    self._silence = 0.0
                    logging.info("[AUDIO] utterance captured (%d bytes)", len(pcm))
                    self._mark_speech_end()
                    asyncio.create_task(self._asr_task(pcm))

    async def _asr_task(self, pcm: bytes):
        text = await self.loop.run_in_executor(self.pool, asr_worker, pcm)
        if text.strip():
            logging.info("[ASR] %s", text.strip())
            asyncio.create_task(self._llm_task(text.strip()))

    async def _llm_task(self, prompt: str):
        logging.info("[USER] %s", prompt)
        stream = await self.client.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True
        )

        buf = []
        async for part in stream:
            if self._stop_llm:
                self._stop_llm = False
                logging.info("stopping llm task...")
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
                logging.info("stopping tts task...")
                return
            await self._out_q.put(c)
        logging.debug(
            "[TTS] queued %d chunks (%d bytes)",
            len(chunks),
            sum(len(c) for c in chunks),
        )

    async def _speaker_task(self):
        with sd.RawOutputStream(
            samplerate=TTS_RATE, channels=1, dtype=AUDIO_PRECISION, latency="low"
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
                    logging.info("stopping speaker task...")
                    continue
                self._speaking = False
                logging.debug("[SPEAK] wrote %d bytes", len(data))

    def _handle_interruption(self, buffer):
        if len(buffer) > ASR_INPUT_CHUNK_SIZE:
            logging.info("detected user interruption...")
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
    logging.info("downloading Whisper model if missing …")
    download_model(
        WHISPER_MODEL,
    )
    logging.info("starting LocalVoice (loglevel=%s)", LOGLEVEL)
    LocalVoice().run()
