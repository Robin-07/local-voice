import asyncio
import itertools
import logging
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait

import sounddevice as sd
import webrtcvad
from ollama import AsyncClient

from workers import WORKER_POOL_SIZE, asr_worker, init_child, tts_worker

# Audio config
AUDIO_BUFFER_TYPE = "int16"
RATE, FRAME_MS = 16_000, 20
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2  # 20 ms of int16 audio
SILENCE_THRESHOLD = 1  # 1 s of silence
VAD_MODE = 3
ASR_INPUT_CHUNK_SIZE = RATE * 2 * 2  # 2 s of audio (speech + silence)
TTS_RATE = 22_050

# LLM config
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
SYSTEM_PROMPT = """
You are Local Voice, a helpful AI support agent. Answer the user's queries in a single sentence.
"""

# Logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

spinner = itertools.cycle("◐◓◑◒")
_spinning = False


def spin():
    """Start spinner loop."""
    while _spinning:
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write("\b")


def start_spinner():
    """Start spinner in a separate thread."""
    global _spinning
    if _spinning:
        return
    _spinning = True
    threading.Thread(target=spin, daemon=True).start()


def stop_spinner():
    """Stop spinner."""
    global _spinning
    _spinning = False
    sys.stdout.write(" ")
    sys.stdout.write("\b")


class LocalVoice:
    """
    This class orchestrates a 3-stage streaming pipeline of STT → LLM → TTS
    while keeping the main thread responsive with asyncio and executing
    compute-heavy stages in separate worker processes.

    ----------
    Attributes
    ----------
    client : AsyncClient
        Thin wrapper around Ollama's HTTP API for streamed LLM calls.
    _vad : webrtcvad.Vad
        Aggressive-mode VAD used for endpointing and barge-in detection.
    _silence : float
        Rolling total of consecutive silence seconds detected by VAD.
    _mic_buffer : list[bytes]
        Raw 16-kHz PCM frames accumulated from the microphone callback.
    _out_queue : asyncio.Queue[bytes]
        Async FIFO of TTS chunks awaiting playback.
    loop : asyncio.AbstractEventLoop
        Event loop that owns every coroutine except the PortAudio thread.
    pool : concurrent.futures.ProcessPoolExecutor
        Pool that hosts the heavyweight STT / LLM / TTS workers.
    _system_message : list[dict[str, str]]
        Persistent system prompt injected at the head of every LLM turn.
    _stop_llm : bool
        Flag set by barge-in logic to cancel the current LLM generation.
    _stop_tts : bool
        Flag set to truncate / fade out the current TTS stream.
    _speaking : bool
        True while audio is actively being written to the output stream.
    """

    def __init__(self):
        logger.info("initializing LocalVoice...")
        self.client = AsyncClient(host=OLLAMA_HOST)
        self._vad = webrtcvad.Vad(VAD_MODE)
        self._silence: float = 0.0
        self._mic_buffer: list[bytes] = []
        self._out_queue = asyncio.Queue(maxsize=100)
        # self._play_queue = queue.Queue(maxsize=100)
        self.loop = asyncio.get_event_loop()
        self.pool = ProcessPoolExecutor(
            max_workers=WORKER_POOL_SIZE, initializer=init_child
        )
        self._system_message = [{"role": "system", "content": SYSTEM_PROMPT}]
        # boolean flags
        self._stop_llm = False
        self._stop_tts = False
        self._speaking = False
        # prime worker pool
        self._prime_pool()
        logger.info("initialized LocalVoice...")

    async def _mic_task(self):
        logger.info("listening...")

        def cb(indata, frames, t, status):
            if not self._speaking:
                self._mic_buffer.append(bytes(indata))

        with sd.RawInputStream(
            samplerate=RATE,
            channels=1,
            blocksize=FRAME_BYTES // 2,
            dtype=AUDIO_BUFFER_TYPE,
            callback=cb,
        ):
            buf = bytearray()
            while True:
                if not self._mic_buffer:
                    await asyncio.sleep(0.001)
                    continue
                frame = self._mic_buffer.pop(0)
                buf.extend(frame)

                if self._vad.is_speech(frame, RATE):
                    start_spinner()
                    self._silence = 0.0
                    if self._speaking and len(buf) > ASR_INPUT_CHUNK_SIZE:
                        logger.info("detected user interruption...")
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
                    logger.debug("[AUDIO] utterance captured (%d bytes)", len(pcm))
                    self._mark_speech_end()
                    stop_spinner()
                    asyncio.create_task(self._asr_task(pcm))

    async def _asr_task(self, pcm: bytes):
        text = await self.loop.run_in_executor(self.pool, asr_worker, pcm)
        if text:
            asyncio.create_task(self._llm_task(text))

    async def _llm_task(self, prompt: str):
        logger.info("[USER] %s", prompt)
        stream = await self.client.chat(
            model=LLM_MODEL,
            messages=(self._system_message + [{"role": "user", "content": prompt}]),
            stream=True,
        )
        buf = []
        async for part in stream:
            if self._stop_llm:
                self._stop_llm = False
                logger.debug("stopping llm task...")
                return
            token = part["message"]["content"]
            buf.append(token)
            if any(token.endswith(p) for p in ".?!,") and len(buf) >= 7:
                chunk = "".join(buf)
                buf.clear()
                await self._tts_task(chunk)
        if buf:
            chunk = "".join(buf)
            if chunk:
                await self._tts_task(chunk)

    async def _tts_task(self, text: str):
        logger.info("[LocalVoice] %s", text)
        chunks = await self.loop.run_in_executor(self.pool, tts_worker, text)
        for c in chunks:
            if self._stop_tts:
                self._stop_tts = False
                logger.debug("stopping tts task...")
                return
            await self._out_queue.put(c)
        logger.debug(
            "[TTS] queued %d chunks (%d bytes)",
            len(chunks),
            sum(len(c) for c in chunks),
        )

    # async def _speaker_task_async(self):
    #     event = asyncio.Event()

    #     def cb(outdata, frames, t, status):
    #         try:
    #             data = self._play_queue.get_nowait()
    #             pcm = np.frombuffer(data, dtype=AUDIO_BUFFER_TYPE)
    #             outdata[:] = pcm.reshape(-1, 1)
    #         except queue.Empty:
    #             outdata.fill(0)

    #     with sd.OutputStream(
    #         samplerate=TTS_RATE,
    #         channels=1,
    #         dtype=AUDIO_BUFFER_TYPE,
    #         latency="low",
    #         callback=cb,
    #     ):
    #         await event.wait()

    async def _speaker_task(self):
        with sd.RawOutputStream(
            samplerate=TTS_RATE, channels=1, dtype=AUDIO_BUFFER_TYPE, latency="low"
        ) as out:
            first_chunk = True
            while True:
                data = await self._out_queue.get()
                if first_chunk:
                    self._mark_playback_start()
                    first_chunk = False
                self._speaking = True
                out.write(data)
                if not self._speaking:
                    try:
                        while True:
                            self._out_queue.get_nowait()
                            self._out_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    logger.debug("stopping speaker task...")
                    continue
                self._speaking = False
                logger.debug("[SPEAK] wrote %d bytes", len(data))

    def _handle_interruption(self):
        self._speaking = False
        self._stop_llm = True
        self._stop_tts = True

    def _mark_speech_end(self):
        self._speech_end_ts = time.monotonic()

    def _mark_playback_start(self):
        now = time.monotonic()
        ms = (now - self._speech_end_ts) * 1000
        logger.debug(f"Round-trip latency: {ms:.0f} ms")

    def _prime_pool(self):
        logger.info("priming worker pool...")
        noop = lambda *args, **kwargs: None
        futures = [self.pool.submit(noop) for _ in range(WORKER_POOL_SIZE)]
        wait(futures)

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
    LocalVoice().run()
