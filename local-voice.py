import logging
import math
import os
import sys
import time
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Thread, Lock

import numpy as np
import ollama
import pyaudio
import pyttsx3
from faster_whisper import WhisperModel

os.environ["OLLAMA_HOST"] = "localhost:11434"


class LocalVoice:
    '''
    Provides a function 'start_session' to start a Real-time Voice Chat.
    Concurrently records audio input, transcribes it, generates a response and plays it back.
    '''
    def __init__(self, whisper_model: str = "base.en", llm: str = "phi3", log_level: str = "DEBUG"):
        self._asr_model = WhisperModel(
            whisper_model, 
            device="cuda", 
            compute_type="int8_float16",
            download_root="./models/whisper",
        )
        self._llm = llm
        self._speech_engine = pyttsx3.init()
        self._pa = pyaudio.PyAudio()
        self._input_audio_buffer = bytearray()
        self._processed_audio_buffer = bytearray()
        self._max_silence_seconds = 1
        self._min_listen_amplitude = 50
        self._input_audio_rate = 44100
        self._input_audio_chunk = 1024
        self._recent_chunk_size = (self._max_silence_seconds * self._input_audio_rate)
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self._input_audio_rate,
            input=True,
            frames_per_buffer=self._input_audio_chunk,
            input_device_index=1,
            stream_callback=self._listen_callback,
        )
        # Set up worker threads
        self._input_audio_processor = Thread(target=self._process_input_audio)
        self._transcriber = Thread(target=self._transcribe)
        self._generator = Thread(target=self._generate)
        self._speaker = Thread(target=self._speak)
        # Set up queues for inter-thread communication
        self._input_audio_queue: Queue = Queue()
        self._transcript_queue: Queue = Queue()
        self._response_queue: Queue = Queue()
        # Set up events for thread signaling
        self._stop = Event()
        self._assistant_speaking = Event()
        # Set up logger
        self._logger = self._get_logger(log_level)

    def _get_logger(self, log_level):
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(lineno)s - %(funcName)s() - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
        return logger
    
    def start_session(self):
        self._logger.info("Starting session...")
        try:
            self._stream.start_stream()
            worker_threads = [
                self._input_audio_processor,
                self._transcriber,
                self._generator,
                self._speaker,
            ]
            for thread in worker_threads:
                thread.start()
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._logger.info("Stopping session...")
            self._stop.set()
            for thread in worker_threads:
                thread.join()
            self._logger.debug("Cleaning up audio stream...")
            self._stream.stop_stream()
            self._stream.close()        
            self._pa.terminate()
            self._logger.debug("Exited successfully!")
            sys.exit()

    def _listen_callback(self, in_data, frame_count, time_info, status):
        if not self._assistant_speaking.is_set():
            self._input_audio_buffer.extend(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_input_audio(self):
        while self._running:
            if len(self._input_audio_buffer) >= self._recent_chunk_size:
                recent_chunk = self._input_audio_buffer[:self._recent_chunk_size]
                self._input_audio_buffer = self._input_audio_buffer[self._recent_chunk_size:]
                # Amplitude check to detect end of user speech
                chunk_linear_rms = np.sqrt(np.mean(recent_chunk**2))
                chunk_amplitude = 20 * math.log10(chunk_linear_rms)
                if chunk_amplitude < self._min_listen_amplitude:
                    self._logger.debug("Detected silence in audio")
                    if len(self._processed_audio_buffer):
                        self._input_audio_queue.put(self._processed_audio_buffer)
                        self._processed_audio_buffer = bytearray()
                else:
                    self._logger.debug("Added processed audio chunk to buffer")
                    self._processed_audio_buffer.extend(recent_chunk)
        self._logger.debug("Stopped input audio processor")
            
    def _transcribe(self):
        while self._running:
            try:
                audio_data = self._input_audio_queue.get(block=False)
            except QueueEmpty:
                self._logger.debug("input audio queue empty")
                time.sleep(0.1)
                continue
            parsed_audio = np.frombuffer(audio_data, np.float32).flatten() / 32768.0
            self._logger.debug("Transcribing audio...")
            segments, _ = self._asr_model.transcribe(
                audio=parsed_audio, 
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=100),
                condition_on_previous_text=True,
            )
            self._input_audio_queue.task_done()
            words = " ".join([segment.text for segment in segments])
            self._logger.debug("Empty transcript generated")
            if words:
                self._transcript_queue.put(words)
                self._logger.debug(f"Added {len(words)} words to transcript queue")
        self._logger.debug("Stopped transcriber")

    def _generate(self):
        while self._running:
            try:
                words = self._transcript_queue.get(block=False)
            except QueueEmpty:
                self._logger.debug("transcript queue empty")
                time.sleep(0.1)
                continue
            self._logger.info(f"User: {words}")
            response = ollama.generate(
                model=self._llm, 
                prompt=words, 
                system="You are a helpful AI Assistant."
            )
            response = response['response']
            self._logger.info(f"LocalVoice: {response}")
            self._transcript_queue.task_done()
            self._response_queue.put(response)
            self._logger.debug(f"Added {len(response)} words to response queue")
        self._logger.debug("Stopped generator")

    def _speak(self):
        while self._running:
            try:
                response = self._response_queue.get(block=False)
            except QueueEmpty:
                self._logger.debug("response queue empty")
                time.sleep(0.1)
                continue
            self._speech_engine.say(response)
            self._assistant_speaking.set()
            self._speech_engine.runAndWait()
            self._assistant_speaking.clear()
            self._response_queue.task_done()
        self._logger.debug("Stopped speaker")

    @property
    def _running(self) -> bool:
        return not self._stop.is_set()


if __name__ == '__main__':
    LocalVoice().start_session()