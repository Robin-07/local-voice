import numpy as np
import pyaudio
import pyttsx3
from faster_whisper import WhisperModel
from gpt4all import GPT4All


class LocalVoice:
    '''
    Provides a public method 'listen' which takes audio input,
    transcribes it, produces a response and plays it back to the user.
    '''
    def __init__(self, model_size: str = "base.en"):
        print("INITIALIZING...")
        self._asr_model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8",
            download_root="./models/whisper",
        )
        self._llm = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", model_path='models/gpt4all/', allow_download=False)
        self._system_template = 'Respond like an AI assistant'
        self._speech_engine = pyttsx3.init()
        self._chunk = 16384
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 16384
        self._vad_filter = True
        pa = pyaudio.PyAudio()
        self._stream = pa.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=1,
        )
    
    def listen(self) -> None:
        print("Listening...")
        with self._llm.chat_session(self._system_template):
            while True:
                frames = [self._stream.read(self._chunk) for i in range(0, int(self._rate / self._chunk * 2))]
                transcript = self._transcribe(b"".join(frames))
                if transcript:
                    self._respond(transcript)
                else:
                    self._stream.stop_stream()
                    self._stream.close()
                    break
    
    def _transcribe(self, data) -> str:
        audio_data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        segments, _ = self._asr_model.transcribe(
            audio_data, 
            beam_size=5,
            vad_filter=self._vad_filter,
            vad_parameters=dict(min_silence_duration_ms=1000),
            condition_on_previous_text=True,
        )
        words_spoken = [segment.text for segment in segments]
        return " ".join(words_spoken)

    def _respond(self, prompt):
        print(prompt)
        response = self._llm.generate(prompt, max_tokens=30)
        print(response)
        self._speech_engine.say(response)
        self._speech_engine.runAndWait()


if __name__ == '__main__':
    LocalVoice().listen()