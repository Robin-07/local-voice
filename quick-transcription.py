import time

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


class QuickTranscription:
    '''
    Provides a public method 'listen' to start the audio stream along 
    with the live transcription.
    '''
    def __init__(self, model_size: str = "base"):
        print("INITIALIZING...")
        self._model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8",
            download_root="./models",
        )
        self._chunk = 16000
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 16000
        self._vad_filter = True
        self._input_device_index = 1
        print("READY")
    
    def listen(self) -> None:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._input_device_index,
            stream_callback=self._transcribe,
        )
        stream.start_stream()
        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    def _transcribe(self, in_data, frame_count, time_info, status) -> tuple:
        audio_data = np.frombuffer(in_data, np.int16).flatten().astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio_data, 
            beam_size=5, 
            vad_filter=self._vad_filter,
            vad_parameters=dict(min_silence_duration_ms=1000),
        )
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        return (in_data, pyaudio.paContinue)


if __name__ == '__main__':
    q = QuickTranscription()
    q.listen()