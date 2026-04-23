'''from faster_whisper import WhisperModel
import os

def transcribe_pi(file_path):
    # 'tiny.en' is fastest for Pi. 
    # Use 'base.en' if you need better accuracy and have a Pi 4 or 5.
    model_size = "tiny.en"

    # Run on CPU with INT8 quantization to save memory
    print(f"Loading {model_size} model...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing {file_path}...")
    segments, info = model.transcribe(file_path, beam_size=5)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "
    
    return full_text

if __name__ == "__main__":
    audio_file = "camera_started.mp3" # Ensure this file is in the same folder
    if os.path.exists(audio_file):
        result = transcribe_pi(audio_file)
        with open("output.txt", "w") as f:
            f.write(result)
        print("\nTranscription complete. Saved to output.txt")
    else:
        print("File not found!")
        '''

import os
import sys
import time
import io
import queue
import speech_recognition as sr
from faster_whisper import WhisperModel
from ctypes import *

# --- 1. Aggressively silence ALSA errors ---
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# --- 2. Initialize Whisper ---
# Using 'tiny.en' for Raspberry Pi speed
print("Loading Whisper Model (tiny.en)...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

audio_queue = queue.Queue()

def audio_callback(recognizer, audio):
    audio_queue.put(audio)

def start_listening():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Increase this if the Pi cuts you off too soon while you're thinking
    recognizer.pause_threshold = 1.5 
    
    stop_phrase = "over and out"
    full_transcript = []

    with mic as source:
        print("Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    # Background listener starts here
    stop_listening = recognizer.listen_in_background(mic, audio_callback, phrase_time_limit=10)
    
    print("\n>>> SYSTEM READY. Speak now. (Say 'over and out' to finish) <<<\n")

    try:
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Convert to audio bytes for Whisper
                wav_data = io.BytesIO(audio_data.get_wav_data())
                
                # Transcribe with beam_size=1 for maximum speed on Pi
                segments, _ = model.transcribe(wav_data, beam_size=1)

                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue
                    
                    # Check for stop phrase
                    if stop_phrase.lower() in text.lower():
                        # Remove the stop phrase from the final result
                        cleaned = text.lower().replace(stop_phrase.lower(), "").strip()
                        if cleaned:
                            full_transcript.append(cleaned)
                            print(f"Captured: {cleaned}")
                        
                        print("\n[Stop phrase detected. Saving and Exiting...]")
                        stop_listening(wait_for_stop=False)
                        return " ".join(full_transcript)
                    
                    print(f"Captured: {text}")
                    full_transcript.append(text)
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        return " ".join(full_transcript)

if __name__ == "__main__":
    final_text = start_listening()
    
    if final_text:
        with open("transcription.txt", "w") as f:
            f.write(final_text)
        print(f"\n--- FINAL RESULT ---\n{final_text}\n")
    else:
        print("Nothing recorded.")