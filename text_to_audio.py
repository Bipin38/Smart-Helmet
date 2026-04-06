from gtts import gTTS
import os

def text_to_speech(text, filename="script_stopped.mp3", lang='en'):
    try:
        # Create the gTTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save the audio file
        tts.save(filename)
        print(f"Success! Audio saved as {filename}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
my_text = "Stopped. You collected a good amount of data. Thank you."
text_to_speech(my_text)