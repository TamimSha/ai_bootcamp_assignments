# Tamim Shaban

import speech_recognition as sr
import time
import playsound
import os
from gtts import gTTS
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://quotes.toscrape.com"
html = urlopen(url)
soup = BeautifulSoup(html, "html.parser")

type(soup)
all_link = soup.findAll('div', {'class': 'qoute'})
str_cells = str(all_link)
cleartext = BeautifulSoup(str_cells, "html.parser").get_text()

def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def get_audio():
    r = sr.Recognizer()
    wtih sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
    try:
        said = r.recognize_google(audio)
        print(said)
    except Exception as e:
        print("Exception: " + str(e))
    return said

text = get_audio()

if "option one" in text:
    speak("output one")
elif "option two" in text:
    speak("output two")
elif "give me a qoute" in text:
    speak(cleartext)
