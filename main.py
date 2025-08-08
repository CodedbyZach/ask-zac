import sys, os
sys.stderr = open(os.devnull, 'w')
os.environ["JACK_NO_MSG"] = "1"
os.environ["SDL_JACK_NO_MSG"] = "1"
os.environ["ALSA_LOGLEVEL"] = "none"
os.environ["ALSA_DEBUG"] = "0"

import openai
import speech_recognition as sr
from dotenv import load_dotenv
import requests
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import sys
import subprocess

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
openai.api_key = OPENAI_API_KEY

recognizer = sr.Recognizer()
mic = sr.Microphone()

def web_search(query):
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        result = data.get("organic_results", [{}])[0]
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if title or snippet:
            return f"{title}: {snippet}"
        else:
            return "No relevant web results found."
    except Exception as e:
        return f"Web search failed: {e}"

def ask_openai(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer as helpfully as possible."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

def speak_openai(text, on_done, voice="alloy"):
    def tts_thread():
        try:
            if not text.strip():
                text_to_speak = "Sorry, I don't know."
            else:
                text_to_speak = text.strip()
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text_to_speak
            )
            with open("output.wav", "wb") as f:
                f.write(response.content)
            # Play audio, suppress all terminal output
            subprocess.run(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', 'output.wav'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            pass
        finally:
            on_done()
    threading.Thread(target=tts_thread, daemon=True).start()

class AskZacApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AskZac")
        self.root.geometry("700x400")

        self.text_area = ScrolledText(root, wrap=tk.WORD, height=16, width=85, state='normal')
        self.text_area.pack(pady=8)

        self.entry = tk.Entry(root, width=70)
        self.entry.pack(padx=8, pady=3)
        self.entry.bind('<Return>', self.on_enter)

        self.append("AskZac: Hi, I am AskZac. Say 'Hey, GPT' or type your question. (Say/type 'q' to quit)")
        speak_openai("Hi, I am AskZac. Say Hey, GPT to begin.", self.start_wakeword_loop)

    def append(self, msg):
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.see(tk.END)

    def on_enter(self, event=None):
        user_input = self.entry.get().strip()
        self.entry.delete(0, tk.END)
        if user_input.lower() == "q":
            self.root.destroy()
            sys.exit(0)
        elif user_input:
            self.ask_and_speak(user_input)

    def start_wakeword_loop(self):
        threading.Thread(target=self.wakeword_loop, daemon=True).start()

    def wakeword_loop(self):
        while True:
            self.append("Listening for: 'Hey, GPT' ...")
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                self.append("Recognizing speech (wake word)...")
                try:
                    wake_text = recognizer.recognize_google(audio)
                    self.append(f"You said: {wake_text}")
                except:
                    self.append("Sorry, I didn't catch that. Say 'Hey, GPT' to start.")
                    continue
                if wake_text.strip().lower() == "q":
                    self.append("Goodbye!")
                    self.root.destroy()
                    sys.exit(0)
                if "hey gpt" in wake_text.strip().lower():
                    self.append("Heard wake word! Listening for your question...")
                    self.listen_for_question()
                    break  # Exit loop until speech completes
                else:
                    self.append("Didn't hear 'Hey, GPT'. Waiting...")
            except Exception as e:
                self.append(f"Wakeword speech error: {e}")
                continue

    def listen_for_question(self):
        self.append("Listening for your question...")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            self.append("Recognizing your question...")
            try:
                query = recognizer.recognize_google(audio)
                self.append(f"You said: {query}")
            except:
                self.append("Sorry, I didn't catch that. Say 'Hey, GPT' again to start.")
                self.start_wakeword_loop()
                return
            if query.strip().lower() == "q":
                self.append("Goodbye!")
                self.root.destroy()
                sys.exit(0)
            self.ask_and_speak(query)
        except Exception as e:
            self.append(f"Speech error: {e}")
            self.start_wakeword_loop()

    def ask_and_speak(self, query):
        self.append(f"You: {query}")
        self.append("Thinking...")
        answer = ask_openai(query)
        if answer:
            self.append(f"AskZac: {answer}")
        else:
            self.append("ChatGPT failed. Using web search fallback...")
            answer = web_search(query)
            self.append(f"AskZac (web): {answer}")
        # After speaking, go back to wakeword loop
        speak_openai(answer, self.start_wakeword_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AskZacApp(root)
    root.mainloop()