import sys, os, re, difflib, time, threading, subprocess, json
sys.stderr = open(os.devnull, 'w')
os.environ["JACK_NO_MSG"] = "1"
os.environ["SDL_JACK_NO_MSG"] = "1"
os.environ["ALSA_LOGLEVEL"] = "none"
os.environ["ALSA_DEBUG"] = "0"

import openai
import speech_recognition as sr
from dotenv import load_dotenv
import requests
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# =============== Config ===============
WAKE_CANONICAL = "gpt"
WAKE_CANONICAL_SPOKEN = "gee pee tee"
WAKE_TIMEOUT_S   = 6.0
WAKE_PHRASE_MAXS = 5.0
QUESTION_TIMEOUT = 10.0
QUESTION_MAXS    = 14.0
UNCERTAIN_TOKEN  = "<i-dont-know>"
USER_ZIP         = "14580"  # Webster, NY
TZ               = "America/New_York"
SEARCH_RESULTS_N = 3
# =====================================

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
openai.api_key = OPENAI_API_KEY

recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.25
recognizer.non_speaking_duration = 0.2
recognizer.phrase_threshold = 0.1
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 300
mic = sr.Microphone()

def quick_calibrate(seconds=0.6):
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=seconds)
            recognizer.energy_threshold = max(150, recognizer.energy_threshold)
    except Exception:
        pass

# -------- Wake word helpers ----------
COMMON_EQUIVS = {
    r"\bgee\b": "g", r"\bje\b": "g", r"\bjee\b": "g",
    r"\bpea\b": "p", r"\bpee\b": "p",
    r"\btea\b": "t", r"\btee\b": "t",
}
WAKE_PAT = re.compile(r"\b(g\.?\s*p\.?\s*t)\b", re.I)

def normalize_letters(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[._\-,:;!?]", " ", s)
    for pat, repl in COMMON_EQUIVS.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", "", s)
    return s

def contains_wake(raw: str) -> bool:
    if not raw: return False
    if WAKE_PAT.search(raw): return True
    n = normalize_letters(raw)
    if "gpt" in n: return True
    r1 = difflib.SequenceMatcher(None, raw.lower(), WAKE_CANONICAL).ratio()
    r2 = difflib.SequenceMatcher(None, raw.lower(), WAKE_CANONICAL_SPOKEN).ratio()
    return max(r1, r2) >= 0.75

def split_after_wake(raw: str) -> str:
    if not raw: return ""
    s = raw.strip()
    m = WAKE_PAT.search(s)
    if m:
        return s[m.end():].lstrip(" ,:-").strip()
    lower = s.lower()
    for form in ["gpt", "g.p.t", "g p t", "gee pee tee"]:
        idx = lower.find(form)
        if idx != -1:
            return s[idx+len(form):].lstrip(" ,:-").strip()
    if contains_wake(s):
        tokens = s.split()
        out = []
        removed = 0
        for tk in tokens:
            tk_clean = normalize_letters(tk)
            if removed < 3 and tk_clean in {"g", "p", "t", "gpt"}:
                removed += 1; continue
            if removed == 0 and tk.lower() in {"gee","pee","tea","tee"}:
                removed += 1; continue
            out.append(tk)
        return " ".join(out).lstrip(" ,:-").strip()
    return ""
# -------------------------------------

# ---------- Data fetchers ----------
def web_search_structured(query, n=SEARCH_RESULTS_N):
    """
    Return top N SerpAPI organic results as a list of {title, snippet, link, position}.
    """
    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": SERPAPI_KEY, "engine": "google"}
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        out = []
        for item in data.get("organic_results", [])[:n]:
            out.append({
                "position": item.get("position"),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return out
    except Exception as e:
        return [{"position": 0, "title": "Search failed", "snippet": str(e), "link": ""}]

def geocode_zip(zip_code):
    # Zippopotam.us is free/no key
    r = requests.get(f"http://api.zippopotam.us/us/{zip_code}", timeout=10)
    r.raise_for_status()
    j = r.json()
    place = j["places"][0]
    lat = float(place["latitude"])
    lon = float(place["longitude"])
    place_name = f'{place["place name"]}, {place["state abbreviation"]}'
    return lat, lon, place_name

def fetch_weather_zip(zip_code, tz=TZ):
    """
    Open-Meteo with imperial units directly (°F, mph, inches).
    """
    lat, lon, place = geocode_zip(zip_code)
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max","temperature_2m_min",
            "precipitation_probability_max","precipitation_sum",
            "windspeed_10m_max","sunrise","sunset"
        ]),
        "timezone": tz,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    d = r.json().get("daily", {})
    out = {
        "place": place,
        "dates": d.get("time", []),
        "tmax": d.get("temperature_2m_max", []),
        "tmin": d.get("temperature_2m_min", []),
        "popmax": d.get("precipitation_probability_max", []),
        "precip": d.get("precipitation_sum", []),
        "windmax": d.get("windspeed_10m_max", []),
        "sunrise": d.get("sunrise", []),
        "sunset": d.get("sunset", []),
        "lat": lat, "lon": lon
    }
    return out
# -----------------------------------

def ask_openai(prompt):
    """
    Returns model text or the exact token <i-dont-know> if uncertain.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. If you are not confident you know the answer, "
                        "or the answer depends on up-to-date web info you don't have, reply EXACTLY with "
                        f"{UNCERTAIN_TOKEN} and nothing else."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.lower() in {UNCERTAIN_TOKEN, "<i dont know>", "<i_dont_know>", "<idontknow>"}:
            return UNCERTAIN_TOKEN
        return text if text else UNCERTAIN_TOKEN
    except Exception:
        return UNCERTAIN_TOKEN

def ask_openai_style_weather(summary_dict):
    """
    Turn structured imperial weather into a natural sentence.
    """
    try:
        msg = json.dumps(summary_dict)
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content":
                 "Rewrite the given weather data into a single friendly sentence for a voice assistant. "
                 "Use °F, mph, and inches. Say the city correctly from the data, keep it concise, "
                 "include today's high/low, mention precip chances if notable, and a brief wind note. No emojis."},
                {"role": "user", "content": msg}
            ],
            max_tokens=120
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        try:
            p = summary_dict
            t = p['today']
            return (f"{p['place']} today: high {round(t['tmax'])}°F, low {round(t['tmin'])}°F, "
                    f"{t['popmax']}% precip chance ({t['precip']:.2f} in), winds up to {round(t['windmax'])} mph.")
        except Exception:
            return "Here's the local forecast."

def ask_openai_from_search(query, results):
    """
    Feed top search results to GPT to synthesize a concise, human answer.
    """
    try:
        payload = {
            "query": query,
            "results": results
        }
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system", "content":
                 "You are a concise assistant. Based ONLY on the provided search snippets, "
                 "answer the user's question in 1–3 sentences. If the snippets are insufficient, say '<i-dont-know>'."},
                {"role": "user", "content": json.dumps(payload)}
            ],
            max_tokens=180
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.lower() in {UNCERTAIN_TOKEN, "<i dont know>", "<i_dont_know>", "<idontknow>"}:
            return UNCERTAIN_TOKEN
        return text
    except Exception as e:
        return UNCERTAIN_TOKEN

def speak_openai(text, on_done, voice="alloy"):
    def tts_thread():
        try:
            spoken = text.strip() if text and text.strip() else "Sorry, I don't know."
            resp = openai.audio.speech.create(model="tts-1", voice=voice, input=spoken)
            with open("output.wav", "wb") as f:
                f.write(resp.content)
            subprocess.run(['ffplay','-nodisp','-autoexit','-loglevel','quiet','output.wav'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        finally:
            on_done()
    threading.Thread(target=tts_thread, daemon=True).start()

def looks_like_weather(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "weather","forecast","temperature","rain","snow","wind","sunrise","sunset"
    ])

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

        self.listening_enabled = True

        self.append("AskZac: Say 'GPT, <your message>' or type. ('q' to quit)")
        quick_calibrate(0.6)
        self.start_continuous_loop()

    def append(self, msg):
        self.text_area.insert(tk.END, msg + "\n"); self.text_area.see(tk.END)

    def on_enter(self, event=None):
        txt = self.entry.get().strip(); self.entry.delete(0, tk.END)
        if txt.lower() == "q":
            self.root.destroy(); sys.exit(0)
        if txt:
            self.ask_and_speak(txt)

    def start_continuous_loop(self):
        threading.Thread(target=self.continuous_loop, daemon=True).start()

    def continuous_loop(self):
        while True:
            if not self.listening_enabled:
                time.sleep(0.05); continue
            self.append("Listening… (say: 'GPT, hello')")
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recognizer.listen(
                        source,
                        timeout=WAKE_TIMEOUT_S,
                        phrase_time_limit=WAKE_PHRASE_MAXS
                    )
                try:
                    heard = recognizer.recognize_google(audio, language="en-US")
                    self.append(f"You said: {heard}")
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.append(f"ASR error: {e}")
                    continue

                if heard.strip().lower() == "q":
                    self.append("Goodbye!"); self.root.destroy(); sys.exit(0)

                if contains_wake(heard):
                    remainder = split_after_wake(heard)
                    if remainder:
                        self.ask_and_speak(remainder)
                    else:
                        self.append("Heard 'GPT'. What's up?")
                        try:
                            with mic as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                                audio2 = recognizer.listen(source, timeout=QUESTION_TIMEOUT, phrase_time_limit=QUESTION_MAXS)
                            try:
                                q = recognizer.recognize_google(audio2, language="en-US").strip()
                                if q.lower() == "q":
                                    self.append("Goodbye!"); self.root.destroy(); sys.exit(0)
                                if q:
                                    self.ask_and_speak(q)
                            except sr.UnknownValueError:
                                self.append("Didn't catch that—try again with 'GPT, …'")
                            except sr.RequestError as e:
                                self.append(f"ASR error: {e}")
                        except sr.WaitTimeoutError:
                            self.append("Timed out—say 'GPT, …' again.")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.append(f"Mic error: {e}")
                continue

    def ask_and_speak(self, query):
        self.append(f"You: {query}")
        self.append("Thinking...")
        answer = ask_openai(query)

        # Weather-smart path (imperial units)
        if answer == UNCERTAIN_TOKEN and looks_like_weather(query):
            try:
                w = fetch_weather_zip(USER_ZIP, tz=TZ)
                today = {
                    "date": w["dates"][0],
                    "tmax": float(w["tmax"][0]),
                    "tmin": float(w["tmin"][0]),
                    "popmax": int(w["popmax"][0]),
                    "precip": float(w["precip"][0]),   # inches
                    "windmax": float(w["windmax"][0]), # mph
                    "sunrise": w["sunrise"][0],
                    "sunset": w["sunset"][0],
                }
                summary = {"zip": USER_ZIP, "place": w["place"], "today": today}
                styled = ask_openai_style_weather(summary)
                self.append(f"AskZac: {styled}")
                self.listening_enabled = False
                speak_openai(styled, on_done=lambda: self._resume_after_tts())
                return
            except Exception as e:
                self.append(f"Weather fetch failed: {e}. Using web search…")
                # fall through to generic search synth

        # Generic search → GPT synth for anything else
        if answer == UNCERTAIN_TOKEN:
            self.append(f"AskZac: {UNCERTAIN_TOKEN}")
            results = web_search_structured(query, n=SEARCH_RESULTS_N)
            synth = ask_openai_from_search(query, results)
            if synth == UNCERTAIN_TOKEN:
                # last resort: show first snippet
                fallback = results[0]["title"] + ": " + results[0]["snippet"]
                self.append(f"AskZac (web): {fallback}")
                to_say = fallback
            else:
                self.append(f"AskZac (web): {synth}")
                to_say = synth
            self.listening_enabled = False
            speak_openai(to_say, on_done=lambda: self._resume_after_tts())
            return

        # Normal path
        self.append(f"AskZac: {answer}")
        self.listening_enabled = False
        speak_openai(answer, on_done=lambda: self._resume_after_tts())

    def _resume_after_tts(self):
        self.listening_enabled = True

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"+{1920}+{0}")
    app = AskZacApp(root)
    root.mainloop()