# main.py — AskZac (Tkinter, auto-mic, always-listen, TTS, router/search/weather/dictionary)
import sys, os, re, difflib, time, threading, subprocess, json, queue, shutil, platform, tempfile

# -------------------- Debug / env --------------------
DEBUG = os.getenv("DEBUG", "1") == "1"
if not DEBUG:
    sys.stderr = open(os.devnull, 'w')

os.environ["JACK_NO_MSG"] = "1"
os.environ["SDL_JACK_NO_MSG"] = "1"
os.environ["ALSA_LOGLEVEL"] = "none"
os.environ["ALSA_DEBUG"] = "0"
os.environ.setdefault("SDL_AUDIODRIVER", "pulse")  # prefer Pulse/PipeWire on Linux

from dotenv import load_dotenv
load_dotenv()

# -------------------- OpenAI wrapper --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    _oa = OpenAI(api_key=OPENAI_API_KEY or None)
    def chat_create(**kw): return _oa.chat.completions.create(**kw)
    def tts_create(**kw):  return _oa.audio.speech.create(**kw)
    def whisper_api_transcribe(path):
        with open(path, "rb") as f:
            tr = _oa.audio.transcriptions.create(model="whisper-1", file=f)
        return tr.text
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY or None
    def chat_create(**kw): return openai.chat.completions.create(**kw)
    def tts_create(**kw):  return openai.audio.speech.create(**kw)
    def whisper_api_transcribe(path):
        with open(path, "rb") as f:
            tr = openai.audio.transcriptions.create(model="whisper-1", file=f)
        return tr["text"]

import requests
import speech_recognition as sr
import tkinter as tk
from tkinter import scrolledtext

# -------------------- Config --------------------
WAKE_CANONICAL = "gpt"
WAKE_CANONICAL_SPOKEN = "gee pee tee"
WAKE_TIMEOUT_S   = 6.0
WAKE_PHRASE_MAXS = 7.0
QUESTION_TIMEOUT = 10.0
QUESTION_MAXS    = 14.0
UNCERTAIN_TOKEN  = "<i-dont-know>"
USER_ZIP         = "14580"
TZ               = "America/New_York"
SEARCH_RESULTS_N = 3
DICT_LANG        = "en"
ROUTER_MODEL     = "gpt-4o"
ANSWER_MODEL     = "gpt-4o"

RESPOND_WITHOUT_WAKE = os.getenv("RESPOND_WITHOUT_WAKE", "1") == "1"
ASR_ENGINE = os.getenv("ASR_ENGINE", "google")  # google | whisper_api | local

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MIC_INDEX   = os.getenv("MIC_INDEX")
MIC_NAME    = os.getenv("MIC_NAME", "")

# -------------------- Speech setup --------------------
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 150
recognizer.pause_threshold = 0.20
recognizer.non_speaking_duration = 0.15
recognizer.phrase_threshold = 0.10

def list_mics():
    try: return sr.Microphone.list_microphone_names()
    except Exception: return []

def pick_mic():
    """Auto-pick the capture device.

    Priority:
      1) MIC_INDEX / MIC_NAME if provided
      2) On Linux, 'pulse' (system default from PipeWire/PulseAudio)
      3) First non-HDMI device that looks like a mic (USB/Microphone/etc.)
      4) fall back to None (PyAudio default)
    """
    names = list_mics()

    # 1) Respect explicit env
    if MIC_INDEX:
        try:
            idx = int(MIC_INDEX)
            return idx, names
        except ValueError:
            pass
    if MIC_NAME:
        for i, n in enumerate(names):
            if MIC_NAME.lower() in (n or "").lower():
                return i, names

    # 2) Linux 'pulse' default
    if sys.platform.startswith("linux"):
        for i, n in enumerate(names):
            if (n or "").strip().lower() == "pulse":
                return i, names

    # 3) Heuristic for real microphones
    PREFER = ("mic", "microphone", "usb", "emeet", "webcam", "yeti", "hyperx", "logitech", "c920", "audio", "line in")
    AVOID  = ("hdmi", "spdif", "iec958", "a52", "dmix", "surround", "phoneline", "modem", "rear", "center_lfe", "side",
              "default", "sysdefault", "front", "iec", "hdmi 1", "hdmi 2", "hdmi 3")
    best = None
    for i, n in enumerate(names):
        nl = (n or "").lower()
        if any(a in nl for a in AVOID): continue
        if any(p in nl for p in PREFER):
            best = i
            break
    if best is not None:
        return best, names

    # 4) Fall back to PyAudio default
    return None, names

MIC_IDX, MIC_NAMES = pick_mic()
mic = sr.Microphone(device_index=MIC_IDX)

def quick_calibrate(seconds=1.2):
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=seconds)
            recognizer.energy_threshold = max(100, min(500, recognizer.energy_threshold))
    except Exception as e:
        print("Calibrate error:", e)

def _which(cmd): return shutil.which(cmd) is not None

def _play_wav(path: str):
    try:
        if sys.platform.startswith("linux"):
            if _which("pw-play"):
                subprocess.run(["pw-play", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
            if _which("paplay"):
                subprocess.run(["paplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
        if sys.platform.startswith("win"):
            cmd = ["powershell","-NoProfile","-Command",
                   f"$p=New-Object System.Media.SoundPlayer('{os.path.abspath(path)}');$p.PlaySync();"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
        if _which("ffplay"):
            subprocess.run(["ffplay","-nodisp","-autoexit","-loglevel","quiet", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
        if _which("afplay"):
            subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
        if _which("aplay"):
            subprocess.run(["aplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return
    except Exception:
        pass

# -------------------- Wake helpers --------------------
COMMON_EQUIVS = {r"\bgee\b":"g", r"\bje\b":"g", r"\bjee\b":"g",
                 r"\bpea\b":"p", r"\bpee\b":"p",
                 r"\btea\b":"t", r"\btee\b":"t"}
WAKE_PAT = re.compile(r"\b(g\.?\s*p\.?\s*t)\b", re.I)

def normalize_letters(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[._\-,:;!?]", " ", s)
    for pat, repl in COMMON_EQUIVS.items():
        s = re.sub(pat, repl)
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
    if m: return s[m.end():].lstrip(" ,:-").strip()
    lower = s.lower()
    for form in ["gpt", "g.p.t", "g p t", "gee pee tee"]:
        i = lower.find(form)
        if i != -1: return s[i+len(form):].lstrip(" ,:-").strip()
    if contains_wake(s):
        tokens = s.split(); out = []; removed = 0
        for tk in tokens:
            tkc = normalize_letters(tk)
            if removed < 3 and tkc in {"g","p","t","gpt"}: removed += 1; continue
            if removed == 0 and tk.lower() in {"gee","pee","tea","tee"}: removed += 1; continue
            out.append(tk)
        return " ".join(out).lstrip(" ,:-").strip()
    return ""

# -------------------- Data fetchers --------------------
def web_search_structured(query, n=SEARCH_RESULTS_N):
    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": SERPAPI_KEY, "engine": "google"}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
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
        return [{"position":0,"title":"Search failed","snippet":str(e),"link":""}]

def geocode_zip(zip_code):
    r = requests.get(f"http://api.zippopotam.us/us/{zip_code}", timeout=10)
    r.raise_for_status()
    j = r.json(); place = j["places"][0]
    lat = float(place["latitude"]); lon = float(place["longitude"])
    place_name = f'{place["place name"]}, {place["state abbreviation"]}'
    return lat, lon, place_name

def fetch_weather_zip(zip_code, tz=TZ):
    lat, lon, place = geocode_zip(zip_code)
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max","temperature_2m_min",
            "precipitation_probability_max","precipitation_sum",
            "windspeed_10m_max","sunrise","sunset"
        ]),
        "timezone": tz, "temperature_unit":"fahrenheit",
        "windspeed_unit":"mph", "precipitation_unit":"inch",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    d = r.json().get("daily", {})
    return {
        "place": place, "dates": d.get("time", []),
        "tmax": d.get("temperature_2m_max", []),
        "tmin": d.get("temperature_2m_min", []),
        "popmax": d.get("precipitation_probability_max", []),
        "precip": d.get("precipitation_sum", []),
        "windmax": d.get("windspeed_10m_max", []),
        "sunrise": d.get("sunrise", []),
        "sunset": d.get("sunset", []),
        "lat": lat, "lon": lon
    }

# -------------------- Dictionary helpers --------------------
DICT_DEFINE_PATTERNS = [
    r"^\s*(define|definition of)\s+(?P<term>.+?)\s*\??$",
    r"^\s*what\s+does\s+(?P<term>.+?)\s+mean\s*\??$",
    r"^\s*(meaning of)\s+(?P<term>.+?)\s*\??$",
]
DICT_SYNONYM_PATTERNS = [
    r"^\s*(synonyms?|syns?)\s+(for|of)\s+(?P<term>.+?)\s*\??$",
    r"^\s*what\s+are\s+synonyms?\s+(for|of)\s+(?P<term>.+?)\s*\??$",
]
DICT_ANTONYM_PATTERNS = [
    r"^\s*(antonyms?|ants?)\s+(for|of)\s+(?P<term>.+?)\s*\??$",
]
DICT_PRONOUNCE_PATTERNS = [
    r"^\s*(pronounce|pronunciation of)\s+(?P<term>.+?)\s*\??$",
    r"^\s*how\s+do\s+you\s+pronounce\s+(?P<term>.+?)\s*\??$",
]
DICT_SPELL_PATTERNS = [
    r"^\s*(how\s+do\s+you\s+spell|spell)\s+(?P<term>.+?)\s*\??$",
]

def looks_like_dictionary(q: str) -> bool:
    ql = q.strip().lower()
    pats = DICT_DEFINE_PATTERNS + DICT_SYNONYM_PATTERNS + DICT_ANTONYM_PATTERNS + DICT_PRONOUNCE_PATTERNS + DICT_SPELL_PATTERNS
    return any(re.match(p, ql) for p in pats)

def parse_dictionary_query(q: str):
    ql = q.strip().lower()
    for p in DICT_DEFINE_PATTERNS:
        m = re.match(p, ql)
        if m: return ("define", m.group("term"))
    for p in DICT_SYNONYM_PATTERNS:
        m = re.match(p, ql)
        if m: return ("synonyms", m.group("term"))
    for p in DICT_ANTONYM_PATTERNS:
        m = re.match(p, ql)
        if m: return ("antonyms", m.group("term"))
    for p in DICT_PRONOUNCE_PATTERNS:
        m = re.match(p, ql)
        if m: return ("pronounce", m.group("term"))
    for p in DICT_SPELL_PATTERNS:
        m = re.match(p, ql)
        if m: return ("spell", m.group("term"))
    toks = re.findall(r"[A-Za-z\-']+", q)
    return ("define", toks[-1] if toks else q.strip())

def _clean_term(term: str) -> str:
    return term.strip().strip('"\''"“”‘’.,:;!?()[]{}")

def fetch_dictionary(term: str):
    try:
        t = _clean_term(term)
        url = f"https://api.dictionaryapi.dev/api/v2/entries/{DICT_LANG}/{requests.utils.quote(t)}"
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return None
        data = r.json()
        if not isinstance(data, list) or not data: return None
        entry = data[0]; word = entry.get("word", t)
        phon = ""
        for ph in entry.get("phonetics", []):
            if ph.get("text"): phon = ph["text"]; break
        meanings = entry.get("meanings", [])
        defs, syns, ants = [], set(), set()
        for m in meanings:
            pos = m.get("partOfSpeech", "")
            for d in m.get("definitions", [])[:1]:
                definition = d.get("definition", "")
                if definition:
                    defs.append((pos, definition, d.get("example")))
                for s in d.get("synonyms", []):
                    if isinstance(s, str): syns.add(s)
                for a in d.get("antonyms", []):
                    if isinstance(a, str): ants.add(a)
            for s in m.get("synonyms", []):
                if isinstance(s, str): syns.add(s)
            for a in m.get("antonyms", []):
                if isinstance(a, str): ants.add(a)
        return {"word":word,"phon":phon,"defs":defs,
                "synonyms":sorted(list(syns))[:10],
                "antonyms":sorted(list(ants))[:10]}
    except Exception:
        return None

def format_dictionary_response(mode: str, term: str, info: dict) -> str:
    if not info: return "I couldn't find that in the dictionary."
    w = info["word"]; phon = f" {info['phon']}" if info.get("phon") else ""
    if mode in ("define","spell","pronounce"):
        bits = []
        for i, (pos, definition, example) in enumerate(info["defs"][:2], start=1):
            pos_txt = f"{pos}" if pos else "def"
            if example: bits.append(f"{i}) {pos_txt}: {definition} — e.g., “{example}”.")
            else:       bits.append(f"{i}) {pos_txt}: {definition}.")
        if not bits: bits = ["No formal definition found."]
        extra = []
        if mode == "spell": extra.append(f"Spelling: {w}.")
        if mode == "pronounce" and info.get("phon"): extra.append(f"Pronounced as {info['phon']}.")
        synbit = (" Synonyms: " + ", ".join(info["synonyms"][:6]) + ".") if info["synonyms"] else ""
        return f"{w}{phon}: " + " ".join(bits) + ((" " + " ".join(extra)) if extra else "") + synbit
    elif mode == "synonyms":
        return f"Synonyms of {w}:{phon} " + (", ".join(info["synonyms"][:10]) + "." if info["synonyms"] else "None found.")
    elif mode == "antonyms":
        return f"Antonyms of {w}:{phon} " + (", ".join(info["antonyms"][:10]) + "." if info["antonyms"] else "None found.")
    return "I couldn't process that dictionary request."

# -------------------- STT (Google → Whisper API → Local) --------------------
def stt_transcribe(audio_obj) -> str:
    if ASR_ENGINE == "google":
        try:
            return recognizer.recognize_google(audio_obj, language="en-US")
        except sr.RequestError:
            if not OPENAI_API_KEY: raise
        except sr.UnknownValueError:
            raise

    if ASR_ENGINE in ("google","whisper_api"):
        try:
            wav = audio_obj.get_wav_data(convert_rate=16000, convert_width=2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav); tmp = tf.name
            text = whisper_api_transcribe(tmp)
            os.unlink(tmp)
            return (text or "").strip()
        except Exception as e:
            if ASR_ENGINE == "whisper_api":
                raise sr.RequestError(f"Whisper API failed: {e}")

    if ASR_ENGINE == "local":
        try:
            from faster_whisper import WhisperModel
            wav = audio_obj.get_wav_data(convert_rate=16000, convert_width=2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav); tmp = tf.name
            device = "cuda" if shutil.which("nvidia-smi") else "cpu"
            model = WhisperModel("base", device=device, compute_type=("float16" if device=="cuda" else "int8"))
            segments, _ = model.transcribe(tmp, language="en", vad_filter=True)
            text = " ".join(s.text for s in segments).strip()
            os.unlink(tmp)
            if not text: raise sr.UnknownValueError("No speech recognized (local).")
            return text
        except Exception as e:
            raise sr.RequestError(f"Local Whisper failed: {e}")

    raise sr.UnknownValueError("No speech recognized.")

# -------------------- GPT helpers --------------------
def _extract_json_maybe(s: str):
    try: return json.loads(s)
    except Exception: pass
    try:
        i = s.find('{'); j = s.rfind('}')
        if i != -1 and j != -1 and j > i: return json.loads(s[i:j+1])
    except Exception: return None

def ask_router(query: str) -> dict:
    try:
        msg = {
            "query": query,
            "rules": {
                "actions": ["dict","weather","search","answer"],
                "dict_criteria": "meaning/definition/spelling/pronunciation/synonyms/antonyms of an English word/idiom/acronym",
                "weather_criteria": "asks about forecast, temperature, rain/snow, wind, sunrise/sunset",
                "search_criteria": "needs up-to-date facts, named entities, prices, news, schedules, laws, specs, or anything uncertain",
                "answer_criteria": "general knowledge or reasoning you can answer confidently without browsing",
                "tie_breaker": "if unsure between answer and search, choose search"
            }
        }
        resp = chat_create(
            model=ROUTER_MODEL, temperature=0,
            messages=[
                {"role":"system","content":"You are a strict router. Choose a single action. Return ONLY JSON."},
                {"role":"user","content": json.dumps(msg)}
            ],
            max_tokens=80
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _extract_json_maybe(raw) or {}
        act = (data.get("action") or "").lower()
        if act not in {"dict","weather","search","answer"}:
            return {"action":"answer"}
        return data
    except Exception:
        return {"action":"answer"}

def ask_openai_answer(prompt):
    try:
        resp = chat_create(
            model=ANSWER_MODEL, temperature=0,
            messages=[
                {"role":"system","content":("You are a careful assistant. If unsure or needs browsing, reply EXACTLY with "
                                            f"{UNCERTAIN_TOKEN}.")},
                {"role":"user","content":prompt}
            ],
            max_tokens=300
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.lower() in {UNCERTAIN_TOKEN,"<i dont know>","<i_dont_know>","<idontknow>"}:
            return UNCERTAIN_TOKEN
        return text if text else UNCERTAIN_TOKEN
    except Exception as e:
        return f"Error: {e}"

def ask_openai_style_weather(summary_dict):
    try:
        msg = json.dumps(summary_dict)
        resp = chat_create(
            model=ANSWER_MODEL, temperature=0.3,
            messages=[
                {"role":"system","content":"Rewrite into one friendly sentence using °F, mph, inches; city name as given; concise."},
                {"role":"user","content":msg}
            ],
            max_tokens=120
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        try:
            p = summary_dict; t = p['today']
            return (f"{p['place']} today: high {round(t['tmax'])}°F, low {round(t['tmin'])}°F, "
                    f"{t['popmax']}% precip ({t['precip']:.2f} in), winds up to {round(t['windmax'])} mph.")
        except Exception:
            return "Here's the local forecast."

def ask_openai_from_search(query, results):
    try:
        payload = {"query": query, "results": results}
        resp = chat_create(
            model=ANSWER_MODEL, temperature=0.2,
            messages=[
                {"role":"system","content":"Answer using ONLY these snippets in 1–3 sentences; else return <i-dont-know>."},
                {"role":"user","content": json.dumps(payload)}
            ],
            max_tokens=180
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.lower() in {UNCERTAIN_TOKEN,"<i dont know>","<i_dont_know>","<idontknow>"}:
            return UNCERTAIN_TOKEN
        return text
    except Exception:
        return UNCERTAIN_TOKEN

def speak_openai(text, on_done, voice="alloy"):
    def tts_thread():
        try:
            spoken = text.strip() if text and text.strip() else "Sorry, I don't know."
            resp = tts_create(model="tts-1", voice=voice, input=spoken)
            with open("output.wav","wb") as f: f.write(resp.content)
            _play_wav("output.wav")
        except Exception as e:
            print("TTS error:", e)
        finally:
            on_done()
    threading.Thread(target=tts_thread, daemon=True).start()

def looks_like_weather(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["weather","forecast","temperature","rain","snow","wind","sunrise","sunset"])

# -------------------- Tk UI --------------------
class TkUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AskZac")
        self.root.geometry("920x600")

        self.header = tk.Frame(self.root, bg="#0b1020")
        self.header.pack(fill="x")
        tk.Label(self.header, text=" AskZac ", fg="#e5e7eb", bg="#0b1020",
                 font=("Inter", 14, "bold")).pack(side="left", padx=8, pady=6)
        tk.Label(self.header, text="Say “GPT, <your message>” — always listening.",
                 fg="#9ca3af", bg="#0b1020", font=("Inter", 10)).pack(side="left")

        self.chat = scrolledtext.ScrolledText(self.root, wrap="word", bg="#111827", fg="#e5e7eb",
                                              insertbackground="#e5e7eb", font=("Inter", 11))
        self.chat.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat.config(state="disabled")

        bottom = tk.Frame(self.root, bg="#0b1020")
        bottom.pack(fill="x", padx=8, pady=8)
        self.entry = tk.Entry(bottom, bg="#1f2937", fg="#e5e7eb", insertbackground="#e5e7eb",
                              font=("Inter", 11))
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,8))
        self.btn = tk.Button(bottom, text="Send", bg="#6366f1", fg="white",
                             activebackground="#4f46e5", command=self.on_send)
        self.btn.pack(side="right")
        self.entry.bind("<Return>", lambda e: self.on_send())

        self.queue = queue.Queue()
        self._poll_queue()

    def _poll_queue(self):
        try:
            while True:
                role, text = self.queue.get_nowait()
                self._append(role, text)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _append(self, role, text):
        self.chat.config(state="normal")
        prefix = "You: " if role=="user" else "Bot: "
        self.chat.insert("end", prefix + text + "\n")
        self.chat.see("end")
        self.chat.config(state="disabled")

    def add(self, role, text):
        self.queue.put((role, text))

    def on_send(self):
        t = self.entry.get().strip()
        if not t: return
        self.entry.delete(0, "end")
        app.ask_and_speak(t)

# -------------------- App (always-listen) --------------------
class AskZacApp:
    def __init__(self, ui: TkUI):
        self.ui = ui
        self.listening_enabled = True
        self.requests_q = queue.Queue()
        self.stop_bg = None
        self._lock = threading.Lock()

    def ui_add(self, role, text):
        self.ui.add(role, text)

    def start_background(self):
        threading.Thread(target=self._process_queue_loop, daemon=True).start()
        self._start_bg_listener()

    def _start_bg_listener(self):
        if self.stop_bg is not None:
            return
        try:
            self.stop_bg = recognizer.listen_in_background(
                mic, self._asr_callback, phrase_time_limit=WAKE_PHRASE_MAXS
            )
            self.ui_add("bot", "Background listener ON")
        except Exception as e:
            self.ui_add("bot", f"Mic start error: {e}")
            self.stop_bg = None

    def _stop_bg_listener(self):
        if self.stop_bg:
            try: self.stop_bg(wait_for_stop=False)
            except Exception: pass
            self.stop_bg = None

    def _asr_callback(self, recognizer_obj, audio_obj):
        if not self.listening_enabled:
            return
        self.ui_add("bot", "Heard audio… transcribing…")
        try:
            heard = stt_transcribe(audio_obj).strip()
        except sr.UnknownValueError:
            self.ui_add("bot", "ASR: didn't catch that.")
            return
        except sr.RequestError as e:
            self.ui_add("bot", f"ASR error: {e}")
            return
        except Exception as e:
            self.ui_add("bot", f"ASR unexpected: {e}")
            return
        self.requests_q.put(heard)

    def _process_queue_loop(self):
        while True:
            heard = self.requests_q.get()
            if not heard: continue
            self.ui_add("bot", f"You said: {heard}")

            lt = heard.strip().lower()
            if lt == "q":
                self.ui_add("bot", "Goodbye!")
                os._exit(0)

            if contains_wake(heard):
                remainder = split_after_wake(heard)
                if remainder:
                    self.ask_and_speak(remainder)
                else:
                    self.ui_add("bot", "Heard 'GPT'. What's up?")
                    try:
                        self._stop_bg_listener()
                        with mic as source:
                            recognizer.adjust_for_ambient_noise(source, duration=0.4)
                            audio2 = recognizer.listen(source, timeout=QUESTION_TIMEOUT, phrase_time_limit=QUESTION_MAXS)
                        try:
                            q = stt_transcribe(audio2).strip()
                            if q.lower() == "q":
                                self.ui_add("bot", "Goodbye!"); os._exit(0)
                            if q: self.ask_and_speak(q)
                        except sr.UnknownValueError:
                            self.ui_add("bot", "Didn't catch that—try again with 'GPT, …'")
                        except sr.RequestError as e:
                            self.ui_add("bot", f"ASR error: {e}")
                    except sr.WaitTimeoutError:
                        self.ui_add("bot", "Timed out—say 'GPT, …' again.")
                    finally:
                        self._start_bg_listener()
            else:
                if RESPOND_WITHOUT_WAKE:
                    self.ask_and_speak(heard)
                else:
                    self.ui_add("bot", "Say: “GPT, …”")

    # -------- Routing + actions --------
    def ask_and_speak(self, query):
        with self._lock:
            self.ui_add("user", query)
            self.ui_add("bot", "Thinking...")

            route = ask_router(query)
            action = (route.get("action") or "answer").lower()

            if action == "dict":
                term = route.get("term")
                mode, parsed_term = parse_dictionary_query(query)
                term = term or parsed_term
                info = fetch_dictionary(term)
                if info:
                    resp = format_dictionary_response(mode, term, info)
                    self._speak_with_ducking(resp); return
                else:
                    action = "search"

            if action == "weather" or (action == "search" and looks_like_weather(query)):
                try:
                    w = fetch_weather_zip(USER_ZIP, tz=TZ)
                    today = {
                        "date": w["dates"][0],
                        "tmax": float(w["tmax"][0]),
                        "tmin": float(w["tmin"][0]),
                        "popmax": int(w["popmax"][0]),
                        "precip": float(w["precip"][0]),
                        "windmax": float(w["windmax"][0]),
                        "sunrise": w["sunrise"][0],
                        "sunset": w["sunset"][0],
                    }
                    summary = {"zip": USER_ZIP, "place": w["place"], "today": today}
                    styled = ask_openai_style_weather(summary)
                    self._speak_with_ducking(styled); return
                except Exception as e:
                    self.ui_add("bot", f"Weather fetch failed: {e}. Continuing…")

            if action == "search":
                q = route.get("query") or query
                results = web_search_structured(q, n=SEARCH_RESULTS_N)
                synth = ask_openai_from_search(q, results)
                to_say = synth
                if synth == UNCERTAIN_TOKEN:
                    to_say = (results[0]["title"] + ": " + results[0]["snippet"]) if results else "No results."
                self._speak_with_ducking(to_say); return

            answer = ask_openai_answer(query)
            if answer == UNCERTAIN_TOKEN:
                if looks_like_weather(query):
                    try:
                        w = fetch_weather_zip(USER_ZIP, tz=TZ)
                        today = {
                            "date": w["dates"][0],
                            "tmax": float(w["tmax"][0]),
                            "tmin": float(w["tmin"][0]),
                            "popmax": int(w["popmax"][0]),
                            "precip": float(w["precip"][0]),
                            "windmax": float(w["windmax"][0]),
                            "sunrise": w["sunrise"][0],
                            "sunset": w["sunset"][0],
                        }
                        summary = {"zip": USER_ZIP, "place": w["place"], "today": today}
                        styled = ask_openai_style_weather(summary)
                        self._speak_with_ducking(styled); return
                    except Exception:
                        pass
                results = web_search_structured(query, n=SEARCH_RESULTS_N)
                synth = ask_openai_from_search(query, results)
                to_say = synth if synth != UNCERTAIN_TOKEN else ((results[0]["title"] + ": " + results[0]["snippet"]) if results else "No results.")
                self._speak_with_ducking(to_say); return

            self._speak_with_ducking(answer)

    def _speak_with_ducking(self, text):
        self.ui_add("bot", text)
        self.listening_enabled = False
        self._stop_bg_listener()
        speak_openai(text, on_done=self._resume_after_tts)

    def _resume_after_tts(self):
        time.sleep(0.15)
        self.listening_enabled = True
        self._start_bg_listener()

# -------------------- Boot --------------------
ui = TkUI()
app = AskZacApp(ui)

def boot():
    if not OPENAI_API_KEY:
        ui.add("bot", "ERROR: Set OPENAI_API_KEY in .env")
        return
    quick_calibrate(1.2)
    try:
        ui.add("bot", "Inputs:\n" + "\n".join(f"{i}: {n}" for i,n in enumerate(MIC_NAMES)))
    except Exception:
        pass
    chosen = MIC_NAMES[MIC_IDX] if MIC_IDX is not None and 0 <= MIC_IDX < len(MIC_NAMES) else "PyAudio default"
    ui.add("bot", f"Mic ready. energy_threshold={recognizer.energy_threshold}, device_index={MIC_IDX} ({chosen})")
    ui.add("bot", f"RESPOND_WITHOUT_WAKE={'ON' if RESPOND_WITHOUT_WAKE else 'OFF'}  ASR_ENGINE={ASR_ENGINE}")
    app.start_background()

ui.root.after(100, boot)
ui.root.mainloop()