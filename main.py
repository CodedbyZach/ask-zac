import os, sys, re, difflib, threading, subprocess, json, math, requests, openai, tempfile, urllib.request, time
import speech_recognition as sr
from dotenv import load_dotenv
from requests.exceptions import ChunkedEncodingError, ConnectionError
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QTime, QDate
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QFont, QPainterPath, QRadialGradient, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QGraphicsDropShadowEffect
)

# ================== Config ==================
WAKE_CANONICAL = "gpt"
WAKE_CANONICAL_SPOKEN = "gee pee tee"
WAKE_TIMEOUT_S   = 6.0
WAKE_PHRASE_MAXS = 5.0
QUESTION_TIMEOUT = 10.0
QUESTION_MAXS    = 14.0
UNCERTAIN_TOKEN  = "<i-dont-know>"
TZ               = "America/New_York"
SEARCH_RESULTS_N = 3
FULLSCREEN_ON_SECOND = True
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
USER_ZIP = os.getenv("USER_ZIP", "90210").split('#')[0].strip()
TIMER_RING_SOUND = os.getenv("TIMER_RING_SOUND", "Sounds/alarm-1.mp3").split('#')[0].strip() # Change alarm-1 to alarm-2 for a special suprise.
SECOND_MONITOR_INDEX = int(os.getenv("MONITOR_NUMBER", "0").split('#')[0].strip())
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5").split('#')[0].strip()
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1").split('#')[0].strip()
openai.api_key = OPENAI_API_KEY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.getenv("MUSIC_DIR", "Music").split('#')[0].strip()
MUSIC_EXTS = {".mp3", ".wav"}


# ====== Cross-platform volume dimmer (Windows + Linux) ======
_VOLUME_AVAILABLE = False
_VOLUME_MODE = None  # "windows" or "linux"
_baseline_volume = None  # scalar 0.0–1.0
_volume_lock = threading.Lock()

try:
    if sys.platform.startswith("win"):
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        _VOLUME_AVAILABLE = True
        _VOLUME_MODE = "windows"
        print("[Volume] pycaw available, Windows dimming ENABLED", flush=True)

    elif sys.platform.startswith("linux"):
        from shutil import which
        if which("pactl") is not None:
            _VOLUME_AVAILABLE = True
            _VOLUME_MODE = "linux"
            print("[Volume] pactl available, Linux dimming ENABLED", flush=True)
        else:
            print("[Volume] pactl not found, Linux dimming DISABLED", flush=True)

    else:
        print(f"[Volume] Unsupported platform {sys.platform}, dimming DISABLED", flush=True)

except Exception as e:
    print(f"[Volume] init error: {e}", flush=True)
    _VOLUME_AVAILABLE = False
    _VOLUME_MODE = None


def _get_current_volume_scalar():
    """
    Returns current master volume as a scalar 0.0–1.0 for the active platform.
    """
    if not _VOLUME_AVAILABLE:
        raise RuntimeError("Volume not available")

    if _VOLUME_MODE == "windows":
        # pycaw: AudioEndpointVolume scalar
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return float(volume.GetMasterVolumeLevelScalar())

    if _VOLUME_MODE == "linux":
        # pactl: parse "%"" from default sink
        out = subprocess.check_output(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")

        # Example line:
        # "Volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB"
        m = re.search(r"/\s*(\d+)%", out)
        if not m:
            raise RuntimeError(f"Cannot parse pactl output: {out!r}")
        pct = int(m.group(1))
        return max(0.0, min(1.0, pct / 100.0))

    raise RuntimeError("Unknown volume mode")


def _set_volume_scalar(scalar: float):
    """
    Set master volume to scalar 0.0–1.0 for the active platform.
    """
    scalar = max(0.0, min(1.0, float(scalar)))
    if not _VOLUME_AVAILABLE:
        return

    if _VOLUME_MODE == "windows":
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(scalar, None)
        except Exception as e:
            print(f"[Volume] Windows set failed: {e}", flush=True)
        return

    if _VOLUME_MODE == "linux":
        try:
            pct = max(0, min(100, int(round(scalar * 100))))
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[Volume] Linux set failed: {e}", flush=True)
        return


def _fade_to(target: float, duration: float = 0.4, steps: int = 10):
    """
    Smoothly fade master volume to 'target' in [0.0, 1.0] over duration,
    remembering a baseline so we can restore after ducking.
    """
    global _baseline_volume
    if not _VOLUME_AVAILABLE:
        print("[Volume] _fade_to called but dimming unavailable", flush=True)
        return

    target = max(0.0, min(1.0, float(target)))

    with _volume_lock:
        try:
            start = _get_current_volume_scalar()
            print(f"[Volume] current={start:.3f} target={target:.3f}", flush=True)
        except Exception as e:
            print(f"[Volume] read failed: {e}", flush=True)
            return

        # Store original volume once, when we first dim downward
        if _baseline_volume is None and target < start:
            _baseline_volume = start
            print(f"[Volume] baseline set to {start:.3f}", flush=True)

        if steps <= 0 or duration <= 0:
            _set_volume_scalar(target)
            return

        delta = target - start
        for i in range(steps):
            v = start + delta * (i + 1) / steps
            _set_volume_scalar(v)
            time.sleep(duration / steps)


def dim_system_volume():
    """
    Called on wake-word: duck everything by lowering master volume.
    """
    if not _VOLUME_AVAILABLE:
        print("[Volume] dim_system_volume called but dimming unavailable", flush=True)
        return

    print("[Volume] dim_system_volume → fading to 0.25", flush=True)
    threading.Thread(
        target=lambda: _fade_to(0.25, duration=0.25, steps=8),
        daemon=True
    ).start()


def restore_system_volume():
    """
    Called after TTS finishes: restore master volume back to the
    previous baseline captured before the last dim.
    """
    global _baseline_volume
    if not _VOLUME_AVAILABLE or _baseline_volume is None:
        print("[Volume] restore_system_volume skipped (no baseline)", flush=True)
        return

    prev = _baseline_volume
    _baseline_volume = None
    print(f"[Volume] restore_system_volume → fading back to {prev:.3f}", flush=True)
    threading.Thread(
        target=lambda: _fade_to(prev, duration=0.25, steps=8),
        daemon=True
    ).start()


def normalize_song_key(text: str) -> str:
    """Lowercase, strip punctuation, compress spaces; used for fuzzy match keys."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_music_files():
    """
    Yield (display_name, full_path) for every audio file in Music/.
    Display name is filename without extension, for TTS like 'Billy Joel - Piano Man'.
    """
    root_dir = os.path.join(BASE_DIR, MUSIC_DIR)
    try:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                _, ext = os.path.splitext(name)
                if ext.lower() in MUSIC_EXTS:
                    full = os.path.join(root, name)
                    base = os.path.splitext(name)[0]
                    yield base, full
    except Exception as e:
        print(f"[Music] Error scanning {root_dir}: {e}", flush=True)


def find_best_music_match(query_text: str):
    """
    Fuzzy-match the user query against 'Artist - Title' style names in Music/.
    Returns (display_name, full_path) or None.
    """
    search = normalize_song_key(query_text)
    if not search:
        return None

    best = None
    best_score = 0.0
    q_tokens = set(search.split())

    for base, full in iter_music_files():
        key = normalize_song_key(base)
        if not key:
            continue

        score = difflib.SequenceMatcher(None, search, key).ratio()

        # Small bonus if every query word appears in the filename key
        if q_tokens and q_tokens.issubset(set(key.split())):
            score += 0.15

        if score > best_score:
            best_score = score
            best = (base, full)

    # Threshold is intentionally loose so 'play Billy Joel'
    # still hits 'Billy Joel - Piano Man'.
    if best and best_score >= 0.45:
        return best
    return None


def extract_music_query(full_query: str):
    """
    If the query looks like a 'play' command, return the song/artist portion.
    Otherwise return None.
    Examples:
      'play billy joel piano man'
      'play the song piano man by billy joel'
      'can you play billy joel'
    """
    s = full_query.strip()
    low = s.lower()

    prefixes = [
        "play ",
        "play the song ",
        "play song ",
        "play the track ",
        "can you play ",
        "please play ",
    ]

    tail = None
    for p in prefixes:
        if low.startswith(p):
            tail = s[len(p):].strip()
            break

    if tail is None:
        return None

    # Strip services etc: "on Spotify", "on YouTube", "for me", etc.
    tail = re.sub(r"\b(on|in)\s+(spotify|youtube|apple music|amazon music)\b", "", tail, flags=re.I)
    tail = re.sub(r"\bfor me\b", "", tail, flags=re.I)

    tail = tail.strip(" ,.")
    return tail or None

# Track music playback so we can stop / resume
_current_music_proc = None
_current_music_path = None
_music_lock = threading.Lock()

def play_music_file(path: str):
    """Spawn ffplay in a background thread to play a local track and remember it."""
    global _current_music_proc, _current_music_path

    def _runner(proc_obj):
        global _current_music_proc
        try:
            proc_obj.wait()
        finally:
            with _music_lock:
                if _current_music_proc is proc_obj:
                    _current_music_proc = None

    try:
        with _music_lock:
            _current_music_path = path
            _current_music_proc = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            t = threading.Thread(target=_runner, args=(_current_music_proc,), daemon=True)
            t.start()
    except Exception as e:
        print(f"[Music playback error] {e}", flush=True)


def stop_music():
    """Stop the currently playing song (if any)."""
    global _current_music_proc
    with _music_lock:
        proc = _current_music_proc
        _current_music_proc = None
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def resume_music():
    """
    Restart the last song from the beginning.
    (ffplay cannot resume from the middle without a proper player API.)
    """
    with _music_lock:
        path = _current_music_path
    if path:
        play_music_file(path)

# ====== Speech Recog ======
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
    r"(?i)\b[dDbBpP]\s*P\s*T\b": "gpt",  # DPT, BPT, PPT → GPT
    r"(?i)\b[gG]\s*P\s*T\b": "gpt",      # GPT (normalized case-insensitive)
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
            if removed == 0 and tk.lower() in {"gee","pea","pee","tea","tee"}:
                removed += 1; continue
            out.append(tk)
        return " ".join(out).lstrip(" ,:-").strip()
    return ""
# -------------------------------------

# ---------- Data fetchers ----------
def web_search_structured(query, n=SEARCH_RESULTS_N):
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
    r = requests.get(f"http://api.zippopotam.us/us/{zip_code}", timeout=10)
    r.raise_for_status()
    j = r.json()
    place = j["places"][0]
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
        "timezone": tz,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
    }

    url = "https://api.open-meteo.com/v1/forecast"
    try:
        r = requests.get(url, params=params, timeout=(15, 15))
        r.raise_for_status()
    except (ChunkedEncodingError, ConnectionError):
        # retry once
        r = requests.get(url, params=params, timeout=(15, 15))
        r.raise_for_status()

    d = r.json().get("daily", {})
    return {
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
# -----------------------------------

# ---------- OpenAI helpers ----------
def ask_openai(prompt):
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a careful assistant. If unsure, lacking fresh web info, or the user asks about the weather, reply EXACTLY with {UNCERTAIN_TOKEN}. Keep your answers to no more than two sentences unless more is absolutely necessary."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            timeout=15
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.lower() in {UNCERTAIN_TOKEN, "<i dont know>", "<i_dont_know>", "<idontknow>"}:
            return UNCERTAIN_TOKEN
        return text if text else UNCERTAIN_TOKEN
    except Exception:
        return UNCERTAIN_TOKEN


def ask_openai_style_weather(summary_dict):
    """Always phrase weather nicely (no raw numbers)."""
    try:
        msg = json.dumps(summary_dict)
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role":"system","content":
                 "Turn the given weather data into ONE short, natural sentence for a voice assistant. Round temperatures to the nearest whole number and say 'degrees' (no ° symbol, no F). Include the city, today's high/low, notable precip %, and brief wind in mph."},
                {"role":"user","content":msg}
            ],
            max_tokens=120,
            timeout=15
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        try:
            p = summary_dict; t = p['today']
            return (f"{p['place']} today: high {round(t['tmax'])} degrees, low {round(t['tmin'])} degrees, "
                    f"{t['popmax']}% precip ({t['precip']:.2f} in), winds up to {round(t['windmax'])} mph.")
        except Exception:
            return "Here's the local forecast."


def ask_openai_from_search(query, results):
    """Synthesize an answer from snippets (no raw snippets to user)."""
    try:
        payload = {"query": query, "results": results}
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role":"system","content":
                 "You are a concise assistant. Using ONLY the provided search snippets, "
                 "answer the user's question in a natural 1–2 sentence reply."},
                {"role":"user","content": json.dumps(payload)}
            ],
            max_tokens=180,
            timeout=15
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and text.lower() not in {UNCERTAIN_TOKEN, "<i dont know>", "<i_dont_know>", "<idontnow>"}:
            return text
        return UNCERTAIN_TOKEN
    except Exception:
        return UNCERTAIN_TOKEN


def refine_text_with_openai(query, context_text):
    """Last-resort phrasing pass so we NEVER speak raw data."""
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,
            messages=[
                {"role":"system","content":
                 "Rephrase the given terse data into a single, friendly sentence suitable for a voice assistant. "
                 "Be precise and concise."},
                {"role":"user","content": json.dumps({"query": query, "data": context_text})}
            ],
            max_tokens=120,
            timeout=15
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else context_text
    except Exception:
        return context_text


def ask_openai_with_timer_detection(prompt):
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. "
                        "If the user requests a timer, respond ONLY as a JSON object "
                        "with this exact format: {\"timer_seconds\": <number>}. "
                        "If no timer is requested, respond as "
                        "{\"timer_seconds\": null, \"answer\": \"<short reply>\"}."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            timeout=15
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"[DEBUG] GPT raw output: {text}")

        data = json.loads(text)
        if data.get("timer_seconds") is not None:
            return {"_timer_seconds": int(data["timer_seconds"])}
        return {"_answer": data.get("answer", "")}

    except Exception as e:
        print(f"[ERROR] Timer detection failed: {e}")
        return {"_answer": UNCERTAIN_TOKEN}


def speak_openai(text, on_done, voice="alloy"):
    def tts_thread():
        try:
            spoken = text.strip() if text and text.strip() else "Sorry, I don't know."
            try:
                resp = openai.audio.speech.create(model=TTS_MODEL, voice=voice, input=spoken, timeout=60)
            except TypeError:
                resp = openai.audio.speech.create(model=TTS_MODEL, voice=voice, input=spoken)
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

# ====== Timer logic ======
def start_timer(seconds, auto_dismiss=True):
    """Starts a timer and shows the top-right bubble with time remaining."""

    win.startTimerSig.emit(seconds)

    def timer_thread():
        threading.Event().wait(seconds)
        try:
            sound_source = TIMER_RING_SOUND
            if sound_source.startswith(("http://", "https://")):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(sound_source)[1])
                urllib.request.urlretrieve(sound_source, tmp_file.name)
                sound_source = tmp_file.name

            subprocess.run(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', sound_source],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[Timer sound error] {e}")

        if auto_dismiss:
            self_ref = win
            self_ref.statusSig.emit("Idle")
            self_ref.wakeModeSig.emit('off')
            self_ref.stopTimerSig.emit()

    threading.Thread(target=timer_thread, daemon=True).start()

def format_duration_human(seconds):
    parts = []
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m > 0:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    if s > 0 or not parts:
        parts.append(f"{s} second" + ("s" if s != 1 else ""))
    return ", ".join(parts)

# ========= State-aware Wake Bar =========
class WakeBar(QWidget):
    """Modes:
       - 'off'      : hidden
       - 'listen'   : solid blue (breathing)
       - 'think'    : cross-fade blue -> orange and stay visible
       - 'speaking' : keep orange and fade opacity smoothly to 0 (still animating)
    """
    def __init__(self, parent=None, height=18):
        super().__init__(parent)
        self.setFixedHeight(height)
        self._active = False
        self._t = 0.0
        self._mode = 'off'
        self._orange_mix = 0.0
        self._fade = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer_seconds_left = 0
        self.hide()

    def setMode(self, mode: str):
        mode = mode.lower()
        if mode == 'off':
            self._mode = 'off'
            self._orange_mix = 0.0
            self._fade = 0.0
            self.showActive(False)
            return

        if mode == 'listen':
            self._mode = 'listen'
            self._orange_mix = 0.0
            self._fade = 0.0
        elif mode == 'think':
            self._mode = 'think'
            self._orange_mix = 0.0
            self._fade = 0.0
        elif mode == 'speaking':
            self._mode = 'speaking'
            self._fade = 0.0

        self.showActive(True)

    def showActive(self, active: bool):
        self._active = active
        if active:
            self._t = 0.0
            self._timer.start(16)  # ~60fps
            self.show()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def _tick(self):
        self._t += 0.035

        if self._mode == 'think':
            self._orange_mix = min(1.0, self._orange_mix + 0.05)
            self._fade = 0.0
        elif self._mode == 'speaking':
            self._fade = min(1.0, self._fade + 0.04)
            if self._fade >= 1.0:
                self.setMode('off')
                return

        self.update()

    def _mix(self, a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    def _mixColor(self, c1: QColor, c2: QColor, t: float, alpha: int) -> QColor:
        return QColor(
            self._mix(c1.red(),   c2.red(),   t),
            self._mix(c1.green(), c2.green(), t),
            self._mix(c1.blue(),  c2.blue(),  t),
            alpha
        )

    def paintEvent(self, event):
        if not self._active:
            return
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # --- Top edge waveform ---
        path = QPainterPath()
        amp = max(2.0, h * 0.35)
        freq = 2.0
        phase = self._t * 2.2
        path.moveTo(0, h)
        x = 0
        step = max(4, int(w / 120))
        while x <= w:
            y_top = h - 1 - amp * (0.5 + 0.5 * math.sin((x / max(1, w)) * math.tau * freq + phase))
            path.lineTo(x, y_top)
            x += step
        path.lineTo(w, h)
        path.closeSubpath()

        # breathing base alpha
        pulse = 0.6 + 0.4 * math.sin(self._t * 2.0)
        base_alpha = int(150 + 70 * pulse)
        base_alpha = int(base_alpha * (1.0 - self._fade))

        # Palettes
        blue_left  = QColor(26, 116, 240)
        blue_mid   = QColor(0, 201, 255)
        blue_right = QColor(26, 116, 240)

        orange_left  = QColor(255, 149, 0)
        orange_mid   = QColor(255, 196, 0)
        orange_right = QColor(255, 149, 0)

        tcol = self._orange_mix  # 0=blue, 1=orange

        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.00, self._mixColor(blue_left,  orange_left,  tcol, base_alpha))
        grad.setColorAt(0.50, self._mixColor(blue_mid,   orange_mid,   tcol, min(255, base_alpha + 25)))
        grad.setColorAt(1.00, self._mixColor(blue_right, orange_right, tcol, base_alpha))
        p.fillPath(path, grad)

        # "comet" highlight
        comet_blue_inner = QColor(200, 255, 255)
        comet_blue_mid   = QColor(0, 220, 255)
        comet_warm_inner = QColor(255, 230, 200)
        comet_warm_mid   = QColor(255, 170, 60)

        cx = (0.5 * (math.sin(self._t * 1.4) + 1.0)) * w
        comet = QRadialGradient(cx, h * 0.55, h * 1.8)
        comet.setColorAt(0.00, self._mixColor(comet_blue_inner, comet_warm_inner, tcol, int(200 * (1.0 - self._fade))))
        comet.setColorAt(0.40, self._mixColor(comet_blue_mid,   comet_warm_mid,   tcol, int(150 * (1.0 - self._fade))))
        comet.setColorAt(1.00, self._mixColor(QColor(0,0,0,0),  QColor(0,0,0,0),  0.0, 0))
        p.setBrush(comet)
        p.setPen(Qt.NoPen)
        p.drawPath(path)

        p.setOpacity(0.9 * (1.0 - self._fade))
        p.strokePath(path, QPen(self._mixColor(QColor(0,230,255,180), QColor(255,190,90,180), tcol, 180), 2))
        # (Removed drawing raw timer here; bubble handles it.)

# ========= Clock (Echo Show vibe) =========
class ClockWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setStyleSheet("color:#e8f2ff;")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    def _tick(self):
        t = QTime.currentTime().toString("h:mm AP")
        d = QDate.currentDate().toString("dddd, MMMM d")
        self.setText(f"{t}  •  {d}")
        f = QFont("Segoe UI", 20, QFont.Medium)
        self.setFont(f)

# ========= Mic Listener Thread =========
class MicListener(QThread):
    append = pyqtSignal(str)
    query = pyqtSignal(str)
    exit_signal = pyqtSignal()
    wake = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.listening_enabled = True
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            if not self.listening_enabled:
                self.msleep(50)
                continue

            self.status.emit("Listening")
            print("Listening… (say: 'GPT, hello')", flush=True)

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
                    print(f"You said: {heard}", flush=True)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"ASR error: {e}", flush=True)
                    continue

                if heard.strip().lower() == "q":
                    print("Goodbye!", flush=True)
                    self.exit_signal.emit()
                    return

                if contains_wake(heard):
                    self.wake.emit()
                    remainder = split_after_wake(heard)
                    if remainder:
                        self.query.emit(remainder)
                    else:
                        print("Heard 'GPT'. What's up?", flush=True)
                        try:
                            with mic as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                                audio2 = recognizer.listen(
                                    source,
                                    timeout=QUESTION_TIMEOUT,
                                    phrase_time_limit=QUESTION_MAXS
                                )
                            try:
                                q = recognizer.recognize_google(audio2, language="en-US").strip()
                                if q.lower() == "q":
                                    print("Goodbye!", flush=True)
                                    self.exit_signal.emit()
                                    return
                                if q:
                                    self.query.emit(q)
                            except sr.UnknownValueError:
                                print("Didn't catch that, try again with 'GPT, …'", flush=True)
                                restore_system_volume()
                            except sr.RequestError as e:
                                print(f"ASR error: {e}", flush=True)
                        except sr.WaitTimeoutError:
                            print("Timed out. Say 'GPT, …' again.", flush=True)
                            restore_system_volume()
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Mic error: {e}", flush=True)
                continue

# ========= Main Window =========
class AskZacWindow(QMainWindow):
    appendSignal = pyqtSignal(str)
    statusSig = pyqtSignal(str)
    wakeModeSig = pyqtSignal(str)
    startTimerSig = pyqtSignal(int)
    stopTimerSig  = pyqtSignal()

    def _update_timer_label(self):
        if self._timer_seconds_left > 0:
            hrs = self._timer_seconds_left // 3600
            mins = (self._timer_seconds_left % 3600) // 60
            secs = self._timer_seconds_left % 60
            self.timerLabel.setText(f"⏱ {hrs:02}:{mins:02}:{secs:02}")
            self.timerLabel.adjustSize()
            cw = self.centralWidget()
            self.timerLabel.move(cw.width() - self.timerLabel.width() - 20, 14)
            self.timerLabel.show()
            self.timerLabel.raise_()
            self._timer_seconds_left -= 1
        else:
            self.timerLabel.setText("")
            self.timerLabel.hide()
            self._timer_qtimer.stop()

    def resizeEvent(self, event):
        cw = self.centralWidget()
        self.timerLabel.move(cw.width() - self.timerLabel.width() - 20, 14)
        super(AskZacWindow, self).resizeEvent(event)
        self.timerLabel.raise_()

    def _start_timer_ui(self, seconds: int):
        self._timer_seconds_left = seconds
        self._update_timer_label()
        self.timerLabel.raise_()
        self._timer_qtimer.start(1000)

    def _stop_timer_ui(self):
        self.timerLabel.setText("")
        self.timerLabel.hide()
        self._timer_qtimer.stop()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle("AskZac")
        # Alexa-like deep blue gradient background + timer bubble style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0,y1:0, x2:0,y2:1,
                                stop:0 #0a1026, stop:0.6 #0d1a3a, stop:1 #0b1631);
            }
            QLabel#title {
                color: #e8f2ff; font-size: 44px; font-weight: 700; letter-spacing: 0.6px;
            }
            QLabel#status {
                color: #9fd7ff; font-size: 20px; font-weight: 500;
            }
            QTextEdit {
                background: transparent;
                color: #eef7ff;
                border: none;
                padding: 0px;
                font-family: Segoe UI, Roboto, "Fira Sans", Arial;
                font-size: 28px;
            }
            QLabel#timerBubble {
                color: #ffffff;
                background: rgba(40,45,60,220);        /* slightly lighter gray bubble */
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 6px 12px;
                font-size: 18px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
        """)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(40, 30, 40, 24)
        root.setSpacing(12)

        # Centered clock
        self.clock = ClockWidget(self)
        root.addWidget(self.clock, 0, Qt.AlignHCenter)

        # Timer bubble (top-right)
        self.timerLabel = QLabel("", central)
        self.timerLabel.setObjectName("timerBubble")
        self.timerLabel.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.timerLabel.hide()
        # Monospace-ish digits for stable width (fallbacks ok)
        self.timerLabel.setFont(QFont("Consolas", 18, QFont.DemiBold))
        # Subtle shadow for pop
        shadow = QGraphicsDropShadowEffect(self.timerLabel)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.timerLabel.setGraphicsEffect(shadow)
        self.timerLabel.move(self.width() - 160, 14)
        self.timerLabel.raise_()

        self._timer_seconds_left = 0
        self._timer_qtimer = QTimer(self)
        self._timer_qtimer.timeout.connect(self._update_timer_label)
        self.startTimerSig.connect(self._start_timer_ui)
        self.stopTimerSig.connect(self._stop_timer_ui)

        self.statusLabel = QLabel("Idle", self); self.statusLabel.setObjectName("status")
        root.addWidget(self.statusLabel, 0, Qt.AlignLeft)

        self.textArea = QTextEdit(self); self.textArea.setReadOnly(True)
        self.textArea.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textArea.setAlignment(Qt.AlignHCenter)
        self.timerLabel.setFont(QFont("Segoe UI", 18, QFont.DemiBold))
        root.addWidget(self.textArea, 1)
        self.timerLabel.setTextFormat(Qt.PlainText)

        self.wakeBar = WakeBar(self, height=18)
        root.addWidget(self.wakeBar)

        self.setCentralWidget(central)
        self.appendSignal.connect(self._append)
        self.statusSig.connect(self.set_status)
        self.wakeModeSig.connect(self.wakeBar.setMode)

        self.listener = MicListener(self)
        self.listener.append.connect(self.append)
        self.listener.query.connect(self.ask_and_speak)
        self.listener.exit_signal.connect(self.close)
        self.listener.wake.connect(self.onWake)
        self.listener.status.connect(lambda s: self.set_status(s))

        self.listening_enabled = True
        self.append("Say “GPT, …”")
        self.set_status("Idle")
        quick_calibrate(0.6)
        self.listener.start()

    def _maybe_handle_music_command(self, query: str) -> bool:
        """
        If query starts with 'play ...', try to resolve it to a local file in Music/.
        On success: say 'Playing ...', then AFTER TTS finishes start the song.
        On failure: say an error. If it's not a play command at all, return False.
        """
        music_query = extract_music_query(query)
        if not music_query:
            return False

        match = find_best_music_match(music_query)
        if not match:
            msg = "I could not find that song in your Music folder."
            self.append(msg)
            self.wakeModeSig.emit('speaking')
            self.statusSig.emit("Speaking")
            speak_openai(msg, on_done=self._resume_after_tts)
            return True

        display_name, full_path = match
        msg = f"Playing {display_name}."
        self.append(msg)
        self.wakeModeSig.emit('speaking')
        self.statusSig.emit("Speaking")

        def after_tts():
            # Normal volume restore + resume listening
            self._resume_after_tts()
            # Now start the music at full volume
            play_music_file(full_path)

        speak_openai(msg, on_done=after_tts)
        return True

    # ----- Placement on second monitor -----
    def place_on_second_monitor(self, index=SECOND_MONITOR_INDEX, fullscreen=FULLSCREEN_ON_SECOND):
        app = QApplication.instance()
        screens = app.screens()
        if len(screens) > index:
            geo = screens[index].geometry()
            if fullscreen:
                self.setGeometry(geo)
                self.showFullScreen()
            else:
                self.resize(int(geo.width()*0.8), int(geo.height()*0.8))
                x = geo.x() + (geo.width()-self.width())//2
                y = geo.y() + (geo.height()-self.height())//3
                self.move(x, y)
        else:
            if fullscreen:
                self.showFullScreen()

    # ----- UI helpers -----
    def append(self, msg: str):
        self.appendSignal.emit(msg)

    def _append(self, msg: str):
        self.textArea.clear()
        self.textArea.setText(msg)
        self.textArea.moveCursor(self.textArea.textCursor().End)

    def set_status(self, status: str):
        if status == "Listening" and not self.listening_enabled:
            return
        self.statusLabel.setText(status)

    # ----- Wake bar control -----
    def onWake(self):
        # Dim system volume while we're in wake / listen mode
        dim_system_volume()
        self.wakeBar.setMode('listen')

    # ----- Core logic -----
    def ask_and_speak(self, query: str):
        print(f"You: {query}", flush=True)
        print("Thinking...", flush=True)

        # ====== Stop / Resume music by voice ======
        qlow = query.strip().lower()

        if qlow in {"stop", "stop music", "stop the music", "stop song", "stop the song"}:
            stop_music()
            self.append("Stopping music.")
            self.wakeModeSig.emit('speaking')
            self.statusSig.emit("Speaking")
            speak_openai("Stopping music.", on_done=self._resume_after_tts)
            return

        if qlow in {"resume", "resume music", "continue music", "continue the music"}:
            if _current_music_path:
                resume_music()
                self.append("Resuming music.")
                self.wakeModeSig.emit('speaking')
                self.statusSig.emit("Speaking")
                speak_openai("Resuming music.", on_done=self._resume_after_tts)
            else:
                self.append("There is no music to resume.")
                self.wakeModeSig.emit('speaking')
                self.statusSig.emit("Speaking")
                speak_openai("There is no music to resume.", on_done=self._resume_after_tts)
            return

        self.pause_listening()
        self.set_status("Thinking")
        self.textArea.clear()

        self.wakeBar.setMode('think')

        QTimer.singleShot(30000, lambda: (
            self.wakeBar.setMode('off'),
            self.set_status('Listening'),
            self.resume_listening()
        ) if getattr(self.wakeBar, "_mode", "off") == "think" else None)

        def speak_and_fade(text_to_say: str):
            self.wakeModeSig.emit('speaking')
            self.statusSig.emit("Speaking")
            speak_openai(text_to_say, on_done=self._resume_after_tts)

        def worker():
            # 1) Local music playback intent: 'play ...'
            if self._maybe_handle_music_command(query):
                # Music handled, nothing else to do
                return
            parsed = ask_openai_with_timer_detection(query)
            if "_timer_seconds" in parsed:
                secs = parsed["_timer_seconds"]
                human_time = format_duration_human(secs)
                text_to_speak = f"Timer set for {human_time}."
                self.append(text_to_speak)
                speak_openai(text_to_speak, on_done=lambda: self._resume_after_tts())
                self.wakeModeSig.emit('off')
                self.statusSig.emit("Idle")
                start_timer(secs, auto_dismiss=True)
                return

            answer = ask_openai(query)

            if answer == UNCERTAIN_TOKEN and looks_like_weather(query):
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
                    phrased = ask_openai_style_weather(summary)
                    self.append(f"AskZac: {phrased}")
                    speak_and_fade(phrased)
                    return
                except Exception as e:
                    print(f"Weather fetch failed: {e}. Checking the web…", flush=True)

            if answer == UNCERTAIN_TOKEN:
                print("Checking the web…", flush=True)
                results = web_search_structured(query, n=SEARCH_RESULTS_N)
                synth = ask_openai_from_search(query, results)
                if synth == UNCERTAIN_TOKEN:
                    if results:
                        blob = f"{results[0].get('title','')}: {results[0].get('snippet','')}"
                    else:
                        blob = "No reliable snippets were available."
                    phrased = refine_text_with_openai(query, blob)
                    self.append(f"AskZac (web): {phrased}")
                    speak_and_fade(phrased)
                else:
                    self.append(f"AskZac (web): {synth}")
                    speak_and_fade(synth)
                return

            self.append(f"AskZac: {answer}")
            speak_and_fade(answer)

        threading.Thread(target=worker, daemon=True).start()

    def pause_listening(self):
        self.listening_enabled = True and False  # keep semantics explicit
        self.listener.listening_enabled = False

    def resume_listening(self):
        self.listening_enabled = True
        self.listener.listening_enabled = True

    def _resume_after_tts(self):
        # Bring volume back up to whatever it was before dimming
        restore_system_volume()
        QTimer.singleShot(0, self.resume_listening)
        self.statusSig.emit("Listening")
        self.wakeModeSig.emit('off')

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()

    def closeEvent(self, event):
        try:
            self.listener.stop()
            self.listener.wait(500)
        except Exception:
            pass
        event.accept()

# ====== Main ======
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AskZacWindow()
    win.show()
    win.place_on_second_monitor(index=SECOND_MONITOR_INDEX, fullscreen=FULLSCREEN_ON_SECOND)
    sys.exit(app.exec_())