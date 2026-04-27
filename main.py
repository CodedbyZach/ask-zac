import os, sys, re, difflib, threading, subprocess, json, math, requests, tempfile, urllib.request, time # type: ignore
import google.generativeai as genai
import speech_recognition as sr # type: ignore
import pyttsx3 # type: ignore
from dotenv import load_dotenv # type: ignore
from requests.exceptions import ChunkedEncodingError, ConnectionError # type: ignore
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QTime, QDate # type: ignore
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QFont, QPainterPath, QRadialGradient, QPen # type: ignore
from PyQt5.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QGraphicsDropShadowEffect
)

WAKE_CANONICAL = "gpt"
WAKE_CANONICAL_SPOKEN = "PPT"
WAKE_TIMEOUT_S   = 6.0
WAKE_PHRASE_MAXS = 5.0
QUESTION_TIMEOUT = 10.0
QUESTION_MAXS    = 14.0
UNCERTAIN_TOKEN  = "<i-dont-know>"
SEARCH_RESULTS_N = 3
FULLSCREEN_ON_SECOND = True
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[{"google_search_retrieval": {}}]
)

USER_ZIP = os.getenv("USER_ZIP", "90210").split('#')[0].strip()
TIMER_RING_SOUND = os.getenv("TIMER_RING_SOUND", "Sounds/alarm-1.mp3").split('#')[0].strip()
SECOND_MONITOR_INDEX = int(os.getenv("MONITOR_NUMBER", "0").split('#')[0].strip())
TZ        = os.getenv("TZ", "America/New_York").split('#')[0].strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.getenv("MUSIC_DIR", "Music").split('#')[0].strip()
MUSIC_EXTS = {".mp3", ".wav"}

_VOLUME_AVAILABLE = False
_VOLUME_MODE = None  
_baseline_volume = None  
_volume_lock = threading.Lock()

try:
    if sys.platform.startswith("win"):
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore
        _VOLUME_AVAILABLE = True
        _VOLUME_MODE = "windows"
    elif sys.platform.startswith("linux"):
        from shutil import which
        if which("pactl") is not None:
            _VOLUME_AVAILABLE = True
            _VOLUME_MODE = "linux"
except Exception:
    _VOLUME_AVAILABLE = False

def _get_current_volume_scalar():
    if not _VOLUME_AVAILABLE: return 1.0
    if _VOLUME_MODE == "windows":
        interface = AudioUtilities.GetSpeakers().Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return float(cast(interface, POINTER(IAudioEndpointVolume)).GetMasterVolumeLevelScalar())
    if _VOLUME_MODE == "linux":
        out = subprocess.check_output(["pactl", "get-sink-volume", "@DEFAULT_SINK@"]).decode("utf-8")
        m = re.search(r"/\s*(\d+)%", out)
        return int(m.group(1)) / 100.0 if m else 1.0

def _set_volume_scalar(scalar: float):
    if not _VOLUME_AVAILABLE: return
    scalar = max(0.0, min(1.0, float(scalar)))
    if _VOLUME_MODE == "windows":
        interface = AudioUtilities.GetSpeakers().Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        cast(interface, POINTER(IAudioEndpointVolume)).SetMasterVolumeLevelScalar(scalar, None)
    elif _VOLUME_MODE == "linux":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(scalar*100)}%"])

def _fade_to(target: float, duration: float = 0.25, steps: int = 8):
    global _baseline_volume
    if not _VOLUME_AVAILABLE: return
    with _volume_lock:
        start = _get_current_volume_scalar()
        if _baseline_volume is None and target < start: _baseline_volume = start
        delta = (target - start) / steps
        for i in range(steps):
            _set_volume_scalar(start + delta * (i + 1))
            time.sleep(duration / steps)

def dim_system_volume(): threading.Thread(target=lambda: _fade_to(0.25), daemon=True).start()
def restore_system_volume():
    global _baseline_volume
    if _baseline_volume is not None:
        val = _baseline_volume
        _baseline_volume = None
        threading.Thread(target=lambda: _fade_to(val), daemon=True).start()

def normalize_song_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()

def iter_music_files():
    root_dir = os.path.join(BASE_DIR, MUSIC_DIR)
    if not os.path.exists(root_dir): return
    for root, _, files in os.walk(root_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in MUSIC_EXTS:
                yield os.path.splitext(name)[0], os.path.join(root, name)

def find_best_music_match(query_text: str):
    search = normalize_song_key(query_text)
    if not search: return None
    best, best_score, q_tokens = None, 0.0, set(search.split())
    for base, full in iter_music_files():
        key = normalize_song_key(base)
        score = difflib.SequenceMatcher(None, search, key).ratio()
        if q_tokens.issubset(set(key.split())): score += 0.15
        if score > best_score: best_score, best = score, (base, full)
    return best if best_score >= 0.45 else None

def extract_music_query(full_query: str):
    s = full_query.strip()
    prefixes = ["play the song ", "play song ", "play the track ", "can you play ", "please play ", "play "]
    for p in prefixes:
        if s.lower().startswith(p):
            tail = s[len(p):].strip()
            tail = re.sub(r"\b(on|in)\s+(spotify|youtube|apple music|amazon music)\b", "", tail, flags=re.I)
            return re.sub(r"\bfor me\b", "", tail, flags=re.I).strip(" ,.")
    return None

_music_proc = None
def play_music_file(path: str):
    global _music_proc
    stop_music()
    _music_proc = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path])

def stop_music():
    global _music_proc
    if _music_proc and _music_proc.poll() is None: _music_proc.terminate()
    _music_proc = None

def ask_gemini_unified(prompt):
    system_instr = (
        "You are AskZac, a helpful voice assistant. "
        "If the user wants a timer, respond ONLY with JSON: {\"timer_seconds\": N}. "
        "Otherwise, use Google Search grounding for facts/weather and keep responses to 1-2 natural sentences."
    )
    try:
        response = model.generate_content(f"{system_instr}\n\nUser: {prompt}")
        text = response.text.strip()
        if "timer_seconds" in text:
            try:
                data = json.loads(re.sub(r"```json|```", "", text).strip())
                if data.get("timer_seconds"): return {"_timer_seconds": int(data["timer_seconds"])}
            except: pass
        return text if text else UNCERTAIN_TOKEN
    except Exception as e:
        return f"Gemini error: {str(e)}"

def speak_gemini(text, on_done):
    def tts_thread():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e: print(f"TTS error: {e}")
        finally: on_done()
    threading.Thread(target=tts_thread, daemon=True).start()

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
mic = sr.Microphone()

WAKE_PAT = re.compile(r"\b(g\.?\s*p\.?\s*t)\b", re.I)
def contains_wake(raw: str) -> bool:
    if WAKE_PAT.search(raw): return True
    r1 = difflib.SequenceMatcher(None, raw.lower(), WAKE_CANONICAL).ratio()
    r2 = difflib.SequenceMatcher(None, raw.lower(), WAKE_CANONICAL_SPOKEN).ratio()
    return max(r1, r2) >= 0.75

def split_after_wake(raw: str) -> str:
    m = WAKE_PAT.search(raw)
    if m: return raw[m.end():].lstrip(" ,:-").strip()
    for form in ["gpt", "g.p.t", "g p t"]:
        idx = raw.lower().find(form)
        if idx != -1: return raw[idx+len(form):].lstrip(" ,:-").strip()
    return ""

class WakeBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18); self._active = False; self._t = 0.0; self._mode = 'off'; self._orange_mix = 0.0; self._fade = 0.0
        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self.hide()
    def setMode(self, mode: str):
        self._mode = mode.lower()
        if self._mode == 'off': self._active = False; self.hide(); self._timer.stop()
        else: self._active = True; self.show(); self._t = 0.0; self._timer.start(16)
    def _tick(self):
        self._t += 0.035
        if self._mode == 'think': self._orange_mix = min(1.0, self._orange_mix + 0.05)
        elif self._mode == 'speaking':
            self._fade = min(1.0, self._fade + 0.04)
            if self._fade >= 1.0: self.setMode('off')
        self.update()
    def paintEvent(self, event):
        if not self._active: return
        w, h, p = self.width(), self.height(), QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath(); path.moveTo(0, h)
        for x in range(0, w + 5, 5):
            path.lineTo(x, h - 1 - (h*0.35) * (0.5 + 0.5 * math.sin((x/w)*math.tau*2.0 + self._t*2.2)))
        path.lineTo(w, h); path.closeSubpath()
        alpha = int((150 + 70 * (0.6 + 0.4 * math.sin(self._t*2.0))) * (1.0 - self._fade))
        grad = QLinearGradient(0, 0, w, 0)
        t = self._orange_mix
        grad.setColorAt(0.5, QColor(int(26*(1-t)+255*t), int(116*(1-t)+149*t), int(240*(1-t)), alpha))
        p.fillPath(path, grad)

class MicListener(QThread):
    query, wake, status = pyqtSignal(str), pyqtSignal(), pyqtSignal(str)
    def __init__(self): super().__init__(); self.listening = self.running = True
    def run(self):
        while self.running:
            if not self.listening: self.msleep(50); continue
            self.status.emit("Listening")
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, 0.2)
                    audio = recognizer.listen(source, timeout=WAKE_TIMEOUT_S, phrase_time_limit=WAKE_PHRASE_MAXS)
                heard = recognizer.recognize_google(audio)
                if contains_wake(heard):
                    self.wake.emit()
                    rem = split_after_wake(heard)
                    if not rem:
                        with mic as source: audio2 = recognizer.listen(source, timeout=QUESTION_TIMEOUT, phrase_time_limit=QUESTION_MAXS)
                        rem = recognizer.recognize_google(audio2)
                    if rem: self.query.emit(rem)
            except: pass

class AskZacWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("QMainWindow { background: #0a1026; } QTextEdit { background: transparent; color: #eef7ff; border: none; font-size: 28px; }")
        central = QWidget(self); layout = QVBoxLayout(central); layout.setContentsMargins(40, 30, 40, 24)
        self.clock = QLabel(self); self.clock.setStyleSheet("color:#e8f2ff; font-size: 20px;"); layout.addWidget(self.clock, 0, Qt.AlignHCenter)
        self.statusLabel = QLabel("Idle", self); self.statusLabel.setStyleSheet("color:#9fd7ff; font-size: 18px;"); layout.addWidget(self.statusLabel)
        self.textArea = QTextEdit(self); self.textArea.setReadOnly(True); self.textArea.setAlignment(Qt.AlignHCenter); layout.addWidget(self.textArea, 1)
        self.wakeBar = WakeBar(self); layout.addWidget(self.wakeBar)
        self.setCentralWidget(central)
        
        self.clk_timer = QTimer(self); self.clk_timer.timeout.connect(lambda: self.clock.setText(QTime.currentTime().toString("h:mm AP"))); self.clk_timer.start(1000)
        self.listener = MicListener()
        self.listener.query.connect(self.process_query); self.listener.wake.connect(self.on_wake); self.listener.status.connect(self.statusLabel.setText)
        self.listener.start()

    def on_wake(self): dim_system_volume(); self.wakeBar.setMode('listen')
    def process_query(self, query):
        self.listener.listening = False
        self.statusLabel.setText("Thinking"); self.wakeBar.setMode('think')
        
        def worker():
            mq = extract_music_query(query)
            if mq:
                match = find_best_music_match(mq)
                if match:
                    self.textArea.setText(f"Playing {match[0]}.")
                    speak_gemini(f"Playing {match[0]}.", lambda: (self.done(), play_music_file(match[1])))
                else: speak_gemini("Song not found.", self.done)
                return
            
            res = ask_gemini_unified(query)
            if isinstance(res, dict):
                self.textArea.setText(f"Timer set for {res['_timer_seconds']} seconds.")
                speak_gemini("Timer set.", self.done)
            else:
                self.textArea.setText(res); self.wakeBar.setMode('speaking')
                speak_gemini(res, self.done)
        threading.Thread(target=worker, daemon=True).start()

    def done(self): restore_system_volume(); self.listener.listening = True; self.wakeBar.setMode('off')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AskZacWindow(); win.resize(800, 480); win.show()
    sys.exit(app.exec_())