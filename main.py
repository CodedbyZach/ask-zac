#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Alexa-style full-screen UI on monitor #2, blue wake bar at bottom that appears
# ONLY when the user says "GPT". Now: weather + search results are ALWAYS
# phrased via ChatGPT (no raw snippets).

import sys, os, re, difflib, time, threading, subprocess, json, math
sys.stderr = open(os.devnull, 'w')
os.environ["JACK_NO_MSG"] = "1"
os.environ["SDL_JACK_NO_MSG"] = "1"
os.environ["ALSA_LOGLEVEL"] = "none"
os.environ["ALSA_DEBUG"] = "0"

import openai
import speech_recognition as sr
from dotenv import load_dotenv
import requests

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QTime, QDate
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QFont, QPainterPath, QRadialGradient, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QHBoxLayout
)

# ================== Config ==================
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
SECOND_MONITOR_INDEX = 1      # 0=primary, 1=second
FULLSCREEN_ON_SECOND = True   # Alexa-style: full-screen
WAKE_BAR_MS = 1600            # how long the wake bar pulses after "GPT"
# ===========================================

# ====== Env / API ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
openai.api_key = OPENAI_API_KEY

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
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
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
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role":"system",
                    "content": f"You are a careful assistant. If unsure or lacking fresh web info, reply EXACTLY with {UNCERTAIN_TOKEN}."
                },
                {"role":"user","content":prompt}
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
    """Always phrase weather nicely (no raw numbers)."""
    try:
        msg = json.dumps(summary_dict)
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role":"system","content":
                 "Turn the given weather data into ONE short, natural sentence for a voice assistant. "
                 "Use °F, mph, inches. Include the city, today's high/low, notable precip %, and brief wind."},
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
    """Synthesize an answer from snippets (no raw snippets to user)."""
    try:
        payload = {"query": query, "results": results}
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role":"system","content":
                 "You are a concise assistant. Using ONLY the provided search snippets, "
                 "answer the user's question in a natural 1–2 sentence reply."},
                {"role":"user","content": json.dumps(payload)}
            ],
            max_tokens=180
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and text.lower() not in {UNCERTAIN_TOKEN, "<i dont know>", "<i_dont_know>", "<idontknow>"}:
            return text
        return UNCERTAIN_TOKEN
    except Exception:
        return UNCERTAIN_TOKEN

def refine_text_with_openai(query, context_text):
    """Last-resort phrasing pass so we NEVER speak raw data."""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role":"system","content":
                 "Rephrase the given terse data into a single, friendly sentence suitable for a voice assistant. "
                 "Be precise and concise."},
                {"role":"user","content": json.dumps({"query": query, "data": context_text})}
            ],
            max_tokens=120
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else context_text
    except Exception:
        return context_text

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

# ========= Alexa-style bottom wake bar =========
class WakeBar(QWidget):
    def __init__(self, parent=None, height=18):
        super().__init__(parent)
        self.setFixedHeight(height)
        self._active = False
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def showActive(self, active: bool):
        self._active = active
        if active:
            self._t = 0.0
            self._timer.start(16)  # smoother animation (~60fps)
            self.show()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def _tick(self):
        self._t += 0.035
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # --- Alexa-like undulating top edge ---
        path = QPainterPath()
        amp = max(2.0, h * 0.35)                      # small ripple
        freq = 2.0                                    # ~2 waves across
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

        # --- Core cyan/royal gradient (Alexa vibe) ---
        pulse = 0.6 + 0.4 * math.sin(self._t * 2.0)
        base_alpha = int(150 + 70 * pulse)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.00, QColor(26, 116, 240, base_alpha))   # royal blue
        grad.setColorAt(0.50, QColor(0, 201, 255, min(255, base_alpha + 25)))  # cyan
        grad.setColorAt(1.00, QColor(26, 116, 240, base_alpha))
        p.fillPath(path, grad)

        # --- Traveling "comet" highlight ---
        cx = (0.5 * (math.sin(self._t * 1.4) + 1.0)) * w
        comet = QRadialGradient(cx, h * 0.55, h * 1.8)
        comet.setColorAt(0.00, QColor(200, 255, 255, 200))
        comet.setColorAt(0.40, QColor(0, 220, 255, 150))
        comet.setColorAt(1.00, QColor(0, 220, 255, 0))
        p.setBrush(comet)
        p.setPen(Qt.NoPen)
        p.drawPath(path)

        # --- Subtle top edge glow ---
        p.setOpacity(0.9)
        p.strokePath(path, QPen(QColor(0, 230, 255, 180), 2))

# ========= Clock (Echo Show vibe) =========
class ClockWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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
    wake = pyqtSignal()       # emitted when "GPT" is detected
    status = pyqtSignal(str)  # e.g., "Listening"

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
            self.append.emit("Listening… (say: 'GPT, hello')")

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
                    self.append.emit(f"You said: {heard}")
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.append.emit(f"ASR error: {e}")
                    continue

                if heard.strip().lower() == "q":
                    self.append.emit("Goodbye!")
                    self.exit_signal.emit()
                    return

                if contains_wake(heard):
                    self.wake.emit()  # <-- show the blue bar
                    remainder = split_after_wake(heard)
                    if remainder:
                        self.query.emit(remainder)
                    else:
                        self.append.emit("Heard 'GPT'. What's up?")
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
                                    self.append.emit("Goodbye!")
                                    self.exit_signal.emit()
                                    return
                                if q:
                                    self.query.emit(q)
                            except sr.UnknownValueError:
                                self.append.emit("Didn't catch that—try again with 'GPT, …'")
                            except sr.RequestError as e:
                                self.append.emit(f"ASR error: {e}")
                        except sr.WaitTimeoutError:
                            self.append.emit("Timed out—say 'GPT, …' again.")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.append.emit(f"Mic error: {e}")
                continue

# ========= Main Window (Alexa screen styling) =========
class AskZacWindow(QMainWindow):
    appendSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AskZac")
        # Alexa-like deep blue gradient background
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0,y1:0, x2:0,y2:1,
                                stop:0 #0a1026, stop:0.6 #0d1a3a, stop:1 #0b1631); }
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
        """)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(40, 30, 40, 24)
        root.setSpacing(12)

        top = QHBoxLayout()
        self.title = QLabel("AskZac", self); self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top.addWidget(self.title, 1)

        self.clock = ClockWidget(self)
        top.addWidget(self.clock, 0, Qt.AlignRight)
        root.addLayout(top)

        self.statusLabel = QLabel("Idle", self); self.statusLabel.setObjectName("status")
        root.addWidget(self.statusLabel, 0, Qt.AlignLeft)

        self.textArea = QTextEdit(self); self.textArea.setReadOnly(True)
        self.textArea.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textArea.setAlignment(Qt.AlignHCenter)  # center text horizontally like Echo Show
        root.addWidget(self.textArea, 1)

        # Removed the manual send/typing row to match Alexa-style voice-first UI

        self.wakeBar = WakeBar(self, height=18)
        root.addWidget(self.wakeBar)

        self.setCentralWidget(central)
        self.appendSignal.connect(self._append)

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
        self.textArea.append(msg)
        self.textArea.moveCursor(self.textArea.textCursor().End)

    def set_status(self, status: str):
        self.statusLabel.setText(status)

    def toggle_listening(self):
        # Kept for completeness; safe if called elsewhere
        self.listening_enabled = not self.listening_enabled
        self.listener.listening_enabled = self.listening_enabled
        self.set_status("Listening" if self.listening_enabled else "Paused")

    def pause_listening(self):
        self.listening_enabled = False
        self.listener.listening_enabled = False
        self.set_status("Paused")

    def resume_listening(self):
        self.listening_enabled = True
        self.listener.listening_enabled = True
        self.set_status("Listening")

    # ----- Wake bar control -----
    def onWake(self):
        self.wakeBar.showActive(True)
        QTimer.singleShot(WAKE_BAR_MS, lambda: self.wakeBar.showActive(False))

    # ----- Core logic (ALWAYS phrased via GPT) -----
    def ask_and_speak(self, query: str):
        self.append(f"You: {query}")
        self.append("Thinking...")
        self.set_status("Thinking")

        def worker():
            answer = ask_openai(query)

            # Weather-smart path (always phrased)
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
                    self.pause_listening()
                    self.set_status("Speaking")
                    speak_openai(phrased, on_done=lambda: self._resume_after_tts())
                    return
                except Exception as e:
                    self.append(f"Weather fetch failed: {e}. Checking the web…")

            # Web search synth (never raw)
            if answer == UNCERTAIN_TOKEN:
                self.append("Checking the web…")
                results = web_search_structured(query, n=SEARCH_RESULTS_N)
                synth = ask_openai_from_search(query, results)
                if synth == UNCERTAIN_TOKEN:
                    if results:
                        blob = f"{results[0].get('title','')}: {results[0].get('snippet','')}"
                    else:
                        blob = "No reliable snippets were available."
                    phrased = refine_text_with_openai(query, blob)
                    self.append(f"AskZac (web): {phrased}")
                    to_say = phrased
                else:
                    self.append(f"AskZac (web): {synth}")
                    to_say = synth
                self.pause_listening()
                self.set_status("Speaking")
                speak_openai(to_say, on_done=lambda: self._resume_after_tts())
                return

            # Normal path (already phrased answer)
            self.append(f"AskZac: {answer}")
            self.pause_listening()
            self.set_status("Speaking")
            speak_openai(answer, on_done=lambda: self._resume_after_tts())

        threading.Thread(target=worker, daemon=True).start()

    def _resume_after_tts(self):
        self.resume_listening()
        self.set_status("Listening")

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