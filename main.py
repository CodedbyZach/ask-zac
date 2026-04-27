import os, sys, re, threading, subprocess, json, math, requests, time
from google import genai
from google.genai import types
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QTime
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QPainterPath
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit

load_dotenv()
WAKE_CANONICAL = "gpt"
WAKE_TIMEOUT_S = 6.0
QUESTION_TIMEOUT = 8.0
GEMINI_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

USER_ZIP = os.getenv("USER_ZIP", "90210")
MUSIC_DIR = os.getenv("MUSIC_DIR", "Music")
MUSIC_EXTS = {".mp3", ".wav"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _set_vol(val):
    if sys.platform == "linux":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(val*100)}%"], capture_output=True)

def dim_system_volume(): threading.Thread(target=lambda: _set_vol(0.25), daemon=True).start()
def restore_system_volume(): threading.Thread(target=lambda: _set_vol(1.0), daemon=True).start()

class AIWorker(QThread):
    finished_sig = pyqtSignal(str, bool)
    def __init__(self, query):
        super().__init__()
        self.query = query
    def run(self):
        mq = None
        if self.query.lower().startswith("play "):
            mq = self.query[5:].strip()
        if mq:
            root_dir = os.path.join(BASE_DIR, MUSIC_DIR)
            for r, _, files in os.walk(root_dir):
                for f in files:
                    if mq.lower() in f.lower() and os.path.splitext(f)[1].lower() in MUSIC_EXTS:
                        self.finished_sig.emit(os.path.join(r, f), True)
                        return
        try:
            sys_instr = f"You are AskZac. Respond in 1-2 sentences. Use Google Search. The user's location zip code is {USER_ZIP}. If a timer is requested, respond ONLY with JSON: {{\"timer_seconds\": N}}."
            config = types.GenerateContentConfig(
                system_instruction=sys_instr,
                tools=[{"google_search": {}}]
            )
            res = client.models.generate_content(
                model=GEMINI_MODEL,
                config=config,
                contents=self.query
            )
            self.finished_sig.emit(res.text.strip(), False)
        except Exception as e:
            self.finished_sig.emit(f"API Error: {str(e)}", False)

class WakeBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18); self.mode = 'off'; self.t = 0.0; self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick); self.hide()
    def setMode(self, m):
        self.mode = m
        if m == 'off': self.hide(); self.timer.stop()
        else: self.show(); self.timer.start(16)
    def tick(self): self.t += 0.05; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), 0)
        col = QColor(26, 116, 240) if self.mode == 'listen' else QColor(255, 149, 0)
        grad.setColorAt(0.5, col); p.fillPath(self.get_path(), grad)
    def get_path(self):
        path = QPainterPath(); path.moveTo(0, 18)
        for x in range(0, self.width()+5, 5):
            path.lineTo(x, 10 + 5 * math.sin(x/50 + self.t))
        path.lineTo(self.width(), 18); return path

class AskZacWindow(QMainWindow):
    trigger_query_sig = pyqtSignal(str)
    update_ui_sig = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint); self.setStyleSheet("background:#0a1026;")
        c = QWidget(); l = QVBoxLayout(c); self.setCentralWidget(c)
        self.clock = QLabel(); self.clock.setStyleSheet("color:white; font-size:20px;"); l.addWidget(self.clock, 0, Qt.AlignHCenter)
        self.area = QTextEdit(); self.area.setReadOnly(True); self.area.setStyleSheet("color:white; border:none; font-size:28px;"); l.addWidget(self.area)
        self.bar = WakeBar(); l.addWidget(self.bar)
        self.tm = QTimer(); self.tm.timeout.connect(lambda: self.clock.setText(QTime.currentTime().toString("h:mm AP"))); self.tm.start(1000)
        self.trigger_query_sig.connect(self.process_query)
        self.update_ui_sig.connect(self.safe_update_ui)
        threading.Thread(target=self.mic_loop, daemon=True).start()

    def mic_loop(self):
        r, m = sr.Recognizer(), sr.Microphone()
        while True:
            try:
                with m as s:
                    r.adjust_for_ambient_noise(s, 0.3)
                    audio = r.listen(s, timeout=WAKE_TIMEOUT_S, phrase_time_limit=3.0)
                if WAKE_CANONICAL in r.recognize_google(audio).lower():
                    dim_system_volume()
                    self.update_ui_sig.emit("Listening...", "listen")
                    with m as s:
                        audio2 = r.listen(s, timeout=QUESTION_TIMEOUT, phrase_time_limit=10.0)
                    query_text = r.recognize_google(audio2)
                    self.trigger_query_sig.emit(query_text)
            except Exception:
                if getattr(self.bar, "mode", "off") == "listen":
                    self.update_ui_sig.emit("", "off")
                    restore_system_volume()

    def safe_update_ui(self, text, mode):
        self.area.setText(text); self.bar.setMode(mode)

    def process_query(self, query):
        self.safe_update_ui("Thinking...", "think")
        self.worker = AIWorker(query)
        self.worker.finished_sig.connect(self.handle_ai_done)
        self.worker.start()

    def handle_ai_done(self, res, is_music):
        if is_music:
            self.area.setText("Playing music...")
            self.speak_and_done("Playing music.", lambda: subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', res]))
        else:
            self.area.setText(res); self.bar.setMode('speaking')
            self.speak_and_done(res, None)

    def speak_and_done(self, text, callback):
        def run():
            try:
                audio_file = os.path.join(BASE_DIR, "temp_response.mp3")
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(audio_file)
                subprocess.run(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', audio_file])
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except: pass
            if callback: callback()
            restore_system_volume()
            self.update_ui_sig.emit("", "off")
        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AskZacWindow(); win.resize(800, 480); win.show()
    sys.exit(app.exec_())