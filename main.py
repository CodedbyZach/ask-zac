import os, sys, re, threading, subprocess, json, math, requests, time, io
from google import genai
from google.genai import types
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QTime, QObject
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QPainterPath
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QSizePolicy

load_dotenv()
WAKE_CANONICAL = "gpt"
WAKE_TIMEOUT_S = 6.0
QUESTION_TIMEOUT = 8.0
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
TTS_TEMPO = 1.15

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

USER_ZIP = os.getenv("USER_ZIP", "90210")
MUSIC_DIR = os.getenv("MUSIC_DIR", "Music")
MUSIC_EXTS = {".mp3", ".wav"}
MONITOR_NUMBER = int(os.getenv("MONITOR_NUMBER", "0"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, os.getenv("SOUNDS_DIR", "Sounds"))
ALARM_SOUND = os.getenv("ALARM_SOUND", "alarm-1.mp3")
ALARM_PATH = os.path.join(SOUNDS_DIR, ALARM_SOUND)
SONG_STOPWORDS = {"play", "the", "song", "track", "please", "me", "a", "can", "you", "some"}
WAKE_VOLUME_DIM_FACTOR = 0.6
WAKE_VOLUME_DIM_MIN_PERCENT = int(os.getenv("WAKE_VOLUME_DIM_MIN_PERCENT", "20"))
MUSIC_DUCK_PERCENT = 20
STOP_PHRASES = {"stop", "stop music", "stop the music", "stop song", "stop the song"}
API_HEALTH_CHECK_INTERVAL_S = 30
TIMER_LABEL_STYLE = "color:#ff9500; font-size:72px; font-weight:bold;"
SONG_LABEL_STYLE = "color:#ff9500; font-size:26px; font-weight:bold;"

app_exiting = threading.Event()

if not os.path.isfile(ALARM_PATH):
    print(f"Warning: ALARM_SOUND '{ALARM_SOUND}' was not found in {SOUNDS_DIR}")

def _get_system_volume_percent():
    if sys.platform != "linux":
        return None
    try:
        out = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], capture_output=True, text=True, timeout=2).stdout
        match = re.search(r'(\d+)%', out)
        return int(match.group(1)) if match else None
    except Exception:
        return None

def _set_system_volume_percent(percent):
    if sys.platform == "linux":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(percent)}%"], capture_output=True)

def _find_sink_input_for_pid(pid):
    try:
        out = subprocess.run(["pactl", "list", "sink-inputs"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return None
    current_id = None
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r'Sink Input #(\d+)', line)
        if m:
            current_id = m.group(1)
        elif current_id and line.startswith("application.process.id"):
            m2 = re.search(r'"(\d+)"', line)
            if m2 and int(m2.group(1)) == pid:
                return current_id
    return None

def _set_sink_input_volume(sink_input_id, percent):
    if sink_input_id is None or sys.platform != "linux":
        return
    subprocess.run(["pactl", "set-sink-input-volume", sink_input_id, f"{int(percent)}%"], capture_output=True)

class ExitSignal(QObject):
    exit_sig = pyqtSignal()

active_players = []
active_players_lock = threading.Lock()

def register_player(proc):
    with active_players_lock:
        active_players.append(proc)

def unregister_player(proc):
    with active_players_lock:
        if proc in active_players:
            active_players.remove(proc)

def kill_active_players():
    with active_players_lock:
        procs = list(active_players)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass

def stdin_exit_listener(exit_signal):
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "exit":
            app_exiting.set()
            kill_active_players()
            print("Exited.")
            exit_signal.exit_sig.emit()
            break

class AIWorker(QThread):
    finished_sig = pyqtSignal(str, bool)
    def __init__(self, query):
        super().__init__()
        self.query = query

    @staticmethod
    def _clean_song_query(text):
        words = re.findall(r"[a-z0-9']+", text.lower())
        words = [w for w in words if w not in SONG_STOPWORDS]
        return " ".join(words)

    @staticmethod
    def _find_song(query, root_dir):
        tokens = query.split()
        if not tokens:
            return None
        for r, _, files in os.walk(root_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() not in MUSIC_EXTS:
                    continue
                name = f.lower()
                if all(t in name for t in tokens):
                    return os.path.join(r, f)
        return None

    def run(self):
        mq = None
        if self.query.lower().startswith("play "):
            mq = self._clean_song_query(self.query)
        if mq:
            root_dir = os.path.join(BASE_DIR, MUSIC_DIR)
            song_path = self._find_song(mq, root_dir)
            if song_path:
                self.finished_sig.emit(song_path, True)
            else:
                self.finished_sig.emit(f"Song '{mq}' not found in the music folder.", False)
            return
        try:
            sys_instr = f"You are AskZac. Do not use MarkDown format as it will not display correctly, nor use quotes unless you want it to read as inches. Do not use any unicode symbols or emoji, plain text only. Shorten your responses a bit. Respond in 1-2 sentences. Use Google Search. The user's location zip code is {USER_ZIP}. If a timer is requested, respond ONLY with JSON: {{\"timer_seconds\": N}}."
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
    refresh_status_sig = pyqtSignal()
    play_song_sig = pyqtSignal(str)
    api_status_sig = pyqtSignal(bool)
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint); self.setStyleSheet("background:#0a1026;")
        c = QWidget(); l = QVBoxLayout(c); self.setCentralWidget(c)

        top_row = QHBoxLayout()
        self.clock = QLabel(); self.clock.setStyleSheet("color:white; font-size:20px;")
        self.api_status_label = QLabel("Status: …"); self.api_status_label.setStyleSheet("color:#888; font-size:14px;")
        top_row.addStretch(); top_row.addWidget(self.clock); top_row.addStretch(); top_row.addWidget(self.api_status_label)
        l.addLayout(top_row)

        self.now_playing_label = QLabel(); self.now_playing_label.setAlignment(Qt.AlignCenter)
        self.now_playing_label.setStyleSheet("color:#1a74f0; font-size:20px; font-weight:bold;")
        self.now_playing_label.setWordWrap(True)
        self.now_playing_label.hide(); l.addWidget(self.now_playing_label)

        self.timer_label = QLabel(); self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setWordWrap(True)
        self.timer_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.timer_label.hide(); l.addWidget(self.timer_label)

        self.area = QTextEdit(); self.area.setReadOnly(True); self.area.setStyleSheet("color:white; border:none; font-size:28px;"); l.addWidget(self.area)
        self.bar = WakeBar(); l.addWidget(self.bar)
        self.tm = QTimer(); self.tm.timeout.connect(lambda: self.clock.setText(QTime.currentTime().toString("h:mm AP"))); self.tm.start(1000)
        self.timer_countdown = QTimer(self); self.timer_countdown.timeout.connect(self._tick_timer_countdown)
        self.timer_remaining = 0
        self.music_proc = None
        self.current_song = None
        self._cached_volume = None
        self.trigger_query_sig.connect(self.process_query)
        self.update_ui_sig.connect(self.safe_update_ui)
        self.refresh_status_sig.connect(self._refresh_status_labels)
        self.play_song_sig.connect(self.play_song)
        self.api_status_sig.connect(self._set_api_status)
        threading.Thread(target=self.mic_loop, daemon=True).start()
        threading.Thread(target=self._api_health_loop, daemon=True).start()

    def mic_loop(self):
        r, m = sr.Recognizer(), sr.Microphone()
        r.pause_threshold = 0.5
        while not app_exiting.is_set():
            try:
                with m as s:
                    r.adjust_for_ambient_noise(s, 0.3)
                    audio = r.listen(s, timeout=WAKE_TIMEOUT_S, phrase_time_limit=1.5)
                    if WAKE_CANONICAL not in r.recognize_google(audio).lower():
                        continue
                    self._dim_for_listening()
                    self.update_ui_sig.emit("Listening...", "listen")
                    audio2 = r.listen(s, timeout=QUESTION_TIMEOUT, phrase_time_limit=10.0)
                query_text = r.recognize_google(audio2)
                self.trigger_query_sig.emit(query_text)
            except Exception:
                if app_exiting.is_set():
                    return
                if getattr(self.bar, "mode", "off") == "listen":
                    self.update_ui_sig.emit("", "off")
                    self._restore_volume()

    def safe_update_ui(self, text, mode):
        self.area.setText(text); self.bar.setMode(mode)

    def process_query(self, query):
        if query.strip().lower().rstrip(".!") in STOP_PHRASES:
            was_playing = self.current_song is not None
            self.stop_song()
            self._restore_volume()
            self.safe_update_ui("Stopped." if was_playing else "Nothing is playing.", "off")
            return
        self.safe_update_ui("Thinking...", "think")
        self.worker = AIWorker(query)
        self.worker.finished_sig.connect(self.handle_ai_done)
        self.worker.start()

    def _parse_song_meta(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        if " - " in name:
            artist, title = name.split(" - ", 1)
        else:
            artist, title = "Unknown Artist", name
        return {"artist": artist.strip(), "title": title.strip()}

    def play_song(self, path):
        self.stop_song()
        meta = self._parse_song_meta(path)
        self.current_song = meta
        self._refresh_status_labels()
        proc = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path])
        self.music_proc = proc
        register_player(proc)
        def watch():
            proc.wait()
            unregister_player(proc)
            if app_exiting.is_set():
                return
            if self.music_proc is proc:
                self.music_proc = None
                self.current_song = None
                self.refresh_status_sig.emit()
        threading.Thread(target=watch, daemon=True).start()

    def stop_song(self):
        proc = self.music_proc
        if proc and proc.poll() is None:
            proc.terminate()
        self.music_proc = None
        self.current_song = None
        self._refresh_status_labels()

    def handle_ai_done(self, res, is_music):
        if is_music:
            meta = self._parse_song_meta(res)
            msg = f"Playing {meta['title']} by {meta['artist']}."
            self.area.setText(msg); self.bar.setMode('speaking')
            self.speak_and_done(msg, lambda: self.play_song_sig.emit(res))
            return
        seconds = self._parse_timer_seconds(res)
        if seconds is not None:
            self.start_timer(seconds)
        else:
            self.area.setText(res); self.bar.setMode('speaking')
            self.speak_and_done(res, None)

    @staticmethod
    def _parse_timer_seconds(text):
        match = re.search(r'\{[^{}]*"timer_seconds"\s*:\s*(\d+(?:\.\d+)?)[^{}]*\}', text)
        if not match:
            return None
        try:
            return json.loads(match.group(0)).get("timer_seconds")
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _format_duration(seconds):
        seconds = int(round(seconds))
        minutes, secs = divmod(seconds, 60)
        parts = []
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        return " and ".join(parts)

    def start_timer(self, seconds):
        msg = f"Timer set for {self._format_duration(seconds)}."
        self.area.setText(msg); self.bar.setMode('speaking')
        self.speak_and_done(msg, None)
        self.timer_remaining = int(round(seconds))
        self.timer_countdown.start(1000)
        self._refresh_status_labels()

    def _tick_timer_countdown(self):
        self.timer_remaining -= 1
        if self.timer_remaining <= 0:
            self.timer_countdown.stop()
            self.timer_done()
        else:
            self._refresh_status_labels()

    def _refresh_status_labels(self):
        timer_active = self.timer_countdown.isActive()
        song_text = None
        if self.current_song:
            song_text = f"{self.current_song['title']} — {self.current_song['artist']}"

        if timer_active:
            minutes, secs = divmod(max(self.timer_remaining, 0), 60)
            self.timer_label.setStyleSheet(TIMER_LABEL_STYLE)
            self.timer_label.setText(f"{minutes:02d}:{secs:02d}")
            self.timer_label.show()
        elif song_text:
            self.timer_label.setStyleSheet(SONG_LABEL_STYLE)
            self.timer_label.setText(song_text)
            self.timer_label.show()
        else:
            self.timer_label.hide()

        if timer_active and song_text:
            self.now_playing_label.setText(song_text)
            self.now_playing_label.show()
        else:
            self.now_playing_label.hide()

    def timer_done(self):
        self._refresh_status_labels()
        if not os.path.isfile(ALARM_PATH):
            msg = f"Alarm sound '{ALARM_SOUND}' was not found in the Sounds folder."
            self.area.setText(msg); self.bar.setMode('speaking')
            self.speak_and_done(msg, None)
            return
        self.area.setText("Time's up!"); self.bar.setMode('speaking')
        def run():
            proc = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', ALARM_PATH])
            register_player(proc)
            proc.wait()
            unregister_player(proc)
            if app_exiting.is_set():
                return
            self.update_ui_sig.emit("", "off")
        threading.Thread(target=run, daemon=True).start()

    def speak_and_done(self, text, callback):
        def run():
            music_proc = self.music_proc
            duck_sink_id = None
            if music_proc and music_proc.poll() is None:
                duck_sink_id = _find_sink_input_for_pid(music_proc.pid)
                if duck_sink_id:
                    _set_sink_input_volume(duck_sink_id, MUSIC_DUCK_PERCENT)
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                proc = subprocess.Popen(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet',
                     '-af', f'atempo={TTS_TEMPO}', '-f', 'mp3', 'pipe:0'],
                    stdin=subprocess.PIPE
                )
                register_player(proc)
                proc.communicate(input=buf.getvalue())
                unregister_player(proc)
            except: pass
            if app_exiting.is_set():
                return
            if duck_sink_id and music_proc.poll() is None:
                _set_sink_input_volume(duck_sink_id, 100)
            if callback: callback()
            self._restore_volume()
            self.update_ui_sig.emit("", "off")
        threading.Thread(target=run, daemon=True).start()

    def _dim_for_listening(self):
        if self._cached_volume is not None:
            return
        current = _get_system_volume_percent()
        if current is None or current <= WAKE_VOLUME_DIM_MIN_PERCENT:
            return
        self._cached_volume = current
        _set_system_volume_percent(round(current * WAKE_VOLUME_DIM_FACTOR))

    def _restore_volume(self):
        cached = self._cached_volume
        if cached is None:
            return
        self._cached_volume = None
        threading.Thread(target=lambda: _set_system_volume_percent(cached), daemon=True).start()

    def _api_health_loop(self):
        while not app_exiting.is_set():
            try:
                client.models.list(config={"page_size": 1})
                is_up = True
            except Exception:
                is_up = False
            if app_exiting.is_set():
                return
            self.api_status_sig.emit(is_up)
            app_exiting.wait(API_HEALTH_CHECK_INTERVAL_S)

    def _set_api_status(self, is_up):
        if is_up:
            self.api_status_label.setText("● Status: Up")
            self.api_status_label.setStyleSheet("color:#2ecc71; font-size:14px;")
        else:
            self.api_status_label.setText("● Status: Down")
            self.api_status_label.setStyleSheet("color:#e74c3c; font-size:14px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    screens = app.screens()
    screen = screens[MONITOR_NUMBER] if 0 <= MONITOR_NUMBER < len(screens) else screens[0]
    geometry = screen.geometry()

    win = AskZacWindow()
    win.setFixedSize(800, 480)
    win.move(geometry.left(), geometry.top())
    win.show()
    win.windowHandle().setScreen(screen)
    win.move(geometry.left(), geometry.top())

    exit_signal = ExitSignal()
    exit_signal.exit_sig.connect(app.quit)
    threading.Thread(target=stdin_exit_listener, args=(exit_signal,), daemon=True).start()

    print("Running AskZac.")
    sys.exit(app.exec_())