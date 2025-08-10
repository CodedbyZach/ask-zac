import sys, os, re, difflib, time, threading, subprocess, json
sys.stderr = open(os.devnull, 'w')
os.environ["JACK_NO_MSG"] = "1"
os.environ["SDL_JACK_NO_MSG"] = "1"
os.environ["ALSA_LOGLEVEL"] = "none"
os.environ["ALSA_DEBUG"] = "0"

# --- Force the same backend that worked in your test ---
os.environ.setdefault("PYWEBVIEW_GUI", "qt")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # use 'wayland' if you prefer

import openai
import speech_recognition as sr
from dotenv import load_dotenv
import requests
import webview  # pywebview

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
DICT_LANG        = "en"     # dictionaryapi.dev
ROUTER_MODEL     = "gpt-4o"
ANSWER_MODEL     = "gpt-4o"

# Second screen placement (left edge at x=1920, y=0). Adjust if needed.
SECOND_SCREEN_X  = 1920
SECOND_SCREEN_Y  = 0
WIN_WIDTH        = 900
WIN_HEIGHT       = 560
# =====================================

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
openai.api_key = OPENAI_API_KEY

# Speech setup
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
    """Return top N SerpAPI organic results as a list of {title, snippet, link, position}."""
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
    lat = float(place["latitude"])
    lon = float(place["longitude"])
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

# ---------- Dictionary helpers ----------
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
    r"^\s*what\s+are\s+antonyms?\s+(for|of)\s+(?P<term>.+?)\s*\??$",
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
    term = term.strip().strip('"\''"“”‘’.,:;!?()[]{}")
    return term

def fetch_dictionary(term: str):
    try:
        t = _clean_term(term)
        url = f"https://api.dictionaryapi.dev/api/v2/entries/{DICT_LANG}/{requests.utils.quote(t)}"
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        entry = data[0]
        word = entry.get("word", t)
        phon = ""
        for ph in entry.get("phonetics", []):
            if ph.get("text"):
                phon = ph["text"]; break
        meanings = entry.get("meanings", [])
        defs = []
        syns = set()
        ants = set()
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
        return {
            "word": word,
            "phon": phon,
            "defs": defs,
            "synonyms": sorted(list(syns))[:10],
            "antonyms": sorted(list(ants))[:10],
        }
    except Exception:
        return None

def format_dictionary_response(mode: str, term: str, info: dict) -> str:
    if not info: return "I couldn't find that in the dictionary."
    w = info["word"]
    phon = f" {info['phon']}" if info.get("phon") else ""
    if mode in ("define","spell","pronounce"):
        bits = []
        for i, (pos, definition, example) in enumerate(info["defs"][:2], start=1):
            pos_txt = f"{pos}" if pos else "def"
            if example:
                bits.append(f"{i}) {pos_txt}: {definition} — e.g., “{example}”.")
            else:
                bits.append(f"{i}) {pos_txt}: {definition}.")
        if not bits:
            bits = ["No formal definition found."]
        extra = []
        if mode == "spell":
            extra.append(f"Spelling: {w}.")
        if mode == "pronounce" and info.get("phon"):
            extra.append(f"Pronounced as {info['phon']}.")
        synbit = ""
        if info["synonyms"]:
            synbit = " Synonyms: " + ", ".join(info["synonyms"][:6]) + "."
        return f"{w}{phon}: " + " ".join(bits) + ((" " + " ".join(extra)) if extra else "") + synbit
    elif mode == "synonyms":
        if info["synonyms"]:
            return f"Synonyms of {w}:{phon} " + ", ".join(info["synonyms"][:10]) + "."
        return f"No common synonyms found for {w}."
    elif mode == "antonyms":
        if info["antonyms"]:
            return f"Antonyms of {w}:{phon} " + ", ".join(info["antonyms"][:10]) + "."
        return f"No common antonyms found for {w}."
    return "I couldn't process that dictionary request."
# -------------------------------------

# ---------- GPT helpers ----------
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
        resp = openai.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            messages=[
                {"role":"system","content":
                 "You are a strict router. Choose the single best action for the user's query.\n"
                 "Return ONLY valid JSON with keys as specified. Do NOT include natural language."},
                {"role":"user","content": json.dumps(msg)}
            ],
            max_tokens=80
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        act = (data.get("action") or "").lower()
        if act not in {"dict","weather","search","answer"}:
            return {"action":"answer"}
        return data
    except Exception:
        return {"action":"answer"}

def ask_openai_answer(prompt):
    try:
        resp = openai.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=0,
            messages=[
                {"role": "system",
                 "content": ("You are a careful assistant. If you are not confident you know the answer, "
                             "or the answer depends on up-to-date web info you don't have, reply EXACTLY with "
                             f"{UNCERTAIN_TOKEN} and nothing else.")},
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
    try:
        msg = json.dumps(summary_dict)
        resp = openai.chat.completions.create(
            model=ANSWER_MODEL,
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
    try:
        payload = {"query": query, "results": results}
        resp = openai.chat.completions.create(
            model=ANSWER_MODEL,
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
    except Exception:
        return UNCERTAIN_TOKEN
# -----------------------------------

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

# ===================== UI (pywebview + HTML) =====================
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AskZac</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.17/400.css">
<script src="https://cdn.tailwindcss.com"></script>
<style>
  :root { color-scheme: dark; }
  body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji"; background: radial-gradient(1200px 600px at 20% -20%, #1f2937 0%, #0b1020 60%, #04070e 100%); }
  .glass { background: rgba(255,255,255,0.06); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.06); }
  .msg { animation: fadeIn .15s ease-out; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(4px);} to { opacity:1; transform: translateY(0);} }
  .badge { font-size:.70rem; padding:2px 8px; border-radius:9999px; border:1px solid rgba(255,255,255,.18); }
  .scrollbar::-webkit-scrollbar{ width:8px; } .scrollbar::-webkit-scrollbar-thumb{ background:rgba(255,255,255,0.15); border-radius:9999px; }
</style>
</head>
<body class="text-gray-100">
  <div class="max-w-3xl mx-auto p-4 min-h-screen flex flex-col">
    <header class="flex items-center gap-3 py-2">
      <div class="h-9 w-9 rounded-2xl bg-indigo-500/70 grid place-items-center shadow-lg">🤖</div>
      <div>
        <h1 class="text-lg font-semibold tracking-tight">AskZac</h1>
        <p class="text-[11px] text-gray-300/80">Say “GPT, &lt;your message&gt;” — always listening.</p>
      </div>
      <div class="ml-auto flex items-center gap-2">
        <span class="badge glass">Qt</span>
      </div>
    </header>

    <div id="chat" class="flex-1 glass rounded-3xl p-4 overflow-y-auto scrollbar">
      <div class="text-sm text-gray-300/80">Listening… (say: “GPT, …”)</div>
    </div>

    <div class="mt-3 glass rounded-2xl p-3">
      <form id="composer" class="flex items:end gap-2">
        <textarea id="input" rows="2" placeholder="Type here and press Enter"
          class="flex-1 bg-transparent outline-none resize-none p-2 text-sm"></textarea>
        <button id="sendBtn" class="px-4 py-2 rounded-xl bg-indigo-500 hover:bg-indigo-600 transition disabled:opacity-50">Send</button>
      </form>
      <div class="text-[11px] text-gray-400/80 mt-1">Shortcuts: “/” to focus, Enter to send.</div>
    </div>
  </div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const composer = document.getElementById('composer');

function addMsg(role, text, meta) {
  const wrap = document.createElement('div');
  const isUser = role === 'user';
  wrap.className = 'msg my-2 flex ' + (isUser ? 'justify-end' : 'justify-start');

  const bubble = document.createElement('div');
  bubble.className = 'max-w-[85%] rounded-2xl px-4 py-3 ' + (isUser ? 'bg-indigo-600/80' : 'bg-white/5');
  bubble.innerText = text;
  wrap.appendChild(bubble);

  if (meta && meta.route) {
    const badge = document.createElement('span');
    const colors = {
      dict: 'bg-amber-500/20 text-amber-200',
      weather: 'bg-sky-500/20 text-sky-200',
      search: 'bg-violet-500/20 text-violet-200',
      answer: 'bg-emerald-500/20 text-emerald-200',
      system: 'bg-white/10 text-white/70'
    };
    badge.className = 'badge ml-2 ' + (colors[meta.route] || 'bg-white/10 text-white/70');
    badge.textContent = meta.route;
    wrap.appendChild(badge);
  }

  chat.appendChild(wrap);

  if (meta && meta.searchResults && Array.isArray(meta.searchResults) && meta.searchResults.length) {
    const list = document.createElement('div');
    list.className = 'ml-2 mt-1 space-y-1';
    meta.searchResults.forEach(r => {
      const a = document.createElement('a');
      a.href = r.link; a.target = '_blank'; a.rel = 'noreferrer';
      a.className = 'block text-xs text-violet-200 hover:underline';
      a.textContent = `• ${r.title}` + (r.snippet ? ` — ${r.snippet}` : '');
      list.appendChild(a);
    });
    chat.appendChild(list);
  }

  chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
}

window.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== input) {
    e.preventDefault();
    input.focus();
  }
});

composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  addMsg('user', q);
  input.value = '';
  sendBtn.disabled = true;
  if (window.pywebview && window.pywebview.api && window.pywebview.api.send_query) {
    window.pywebview.api.send_query(q).finally(() => { sendBtn.disabled = false; });
  } else {
    addMsg('bot', 'Backend not ready', {route:'system'});
    sendBtn.disabled = false;
  }
});

// Called by Python to inject a message
function py_add_msg(role, text, meta_json) {
  let meta = null;
  try { meta = meta_json ? JSON.parse(meta_json) : null; } catch {}
  addMsg(role, text, meta || {});
}
</script>
</body>
</html>
"""

# ===================== App wrapper =====================
class AskZacApp:
    def __init__(self, window: webview.Window):
        self.window = window
        self.listening_enabled = True

    def ui_add(self, role, text, meta=None):
        meta_s = json.dumps(meta or {})
        def esc(s): return s.replace("\\","\\\\").replace("`","\\`").replace("\n","\\n").replace("'","\\'")
        js = f"py_add_msg('{role}','{esc(text)}','{esc(meta_s)}');"
        try:
            self.window.evaluate_js(js)
        except Exception:
            pass

    def start_mic_loop(self):
        threading.Thread(target=self.continuous_loop, daemon=True).start()

    def continuous_loop(self):
        while True:
            if not self.listening_enabled:
                time.sleep(0.05); continue
            self.ui_add("bot", "Listening… (say: 'GPT, …')", {"route":"system"})
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recognizer.listen(source, timeout=WAKE_TIMEOUT_S, phrase_time_limit=WAKE_PHRASE_MAXS)
                try:
                    heard = recognizer.recognize_google(audio, language="en-US")
                    self.ui_add("bot", f"You said: {heard}", {"route":"system"})
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    self.ui_add("bot", f"ASR error: {e}", {"route":"system"})
                    continue

                if heard.strip().lower() == "q":
                    self.ui_add("bot", "Goodbye!", {"route":"system"})
                    os._exit(0)

                if contains_wake(heard):
                    remainder = split_after_wake(heard)
                    if remainder:
                        self.ask_and_speak(remainder)
                    else:
                        self.ui_add("bot", "Heard 'GPT'. What's up?", {"route":"system"})
                        try:
                            with mic as source:
                                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                                audio2 = recognizer.listen(source, timeout=QUESTION_TIMEOUT, phrase_time_limit=QUESTION_MAXS)
                            try:
                                q = recognizer.recognize_google(audio2, language="en-US").strip()
                                if q.lower() == "q":
                                    self.ui_add("bot", "Goodbye!", {"route":"system"})
                                    os._exit(0)
                                if q:
                                    self.ask_and_speak(q)
                            except sr.UnknownValueError:
                                self.ui_add("bot", "Didn't catch that—try again with 'GPT, …'", {"route":"system"})
                            except sr.RequestError as e:
                                self.ui_add("bot", f"ASR error: {e}", {"route":"system"})
                        except sr.WaitTimeoutError:
                            self.ui_add("bot", "Timed out—say 'GPT, …' again.", {"route":"system"})
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.ui_add("bot", f"Mic error: {e}", {"route":"system"})
                continue

    # ---------- Routing + actions ----------
    def ask_and_speak(self, query):
        self.ui_add("user", query)
        self.ui_add("bot", "Thinking...", {"route":"system"})

        route = ask_router(query)
        action = (route.get("action") or "answer").lower()

        if action == "dict":
            term = route.get("term")
            mode, parsed_term = parse_dictionary_query(query)
            term = term or parsed_term
            info = fetch_dictionary(term)
            if info:
                resp = format_dictionary_response(mode, term, info)
                self.ui_add("bot", resp, {"route":"dict"})
                self.listening_enabled = False
                speak_openai(resp, on_done=lambda: self._resume_after_tts())
                return
            else:
                action = "search"  # fallback

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
                self.ui_add("bot", styled, {"route":"weather"})
                self.listening_enabled = False
                speak_openai(styled, on_done=lambda: self._resume_after_tts())
                return
            except Exception as e:
                self.ui_add("bot", f"Weather fetch failed: {e}. Continuing…", {"route":"system"})

        if action == "search":
            q = route.get("query") or query
            results = web_search_structured(q, n=SEARCH_RESULTS_N)
            synth = ask_openai_from_search(q, results)
            if synth == UNCERTAIN_TOKEN:
                fallback = (results[0]["title"] + ": " + results[0]["snippet"]) if results else "No results."
                self.ui_add("bot", fallback, {"route":"search", "searchResults": results})
                to_say = fallback
            else:
                self.ui_add("bot", synth, {"route":"search", "searchResults": results})
                to_say = synth
            self.listening_enabled = False
            speak_openai(to_say, on_done=lambda: self._resume_after_tts())
            return

        # Default direct answer
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
                    self.ui_add("bot", styled, {"route":"weather"})
                    self.listening_enabled = False
                    speak_openai(styled, on_done=lambda: self._resume_after_tts())
                    return
                except Exception:
                    pass
            results = web_search_structured(query, n=SEARCH_RESULTS_N)
            synth = ask_openai_from_search(query, results)
            if synth == UNCERTAIN_TOKEN:
                fallback = (results[0]["title"] + ": " + results[0]["snippet"]) if results else "No results."
                self.ui_add("bot", fallback, {"route":"search", "searchResults": results})
                to_say = fallback
            else:
                self.ui_add("bot", synth, {"route":"search", "searchResults": results})
                to_say = synth
            self.listening_enabled = False
            speak_openai(to_say, on_done=lambda: self._resume_after_tts())
            return

        self.ui_add("bot", answer, {"route":"answer"})
        self.listening_enabled = False
        speak_openai(answer, on_done=lambda: self._resume_after_tts())

    def _resume_after_tts(self):
        self.listening_enabled = True

# JS API object for the page
class UIApi:
    def __init__(self, app: AskZacApp):
        self.app = app
    def send_query(self, text: str):
        self.app.ask_and_speak((text or "").strip())
        return {"ok": True}

# ===================== Boot =====================
def on_webview_ready():
    try:
        window.move(SECOND_SCREEN_X, SECOND_SCREEN_Y)
        window.resize(WIN_WIDTH, WIN_HEIGHT)
    except Exception:
        pass
    quick_calibrate(0.6)
    app.ui_add("bot", "AskZac: Say 'GPT, <your message>' or type. ('q' to quit)", {"route":"system"})
    app.start_mic_loop()

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY in .env"); sys.exit(1)

    # Create API first, pass to window, then wire in the app
    api = UIApi(None)

    window = webview.create_window(
        title="AskZac",
        html=INDEX_HTML,
        width=WIN_WIDTH,
        height=WIN_HEIGHT,
        resizable=True,
        frameless=False,
        easy_drag=False,
        text_select=True,
        zoomable=True,
        js_api=api,   # << v4 style
    )

    # hook the app after window is created so we can call evaluate_js
    app = AskZacApp(window)
    # swap in the real app instance for JS API
    api.app = app

    # Force Qt like your working test
    webview.start(on_webview_ready, window, debug=False, gui='qt')