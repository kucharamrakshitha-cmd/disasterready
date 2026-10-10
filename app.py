"""DisasterReady: learn, certify, report by voice, respond.

Voice reports (Omi) -> memory (Qdrant) -> agents (Lyzr) -> certified rescue team.
"""
import base64, hashlib, hmac, json, os, re, secrets, time, uuid
from typing import Any
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
LYZR_API_KEY = os.environ["LYZR_API_KEY"]
TRIAGE_ID = os.environ["TRIAGE_AGENT_ID"]
PLANNER_ID = os.environ["PLANNER_AGENT_ID"]
SOS_ID = os.environ["SOS_AGENT_ID"]
MANAGER_ID = os.getenv("MANAGER_AGENT_ID", "")
ALLOCATOR_ID = os.getenv("ALLOCATOR_AGENT_ID", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
OMI_APP_ID = os.getenv("OMI_APP_ID", "")
OMI_APP_KEY = os.getenv("OMI_APP_KEY", "")

DEFAULT_OP_PASSWORD = "ready2026"
OPERATOR_EMAIL = os.getenv("OPERATOR_EMAIL", "control@disasterready.demo")
OPERATOR_PASSWORD = os.getenv("OPERATOR_PASSWORD", DEFAULT_OP_PASSWORD)
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)

NOT_EMERGENCY = ("DisasterReady did not hear an emergency in your message. "
                 "If you are in danger, say what happened, where you are, how "
                 "many people are affected and any injuries. If a life is at "
                 "risk, call 112.")
DUPLICATE_SCORE = 0.78  # similarity above this means "same incident"
DEMO_USERS = {"unknown", "test-user", "web-demo"}
AREAS = ["Malakpet", "Dilsukhnagar", "Charminar", "Afzal Gunj", "Mehdipatnam",
         "LB Nagar", "Secunderabad", "Musheerabad", "Kukatpally", "Uppal",
         "Begumpet", "Tolichowki", "Ameerpet", "Madhapur", "Falaknuma"]

embedder = TextEmbedding("BAAI/bge-small-en-v1.5")  # 384 dimensions
qdrant = QdrantClient(url=os.environ["QDRANT_URL"],
                      api_key=os.environ["QDRANT_API_KEY"], timeout=30)
INCIDENTS = {}  # live copy for the dashboard, also saved in Qdrant
PEOPLE = {}     # learners and rescuers, also saved in Qdrant
FILTERED = []   # recent messages that were not emergencies


# ---------- courses ----------
COURSES = [
    {"id": "basics", "level": 0, "requires": None,
     "title": "Disaster Ready Basics",
     "badge": "Disaster Ready Citizen",
     "summary": "For everyone. What to prepare at home and what to do in a "
                "flood, fire or earthquake.",
     "lessons": [
         {"title": "Make a plan and pack a kit", "points": [
             "Agree on a family meeting point and write key phone numbers on paper, not only in your phone.",
             "Pack a go-bag: drinking water (about 3 litres per person per day), dry food, torch, power bank, first aid kit, regular medicines, a whistle and some cash.",
             "Keep copies of ID and important documents in a waterproof cover.",
             "Know the numbers in India: 112 for any emergency, 101 for fire, 108 for an ambulance."]},
         {"title": "Floods", "points": [
             "Move to higher ground early. Do not wait for the water to reach you.",
             "Never walk or drive through moving water. Even ankle-deep fast water can knock an adult over.",
             "Switch off electricity at the mains if you can do it safely, and stay away from fallen power lines.",
             "Avoid contact with flood water and drink only boiled or treated water."]},
         {"title": "Fire", "points": [
             "Get out, stay out, and call 101 or 112. Never go back for belongings.",
             "Smoke rises, so stay low and crawl under it. Cover your nose and mouth with a cloth.",
             "Use the stairs, never the lift. Touch a door with the back of your hand before opening it; if it is hot, find another way.",
             "If your clothes catch fire: stop, drop to the ground and roll."]},
         {"title": "Earthquake", "points": [
             "Drop to the ground, take cover under a sturdy table, and hold on until the shaking stops.",
             "Stay away from windows, shelves and anything that can fall.",
             "If you are outside, move to an open area away from buildings and wires.",
             "After the shaking stops, leave by the stairs and expect aftershocks."]},
         {"title": "Calling for help clearly", "points": [
             "Call 112 first when life is in danger.",
             "Say four things: what happened, where (with a landmark), how many people, and what injuries.",
             "With DisasterReady you can speak the same four things into the Omi app. The control room gets a structured report, the nearest shelters and a certified team within seconds.",
             "Stay reachable and follow the instructions you are given."]},
     ],
     "quiz": [
         {"q": "You are indoors when an earthquake starts. What do you do?",
          "options": ["Run down the stairs immediately", "Drop, take cover and hold on", "Stand next to a window to watch", "Use the lift to get out fast"], "answer": 1},
         {"q": "A room is filling with smoke. How do you move?",
          "options": ["Walk upright quickly", "Stay low and crawl under the smoke", "Wait in the room for it to clear", "Open every window first"], "answer": 1},
         {"q": "Fast flood water is flowing across the road ahead. What is the safe choice?",
          "options": ["Drive through quickly", "Walk through holding hands", "Turn back and go to higher ground", "Wait in the car in the water"], "answer": 2},
         {"q": "Which single number works for any emergency in India?",
          "options": ["100", "112", "139", "1947"], "answer": 1},
         {"q": "What makes an emergency report most useful?",
          "options": ["A long story of how it started", "What happened, where, how many people and injuries", "Only your name and address", "A photo with no location"], "answer": 1},
     ]},
    {"id": "level1", "level": 1, "requires": "basics",
     "title": "Rescuer Level 1: Community First Responder",
     "badge": "Level 1 Rescuer",
     "summary": "Your first rescuer certificate. Stay safe on scene, give "
                "basic first aid and move people to shelter.",
     "lessons": [
         {"title": "Your safety comes first", "points": [
             "Look for hazards before you step in: moving water, fire, live electricity, unstable walls.",
             "Do not become a second casualty. If the scene is not safe, keep people back and report it.",
             "Work in pairs and tell the control room where you are going."]},
         {"title": "Basic first aid", "points": [
             "Severe bleeding: press firmly on the wound with a clean cloth and keep pressing. If it soaks through, add another cloth on top.",
             "Burns: cool under cool running water for about 20 minutes. Remove rings and watches, then cover loosely with a clean cloth. No ice, no toothpaste, and do not burst blisters.",
             "Unconscious but breathing: place the person on their side in the recovery position and watch their breathing.",
             "Not breathing: shout for help, call 108 or 112, and start CPR if you are trained."]},
         {"title": "Moving people to shelter", "points": [
             "Help children, older people, pregnant women and people with disabilities first.",
             "Keep families together and count people when you leave and when you arrive.",
             "Register everyone at the shelter so nobody is reported missing by mistake."]},
     ],
     "quiz": [
         {"q": "You arrive first at a flooded street with a fallen power line. What do you do first?",
          "options": ["Wade in to reach people", "Check hazards, keep people back and report it", "Move the wire with a stick", "Start filming for evidence"], "answer": 1},
         {"q": "A person has a deep cut that is bleeding heavily. What helps most?",
          "options": ["Wash it for ten minutes", "Firm direct pressure with a clean cloth", "Apply turmeric and leave it", "Lift the cloth every minute to check"], "answer": 1},
         {"q": "What is the right first aid for a burn?",
          "options": ["Ice directly on the skin", "Toothpaste or butter", "Cool running water for about 20 minutes", "Burst the blisters"], "answer": 2},
         {"q": "Someone is unconscious but breathing normally. What do you do?",
          "options": ["Sit them upright on a chair", "Give them water to drink", "Place them in the recovery position and monitor", "Leave them and move on"], "answer": 2},
         {"q": "Who do you help to the shelter first?",
          "options": ["Whoever shouts loudest", "Children, older people and people who cannot move easily", "People with the most luggage", "Your own friends"], "answer": 1},
     ]},
    {"id": "level2", "level": 2, "requires": "level1",
     "title": "Rescuer Level 2: Search and Rescue Support",
     "badge": "Level 2 Rescuer",
     "summary": "Support water rescue and evacuations, sort casualties by "
                "urgency and report clearly.",
     "lessons": [
         {"title": "Water rescue: reach, throw, row, go", "points": [
             "Reach with a pole or rope from a safe place first. If you cannot reach, throw something that floats.",
             "Use a boat if one is available. Entering the water yourself is the last resort and only with training and a life jacket.",
             "Never tie a rope around your own body in moving water."]},
         {"title": "Fire and collapsed buildings", "points": [
             "Do not enter a burning or collapsed building. That is the job of the fire service.",
             "Help people who are already out, keep a safe distance, and keep the access road clear for fire engines.",
             "Call out and listen for trapped people, then report exactly where the sounds came from."]},
         {"title": "Sorting casualties by urgency", "points": [
             "People who can walk are usually lower priority. Move them to a safe waiting area.",
             "Top priority: not breathing normally, severe bleeding, or not responding.",
             "Control severe bleeding straight away, then tell the medical team how many people are in each group."]},
         {"title": "Clear communication", "points": [
             "Keep messages short: who you are, where you are, what you see, what you need.",
             "Repeat back instructions so the control room knows you heard them correctly.",
             "Update the control room every time the situation changes."]},
     ],
     "quiz": [
         {"q": "A person is struggling in flood water five metres away. What do you try first?",
          "options": ["Jump in and swim to them", "Reach or throw something that floats from a safe place", "Tie a rope to your waist and wade in", "Wait for the water to drop"], "answer": 1},
         {"q": "A building has partly collapsed and you hear a voice inside. What do you do?",
          "options": ["Climb in alone to pull them out", "Report the exact location and keep the area clear", "Start removing support beams", "Ignore it until morning"], "answer": 1},
         {"q": "Which casualty is the highest priority?",
          "options": ["Someone walking with a sprained wrist", "Someone awake with a small cut", "Someone who is not responding and bleeding heavily", "Someone asking for water"], "answer": 2},
         {"q": "What is the best radio or phone message?",
          "options": ["It is very bad here, send everyone", "Team two at Tolichowki school, 30 people on first floor, need two boats", "Hello, can you hear me, hello", "I will call back later"], "answer": 1},
         {"q": "When do you update the control room?",
          "options": ["Only at the end of the day", "Every time the situation changes", "Only if they call you", "Never, to keep the line free"], "answer": 1},
     ]},
    {"id": "level3", "level": 3, "requires": "level2",
     "title": "Rescuer Level 3: Incident Team Lead",
     "badge": "Level 3 Team Lead",
     "summary": "Lead a rescue team: assign roles, brief the team, manage "
                "shelters and hand over to professional services.",
     "lessons": [
         {"title": "One lead, clear roles", "points": [
             "Every incident needs one team lead, and every member reports to one person.",
             "A leader can manage about five people well. Split larger groups into smaller teams.",
             "There is one line of reporting to the control room, through the lead."]},
         {"title": "Brief the team before moving", "points": [
             "Cover the situation, the task, the hazards, each person's role, how you will communicate and where you will regroup.",
             "Pair people up, and count the team in and out of the scene."]},
         {"title": "Shelters and resources", "points": [
             "Match people to shelters by capacity and facilities, for example medical aid for the injured.",
             "Track how many places are left and ask for more help early, before you run out."]},
         {"title": "Handover and debrief", "points": [
             "When fire, medical or disaster-response services arrive, they take command. Give them a short briefing and follow their lead.",
             "Account for every team member before leaving.",
             "Hold a short debrief, make sure people rest, and record what happened."]},
     ],
     "quiz": [
         {"q": "How many people can one leader manage well?",
          "options": ["About five", "About twenty", "As many as arrive", "Only one"], "answer": 0},
         {"q": "What must a team briefing include?",
          "options": ["Only the address", "Situation, task, hazards, roles, communication and regroup point", "The lead's phone number only", "Nothing, to save time"], "answer": 1},
         {"q": "The fire service arrives at your incident. What happens to command?",
          "options": ["You stay in charge", "They take command and you brief them", "Both teams work separately", "Your team leaves at once without a word"], "answer": 1},
         {"q": "A shelter has 20 places left and you have 60 people to move. What do you do?",
          "options": ["Send all 60 there anyway", "Split them across shelters by capacity and ask for more help early", "Keep everyone at the scene", "Send only the first 20 and stop"], "answer": 1},
         {"q": "What do you do before your team leaves the scene?",
          "options": ["Nothing", "Account for every team member", "Post on social media", "Collect payment"], "answer": 1},
     ]},
]
PRACTICAL = {
    # Optional hands-on workshop for citizens. It does not make a rescuer.
    "basics": {"optional": True, "name": "Community safety workshop",
               "duration": "One 2-hour workshop", "drills": [
        "Use a fire extinguisher on a small practice fire",
        "Practise drop, cover and hold on, and a building evacuation",
        "Pack a family go-bag and check what is missing at home",
        "Bandage a simple wound and make an emergency call clearly"]},
    "level1": {"duration": "One 4-hour session", "drills": [
        "Control bleeding and bandage a wound on a partner",
        "Place a person in the recovery position",
        "CPR on a training manikin",
        "Help and carry a person who cannot walk"]},
    "level2": {"duration": "Two 4-hour sessions", "drills": [
        "Throw-rope and reach rescues at a pool, wearing a life jacket",
        "Sort and tag casualties in a mock incident",
        "Search a building safely in pairs and mark what you find",
        "Radio messages under time pressure"]},
    "level3": {"duration": "One full-day exercise", "drills": [
        "Lead a team through a full mock incident",
        "Give a team briefing and count the team in and out",
        "Plan shelter places for a large group",
        "Hand over to the fire service and run a debrief"]},
}


def needs_practical(cid):
    """Rescuer levels need a confirmed practical before certification."""
    return cid in PRACTICAL and not PRACTICAL[cid].get("optional")


def practical_done(p, cid):
    if PRACTICAL[cid].get("optional"):
        return cid in p.get("workshop", {})
    return cid in p["passed"]


TRAINING_CENTRES = ["Begumpet Fire Station Training Ground",
                    "LB Nagar Indoor Stadium",
                    "Secunderabad Railway Community Centre"]
TRAINING_SLOTS = ["Saturday, 9 AM to 1 PM", "Sunday, 9 AM to 1 PM",
                  "Wednesday, 5 PM to 9 PM"]
COURSE_BY_ID = {c["id"]: c for c in COURSES}
LEVEL_NAME = {0: "Not certified", 1: "Level 1", 2: "Level 2", 3: "Level 3"}

DEMO_RESCUERS = [
    ("Asha R.", "Tolichowki", 3), ("Farhan A.", "Mehdipatnam", 2),
    ("Lakshmi D.", "Charminar", 3), ("Vikram S.", "Secunderabad", 2),
    ("Meena K.", "Ameerpet", 1), ("Joseph P.", "Malakpet", 2),
    ("Sana M.", "Kukatpally", 1), ("Arjun N.", "LB Nagar", 1),
    ("Divya T.", "Begumpet", 2), ("Rahim B.", "Uppal", 1),
]


# ---------- helpers ----------
def now(fmt):
    """Indian Standard Time, so the deployed server shows local times."""
    return time.strftime(fmt, time.gmtime(time.time() + 19800))


def embed(text):
    return next(iter(embedder.embed([text]))).tolist()


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_json(raw):
    """Agents sometimes wrap JSON in text or code fences; pull the object out."""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        raise ValueError("no JSON found")
    return json.loads(match.group(0))


def ask_lyzr(agent_id, uid, session, message):
    r = requests.post(
        "https://agent-prod.studio.lyzr.ai/v3/inference/chat/",
        headers={"Content-Type": "application/json", "x-api-key": LYZR_API_KEY},
        json={"user_id": uid, "agent_id": agent_id,
              "session_id": f"{agent_id}-{session}", "message": message},
        timeout=90,
    )
    return r.json().get("response", "")


def log(inc, step, detail=""):
    inc["steps"].append({"time": now("%H:%M:%S"), "step": step,
                         "detail": detail})
    print(f"[{inc['id'][:8]}] {step}: {detail}")


def send_telegram(text):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=30)
        return r.ok
    except requests.RequestException:
        return False


def notify_caller(uid, message):
    """Reply to the person who reported, through Omi notifications."""
    if not (OMI_APP_ID and OMI_APP_KEY) or uid in DEMO_USERS:
        return False
    try:
        r = requests.post(
            f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification",
            params={"uid": uid, "message": message},
            headers={"Authorization": f"Bearer {OMI_APP_KEY}"}, timeout=30)
        return r.ok
    except requests.RequestException:
        return False


def summary_of(inc):
    return (inc.get("triage") or {}).get("summary") or inc["transcript"]


# ---------- storage in Qdrant ----------
def save(inc):
    """Keep the full incident record in Qdrant so it survives restarts."""
    if not inc.get("stored"):
        return
    try:
        qdrant.set_payload("incidents", payload={"record": inc},
                           points=[inc["id"]])
    except Exception as e:
        print("could not save incident:", e)


def person_text(p):
    return (f"Rescuer based in {p['area']}, Hyderabad. "
            f"Certificate level {p.get('level', 0)}.")


def save_person(p, new_vector=False):
    try:
        if new_vector:
            qdrant.upsert("people", points=[PointStruct(
                id=p["id"], vector=embed(person_text(p)), payload=p)])
        else:
            qdrant.set_payload("people", payload=p, points=[p["id"]])
    except Exception as e:
        print("could not save person:", e)


def load_saved():
    try:
        if not qdrant.collection_exists("people"):
            qdrant.create_collection(
                "people",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE))
        points, _ = qdrant.scroll("people", limit=1000, with_payload=True)
        for p in points:
            if p.payload and p.payload.get("id"):
                PEOPLE[p.payload["id"]] = p.payload
        if not any(p.get("demo") for p in PEOPLE.values()):
            for name, area, level in DEMO_RESCUERS:
                person = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                                               "disasterready:demo:" + name)),
                          "name": name, "email": "", "area": area,
                          "level": level, "available": True, "demo": True,
                          "goal": "rescuer", "passed": {}, "assignment": None}
                PEOPLE[person["id"]] = person
                save_person(person, new_vector=True)
        points, _ = qdrant.scroll("incidents", limit=200, with_payload=True)
        for p in points:
            rec = (p.payload or {}).get("record")
            if rec:
                if rec.get("status") == "processing":
                    rec["status"] = "new"
                INCIDENTS[rec["id"]] = rec
        print(f"loaded {len(INCIDENTS)} incidents and {len(PEOPLE)} people")
    except Exception as e:
        print("could not load saved data:", e)


# ---------- logins ----------
def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                                 100_000).hex()
    return f"{salt}${digest}"


def check_password(password, stored):
    salt = (stored or "$").split("$")[0]
    return hmac.compare_digest(hash_password(password, salt), stored or "")


def make_token(data):
    data = dict(data, exp=time.time() + 7 * 86400)
    body = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
    sig = hmac.new(SECRET_KEY.encode(), body.encode(),
                   hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_token(token):
    try:
        body, sig = token.rsplit(".", 1)
        good = hmac.new(SECRET_KEY.encode(), body.encode(),
                        hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(body))
        return data if data.get("exp", 0) > time.time() else None
    except Exception:
        return None


def with_cookie(data, name, token):
    resp = JSONResponse(data)
    if token:
        resp.set_cookie(name, token, max_age=7 * 86400, httponly=True,
                        samesite="lax")
    else:
        resp.delete_cookie(name)
    return resp


def learner(request):
    data = read_token(request.cookies.get("dr_user", ""))
    return PEOPLE.get(data.get("id")) if data else None


def is_operator(request):
    data = read_token(request.cookies.get("dr_op", ""))
    return bool(data and data.get("op"))


def denied():
    return JSONResponse({"error": "Please log in."}, status_code=401)


def person_id(email):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "disasterready:" + email))


# ---------- who is calling ----------
DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
               "four": "4", "five": "5", "six": "6", "seven": "7",
               "eight": "8", "nine": "9",
               # Omi sometimes writes English number words in Cyrillic.
               "зиро": "0", "зеро": "0", "ван": "1", "уан": "1", "ту": "2",
               "три": "3", "сри": "3", "фри": "3", "фор": "4", "файв": "5",
               "сикс": "6", "севен": "7", "эйт": "8", "ейт": "8",
               "найн": "9"}
LINK_WORDS = re.compile(r"link|линк|code|код|connect|account|аккаунт")


def new_link_code():
    used = {p.get("link_code") for p in PEOPLE.values()}
    while True:
        code = f"{secrets.randbelow(900000) + 100000}"
        if code not in used:
            return code


def spoken_digits(text, drop_repeats=False):
    """'code four 8 two' -> '|482': digits said or written, in order.

    Omi often repeats the last word of one sentence at the start of the
    next ("seven one. One zero"); drop_repeats removes those echoes.
    """
    out = []
    for sentence in re.split(r"[.!?]+", text.lower()):
        part = []
        for token in re.findall(r"[a-zа-яё]+|\d", sentence):
            part.append(DIGIT_WORDS.get(token, token) if token in DIGIT_WORDS
                        or token.isdigit() else "|")
        part = "".join(part)
        if (drop_repeats and out and part and out[-1].isdigit()
                and part[0] == out[-1]):
            part = part[1:]
        out.extend(part)
    return "".join(out)


def one_digit_off(code, said):
    """True if 'said' is the code with one extra digit or one missing."""
    if len(said) == len(code) + 1:
        return any(said[:i] + said[i + 1:] == code for i in range(len(said)))
    if len(said) == len(code) - 1:
        return any(code[:i] + code[i + 1:] == said for i in range(len(code)))
    return False


def link_account(uid, transcript):
    """If the speaker read out their link code, tie this Omi user to the
    DisasterReady account so the control room knows who is calling."""
    if uid in DEMO_USERS or len(transcript) > 300:
        return None
    versions = [spoken_digits(transcript), spoken_digits(transcript, True)]
    coded = [p for p in PEOPLE.values() if p.get("link_code")]
    match = next((p for p in coded
                  if any(p["link_code"] in v for v in versions)), None)
    asked = bool(LINK_WORDS.search(transcript.lower()))
    if not match and asked:
        # Speech-to-text often repeats or drops one digit. Accept a code
        # that is one digit off, but only if exactly one account fits.
        runs = [r for v in versions for r in v.split("|") if r]
        near = [p for p in coded
                if any(one_digit_off(p["link_code"], r) for r in runs)]
        if len(near) == 1:
            match = near[0]
    if not match and asked:
        # The person pressed "Link my Omi now" on the site a moment ago.
        waiting = [p for p in PEOPLE.values()
                   if p.get("link_until", 0) > time.time()]
        if len(waiting) == 1:
            match = waiting[0]
    if not match:
        return None
    match["link_until"] = 0
    for other in PEOPLE.values():
        if other.get("omi_uid") == uid:
            other["omi_uid"] = None
            save_person(other)
    match["omi_uid"] = uid
    save_person(match)
    return match


def caller_info(uid, omi_name=None, geo=None):
    """Name, phone and live location of the person reporting, where known."""
    p = next((x for x in PEOPLE.values() if x.get("omi_uid") == uid), None)
    info = {"linked": bool(p), "name": (p or {}).get("name") or omi_name,
            "name_source": ("DisasterReady account" if p
                            else "Omi profile" if omi_name else None),
            "phone": (p or {}).get("phone") or None,
            "area": (p or {}).get("area")}
    geo = geo if isinstance(geo, dict) else {}
    lat, lng = geo.get("latitude"), geo.get("longitude")
    if (isinstance(lat, (int, float)) and isinstance(lng, (int, float))
            and (lat or lng)):
        info.update(lat=lat, lng=lng, address=geo.get("address"),
                    map="https://www.google.com/maps/search/?api=1&query="
                        f"{lat},{lng}")
    return info


# ---------- rescue team allocation ----------
def team_rules(severity):
    """(team size, minimum level of the team lead) for a severity."""
    return {5: (4, 3), 4: (3, 2), 3: (2, 1)}.get(severity, (1, 1))


def is_free(p):
    return (p.get("level", 0) >= 1 and p.get("available")
            and not p.get("assignment"))


def free_rescuers(location, exclude=()):
    """Free certified rescuers, closest area match first (Qdrant search)."""
    pool = [p for p in PEOPLE.values() if is_free(p) and p["id"] not in exclude]
    order = {}
    try:
        hits = qdrant.query_points(
            "people", query=embed(f"Rescuer based in {location}, Hyderabad."),
            limit=60).points
        order = {str(h.id): n for n, h in enumerate(hits)}
    except Exception as e:
        print("rescuer search failed:", e)
    pool.sort(key=lambda p: order.get(p["id"], 999))
    return pool


def pick_by_rules(pool, location, size, min_lead):
    loc = (location or "").lower()
    ranked = sorted(pool, key=lambda p: (p["area"].lower() not in loc,
                                         -p["level"]))
    team = ranked[:size]
    if team and max(p["level"] for p in team) < min_lead:
        seniors = [p for p in ranked[size:] if p["level"] >= min_lead]
        if seniors:
            team[-1] = seniors[0]
    return team


def allocate_team(inc):
    triage = inc.get("triage") or {}
    size, min_lead = team_rules(inc["severity"])
    pool = free_rescuers(triage.get("location"))[:12]
    inc["team"], inc["team_reason"], inc["team_by"] = [], "", "rules"
    if not pool:
        log(inc, "Rescue team", "No certified rescuers are free right now")
        return
    team = []
    if ALLOCATOR_ID:
        codes = {f"R{n + 1}": p for n, p in enumerate(pool)}
        listing = "\n".join(f"{c} | {p['name']} | level {p['level']} | "
                            f"{p['area']}" for c, p in codes.items())
        try:
            raw = ask_lyzr(
                ALLOCATOR_ID, inc["uid"], inc["id"],
                f"INCIDENT: {json.dumps(triage)}\nTEAM_SIZE: {size}\n"
                f"MIN_LEAD_LEVEL: {min_lead}\nCANDIDATES:\n{listing}")
            choice = parse_json(raw)
            seen = []
            for code in choice.get("team", []):
                p = codes.get(str(code).strip())
                if p and p not in seen:
                    seen.append(p)
            if seen:
                team = seen[:size]
                inc["team_reason"] = str(choice.get("reason", ""))
                inc["team_by"] = "agent"
        except Exception as e:
            print("allocation agent failed, using rules:", e)
    if not team:
        team = pick_by_rules(pool, triage.get("location"), size, min_lead)
    lead = max(team, key=lambda p: p["level"])
    for p in team:
        state = "ready" if p.get("demo") else "assigned"
        p["assignment"] = {"incident_id": inc["id"], "state": state}
        save_person(p)
        inc["team"].append({"id": p["id"], "name": p["name"],
                            "level": p["level"], "area": p["area"],
                            "lead": p is lead, "state": state,
                            "demo": bool(p.get("demo"))})
    who = ("Lyzr Allocation Agent" if inc["team_by"] == "agent"
           else "rule-based fallback")
    log(inc, f"Rescue team allocated ({who})",
        ", ".join(f"{m['name']} (L{m['level']}{', lead' if m['lead'] else ''})"
                  for m in inc["team"])
        + (f". {inc['team_reason']}" if inc["team_reason"] else ""))


def release_team(inc):
    for m in inc.get("team", []):
        p = PEOPLE.get(m["id"])
        if p and (p.get("assignment") or {}).get("incident_id") == inc["id"]:
            p["assignment"] = None
            save_person(p)


# ---------- the agent pipeline ----------
def merge_duplicate(existing, transcript, score):
    """A second report of the same incident: merge it and raise severity."""
    existing["reports"] += 1
    existing["severity"] = min(5, existing["severity"] + 1)
    existing["transcripts"].append(transcript)
    log(existing, f"Duplicate report merged (Qdrant similarity {score:.2f})",
        transcript)
    send_telegram(f"UPDATE: {existing['reports']} reports now confirm: "
                  f"{summary_of(existing)} Severity raised to "
                  f"{existing['severity']}/5.")
    save(existing)


def find_shelters(inc, triage, people):
    hits = qdrant.query_points(
        "shelters",
        query=embed(f"{triage.get('location')} {triage.get('hazard')} shelter"),
        limit=6).points
    options = []
    for h in hits:
        p = h.payload
        options.append({
            "id": str(h.id), "name": p["name"], "area": p["area"],
            "capacity": p["capacity"], "facilities": p["facilities"],
            "free": p.get("available", p["capacity"]),
            "map": "https://www.google.com/maps/search/?api=1&query="
                   + quote_plus(f"{p['name']} {p['area']} Hyderabad"),
        })
    fit = [o for o in options if o["free"] >= people] or options
    inc["shelters"] = fit[:3]


def reserve_shelter(inc, people):
    """Reserve places at the shelter the planner named first in its plan."""
    plan = (inc.get("plan") or "").lower()

    def position(s):
        found = plan.find(s["name"].lower())
        return found if found >= 0 else 10 ** 6

    inc["shelters"].sort(key=position)
    if inc["shelters"] and people:
        first = inc["shelters"][0]
        first["free"] = max(first["free"] - people, 0)
        qdrant.set_payload("shelters", payload={"available": first["free"]},
                           points=[first["id"]])
        inc["reserved"] = {"shelter_id": first["id"], "name": first["name"],
                           "people": people}


def run_pipeline(uid, transcript, source="typed", caller=None):
    inc = {"id": str(uuid.uuid4()), "uid": uid, "source": source,
           "transcript": transcript, "transcripts": [transcript],
           "created": now("%d %b %H:%M"), "ts": time.time(),
           "status": "processing", "stage": 0, "severity": 0, "reports": 1,
           "steps": [], "shelters": [], "team": [], "caller": caller}
    INCIDENTS[inc["id"]] = inc
    try:
        log(inc, "Report received (Omi voice)" if source == "omi"
            else "Report received (typed)", transcript)
        inc["stage"] = 1

        raw = ask_lyzr(
            TRIAGE_ID, uid, inc["id"],
            f"REPORT: {transcript}\n\nNOTES: The report may be in any "
            "language, for example Telugu or Hindi. Always answer in English. "
            "It comes from speech-to-text, so place names may be misspelled. "
            f"Known areas of Hyderabad: {', '.join(AREAS)}. If the location "
            "clearly sounds like one of these, use that spelling. If the "
            "message is not reporting an emergency (a greeting, small talk, a "
            "test or a joke), set hazard to null and severity to 0.")
        try:
            triage = parse_json(raw)
        except ValueError:
            triage = {"location": None, "hazard": "unknown emergency",
                      "severity": 3, "summary": transcript}
        if not triage.get("hazard") or str(triage["hazard"]).lower() in (
                "null", "none", "n/a"):
            print(f"[{inc['id'][:8]}] Not an emergency, filtered: {transcript}")
            del INCIDENTS[inc["id"]]
            FILTERED.insert(0, {"time": now("%d %b %H:%M"), "source": source,
                                "text": transcript[:300]})
            del FILTERED[20:]
            if not notify_caller(uid, NOT_EMERGENCY):
                send_telegram(f"Reply to caller: {NOT_EMERGENCY}")
            return
        inc["triage"] = triage
        inc["severity"] = max(1, min(5, to_int(triage.get("severity")) or 3))
        log(inc, "Triage done (Lyzr Triage Agent)", json.dumps(triage))
        inc["stage"] = 2

        vector = embed(summary_of(inc))
        hits = qdrant.query_points("incidents", query=vector, limit=3,
                                   score_threshold=DUPLICATE_SCORE).points
        for h in hits:
            other = INCIDENTS.get(str(h.id))
            if other and other["status"] in ("new", "dispatched"):
                merge_duplicate(other, transcript, h.score)
                del INCIDENTS[inc["id"]]
                return
        log(inc, "Checked for duplicate reports (Qdrant)", "No duplicate found")

        qdrant.upsert("incidents", points=[PointStruct(
            id=inc["id"], vector=vector,
            payload={"uid": uid, "text": summary_of(inc), "ts": inc["ts"]})])
        inc["stored"] = True
        log(inc, "Incident stored (Qdrant)")

        people = to_int(triage.get("people_count"))
        find_shelters(inc, triage, people)
        log(inc, "Shelters retrieved (Qdrant)",
            " | ".join(f"{s['name']}, {s['area']} ({s['free']} places free)"
                       for s in inc["shelters"]))
        inc["stage"] = 3

        context = "\n".join(
            f"- {s['name']} in {s['area']}. {s['free']} of {s['capacity']} "
            f"places free. Facilities: {s['facilities']}."
            for s in inc["shelters"])
        inc["plan"] = ask_lyzr(
            PLANNER_ID, uid, inc["id"],
            f"INCIDENT: {json.dumps(triage)}\nCONTEXT:\n{context}")
        reserve_shelter(inc, people)
        log(inc, "Plan ready (Lyzr Resource Planner Agent)", inc["plan"])
        inc["stage"] = 4

        allocate_team(inc)
        inc["stage"] = 5

        inc["sos"] = ask_lyzr(
            SOS_ID, uid, inc["id"],
            f"INCIDENT: {json.dumps(triage)}\nPLAN: {inc['plan']}")
        log(inc, "SOS drafted (Lyzr SOS Drafter Agent)", inc["sos"])
        inc["stage"] = 6

        team = ", ".join(f"{m['name']} (Level {m['level']})"
                         for m in inc["team"])
        who = ""
        if caller and (caller.get("name") or caller.get("phone")
                       or caller.get("map")):
            who = "\nCaller: " + ", ".join(
                str(x) for x in (caller.get("name"), caller.get("phone"),
                                 caller.get("map")) if x)
        if send_telegram(inc["sos"] + (f"\nTeam: {team}" if team else "")
                         + who):
            log(inc, "SOS sent to responders (Telegram)")
            inc["stage"] = 7
        else:
            log(inc, "SOS not sent", "Telegram is not configured")
    except Exception as e:
        inc["error"] = str(e)
        log(inc, "ERROR", str(e))
    finally:
        if inc["id"] in INCIDENTS:
            inc["status"] = "new"
            save(inc)


def change_status(inc, status):
    inc["status"] = status
    if status == "dispatched":
        shelter = inc["shelters"][0]["name"] if inc.get("shelters") else None
        message = "Help is on the way."
        if shelter:
            message += f" Nearest shelter: {shelter}."
        told = notify_caller(inc["uid"], message)
        send_telegram(f"Team dispatched for: {summary_of(inc)}")
        if not told:
            told = send_telegram(f"Reply to caller: {message}")
        log(inc, "Team dispatched (operator)",
            f"Caller told: {message}" if told else "Status updated")
    elif status == "resolved":
        res = inc.pop("reserved", None)
        if res:
            try:
                point = qdrant.retrieve("shelters", ids=[res["shelter_id"]])[0]
                cap = point.payload["capacity"]
                free = point.payload.get("available", cap)
                qdrant.set_payload(
                    "shelters",
                    payload={"available": min(cap, free + res["people"])},
                    points=[res["shelter_id"]])
            except Exception as e:
                print("could not release shelter places:", e)
        release_team(inc)
        log(inc, "Incident resolved (operator)",
            "Rescue team released"
            + (f", {res['people']} places freed at {res['name']}" if res else ""))
    save(inc)


# ---------- pages ----------
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
:root{--ink:#13202c;--ink2:#2a3b4a;--paper:#edf1f4;--card:#fff;--line:#d5dce3;--muted:#566473;
--accent:#ffd200;--accent-d:#e9bd00;--brand:#00704a;--ok:#00704a;--okbg:#dcf1e7;--crit:#c8202f;--high:#e07b00;--low:#2c6e9b}
*{box-sizing:border-box}
body{margin:0;font-family:Barlow,Segoe UI,Roboto,Arial,sans-serif;background:var(--paper);color:var(--ink);line-height:1.5}
h1,h2,h3,h4,.brand{font-family:'Barlow Semi Condensed',Segoe UI,Arial,sans-serif;line-height:1.2}
a{color:var(--accent-d)}
.nav{background:var(--ink);color:#fff;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.brand{font-weight:700;font-size:20px;color:#fff;text-decoration:none;display:flex;align-items:center;gap:10px}
.brand i{width:14px;height:14px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 5px rgba(244,81,30,.25);display:inline-block}
.nav nav{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.nav nav a,.nav nav button{color:#d5dcea;text-decoration:none;font-size:14px;padding:7px 12px;border-radius:999px;background:none;border:0;cursor:pointer;font-family:inherit}
.nav nav a:hover,.nav nav button:hover{background:rgba(255,255,255,.12);color:#fff}
main{max-width:1040px;margin:0 auto;padding:22px 16px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:16px}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}.g4{grid-template-columns:repeat(4,1fr)}
button,.btn{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:11px 18px;font:600 16px Barlow,Arial,sans-serif;cursor:pointer;text-decoration:none;display:inline-block}
button:hover,.btn:hover{background:var(--accent-d)}
button.dark,.btn.dark{background:var(--ink)}button.dark:hover,.btn.dark:hover{background:var(--ink2)}
button.green{background:var(--ok)}
button.ghost{background:#fff;color:var(--ink);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:not-allowed}
input,select,textarea{width:100%;padding:11px 12px;border:1px solid #cfc8b8;border-radius:10px;font:inherit;background:#fff;color:var(--ink)}
label{display:block;font-size:13px;font-weight:600;margin:12px 0 5px;color:var(--ink2)}
.muted{color:var(--muted);font-size:14px}
.err{color:var(--crit);font-size:14px;min-height:20px;margin-top:8px}
.badge{display:inline-block;border-radius:999px;padding:4px 11px;font-size:12px;font-weight:600;background:#ece7da;margin:0 4px 4px 0}
.badge.ok{background:var(--okbg);color:var(--ok)}
.badge.dark{background:var(--ink);color:#fff}
.badge.wait{background:#fff0d6;color:#8a5a00}
.hero{background:var(--ink);color:#fff;border-radius:22px;padding:44px 32px;margin-bottom:18px;position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;right:-90px;top:-90px;width:320px;height:320px;border-radius:50%;border:46px solid rgba(244,81,30,.9)}
.hero h1{font-size:42px;margin:0 0 12px;max-width:620px;position:relative;z-index:1}
.hero p{color:#c8d1e2;font-size:17px;max-width:560px;margin:0 0 22px;position:relative;z-index:1}
.hero .row{position:relative;z-index:1}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.step b{display:flex;width:34px;height:34px;border-radius:50%;background:var(--accent);color:#fff;align-items:center;justify-content:center;font-family:'Barlow Semi Condensed',Arial,sans-serif;margin-bottom:10px}
.step h3,.course h3{margin:0 0 6px;font-size:17px}
.stat{text-align:center}.stat b{display:block;font:700 30px 'Barlow Semi Condensed',Arial,sans-serif}.stat span{font-size:13px;color:var(--muted)}
.course{display:flex;flex-direction:column;gap:8px}
.course.locked{opacity:.6}
.course .lvl{font:700 12px 'Barlow Semi Condensed',Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-d)}
.lesson{border-left:4px solid var(--accent);padding:4px 0 4px 16px;margin:18px 0}
.lesson h3{margin:0 0 6px;font-size:17px}.lesson ul{margin:0;padding-left:18px}.lesson li{margin:6px 0}
.q{border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.q p{margin:0 0 8px;font-weight:600}
.q label{font-weight:400;font-size:15px;margin:6px 0;display:flex;gap:9px;align-items:flex-start;color:var(--ink)}
.q input{width:auto;margin-top:4px}
.alert{background:#fff1ec;border:2px solid var(--accent);border-radius:16px;padding:18px;margin-bottom:16px}
.alert h3{margin:0 0 6px}
.cert{border:2px solid var(--ink);border-radius:14px;padding:14px;background:linear-gradient(135deg,#fff,#fbf7ee)}
.cert small{font:700 11px 'Barlow Semi Condensed',Arial,sans-serif;letter-spacing:.1em;text-transform:uppercase;color:var(--accent-d)}
.cert b{display:block;font:600 16px 'Barlow Semi Condensed',Arial,sans-serif;margin:4px 0}
.inc{border-left:8px solid #999}
.inc.crit{border-left-color:var(--crit)}.inc.high{border-left-color:var(--high)}.inc.low{border-left-color:var(--low)}
.inc.done{border-left-color:#8a93a3;opacity:.7}
.top{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:flex-start}
.top h3{margin:0;font-size:18px;flex:1;min-width:220px}
.badge.sev{color:#fff;background:#8a93a3}
.crit .badge.sev{background:var(--crit)}.high .badge.sev{background:var(--high)}.low .badge.sev{background:var(--low)}
.fact{background:var(--paper);border-radius:10px;padding:9px 11px;font-size:15px}
.fact span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.sos{background:#fde9e6;border:1px solid #f2b7ae;border-radius:10px;padding:11px;font-weight:600;margin:12px 0}
.stages{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:12px 0}
.stage{font-size:12px;padding:5px 9px;border-radius:6px;background:#ece7da;color:var(--muted)}
.stage.on{background:var(--okbg);color:var(--ok);font-weight:600}
.arrow{color:#9aa3b2;font-size:12px}
h4{margin:14px 0 5px;font-size:14px}
ul.plain{margin:4px 0;padding-left:20px}ul.plain li{margin:5px 0}
.pre{white-space:pre-wrap;font-size:14px}
details{margin-top:10px}summary{cursor:pointer;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.auth{max-width:440px;margin:30px auto}
.tabs{display:flex;gap:6px;margin-bottom:6px}
.tabs button{flex:1;background:#ece7da;color:var(--ink)}.tabs button.on{background:var(--ink);color:#fff}
.foot{text-align:center;color:var(--muted);font-size:13px;margin-top:26px}
main.wide{max-width:1240px}
.work{display:grid;grid-template-columns:330px 1fr;gap:16px;align-items:start}
.side{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden;position:sticky;top:12px;max-height:calc(100vh - 24px);display:flex;flex-direction:column}
.sidehead{padding:14px 16px;font:600 15px 'Barlow Semi Condensed',Arial,sans-serif;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:baseline}
#list{overflow-y:auto}
.empty{padding:26px 18px;color:var(--muted);text-align:center}
.inrow{display:flex;gap:11px;align-items:center;width:100%;text-align:left;background:#fff;color:var(--ink);border:0;border-bottom:1px solid var(--line);border-left:5px solid transparent;border-radius:0;padding:12px 12px 12px 10px;font:inherit;cursor:pointer}
.inrow:hover{background:#faf7f0}
.inrow.sel{background:#fff4ee;border-left-color:var(--accent)}
.inrow.done{opacity:.55}
.sevbox{flex:none;width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font:700 18px 'Barlow Semi Condensed',Arial,sans-serif;color:#fff;background:#8a93a3}
.crit .sevbox{background:var(--crit)}.high .sevbox{background:var(--high)}.low .sevbox{background:var(--low)}
.rmain{flex:1;min-width:0;display:flex;flex-direction:column}
.rmain b{font-size:14px;line-height:1.3;font-weight:600}
.rmain span{font-size:12px;color:var(--muted);margin-top:2px}
.rstat .badge{margin:0;white-space:nowrap}
.newtag{font:700 11px 'Barlow Semi Condensed',Arial,sans-serif;letter-spacing:.06em;background:var(--accent);color:#fff;border-radius:6px;padding:4px 7px;animation:blink 1.4s infinite}
@keyframes blink{50%{opacity:.45}}
.detail{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:0 20px 20px;overflow:hidden}
.dhead{display:flex;gap:16px;align-items:center;margin:0 -20px;padding:18px 20px;background:#eceff4;flex-wrap:wrap}
.crit .dhead{background:#fbe4e2}.high .dhead{background:#fdeed6}.low .dhead{background:#f6f0cf}
.dsev{flex:none;width:76px;height:76px;border-radius:16px;background:#8a93a3;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center}
.crit .dsev{background:var(--crit)}.high .dsev{background:var(--high)}.low .dsev{background:var(--low)}
.dsev b{font:700 32px/1 'Barlow Semi Condensed',Arial,sans-serif}.dsev span{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-top:3px}
.dtitle{flex:1;min-width:240px}.dtitle h2{margin:0 0 5px;font-size:20px}
.meta{font-size:13px;color:var(--ink2)}
.dact button{font-size:16px;padding:13px 22px}
.flow{display:flex;align-items:center;gap:8px;margin:16px 0;flex-wrap:wrap}
.fstep{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--muted);font-weight:600}
.fstep i{font-style:normal;width:24px;height:24px;border-radius:50%;background:#ece7da;display:flex;align-items:center;justify-content:center;font-size:12px}
.fstep.now{color:var(--ink)}.fstep.now i{background:var(--accent);color:#fff}
.fstep.past{color:var(--ok)}.fstep.past i{background:var(--okbg);color:var(--ok)}
.fline{flex:1;min-width:18px;max-width:70px;height:2px;background:var(--line)}
.panels{gap:12px}
.panel{background:var(--paper);border-radius:12px;padding:14px;font-size:14px}
.plabel{display:flex;justify-content:space-between;font:700 11px 'Barlow Semi Condensed',Arial,sans-serif;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.plabel em{font-style:normal;color:var(--ok)}
.big{display:block;font:700 40px/1.1 'Barlow Semi Condensed',Arial,sans-serif}
.mid{display:block;font:600 16px/1.3 'Barlow Semi Condensed',Arial,sans-serif;margin-bottom:3px}
ul.team{list-style:none;margin:0;padding:0}ul.team li{display:flex;gap:9px;align-items:flex-start;margin:0 0 9px;line-height:1.35}
.dot{flex:none;width:10px;height:10px;border-radius:50%;background:#e0a100;margin-top:5px}.dot.ok{background:var(--ok)}
.said{border-left:4px solid var(--line);padding:2px 0 2px 14px;margin:14px 0}
.said p{margin:4px 0;font-style:italic}
.detail details{border-top:1px solid var(--line);padding-top:10px}
.caller{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;border:1px solid var(--line);border-left:6px solid var(--ink);border-radius:8px;padding:12px 14px;margin-bottom:12px;font-size:14px}
.caller .mid{margin-bottom:2px}
.callbtn{display:inline-block;background:var(--brand);color:#fff;border-radius:8px;padding:8px 14px;font-weight:600;text-decoration:none}
@media(max-width:760px){.caller{grid-template-columns:1fr}}
.sect{border-top:1px solid var(--line);padding-top:12px;margin-top:14px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{background:var(--ink);color:#fff;border-radius:999px;padding:6px 13px;font-size:13px;font-weight:500}
.bar{height:8px;border-radius:99px;background:#e2dccd;overflow:hidden;margin:8px 0 3px}
.bar i{display:block;height:100%;background:var(--ok);border-radius:99px}
.mini{background:var(--paper);border-radius:12px;padding:12px;font-size:14px}
.mini b{font:600 14px 'Barlow Semi Condensed',Arial,sans-serif}
.detail .stages{margin:4px 0 0}
ul.tline{list-style:none;margin:12px 0 0;padding:0 0 0 6px}
ul.tline li{position:relative;padding:0 0 16px 24px;border-left:2px solid var(--line)}
ul.tline li:last-child{border-left-color:transparent;padding-bottom:0}
ul.tline li:before{content:"";position:absolute;left:-8px;top:2px;width:14px;height:14px;border-radius:50%;background:#8a93a3;border:3px solid #fff}
ul.tline li.t-omi:before{background:var(--accent)}ul.tline li.t-lyzr:before{background:#6b4fd8}
ul.tline li.t-qd:before{background:#d6336c}ul.tline li.t-tg:before{background:#1c8fd6}
.thead{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:14px}
.who{font-size:11px;font-weight:600;border-radius:6px;padding:2px 8px;background:#ece7da}
.t-omi .who{background:#ffe3d8;color:#a8380f}.t-lyzr .who{background:#e8e2fb;color:#4a34ab}
.t-qd .who{background:#fbdfe9;color:#a11d4d}.t-tg .who{background:#dceefb;color:#11659b}
.kv{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.kv span{background:var(--paper);border-radius:8px;padding:5px 9px;font-size:13px}
.kv em{font-style:normal;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-right:6px}
.clamp{font-size:13px;color:var(--ink2);margin-top:4px;white-space:pre-wrap;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.banner{display:flex;gap:14px;align-items:center;background:var(--ink);color:#fff;border-radius:14px;padding:14px 18px;margin-bottom:16px}
.banner div{flex:1}
.pulse{flex:none;width:14px;height:14px;border-radius:50%;background:var(--accent);animation:blink 1s infinite}
.video{display:flex;gap:12px;align-items:center;background:var(--ink);color:#fff;border-radius:12px;padding:12px 14px;margin:8px 0 10px;font-size:14px}
.video .muted{color:#aeb9cc}
.play{flex:none;width:38px;height:38px;border-radius:50%;background:var(--accent);position:relative}
.play:after{content:"";position:absolute;left:15px;top:11px;border-left:12px solid #fff;border-top:8px solid transparent;border-bottom:8px solid transparent}
.pathrow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.pathrow span{display:flex;gap:9px;align-items:center;font-size:14px;background:var(--paper);border-radius:10px;padding:10px}
.pathrow i{flex:none;font-style:normal;width:26px;height:26px;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font:600 13px 'Barlow Semi Condensed',Arial,sans-serif}
@media(max-width:760px){.pathrow{grid-template-columns:1fr}}
@media(max-width:900px){.work{grid-template-columns:1fr}.side{position:static;max-height:320px}}
@media(max-width:760px){.g2,.g3,.g4{grid-template-columns:1fr}.hero{padding:30px 20px}.hero h1{font-size:30px}.hero:after{display:none}.facts.g4{grid-template-columns:repeat(2,1fr)}}

/* ---- DisasterReady identity: safety signage. Green for safe, high-vis yellow for action. ---- */
body{font-size:16px}
h1,h2,h3,.brand{letter-spacing:.005em}
h2{font-size:24px;margin:26px 0 10px}
a{color:var(--brand)}
.nav{position:relative;padding-bottom:19px}
.nav:after{content:"";position:absolute;left:0;right:0;bottom:0;height:5px;background:var(--accent)}
.brand{font-size:23px;font-weight:700}
.brand i{width:26px;height:26px;border-radius:6px;background:var(--brand);box-shadow:none;position:relative}
.brand i:after{content:"";position:absolute;left:7px;top:7px;width:9px;height:9px;border-top:3px solid #fff;border-right:3px solid #fff;transform:rotate(45deg) translate(-1px,1px)}
.card,.side,.detail{border-radius:10px;border-color:var(--line)}
button,.btn{background:var(--accent);color:var(--ink);border-radius:8px;font-weight:600}
button:hover,.btn:hover{background:var(--accent-d)}
button.dark,.btn.dark{background:var(--ink);color:#fff}
button.green{background:var(--brand);color:#fff}button.green:hover{background:#005a3b}
button.ghost{background:#fff;color:var(--ink);border:1px solid var(--line)}button.ghost:hover{background:var(--paper)}
button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid var(--brand);outline-offset:2px}
input,select,textarea{border-color:#b9c3cd;border-radius:8px}
.nav nav a,.nav nav button{color:#d3dbe3;font-size:15px;background:none}
.nav nav a:hover,.nav nav button:hover{color:#fff;background:rgba(255,255,255,.12)}
.tabs button{background:var(--paper);color:var(--ink)}.tabs button.on{background:var(--ink);color:#fff}
.badge{background:#e3e8ed}.badge.wait{background:#fff3c2;color:#6b5600}.badge.dark{background:var(--ink);color:#fff}
.hero{background:var(--brand);border-radius:10px}
.hero:after{right:0;top:0;bottom:0;width:38%;height:auto;border:0;border-radius:0;background:repeating-linear-gradient(135deg,var(--accent) 0 34px,transparent 34px 68px);opacity:.95;-webkit-mask-image:linear-gradient(90deg,transparent,#000 55%);mask-image:linear-gradient(90deg,transparent,#000 55%)}
.hero h1{font-size:50px;line-height:1.05;font-weight:700}
.hero p{color:#e2f1ea;font-size:18px}
.hero .btn.dark{border-color:rgba(255,255,255,.35)!important}
.step b{background:var(--ink);color:var(--accent);border-radius:8px}
.lesson{border-left-color:var(--brand)}
.course .lvl,.cert small{color:var(--brand)}
.course .lvl,.cert small,.plabel,.fact span,th,.dsev span,.kv em,.fstep,.newtag{text-transform:none;letter-spacing:0}
.course .lvl,.cert small{font-size:14px;font-weight:600}
.plabel{font-size:14px;font-weight:600;color:var(--muted)}
th{font-size:13px;font-weight:600}
.dsev span{font-size:13px}
.kv em{font-size:12px}
.cert{border-radius:10px;border:1px solid var(--line);border-left:8px solid var(--brand);background:#fff}
.alert{background:#fff8d6;border-color:var(--accent-d);border-radius:10px}
.video{border-radius:8px}.play{background:var(--accent)}.play:after{border-left-color:var(--ink)}
.pathrow i{background:var(--brand)}
.stat b{font-size:36px;font-weight:700}
.stat{border-top:4px solid var(--ink)}
.sevbox{border-radius:8px}.dsev{border-radius:10px}
.inrow.sel{background:#fffbe3;border-left-color:var(--ink)}
.inrow:hover{background:#f6f8fa}
.newtag{background:var(--accent);color:var(--ink);font-size:12px;border-radius:5px}
.fstep.now i{background:var(--accent);color:var(--ink)}
.dhead{background:#e6ebf0}
.crit .dhead{background:#fbe3e4}.high .dhead{background:#fdeeda}.low .dhead{background:#e1edf6}
.sos{background:#fdecec;border-color:#f0b8bc;border-radius:8px}
.panel,.mini,.fact{border-radius:8px}
.chip{background:var(--ink);border-radius:6px}
.banner{border-radius:10px;border-left:10px solid var(--accent)}
.pulse{background:var(--accent)}
.said{border-left-color:var(--brand)}
ul.tline li.t-omi:before{background:var(--high)}
.t-omi .who{background:#fde8cf;color:#8a4a00}
.foot{font-size:14px}
@media(max-width:760px){.hero h1{font-size:34px}.hero:after{display:none}}
@media(prefers-reduced-motion:reduce){.newtag,.pulse{animation:none}}
"""

HOME_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DisasterReady</title><style>__CSS__</style></head><body>
<div class="nav"><a class="brand" href="/"><i></i>DisasterReady</a>
<nav><a href="/learn">Learn and certify</a><a href="/control">Control room</a></nav></div>
<main>
<section class="hero">
<h1>Be ready before the disaster. Be reached when it happens.</h1>
<p>Learn what to do in a flood, fire or earthquake, earn a rescuer certificate, and report an emergency by voice. AI agents turn each report into a plan and send a certified team.</p>
<div class="row"><a class="btn" href="/learn">Start learning</a><a class="btn dark" href="/control">Control room login</a></div>
</section>
<div class="grid g3">
<div class="card step"><b>1</b><h3>Learn</h3><p class="muted">A short basics course for everyone, and three rescuer levels. Rescuer levels need a theory quiz and in-person practical training.</p></div>
<div class="card step"><b>2</b><h3>Report by voice</h3><p class="muted">Speak into the Omi app in English, Hindi or Telugu. Agents extract the location, hazard, people and severity.</p></div>
<div class="card step"><b>3</b><h3>Respond</h3><p class="muted">The control room gets shelters, a response plan and a team of certified rescuers matched to the severity.</p></div>
</div>
<div class="grid g2">
<div class="card"><h3 style="margin-top:0">For citizens and volunteers</h3>
<p class="muted">Register, study the lessons, pass the quiz, then attend practical training. Certified rescuers mark themselves available and confirm when they are assigned to an incident.</p>
<a class="btn" href="/learn">Register or log in</a></div>
<div class="card"><h3 style="margin-top:0">For the control room</h3>
<p class="muted">See live incidents, the agents' work at each stage, shelter capacity and which rescuers are ready, then dispatch.</p>
<a class="btn dark" href="/control">Operator login</a></div>
</div>
<p class="foot">In a life-threatening emergency, call 112 first. DisasterReady is a hackathon prototype built with Omi, Qdrant and Lyzr. Shelters and some rescuers are sample data, and the courses are short awareness training, not an official certification.</p>
</main></body></html>"""

LEARN_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learn | DisasterReady</title><style>__CSS__</style></head><body>
<div class="nav"><a class="brand" href="/"><i></i>DisasterReady</a>
<nav id="nav"><a href="/control">Control room</a></nav></div>
<main id="app"><div class="card">Loading...</div></main>
<script>
const AREAS = __AREAS__, CENTRES = __CENTRES__, SLOTS = __SLOTS__;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let ME = null, OPEN = null, MODE = "register", LAST = "", RESULT = null, EDITING = false, SAVED = 0;
async function api(url, body){
  const opt = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  return (await fetch(url, opt)).json();
}
async function refresh(force){
  const data = await api("/api/me");
  const text = JSON.stringify(data);
  if(!force && text === LAST) return;
  LAST = text; ME = data; render();
}
function render(){
  $("nav").innerHTML = '<a href="/control">Control room</a>' + (ME.user ? '<button onclick="logout()">Log out</button>' : "");
  if(!ME.user){ $("app").innerHTML = authView(); return; }
  $("app").innerHTML = OPEN ? courseView(ME.courses.find(c => c.id === OPEN)) : homeView();
}
function authView(){
  const reg = MODE === "register";
  return `<div class="card auth">
    <div class="tabs"><button class="${reg ? "on" : ""}" onclick="MODE='register';render()">Register</button>
    <button class="${reg ? "" : "on"}" onclick="MODE='login';render()">Log in</button></div>
    <h2>${reg ? "Create your account" : "Welcome back"}</h2>
    <p class="muted">${reg ? "Learn how to stay safe, or train to become a certified rescuer." : "Log in to continue your courses."}</p>
    ${reg ? '<label>Full name</label><input id="name" autocomplete="name">' : ""}
    <label>Email</label><input id="email" type="email" autocomplete="email">
    <label>Password</label><input id="password" type="password" placeholder="${reg ? "At least 6 characters" : ""}">
    ${reg ? '<label>Phone number (optional)</label><input id="phone" type="tel" autocomplete="tel" placeholder="Shown to the control room only when you report an emergency">' : ""}
    ${reg ? `<label>Your area in Hyderabad</label><select id="area">${AREAS.map(a => `<option>${a}</option>`).join("")}</select>
    <label>What do you want to do?</label><select id="goal">
      <option value="citizen">Learn how to be ready for a disaster</option>
      <option value="rescuer">Train to become a certified rescuer</option></select>` : ""}
    <div class="err" id="err"></div>
    <button style="width:100%" onclick="submitAuth()">${reg ? "Create account" : "Log in"}</button></div>`;
}
async function submitAuth(){
  const body = {email:$("email").value, password:$("password").value};
  if(MODE === "register"){ body.name = $("name").value; body.area = $("area").value; body.goal = $("goal").value; body.phone = $("phone").value; }
  const r = await api(MODE === "register" ? "/api/register" : "/api/login", body);
  if(r.error){ $("err").textContent = r.error; return; }
  refresh(true);
}
async function logout(){ await api("/api/logout", {}); OPEN = null; refresh(true); }
function homeView(){
  const u = ME.user, a = ME.assignment;
  const certs = ME.courses.filter(c => c.passed);
  let alert = "";
  if(a){
    alert = `<div class="alert"><h3>${a.state === "ready" ? "You are confirmed for this incident" : "You have been assigned to an incident"}</h3>
      <p style="margin:0 0 8px"><b>${esc(a.summary)}</b></p>
      <p class="muted" style="margin:0 0 10px">Severity ${a.severity}/5 &middot; ${esc(a.location || "Location unknown")} &middot; Team: ${esc(a.team)}${a.shelter ? ` &middot; Shelter: ${esc(a.shelter)}` : ""}</p>
      ${a.state === "ready" ? '<span class="badge ok">Ready, waiting for dispatch</span>' :
        `<div class="row"><button class="green" onclick="answer('ready')">I'm ready</button><button class="ghost" onclick="answer('decline')">I can't go</button></div>`}
      ${a.map ? ` <a href="${esc(a.map)}" target="_blank" rel="noopener">Open shelter in Maps</a>` : ""}</div>`;
  }
  const rescuer = u.level >= 1 ? `<div class="card"><div class="top"><div>
      <h3 style="margin:0 0 4px">Rescuer status</h3>
      <span class="badge dark">${esc(u.level_name)} certified</span>
      <span class="badge ${u.available ? "ok" : "wait"}">${u.available ? "Available for incidents" : "Not available"}</span></div>
      <button class="${u.available ? "ghost" : "green"}" onclick="setAvailable(${!u.available})">${u.available ? "Set as not available" : "Set as available"}</button></div>
      <p class="muted" style="margin:10px 0 0">When you are available, the Allocation Agent can assign you to incidents that match your level and area.</p></div>` : "";
  return `<h1 style="margin:6px 0 4px">Hello, ${esc(u.name)}</h1>
    <p class="muted" style="margin:0 0 16px">${esc(u.area)} &middot; ${u.level >= 1 ? esc(u.level_name) + " rescuer" : (u.goal === "rescuer" ? "Training to become a rescuer" : "Learning to be disaster ready")}</p>
    ${alert}${rescuer}
    <div class="card"><h3 style="margin:0 0 4px">If you report an emergency</h3>
      <p class="muted" style="margin:0 0 10px">People in danger rarely say their name or number. Link your Omi app once, and the control room will know who is calling and how to reach you.</p>
      <div class="grid g2" style="gap:12px">
        <div class="panel"><span class="plabel">Your phone number</span>
          <div class="row"><input id="myphone" type="tel" value="${esc(u.phone)}" placeholder="Phone number" style="flex:1;min-width:150px" onfocus="EDITING=true" onblur="EDITING=false"><button class="dark" onclick="savePhone()">Save</button></div>
          <div class="err" id="perr" style="min-height:0"></div>${Date.now() - SAVED < 60000 ? '<span class="badge ok" style="margin-top:8px">Saved</span>' : u.phone ? '<span class="badge" style="margin-top:8px">Saved on your account</span>' : ""}<div class="muted" style="margin-top:6px">Shown to the control room only on your own emergency reports.</div></div>
        <div class="panel"><span class="plabel">Your Omi app ${u.omi_linked ? "<em>Linked</em>" : ""}</span>
          ${u.omi_linked ? '<div>Your Omi app is linked to this account. Your reports will carry your name and phone number.</div>' : u.link_waiting ? `<span class="badge wait">Waiting for your Omi, up to 5 minutes</span><div style="margin-top:6px">Now open Omi, tap record and say:</div><b class="mid" style="margin-top:4px">"Link my DisasterReady account"</b><div class="muted">Then wait a minute or two. This box will show Linked.</div>` : `<div>Press the button, then say one sentence into Omi.</div><div style="margin:8px 0"><button onclick="startLink()">Link my Omi now</button></div><div class="muted">Or say in Omi: "DisasterReady link code ${esc(u.link_code.split("").join(" "))}"</div>`}</div>
      </div></div>
    <h2 style="margin-bottom:4px">For everyone: be disaster ready</h2>
    <p class="muted" style="margin:0 0 12px">Simple basics for you and your family. Pass the quiz for an awareness certificate. A hands-on community workshop is optional.</p>
    <div class="grid g2">${ME.courses.filter(c => c.level === 0).map(courseCard).join("")}
      <div class="card course" style="background:var(--paper)"><span class="lvl">Optional</span><h3>Community safety workshop</h3>
      <p class="muted" style="margin:0">Two hours, in person: use a fire extinguisher, practise an evacuation, pack a go-bag. Open to everyone after the basics quiz. This is not rescuer training.</p></div></div>
    <h2 style="margin-bottom:4px">Rescuer training: for volunteers who respond to incidents</h2>
    <p class="muted" style="margin:0 0 12px">Different lessons, harder quizzes and longer practical drills than the citizen course. Each level must be confirmed by a trainer.</p>
    <div class="card path"><b>How to become a certified rescuer</b>
      <div class="pathrow"><span><i>1</i>Study the lessons and videos</span><span><i>2</i>Pass the theory quiz</span><span><i>3</i>Attend practical training in person</span><span><i>4</i>A trainer confirms you, and you are certified</span></div>
      <p class="muted" style="margin:10px 0 0">A quiz alone does not make anyone a rescuer. Only people who have completed the practical training are sent to incidents.</p></div>
    <div class="grid g2">${ME.courses.filter(c => c.level > 0).map(courseCard).join("")}</div>
    ${certs.length ? `<h2>Your certificates</h2><div class="grid g2">${certs.map(c => `<div class="cert"><small>${c.has_practical ? "Rescuer certificate: theory and practical" : c.workshop ? "Awareness certificate with hands-on workshop" : "Awareness certificate"}</small><b>${esc(c.badge)}</b><span class="muted">${esc(u.name)} &middot; ${esc(c.passed)}</span></div>`).join("")}</div>` : ""}
    <p class="foot">Prototype training content, not an official certification. Training centres are sample data. In an emergency call 112.</p>`;
}
function courseCard(c){
  const label = c.level === 0 ? "For everyone" : "Rescuer level " + c.level;
  let state, btn = "Start course", ghost = false;
  if(c.passed && c.optional_workshop){
    state = `<span class="badge ok">Passed ${esc(c.passed)}</span>` + (c.workshop ? '<span class="badge ok">Workshop attended</span>' : c.booking ? '<span class="badge wait">Workshop booked</span>' : '<span class="badge">Workshop optional</span>');
    btn = c.workshop || c.booking ? "Review" : "Review or book workshop"; ghost = true;
  }
  else if(c.passed){ state = `<span class="badge ok">Certified ${esc(c.passed)}</span>`; btn = "Review"; ghost = true; }
  else if(c.locked) state = `<span class="badge">Locked: get certified in "${esc(c.requires_title)}" first</span>`;
  else if(c.booking){ state = `<span class="badge ok">Theory passed</span><span class="badge wait">Practical booked, waiting for trainer</span>`; btn = "View booking"; ghost = true; }
  else if(c.theory){ state = `<span class="badge ok">Theory passed</span><span class="badge wait">Practical training needed</span>`; btn = "Book practical training"; }
  else state = `<span class="badge wait">${c.lessons.length} lessons, ${c.quiz.length} questions${c.has_practical ? ", then practical training" : ""}</span>`;
  return `<div class="card course ${c.locked ? "locked" : ""}">
    <span class="lvl">${label}</span><h3>${esc(c.title)}</h3>
    <p class="muted" style="margin:0">${esc(c.summary)}</p>
    <div>${state}</div>
    <div><button class="${ghost ? "ghost" : ""}" ${c.locked ? "disabled" : ""} onclick="openCourse('${c.id}')">${btn}</button></div></div>`;
}
function openCourse(id){ OPEN = id; RESULT = null; render(); window.scrollTo(0, 0); }
function workshopBlock(c){
  const drills = `<ul class="plain">${c.practical.drills.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`;
  let body;
  if(c.workshop) body = `<span class="badge ok">Workshop attended ${esc(c.workshop)}, confirmed by a trainer</span>`;
  else if(c.booking) body = `<div class="alert" style="border-color:var(--ok);background:var(--okbg);margin:10px 0 0"><h3>Your workshop place is booked</h3>
      <p style="margin:0"><b>${esc(c.booking.centre)}</b><br>${esc(c.booking.slot)}</p>
      <p class="muted" style="margin:8px 0 0">After you attend, the trainer confirms it and it is added to your awareness certificate.</p></div>`;
  else if(c.passed) body = `<label>Training centre</label><select id="centre">${CENTRES.map(x => `<option>${esc(x)}</option>`).join("")}</select>
      <label>Session</label><select id="slot">${SLOTS.map(x => `<option>${esc(x)}</option>`).join("")}</select>
      <div style="margin-top:12px"><button onclick="book('${c.id}')">Book a workshop place</button></div>`;
  else body = `<p class="muted">Pass the quiz first. Then you can book a place here.</p>`;
  return `<div class="card" id="practical"><h2 style="margin-top:0">Optional: ${esc(c.practical.name)}</h2>
    <p class="muted">${esc(c.practical.duration)} for citizens and families. It is not rescuer training and you will not be sent to incidents. You will practise:</p>
    ${drills}${body}</div>`;
}
function practicalBlock(c){
  if(c.optional_workshop) return workshopBlock(c);
  if(!c.has_practical) return "";
  const drills = `<ul class="plain">${c.practical.drills.map(d => `<li>${esc(d)}</li>`).join("")}</ul>`;
  let body;
  if(c.passed) body = `<span class="badge ok">Practical training completed and confirmed by a trainer</span>`;
  else if(c.booking) body = `<div class="alert" style="border-color:var(--ok);background:var(--okbg);margin:10px 0 0"><h3>Your practical session is booked</h3>
      <p style="margin:0"><b>${esc(c.booking.centre)}</b><br>${esc(c.booking.slot)}</p>
      <p class="muted" style="margin:8px 0 0">Attend the session. When the trainer confirms that you completed the drills, your certificate is issued and you can be assigned to incidents.</p></div>`;
  else if(c.theory) body = `<label>Training centre</label><select id="centre">${CENTRES.map(x => `<option>${esc(x)}</option>`).join("")}</select>
      <label>Session</label><select id="slot">${SLOTS.map(x => `<option>${esc(x)}</option>`).join("")}</select>
      <div style="margin-top:12px"><button onclick="book('${c.id}')">Book practical training</button></div>`;
  else body = `<p class="muted">Pass the theory quiz first. Then you can book a session here.</p>`;
  return `<div class="card" id="practical"><h2 style="margin-top:0">Step 2: Practical training in person</h2>
    <p class="muted">${esc(c.practical.duration)} with a trainer. Rescue work is physical, so you are certified only after you have done these drills in person:</p>
    ${drills}${body}</div>`;
}
function courseView(c){
  const lessons = c.lessons.map((l, n) => `<div class="lesson"><h3>${n + 1}. ${esc(l.title)}</h3>
    <div class="video"><span class="play"></span><span><b>Video lesson: ${esc(l.title)}</b><br><span class="muted">Video coming soon. The key points are below.</span></span></div>
    <ul>${l.points.map(p => `<li>${esc(p)}</li>`).join("")}</ul></div>`).join("");
  const quiz = c.quiz.map((q, n) => `<div class="q"><p>${n + 1}. ${esc(q.q)}</p>${q.options.map((o, k) => `<label><input type="radio" name="q${n}" value="${k}"> ${esc(o)}</label>`).join("")}</div>`).join("");
  let result = "";
  if(RESULT && RESULT.passed) result = c.has_practical
    ? `<div class="alert" style="border-color:var(--ok);background:var(--okbg)"><h3>Theory passed: ${RESULT.score} of ${RESULT.total}</h3><p style="margin:0">Next, book your practical training below. You are certified only after a trainer confirms it.</p></div>`
    : `<div class="alert" style="border-color:var(--ok);background:var(--okbg)"><h3>Passed: ${RESULT.score} of ${RESULT.total}</h3><p style="margin:0 0 10px">You earned the certificate: <b>${esc(c.badge)}</b>.</p><button class="dark" onclick="OPEN=null;render()">Back to my courses</button></div>`;
  if(RESULT && !RESULT.passed) result = `<div class="alert"><h3>${RESULT.score} of ${RESULT.total} correct</h3><p style="margin:0">You need ${RESULT.need} to pass. Read the lessons again and retry.</p></div>`;
  const done = c.passed || c.theory;
  const quizCard = done && !(RESULT && RESULT.passed)
    ? `<div class="card"><h2 style="margin-top:0">${c.has_practical ? "Step 1: Theory quiz" : "Quiz"}</h2><span class="badge ok">${c.has_practical ? "Theory passed " + esc(c.theory || c.passed) : "Passed " + esc(c.passed)}</span></div>`
    : `<div class="card"><h2 style="margin-top:0">${c.has_practical ? "Step 1: Theory quiz" : "Quiz"}</h2><p class="muted">Answer all questions. You need 4 of 5 to pass.</p>
      ${done ? "" : quiz}<div class="err" id="err"></div>${result}
      ${done ? "" : `<button onclick="submitQuiz('${c.id}', ${c.quiz.length})">Submit answers</button>`}</div>`;
  return `<button class="ghost" onclick="OPEN=null;render()">&larr; Back to my courses</button>
    <div class="card" style="margin-top:14px"><span class="lvl" style="font:600 14px Barlow,Arial;color:var(--brand)">${c.level === 0 ? "For everyone" : "Rescuer level " + c.level}</span>
    <h1 style="margin:6px 0">${esc(c.title)}</h1><p class="muted">${esc(c.summary)}</p>${lessons}</div>
    ${quizCard}${practicalBlock(c)}`;
}
async function submitQuiz(id, count){
  const answers = [];
  for(let n = 0; n < count; n++){
    const pick = document.querySelector(`input[name=q${n}]:checked`);
    if(!pick){ $("err").textContent = "Please answer every question."; return; }
    answers.push(Number(pick.value));
  }
  RESULT = await api("/api/quiz", {course_id:id, answers});
  await refresh(true);
  if(RESULT.passed && $("practical")) $("practical").scrollIntoView({behavior:"smooth"});
}
async function book(id){ await api("/api/practical", {course_id:id, centre:$("centre").value, slot:$("slot").value}); RESULT = null; refresh(true); }
async function startLink(){ await api("/api/link/start", {}); refresh(true); }
async function savePhone(){
  const r = await api("/api/profile", {phone:$("myphone").value});
  if(r.error){ $("perr").textContent = r.error; return; }
  EDITING = false; SAVED = Date.now(); refresh(true);
}
async function setAvailable(v){ await api("/api/availability", {available:v}); refresh(true); }
async function answer(action){ await api("/api/assignment", {action}); refresh(true); }
refresh(true);
setInterval(() => { if(ME && ME.user && !EDITING && !(OPEN && document.querySelector("input[type=radio]"))) refresh(false); }, 6000);
</script></body></html>"""

CONTROL_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Control room | DisasterReady</title><style>__CSS__</style></head><body>
<div class="nav"><a class="brand" href="/"><i></i>DisasterReady</a>
<nav id="nav"><a href="/learn">Learn and certify</a></nav></div>
<main id="app" class="wide"><div class="card">Loading...</div></main>
<script>
const STATUS = {processing:"Agents working", new:"New", dispatched:"Team dispatched", resolved:"Resolved"};
const SEV = {5:"Critical", 4:"Severe", 3:"Serious", 2:"Moderate", 1:"Minor"};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let built = false, DATA = [], SELECTED = null, KNOWN = null, lastDetail = "", lastRoster = "", bannerTimer = null;
async function post(url, body){
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  return r.json();
}
function loginView(){
  built = false; KNOWN = null;
  $("nav").innerHTML = '<a href="/learn">Learn and certify</a>';
  $("app").innerHTML = `<div class="card auth"><h2>Control room login</h2>
    <p class="muted">For operators who watch incidents and dispatch rescue teams.</p>
    <label>Email</label><input id="email" type="email">
    <label>Password</label><input id="password" type="password">
    <div class="err" id="err"></div>
    <button class="dark" style="width:100%" onclick="login()">Log in</button>
    __DEMO_HINT__</div>`;
}
async function login(){
  const r = await post("/api/op/login", {email:$("email").value, password:$("password").value});
  if(r.error){ $("err").textContent = r.error; return; }
  load();
}
async function logout(){ await post("/api/op/logout", {}); loginView(); }
function build(){
  built = true; lastDetail = ""; lastRoster = "";
  $("nav").innerHTML = '<a href="/learn">Learn and certify</a><button onclick="logout()">Log out</button>';
  $("app").innerHTML = `
  <div id="banner"></div>
  <div class="grid g4" style="margin-bottom:16px">
    <div class="card stat" style="margin:0"><b id="s-active">0</b><span>Active incidents</span></div>
    <div class="card stat" style="margin:0"><b id="s-critical">0</b><span>Critical or severe</span></div>
    <div class="card stat" style="margin:0"><b id="s-people">0</b><span>People affected</span></div>
    <div class="card stat" style="margin:0"><b id="s-free">0</b><span>Rescuers free</span></div>
  </div>
  <div class="work">
    <div class="side"><div class="sidehead">Incoming reports <span class="muted">newest first</span></div><div id="list"><div class="empty">No incidents yet.</div></div></div>
    <div id="detail"><div class="card empty">When a report arrives by voice, it appears here with the plan, the shelter and the rescue team.</div></div>
  </div>
  <h2>Tools</h2>
  <div class="grid g2">
    <div class="card" style="margin:0"><h3 style="margin:0 0 8px">Send a typed report</h3>
      <textarea id="report" rows="2" placeholder="Example: Heavy flooding in Tolichowki, thirty people stranded in a school."></textarea>
      <div class="row" style="margin-top:8px"><button onclick="sendReport()">Send report</button><span class="muted" id="sent">Real reports arrive by voice through Omi. Any language works.</span></div></div>
    <div class="card" style="margin:0"><h3 style="margin:0 0 8px">Ask the Manager Agent</h3>
      <input id="question" placeholder="Example: Which incidents need medical teams?">
      <div class="row" style="margin-top:8px"><button class="dark" onclick="ask()">Ask</button></div>
      <p class="muted pre" id="answer" style="margin:8px 0 0"></p></div>
  </div>
  <h2>Practical training to confirm</h2><div class="card"><p class="muted" style="margin-top:0">People who passed a quiz and booked in-person training. Confirm only after the trainer reports that they completed the drills. A rescuer practical certifies the person for incidents. A citizen workshop is recorded on their awareness certificate and does not make them a rescuer.</p><div id="trainees" class="muted">No bookings waiting.</div></div>
  <h2>Certified rescuers</h2><div class="card"><div style="overflow-x:auto"><table><thead><tr><th>Name</th><th>Level</th><th>Area</th><th>Status</th></tr></thead><tbody id="roster"></tbody></table></div></div>
  <h2>Messages that were not emergencies</h2><div class="card"><p class="muted" style="margin-top:0">The Triage Agent filters out greetings and chat, and the caller is asked to describe a real emergency.</p><div id="filtered" class="muted">None yet.</div></div>`;
}
function ago(ts){
  const s = Math.max(0, Date.now() / 1000 - ts);
  if(s < 60) return "just now";
  if(s < 3600) return Math.floor(s / 60) + " min ago";
  if(s < 86400) return Math.floor(s / 3600) + " h ago";
  return Math.floor(s / 86400) + " d ago";
}
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
const cls = i => i.status === "resolved" ? "done" : i.severity >= 4 ? "crit" : i.severity === 3 ? "high" : "low";
const fresh = i => i.status === "new" && Date.now() / 1000 - i.ts < 300;
function title(i){
  const t = i.triage || {};
  if(!t.hazard) return "New report";
  return cap(String(t.hazard)) + (t.location ? " in " + t.location : "");
}
function pick(id){ SELECTED = id; lastDetail = ""; draw(); if(window.innerWidth < 900) $("detail").scrollIntoView({behavior:"smooth"}); }
async function setStatus(id, status){ await post("/incidents/" + id + "/status", {status}); load(); }
async function confirmPractical(person_id, course_id){ await post("/api/op/practical", {person_id, course_id}); lastRoster = ""; load(); }
async function sendReport(){
  const t = $("report");
  if(!t.value.trim()) return;
  await post("/report", {text:t.value});
  t.value = "";
  $("sent").textContent = "Report sent. The agents are working on it.";
  load();
}
async function ask(){
  const q = $("question").value.trim();
  if(!q) return;
  $("answer").textContent = "Thinking...";
  const r = await post("/ask", {question:q});
  $("answer").textContent = r.answer;
}
function secs(t){ const p = String(t || "").split(":").map(Number); return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : null; }
function needs(plan){
  const out = [];
  const add = t => t.split(/[,;]| - /).forEach(x => { x = x.trim().replace(/\.$/, "").replace(/^and\s+/i, ""); if(x && x.length < 60 && !out.includes(cap(x))) out.push(cap(x)); });
  let on = false;
  for(const raw of String(plan || "").split("\n")){
    const line = raw.trim().replace(/^[#*\s]+/, "");
    if(!line) continue;
    const at = line.indexOf(":");
    const head = at >= 0 ? line.slice(0, at) : "";
    if(head && head.length < 40 && /need|suppl|respond|resource/i.test(head)){ on = true; const rest = line.slice(at + 1).trim(); if(rest) add(rest.replace(/^[-\u2022\s]+/, "")); continue; }
    if(head && /shelter/i.test(head) && !/^[-\u2022\d]/.test(line)){ on = false; continue; }
    if(on && /^[-\u2022]/.test(raw.trim())) add(line.replace(/^[-\u2022\s]+/, ""));
    else if(on && !/^[-\u2022\d]/.test(raw.trim())) on = false;
  }
  return out.slice(0, 16);
}
function bar(s){ const pct = s.capacity ? Math.round(100 * s.free / s.capacity) : 0; return `<div class="bar"><i style="width:${pct}%"></i></div><div class="muted">${s.free} of ${s.capacity} places free</div>`; }
function tool(step){
  if(/Omi/.test(step)) return "t-omi";
  if(/Lyzr/.test(step)) return "t-lyzr";
  if(/Qdrant/.test(step)) return "t-qd";
  if(/Telegram/.test(step)) return "t-tg";
  return "t-op";
}
function stepBody(s){
  const d = s.detail || "";
  if(/^Triage done/.test(s.step)){
    try{
      const o = JSON.parse(d);
      return `<div class="kv">${Object.keys(o).filter(k => k !== "summary").map(k => `<span><em>${esc(k.replace(/_/g, " "))}</em>${esc(o[k] === null || o[k] === "" ? "not said" : o[k])}</span>`).join("")}</div>`;
    }catch(e){}
  }
  if(/^Shelters retrieved/.test(s.step)) return `<div class="kv">${d.split(" | ").map(x => `<span>${esc(x)}</span>`).join("")}</div>`;
  return d ? `<div class="clamp">${esc(d)}</div>` : "";
}
function timeline(i){
  return (i.steps || []).map(s => {
    const m = s.step.match(/^(.*?)\s*\((.*)\)$/);
    const name = m ? m[1] : s.step, who = m ? m[2] : "";
    return `<li class="${tool(s.step)}"><div class="thead"><b>${esc(name)}</b>${who ? `<span class="who">${esc(who)}</span>` : ""}<span class="muted">${esc(s.time)}</span></div>${stepBody(s)}</li>`;
  }).join("");
}
function callerBox(i){
  const c = i.caller;
  if(!c) return `<div class="caller"><div><span class="plabel">Caller</span><span class="muted">${i.source === "omi" ? "Reported before caller details were recorded." : "Typed in the control room, so there is no caller to contact."}</span></div></div>`;
  const name = c.name ? `<b class="mid">${esc(c.name)}</b><div class="muted">from their ${esc(c.name_source)}${c.area ? `, lives in ${esc(c.area)}` : ""}</div>` : '<div class="muted">Name not known. The caller has not linked a DisasterReady account.</div>';
  const phone = c.phone ? `<a class="callbtn" href="tel:${esc(c.phone)}">Call ${esc(c.phone)}</a>` : `<div class="muted">${c.linked ? "No phone number saved on their account." : "Not available. Omi does not share phone numbers, so the caller must link an account."}</div>`;
  const where = c.map ? `<b class="mid">${esc(c.address || (c.lat.toFixed(5) + ", " + c.lng.toFixed(5)))}</b><a href="${esc(c.map)}" target="_blank" rel="noopener">Open live location in Maps</a>` : '<div class="muted">Omi did not send a GPS position with this report. Use the spoken location.</div>';
  return `<div class="caller"><div><span class="plabel">Caller</span>${name}</div><div><span class="plabel">Phone</span>${phone}</div><div><span class="plabel">Caller's position</span>${where}</div></div>`;
}
function row(i){
  const src = i.source === "omi" ? "Omi voice" : "Typed";
  return `<button class="inrow ${cls(i)} ${i.id === SELECTED ? "sel" : ""}" onclick="pick('${i.id}')">
    <span class="sevbox">${i.severity || "?"}</span>
    <span class="rmain"><b>${esc(title(i))}</b><span>${ago(i.ts)} &middot; ${src}${i.reports > 1 ? ` &middot; ${i.reports} reports` : ""}</span></span>
    <span class="rstat">${fresh(i) ? '<span class="newtag">NEW</span>' : `<span class="badge">${STATUS[i.status] || esc(i.status)}</span>`}</span></button>`;
}
function detail(i){
  const t = i.triage || {};
  const sev = i.severity || 0;
  const team = i.team || [];
  const ready = team.filter(m => m.state === "ready").length;
  const first = (i.shelters || [])[0];
  const others = (i.shelters || []).slice(1);
  const names = [i.source === "omi" ? "Omi voice" : "Typed report", "Triage Agent", "Qdrant memory", "Planner Agent", "Allocation Agent", "SOS Agent", "Alert sent"];
  const stages = names.map((s, n) => `<span class="stage ${n < i.stage ? "on" : ""}">${n < i.stage ? "&#10003; " : ""}${s}</span>`).join('<span class="arrow">&rarr;</span>');
  const need = needs(i.plan);
  const st = i.steps || [];
  const sent = st.find(x => /^SOS sent/.test(x.step));
  const took = sent && st.length ? secs(sent.time) - secs(st[0].time) : null;
  const order = ["new", "dispatched", "resolved"];
  const at = Math.max(0, order.indexOf(i.status));
  const flow = order.map((s, n) => `<span class="fstep ${n < at ? "past" : n === at ? "now" : ""}"><i>${n < at ? "&#10003;" : n + 1}</i>${STATUS[s]}</span>`).join('<span class="fline"></span>');
  let action = "";
  if(i.status === "processing") action = '<span class="badge wait">Agents are working...</span>';
  if(i.status === "new") action = `<button onclick="setStatus('${i.id}','dispatched')">Dispatch team</button>`;
  if(i.status === "dispatched") action = `<button class="green" onclick="setStatus('${i.id}','resolved')">Mark as resolved</button>`;
  const teamList = team.map(m => `<li><span class="dot ${m.state === "ready" ? "ok" : ""}"></span><span><b>${esc(m.name)}</b>${m.lead ? ' <span class="badge dark">Lead</span>' : ""}${m.demo ? ' <span class="badge">sample</span>' : ""}<br><span class="muted">Level ${m.level} &middot; ${esc(m.area)} &middot; ${m.state === "ready" ? "Ready" : "Waiting to confirm"}</span></span></li>`).join("");
  const said = (i.transcripts || [i.transcript]).map(x => `<p>&ldquo;${esc(x)}&rdquo;</p>`).join("");
  return `<div class="detail ${cls(i)}">
    <div class="dhead">
      <div class="dsev"><b>${sev || "?"}</b><span>${SEV[sev] || "Checking"}</span></div>
      <div class="dtitle"><h2>${esc(t.summary || i.transcript)}</h2>
        <div class="meta">${esc(t.location || "Location not known yet")} &middot; reported ${ago(i.ts)} (${esc(i.created)}) &middot; ${i.source === "omi" ? "by voice through Omi" : "typed report"}${i.reports > 1 ? ` &middot; <b>${i.reports} reports merged</b>` : ""}</div></div>
      <div class="dact">${action}</div>
    </div>
    <div class="flow">${flow}</div>
    ${callerBox(i)}
    <div class="grid g3 panels">
      <div class="panel"><span class="plabel">People affected</span><b class="big">${esc(t.people_count ?? "?")}</b>
        <div class="muted">Injuries: ${esc(t.injuries || "none reported")}</div><div class="muted">Hazard: ${esc(t.hazard || "unknown")}</div></div>
      <div class="panel"><span class="plabel">Take them to</span>${first ? `<b class="mid">${esc(first.name)}</b>
        <div class="muted">${esc(first.area)} &middot; ${esc(first.facilities)}</div>${bar(first)}
        <a href="${esc(first.map)}" target="_blank" rel="noopener">Open in Maps</a>` : '<div class="muted">Finding shelters...</div>'}</div>
      <div class="panel"><span class="plabel">Rescue team ${team.length ? `<em>${ready} of ${team.length} ready</em>` : ""}</span>
        ${team.length ? `<ul class="team">${teamList}</ul>` : `<div class="muted">${i.stage >= 5 ? "No certified rescuers are free right now." : "Allocating..."}</div>`}</div>
    </div>
    ${i.team_reason ? `<p class="muted" style="margin:10px 0 0">Why this team (${i.team_by === "agent" ? "Allocation Agent" : "level and area rules"}): ${esc(i.team_reason)}</p>` : ""}
    ${i.sos ? `<div class="sos"><span class="plabel">Alert sent to responders</span>${esc(i.sos)}</div>` : ""}
    <div class="said"><span class="plabel">What the caller said</span>${said}</div>
    ${need.length ? `<div class="sect"><span class="plabel">What to send</span><div class="chips">${need.map(n => `<span class="chip">${esc(n)}</span>`).join("")}</div></div>` : ""}
    ${i.plan ? `<details id="d-plan" ${need.length ? "" : "open"}><summary>${need.length ? "Full plan from the Planner Agent" : "Response plan"}</summary><div class="pre">${esc(i.plan)}</div></details>` : ""}
    ${others.length ? `<div class="sect"><span class="plabel">Other shelters nearby</span><div class="grid g2" style="gap:10px">${others.map(s => `<div class="mini"><b>${esc(s.name)}</b><div class="muted">${esc(s.area)} &middot; ${esc(s.facilities)}</div>${bar(s)}<a href="${esc(s.map)}" target="_blank" rel="noopener">Open in Maps</a></div>`).join("")}</div></div>` : ""}
    <div class="sect"><span class="plabel">Agent workflow ${took !== null && took >= 0 ? `<em>Report to alert in ${took} seconds</em>` : ""}</span><div class="stages">${stages}</div></div>
    <details id="d-agents"><summary>Show each agent step (${st.length})</summary><ul class="tline">${timeline(i)}</ul></details>
  </div>`;
}
function banner(i){
  $("banner").innerHTML = `<div class="banner"><span class="pulse"></span><div><b>New report received</b><br>${esc((i.triage || {}).summary || i.transcript)}</div><button class="ghost" onclick="pick('${i.id}');$('banner').innerHTML=''">View</button></div>`;
  clearTimeout(bannerTimer); bannerTimer = setTimeout(() => { $("banner").innerHTML = ""; }, 20000);
  try{
    const a = new (window.AudioContext || window.webkitAudioContext)(), o = a.createOscillator(), g = a.createGain();
    o.connect(g); g.connect(a.destination); o.frequency.value = 880; g.gain.value = 0.08; o.start(); o.stop(a.currentTime + 0.25);
  }catch(e){}
}
function draw(){
  const active = DATA.filter(i => i.status !== "resolved");
  $("s-active").textContent = active.length;
  $("s-critical").textContent = active.filter(i => i.severity >= 4).length;
  $("s-people").textContent = active.reduce((n, i) => n + (parseInt((i.triage || {}).people_count) || 0), 0);
  if(!DATA.length) return;
  if(!DATA.some(i => i.id === SELECTED)) SELECTED = DATA[0].id;
  $("list").innerHTML = DATA.map(row).join("");
  const cur = DATA.find(i => i.id === SELECTED);
  const text = JSON.stringify(cur) + ago(cur.ts);
  if(text === lastDetail) return;
  lastDetail = text;
  const open = [...document.querySelectorAll("#detail details")].map(d => [d.id, d.open]);
  $("detail").innerHTML = detail(cur);
  if(open.length && $("detail").dataset.id === cur.id) open.forEach(([id, o]) => { const d = $(id); if(d) d.open = o; });
  $("detail").dataset.id = cur.id;
}
async function load(){
  const res = await fetch("/incidents");
  if(res.status === 401){ if(!$("password")) loginView(); return; }
  if(!built) build();
  DATA = await res.json();
  const ids = DATA.map(i => i.id);
  if(KNOWN){
    const added = DATA.filter(i => !KNOWN.includes(i.id));
    if(added.length){ SELECTED = added[0].id; lastDetail = ""; banner(added[0]); }
  }
  KNOWN = ids;
  draw();
  const extra = await (await fetch("/api/roster")).json();
  const text = JSON.stringify(extra);
  if(text === lastRoster) return;
  lastRoster = text;
  const roster = extra.rescuers;
  $("s-free").textContent = roster.filter(r => r.status === "Free").length;
  $("roster").innerHTML = roster.map(r => `<tr><td>${esc(r.name)}${r.demo ? ' <span class="badge">sample</span>' : ""}</td><td>Level ${r.level}</td><td>${esc(r.area)}</td><td><span class="badge ${r.status === "Free" ? "ok" : r.status === "Not available" ? "" : "wait"}">${esc(r.status)}</span></td></tr>`).join("") || '<tr><td colspan="4" class="muted">No certified rescuers yet.</td></tr>';
  $("trainees").innerHTML = extra.trainees.length ? `<div style="overflow-x:auto"><table><thead><tr><th>Name</th><th>Training for</th><th>Centre</th><th>Session</th><th></th></tr></thead><tbody>${extra.trainees.map(t => `<tr><td>${esc(t.name)}<br><span class="muted">${esc(t.area)}</span></td><td>${esc(t.course)}<br><span class="badge ${t.kind === "Citizen workshop" ? "" : "dark"}" style="margin-top:4px">${esc(t.kind)}</span></td><td>${esc(t.centre)}</td><td>${esc(t.slot)}</td><td><button class="green" onclick="confirmPractical('${t.person_id}','${t.course_id}')">Confirm completed</button></td></tr>`).join("")}</tbody></table></div>` : "No bookings waiting.";
  $("filtered").innerHTML = extra.filtered.length ? `<ul class="plain">${extra.filtered.map(f => `<li><span class="muted">${esc(f.time)} &middot; ${f.source === "omi" ? "Omi voice" : "Typed"}</span><br>&ldquo;${esc(f.text)}&rdquo;</li>`).join("")}</ul>` : "None yet.";
}
load(); setInterval(load, 3000);
</script></body></html>"""

DEMO_HINT = ('<p class="muted" style="margin-top:14px">Demo access for judges: '
             f'<b>{OPERATOR_EMAIL}</b> / <b>{DEFAULT_OP_PASSWORD}</b></p>'
             if OPERATOR_PASSWORD == DEFAULT_OP_PASSWORD else "")


# ---------- web endpoints ----------
app = FastAPI(title="DisasterReady")
load_saved()


@app.get("/", response_class=HTMLResponse)
def home_page():
    return HOME_PAGE.replace("__CSS__", CSS)


@app.get("/learn", response_class=HTMLResponse)
def learn_page():
    return (LEARN_PAGE.replace("__CSS__", CSS)
            .replace("__AREAS__", json.dumps(AREAS))
            .replace("__CENTRES__", json.dumps(TRAINING_CENTRES))
            .replace("__SLOTS__", json.dumps(TRAINING_SLOTS)))


@app.get("/control", response_class=HTMLResponse)
def control_page():
    return (CONTROL_PAGE.replace("__CSS__", CSS)
            .replace("__DEMO_HINT__", DEMO_HINT))


# ----- learner accounts and courses -----
def clean_phone(value):
    """Returns the phone number, None if empty, or False if not valid."""
    text = str(value or "").strip()
    if not text:
        return None
    digits = re.sub(r"[\s\-()]", "", text)
    return digits if re.fullmatch(r"\+?\d{7,15}", digits) else False


@app.post("/api/link/start")
def link_start(request: Request):
    """Open a five-minute window in which this person's Omi can be linked
    by saying 'link my account', without reading out the code."""
    p = learner(request)
    if not p:
        return denied()
    p["link_until"] = time.time() + 300
    save_person(p)
    return {"ok": True}


@app.post("/api/profile")
def profile(request: Request, body: dict = Body(...)):
    p = learner(request)
    if not p:
        return denied()
    phone = clean_phone(body.get("phone"))
    if phone is False:
        return {"error": "Please enter a valid phone number."}
    p["phone"] = phone
    save_person(p)
    return {"ok": True}


@app.post("/api/register")
def register(body: dict = Body(...)):
    name = str(body.get("name", "")).strip()[:60]
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))
    area = body.get("area") if body.get("area") in AREAS else AREAS[0]
    goal = "rescuer" if body.get("goal") == "rescuer" else "citizen"
    phone = clean_phone(body.get("phone"))
    if phone is False:
        return {"error": "Please enter a valid phone number, or leave it empty."}
    if not name:
        return {"error": "Please enter your name."}
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return {"error": "Please enter a valid email address."}
    if len(password) < 6:
        return {"error": "The password needs at least 6 characters."}
    pid = person_id(email)
    if pid in PEOPLE:
        return {"error": "This email is already registered. Please log in."}
    person = {"id": pid, "name": name, "email": email,
              "password": hash_password(password), "area": area, "goal": goal,
              "level": 0, "available": False, "demo": False, "passed": {},
              "assignment": None, "joined": now("%d %b %Y"),
              "phone": phone, "link_code": new_link_code(), "omi_uid": None}
    PEOPLE[pid] = person
    save_person(person, new_vector=True)
    return with_cookie({"ok": True}, "dr_user", make_token({"id": pid}))


@app.post("/api/login")
def login(body: dict = Body(...)):
    email = str(body.get("email", "")).strip().lower()
    person = PEOPLE.get(person_id(email))
    if not person or not check_password(str(body.get("password", "")),
                                        person.get("password")):
        return {"error": "Wrong email or password."}
    return with_cookie({"ok": True}, "dr_user",
                       make_token({"id": person["id"]}))


@app.post("/api/logout")
def logout():
    return with_cookie({"ok": True}, "dr_user", None)


@app.get("/api/me")
def me(request: Request):
    p = learner(request)
    if not p:
        return {"user": None}
    courses = []
    for c in COURSES:
        req = c["requires"]
        courses.append({
            "id": c["id"], "level": c["level"], "title": c["title"],
            "badge": c["badge"], "summary": c["summary"],
            "lessons": c["lessons"],
            "quiz": [{"q": q["q"], "options": q["options"]}
                     for q in c["quiz"]],
            "passed": p["passed"].get(c["id"]),
            "theory": p.get("theory", {}).get(c["id"]),
            "booking": (None if c["id"] not in PRACTICAL
                        or practical_done(p, c["id"])
                        else p.get("practical", {}).get(c["id"])),
            "has_practical": needs_practical(c["id"]),
            "optional_workshop": bool(PRACTICAL.get(c["id"], {})
                                      .get("optional")),
            "workshop": p.get("workshop", {}).get(c["id"]),
            "practical": PRACTICAL.get(c["id"]),
            "locked": bool(req and req not in p["passed"]),
            "requires_title": COURSE_BY_ID[req]["title"] if req else "",
        })
    assignment = None
    a = p.get("assignment")
    inc = INCIDENTS.get(a["incident_id"]) if a else None
    if a and not inc:
        p["assignment"] = None
        save_person(p)
    elif inc:
        t = inc.get("triage") or {}
        first = inc["shelters"][0] if inc.get("shelters") else None
        assignment = {
            "state": a["state"], "summary": summary_of(inc),
            "severity": inc["severity"], "location": t.get("location"),
            "team": ", ".join(m["name"] + (" (lead)" if m["lead"] else "")
                              for m in inc.get("team", [])),
            "shelter": first["name"] if first else None,
            "map": first["map"] if first else None}
    if not p.get("link_code"):
        p["link_code"] = new_link_code()
        save_person(p)
    user = {"phone": p.get("phone") or "", "link_code": p["link_code"],
            "omi_linked": bool(p.get("omi_uid")),
            "link_waiting": p.get("link_until", 0) > time.time(),
            "name": p["name"], "area": p["area"], "goal": p["goal"],
            "level": p["level"], "level_name": LEVEL_NAME[p["level"]],
            "available": bool(p.get("available"))}
    return {"user": user, "courses": courses, "assignment": assignment}


@app.post("/api/quiz")
def quiz(request: Request, body: dict = Body(...)):
    p = learner(request)
    if not p:
        return denied()
    c = COURSE_BY_ID.get(body.get("course_id"))
    answers = body.get("answers") or []
    if not c or len(answers) != len(c["quiz"]):
        return {"error": "Please answer every question."}
    if c["requires"] and c["requires"] not in p["passed"]:
        return {"error": "This course is still locked."}
    total = len(c["quiz"])
    score = sum(1 for q, a in zip(c["quiz"], answers) if q["answer"] == a)
    need = -(-total * 4 // 5)
    passed = score >= need
    if passed and c["id"] not in p["passed"]:
        if needs_practical(c["id"]):
            # Rescuer levels: the quiz is only the theory step.
            p.setdefault("theory", {}).setdefault(c["id"], now("%d %b %Y"))
        else:
            p["passed"][c["id"]] = now("%d %b %Y")
        save_person(p)
    return {"score": score, "total": total, "need": need, "passed": passed}


@app.post("/api/practical")
def book_practical(request: Request, body: dict = Body(...)):
    """A learner books in-person practical training after the theory quiz."""
    p = learner(request)
    if not p:
        return denied()
    cid = body.get("course_id")
    if cid not in PRACTICAL:
        return {"error": "This course has no practical session."}
    quiz_done = (cid in p["passed"] if PRACTICAL[cid].get("optional")
                 else cid in p.get("theory", {}))
    if not quiz_done:
        return {"error": "Pass the quiz first."}
    if practical_done(p, cid):
        return {"ok": True}
    centre = body.get("centre")
    slot = body.get("slot")
    p.setdefault("practical", {})[cid] = {
        "centre": centre if centre in TRAINING_CENTRES else TRAINING_CENTRES[0],
        "slot": slot if slot in TRAINING_SLOTS else TRAINING_SLOTS[0],
        "booked": now("%d %b %Y")}
    save_person(p)
    return {"ok": True}


@app.post("/api/availability")
def availability(request: Request, body: dict = Body(...)):
    p = learner(request)
    if not p:
        return denied()
    if p["level"] >= 1:
        p["available"] = bool(body.get("available"))
        save_person(p)
    return {"ok": True}


@app.post("/api/assignment")
def assignment(request: Request, body: dict = Body(...)):
    """A rescuer confirms they are ready, or says they cannot go."""
    p = learner(request)
    if not p:
        return denied()
    a = p.get("assignment")
    inc = INCIDENTS.get(a["incident_id"]) if a else None
    if not inc:
        return {"ok": True}
    member = next((m for m in inc["team"] if m["id"] == p["id"]), None)
    if body.get("action") == "ready" and member:
        a["state"] = member["state"] = "ready"
        log(inc, "Rescuer confirmed ready", p["name"])
        send_telegram(f"{p['name']} (Level {p['level']}) is ready for: "
                      f"{summary_of(inc)}")
    elif body.get("action") == "decline" and member:
        inc["team"].remove(member)
        p["assignment"] = None
        log(inc, "Rescuer cannot go", p["name"])
        location = (inc.get("triage") or {}).get("location")
        current = [m["id"] for m in inc["team"]]
        spare = pick_by_rules(
            free_rescuers(location, exclude=current + [p["id"]]),
            location, 1, 1)
        if spare:
            s = spare[0]
            state = "ready" if s.get("demo") else "assigned"
            s["assignment"] = {"incident_id": inc["id"], "state": state}
            save_person(s)
            inc["team"].append({"id": s["id"], "name": s["name"],
                                "level": s["level"], "area": s["area"],
                                "lead": False, "state": state,
                                "demo": bool(s.get("demo"))})
            log(inc, "Replacement rescuer allocated (level and area rules)",
                f"{s['name']} (L{s['level']})")
        if inc["team"] and not any(m["lead"] for m in inc["team"]):
            max(inc["team"], key=lambda m: m["level"])["lead"] = True
    save_person(p)
    save(inc)
    return {"ok": True}


# ----- control room -----
@app.post("/api/op/login")
def op_login(body: dict = Body(...)):
    email = str(body.get("email", "")).strip().lower()
    ok = (hmac.compare_digest(email, OPERATOR_EMAIL.lower())
          and hmac.compare_digest(str(body.get("password", "")),
                                  OPERATOR_PASSWORD))
    if not ok:
        return {"error": "Wrong email or password."}
    return with_cookie({"ok": True}, "dr_op", make_token({"op": True}))


@app.post("/api/op/logout")
def op_logout():
    return with_cookie({"ok": True}, "dr_op", None)


@app.post("/api/op/practical")
def confirm_practical(request: Request, body: dict = Body(...)):
    """The trainer or operator confirms a learner completed the practical."""
    if not is_operator(request):
        return denied()
    p = PEOPLE.get(body.get("person_id"))
    c = COURSE_BY_ID.get(body.get("course_id"))
    if not p or not c or c["id"] not in p.get("practical", {}):
        return {"error": "Booking not found."}
    if PRACTICAL[c["id"]].get("optional"):
        # Citizen workshop: recorded on the certificate, no rescuer level.
        p.setdefault("workshop", {}).setdefault(c["id"], now("%d %b %Y"))
        save_person(p)
    elif c["id"] not in p["passed"]:
        p["passed"][c["id"]] = now("%d %b %Y")
        if c["level"] > p["level"]:
            first_certificate = p["level"] == 0
            p["level"] = c["level"]
            if first_certificate:
                p["available"] = True
        save_person(p, new_vector=True)
    return {"ok": True}


@app.get("/api/roster")
def roster(request: Request):
    if not is_operator(request):
        return denied()
    rows = []
    for p in PEOPLE.values():
        if p.get("level", 0) < 1:
            continue
        a = p.get("assignment")
        status = ("Ready for incident" if a and a["state"] == "ready"
                  else "Assigned, waiting to confirm" if a
                  else "Free" if p.get("available") else "Not available")
        rows.append({"name": p["name"], "level": p["level"], "area": p["area"],
                     "status": status, "demo": bool(p.get("demo"))})
    rows.sort(key=lambda r: (-r["level"], r["name"]))
    trainees = []
    for p in PEOPLE.values():
        for cid, b in p.get("practical", {}).items():
            if cid in PRACTICAL and not practical_done(p, cid):
                optional = PRACTICAL[cid].get("optional")
                trainees.append({"person_id": p["id"], "course_id": cid,
                                 "name": p["name"], "area": p["area"],
                                 "kind": ("Citizen workshop" if optional
                                          else "Rescuer practical"),
                                 "course": (PRACTICAL[cid]["name"] if optional
                                            else COURSE_BY_ID[cid]["badge"]),
                                 "centre": b["centre"], "slot": b["slot"]})
    return {"rescuers": rows, "filtered": FILTERED, "trainees": trainees}


@app.get("/incidents")
def incidents(request: Request):
    if not is_operator(request):
        return denied()
    items = list(INCIDENTS.values())
    items.sort(key=lambda i: (i["status"] == "resolved", -i.get("ts", 0)))
    return items


@app.post("/report")
def typed_report(request: Request, background: BackgroundTasks,
                 body: dict = Body(...)):
    """Typed report from the control room, for testing without Omi."""
    if not is_operator(request):
        return denied()
    text = str(body.get("text", "")).strip()
    if text:
        background.add_task(run_pipeline, "web-demo", text)
    return {"status": "ok"}


@app.post("/incidents/{incident_id}/status")
def set_status(incident_id: str, request: Request, body: dict = Body(...)):
    if not is_operator(request):
        return denied()
    inc = INCIDENTS.get(incident_id)
    status = body.get("status")
    if not inc or status not in ("dispatched", "resolved"):
        return {"status": "not found"}
    change_status(inc, status)
    return {"status": "ok"}


@app.post("/ask")
def ask(request: Request, body: dict = Body(...)):
    """Question answering over past incidents, through the Manager Agent."""
    if not is_operator(request):
        return denied()
    question = str(body.get("question", "")).strip()
    if not MANAGER_ID:
        return {"answer": "The Manager Agent is not connected yet."}
    hits = qdrant.query_points("incidents", query=embed(question),
                               limit=5).points
    lines = []
    for h in hits:
        inc = INCIDENTS.get(str(h.id))
        if inc:
            team = ", ".join(m["name"] for m in inc.get("team", [])) or "none"
            lines.append(f"- {summary_of(inc)} | severity {inc['severity']}/5 "
                         f"| status {inc['status']} | reports {inc['reports']} "
                         f"| team: {team} | plan: {inc.get('plan', '')}")
    context = "\n".join(lines) or "none"
    answer = ask_lyzr(MANAGER_ID, "dashboard", str(uuid.uuid4()),
                      f"CONTEXT:\n{context}\n\nQUESTION: {question}")
    return {"answer": answer or "No answer returned.", "context": lines}


# ----- Omi webhook -----
@app.post("/omi/conversation")
def on_conversation(background: BackgroundTasks, uid: str = "unknown",
                    payload: Any = Body(...)):
    """Omi calls this when a conversation is finished."""
    if not isinstance(payload, dict) or payload.get("discarded"):
        return {"status": "ignored"}
    segments = payload.get("transcript_segments", [])
    transcript = " ".join(s.get("text", "") for s in segments).strip()
    if not transcript:
        return {"status": "ok"}
    linked = link_account(uid, transcript)
    if linked:
        message = (f"Your Omi app is now linked to the DisasterReady account "
                   f"of {linked['name']}. The control room will see your name "
                   "and phone number when you report an emergency.")
        FILTERED.insert(0, {"time": now("%d %b %H:%M"), "source": "omi",
                            "text": f"Account linked: {linked['name']}"})
        del FILTERED[20:]
        if not notify_caller(uid, message):
            send_telegram(f"Reply to caller: {message}")
        return {"status": "linked"}
    omi_name = next((s.get("speaker_name") for s in segments
                     if s.get("is_user") and s.get("speaker_name")), None)
    geo = payload.get("geolocation") or (payload.get("structured") or {}).get(
        "geolocation")
    background.add_task(run_pipeline, uid, transcript, "omi",
                        caller_info(uid, omi_name, geo))
    return {"status": "ok"}