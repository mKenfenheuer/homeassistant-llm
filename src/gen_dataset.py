"""Generate the German SFT dataset for Home Assistant tool calling.

Every sample uses a freshly randomised house (rooms, device names, entity_ids,
states), so the model learns to map names from the device list to entity_ids
instead of memorising one house. Prompt and tool schema come from
custom_components/ha_llm_agent/llm_schema.py, the same module the integration
uses at runtime.

Each JSONL row: {"messages": [...], "meta": {"cat": ..., "gold": [...],
"answer_contains": [...]}}. "meta" is ignored by training and used by
src/eval_model.py.

Usage: python src/gen_dataset.py OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

R = random.Random()

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, preposition phrase, compound prefix, slugs
ROOMS = [
    ("Wohnzimmer", "im Wohnzimmer", "Wohnzimmer", ["wohnzimmer", "living_room", "wz"]),
    ("Küche", "in der Küche", "Küchen", ["kueche", "kitchen"]),
    ("Schlafzimmer", "im Schlafzimmer", "Schlafzimmer", ["schlafzimmer", "bedroom", "sz"]),
    ("Bad", "im Bad", "Bad", ["bad", "bathroom", "badezimmer"]),
    ("Badezimmer", "im Badezimmer", "Badezimmer", ["badezimmer", "bathroom"]),
    ("Kinderzimmer", "im Kinderzimmer", "Kinderzimmer", ["kinderzimmer", "kids_room"]),
    ("Büro", "im Büro", "Büro", ["buero", "office"]),
    ("Arbeitszimmer", "im Arbeitszimmer", "Arbeitszimmer", ["arbeitszimmer", "study"]),
    ("Flur", "im Flur", "Flur", ["flur", "hallway", "diele"]),
    ("Esszimmer", "im Esszimmer", "Esszimmer", ["esszimmer", "dining_room"]),
    ("Gästezimmer", "im Gästezimmer", "Gästezimmer", ["gaestezimmer", "guest_room"]),
    ("Keller", "im Keller", "Keller", ["keller", "basement"]),
    ("Garage", "in der Garage", "Garagen", ["garage"]),
    ("Garten", "im Garten", "Garten", ["garten", "garden"]),
    ("Terrasse", "auf der Terrasse", "Terrassen", ["terrasse", "terrace"]),
    ("Balkon", "auf dem Balkon", "Balkon", ["balkon", "balcony"]),
    ("Dachboden", "auf dem Dachboden", "Dachboden", ["dachboden", "attic"]),
    ("Waschküche", "in der Waschküche", "Waschküchen", ["waschkueche", "laundry"]),
    ("Gäste-WC", "im Gäste-WC", "WC", ["gaeste_wc", "wc"]),
    ("Ankleide", "in der Ankleide", "Ankleide", ["ankleide", "dressing_room"]),
    ("Eingang", "im Eingang", "Eingangs", ["eingang", "entrance"]),
    ("Treppenhaus", "im Treppenhaus", "Treppenhaus", ["treppenhaus", "stairs"]),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
INDOOR = [r[0] for r in ROOMS if r[0] not in ("Garten", "Terrasse", "Balkon", "Garage")]

# device type: (label, gender m/f/n/pl, slugs, compound-able)
TYPES = {
    "light": [
        ("Deckenlicht", "n", ["decke", "deckenlicht", "ceiling"], False),
        ("Licht", "n", ["licht", "light"], True),
        ("Deckenlampe", "f", ["deckenlampe", "ceiling_lamp"], False),
        ("Lampe", "f", ["lampe", "lamp"], True),
        ("Stehlampe", "f", ["stehlampe", "floor_lamp"], False),
        ("Leselampe", "f", ["leselampe", "reading_lamp"], False),
        ("LED-Streifen", "m", ["led_strip", "ledstreifen"], False),
        ("Spots", "pl", ["spots", "spotlights"], False),
        ("Wandlampe", "f", ["wandlampe", "wall_lamp"], False),
        ("Lichterkette", "f", ["lichterkette", "fairy_lights"], False),
        ("Pendelleuchte", "f", ["pendelleuchte", "pendant"], False),
        ("Nachttischlampe", "f", ["nachttischlampe", "bedside"], False),
        ("Spiegellicht", "n", ["spiegel", "mirror_light"], False),
    ],
    "switch": [
        ("Steckdose", "f", ["steckdose", "plug", "socket"], False),
        ("Kaffeemaschine", "f", ["kaffeemaschine", "coffee"], False),
        ("Heizlüfter", "m", ["heizluefter", "heater"], False),
        ("Luftentfeuchter", "m", ["entfeuchter", "dehumidifier"], False),
        ("Teichpumpe", "f", ["teichpumpe", "pond_pump"], False),
        ("Bewässerung", "f", ["bewaesserung", "irrigation"], False),
        ("Weihnachtsbaum", "m", ["weihnachtsbaum", "xmas_tree"], False),
        ("Drucker", "m", ["drucker", "printer"], False),
        ("Wasserkocher", "m", ["wasserkocher", "kettle"], False),
        ("Monitor", "m", ["monitor", "screen"], False),
    ],
    "cover": [
        ("Rollladen", "m", ["rollladen", "shutter"], True),
        ("Rollläden", "pl", ["rolllaeden", "shutters"], False),
        ("Jalousie", "f", ["jalousie", "blinds"], False),
        ("Rollo", "n", ["rollo", "blind"], False),
        ("Markise", "f", ["markise", "awning"], False),
    ],
    "climate": [
        ("Heizung", "f", ["heizung", "heating", "thermostat"], True),
        ("Thermostat", "n", ["thermostat", "trv"], False),
        ("Klimaanlage", "f", ["klima", "aircon"], False),
        ("Fußbodenheizung", "f", ["fussbodenheizung", "floor_heating"], False),
    ],
    "fan": [
        ("Ventilator", "m", ["ventilator", "fan"], False),
        ("Deckenventilator", "m", ["deckenventilator", "ceiling_fan"], False),
        ("Lüftung", "f", ["lueftung", "ventilation"], False),
    ],
    "media_player": [
        ("Fernseher", "m", ["tv", "fernseher"], False),
        ("Lautsprecher", "m", ["speaker", "sonos", "lautsprecher"], False),
        ("Soundbar", "f", ["soundbar"], False),
        ("Radio", "n", ["radio"], False),
    ],
}

LOCKS = [("Haustür", "f", ["haustuer", "front_door"]),
         ("Kellertür", "f", ["kellertuer", "basement_door"]),
         ("Terrassentür", "f", ["terrassentuer", "patio_door"]),
         ("Gartentor", "n", ["gartentor", "garden_gate"]),
         ("Hintertür", "f", ["hintertuer", "back_door"])]
VACUUMS = [("Saugroboter", "m", ["saugroboter", "roborock", "robot"]),
           ("Staubsauger", "m", ["staubsauger", "vacuum"])]
SCENES = ["Filmabend", "Gute Nacht", "Kochen", "Entspannen", "Party", "Lesen",
          "Aufwachen", "Abendessen", "Romantisch", "Hell", "Konzentration"]
SCRIPTS = ["Alles aus", "Urlaubsmodus", "Gute-Nacht-Routine", "Kaffee kochen",
           "Haus verlassen", "Ankommen"]
BUTTONS = [("Garagentor", "n", ["garagentor_taster", "garage_door_button"]),
           ("Klingel", "f", ["klingel", "doorbell_button"])]

COLORS = {"rot": "red", "grün": "green", "blau": "blue", "gelb": "yellow",
          "orange": "orange", "lila": "purple", "pink": "pink", "weiß": "white",
          "türkis": "turquoise", "violett": "violet"}
COLOR_TEMPS = {"warmweiß": 2700, "kaltweiß": 6000, "neutralweiß": 4000}

UNKNOWN_DEVICES = [("Sauna", "f"), ("Whirlpool", "m"), ("Pool-Pumpe", "f"),
                   ("Kamin", "m"), ("Aquarium", "n"), ("Rasenmäher", "m"),
                   ("Springbrunnen", "m"), ("Wäschetrockner", "m"),
                   ("Geschirrspüler", "m"), ("Backofen", "m"), ("Klimagerät", "n"),
                   ("Garagentor", "n"), ("Torbeleuchtung", "f")]

ART_ACC = {"m": "den", "f": "die", "n": "das", "pl": "die"}
ART_NOM = {"m": "der", "f": "die", "n": "das", "pl": "die"}
ART_DAT = {"m": "dem", "f": "der", "n": "dem", "pl": "den"}
ALL_ACC = {"m": "alle", "f": "alle", "n": "alle", "pl": "alle"}


def de_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}".replace(".", ",")


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "Deckenlicht"
    gender: str = "n"
    state: str = "off"
    attrs: dict = field(default_factory=dict)

    @property
    def plural(self) -> bool:
        return self.gender == "pl"


class House:
    def __init__(self, size: int):
        self.ents: list[Ent] = []
        self.used_ids: set[str] = set()
        self.used_names: set[str] = set()
        self.cryptic = R.random() < 0.3       # some houses use vendor ids
        self.english_ids = R.random() < 0.4
        n_rooms = max(2, min(len(ROOMS) - 4, size // 4 + R.randint(-1, 2)))
        names = R.sample([r[0] for r in ROOMS if r[0] != "Badezimmer"], n_rooms)
        if "Bad" in names and R.random() < 0.4:
            names[names.index("Bad")] = "Badezimmer"
        self.rooms = names
        self._populate(size)

    # ---- id / name helpers ----
    def _slug(self, room: str | None, type_slugs: list[str], domain: str) -> str:
        rslugs = ROOM_BY_NAME[room][3] if room else ["haus", "home"]
        if self.cryptic and R.random() < 0.6:
            vendor = R.choice(["shelly1", "shellyplus1pm", "hue_color_lamp",
                               "tz3000", "sonoff_basic", "zbmini", "tasmota",
                               "fritz_dect", "homematic_hm", "tuya"])
            tail = "".join(R.choices("0123456789abcdef", k=R.choice([4, 6])))
            base = f"{vendor}_{tail}"
        else:
            rs = rslugs[1] if self.english_ids and len(rslugs) > 1 else rslugs[0]
            ts = type_slugs[1] if self.english_ids and len(type_slugs) > 1 else type_slugs[0]
            base = R.choice([f"{rs}_{ts}", f"{ts}_{rs}", f"{rs}_{ts}"])
        eid = f"{domain}.{base}"
        i = 2
        while eid in self.used_ids:
            eid = f"{domain}.{base}_{i}"
            i += 1
        self.used_ids.add(eid)
        return eid

    def _name(self, room: str | None, label: str, compound: bool) -> str:
        opts = []
        if room:
            opts += [f"{room} {label}", f"{label} {room}"]
            if compound:
                opts += [ROOM_BY_NAME[room][2] + label.lower()]
        else:
            opts += [label]
        name = R.choice(opts)
        i = 2
        base = name
        while name in self.used_names:
            name = f"{base} {i}"
            i += 1
        self.used_names.add(name)
        return name

    def add(self, domain, room, label, gender, slugs, compound=False,
            name=None, state="off", attrs=None):
        eid = self._slug(room, slugs, domain)
        if name is None:
            name = self._name(room, label, compound)
        else:
            self.used_names.add(name)
        e = Ent(eid, name, room, domain, label, gender, state, attrs or {})
        self.ents.append(e)
        return e

    def _populate(self, size: int):
        for room in self.rooms:
            outdoor = room in ("Garten", "Terrasse", "Balkon")
            # lights: 1-3 per room
            for lt in R.sample(TYPES["light"], R.choice([1, 1, 1, 2, 2, 3])):
                on = R.random() < 0.4
                self.add("light", room, lt[0], lt[1], lt[2], lt[3],
                         state="on" if on else "off",
                         attrs={"brightness": R.randint(20, 255)} if on else {})
            if not outdoor and room not in ("Flur", "Treppenhaus", "Gäste-WC") \
                    and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                         state="closed" if pos == 0 else "open",
                         attrs={"current_position": pos})
            if room in ("Terrasse", "Balkon") and R.random() < 0.5:
                self.add("cover", room, "Markise", "f", ["markise", "awning"],
                         state=R.choice(["open", "closed"]),
                         attrs={"current_position": R.choice([0, 100])})
            if not outdoor and room not in ("Garage",) and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:3] if room != "Bad" else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, "Temperatur", "f",
                         ["temperatur", "temperature"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "Luftfeuchtigkeit", "f",
                         ["luftfeuchtigkeit", "humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "Fenster", "n",
                         ["fenster", "window"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Wohnzimmer", "Schlafzimmer", "Küche", "Büro",
                        "Kinderzimmer") and R.random() < 0.4:
                mt = R.choice(TYPES["media_player"])
                playing = R.random() < 0.4
                attrs = {"volume_level": R.choice([0.2, 0.3, 0.35, 0.5])}
                if playing:
                    attrs.update(media_title=R.choice(
                        ["Tagesschau", "Bohemian Rhapsody", "Atemlos",
                         "Tatort", "Die Maus", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "Helene Fischer", ""]))
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""})
            if room in ("Schlafzimmer", "Büro", "Wohnzimmer", "Dachboden") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "Leistung", "f", ["leistung", "power"],
                         name=self._name(room, R.choice(["Steckdose Leistung", "Leistung"]), False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            area = R.choice(self.rooms + [None])
            self.add("alarm_control_panel", None, "Alarmanlage", "f",
                     ["alarmanlage", "alarm"], name="Alarmanlage",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Flur", "Eingang", None]) if lk[0] == "Haustür" else None
                room = room if room in self.rooms else None
                self.add("lock", room, lk[0], lk[1], lk[2], name=lk[0],
                         state=R.choice(["locked", "locked", "unlocked"]))
        if R.random() < 0.35:
            vt = R.choice(VACUUMS)
            self.add("vacuum", None, vt[0], vt[1], vt[2], name=vt[0],
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.6:
            for sc in R.sample(SCENES, R.randint(1, 4)):
                self.add("scene", None, sc, "f", [sc.lower().replace(" ", "_")
                         .replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")],
                         name=sc, state="2026-10-30T19:12:00+00:00")
        if R.random() < 0.35:
            for sc in R.sample(SCRIPTS, R.randint(1, 2)):
                self.add("script", None, sc, "n", [sc.lower().replace(" ", "_")
                         .replace("-", "_")], name=sc, state="off")
        if R.random() < 0.5:
            self.add("sensor", None, "Außentemperatur", "f",
                     ["aussentemperatur", "outdoor_temperature"],
                     name=R.choice(["Außentemperatur", "Temperatur Außen",
                                    "Wetterstation Temperatur"]),
                     state=de_num(round(R.uniform(-6, 31), 1)).replace(",", "."),
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        if "Garage" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garage", "Garagentor", "n",
                     ["garagentor", "garage_door"], name="Garagentor",
                     state=R.choice(["open", "closed", "closed"]),
                     attrs={"current_position": 0})
        # pad tiny houses with extra lights / sensors
        while len(self.ents) < size:
            room = R.choice(self.rooms)
            lt = R.choice(TYPES["light"])
            self.add("light", room, lt[0], lt[1], lt[2], lt[3],
                     state=R.choice(["on", "off"]))

    # ---- queries ----
    def of(self, domain, room="*"):
        return [e for e in self.ents if e.domain == domain
                and (room == "*" or e.area == room)]

    def system_prompt(self, location=None):
        return S.build_system_prompt(
            [(e.eid, e.name, e.area) for e in self.ents], location)

    def state_of(self, ents):
        return S.dumps([S.format_state(e.eid, e.name, e.state, e.attrs)
                        for e in ents])


# --------------------------------------------------------------------------
# references: how the user names a device or a group of devices
# --------------------------------------------------------------------------
@dataclass
class Ref:
    acc: str      # accusative phrase used in commands ("das Licht im Bad")
    nom: str      # nominative phrase for confirmations
    plural: bool
    ents: list


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    opts = ["name", "name"]
    if allow_room_ref and e.area and len(same_type_in_room) == 1:
        opts += ["type_room", "type_room"]
    choice = R.choice(opts)
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        art = R.random() < 0.35 and e.label and e.name.endswith(e.label)
        if art:
            return Ref(f"{ART_ACC[e.gender]} {name}", f"{ART_NOM[e.gender]} {e.name}",
                       e.plural, [e])
        return Ref(name, e.name, e.plural, [e])
    prep = ROOM_BY_NAME[e.area][1]
    return Ref(f"{ART_ACC[e.gender]} {e.label} {prep}",
               f"{ART_NOM[e.gender]} {e.label} {prep}", e.plural, [e])


# group nouns per domain: (acc singular "das Licht", plural "die Lichter")
GROUP = {
    "light": [("das Licht", "das Licht", False), ("die Lichter", "die Lichter", True),
              ("die Lampen", "die Lampen", True), ("die Beleuchtung", "die Beleuchtung", False)],
    "cover": [("die Rollläden", "die Rollläden", True), ("die Rollos", "die Rollos", True),
              ("die Jalousien", "die Jalousien", True)],
    "climate": [("die Heizung", "die Heizung", False), ("die Heizungen", "die Heizungen", True)],
    "media_player": [("die Musik", "die Musik", False)],
}
ALL_GROUP = {
    "light": [("alle Lichter", "Alle Lichter"), ("alle Lampen", "Alle Lampen"),
              ("überall das Licht", "Das Licht im ganzen Haus")],
    "cover": [("alle Rollläden", "Alle Rollläden"), ("alle Rollos", "Alle Rollos")],
    "climate": [("alle Heizungen", "Alle Heizungen")],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    acc, nom, pl = R.choice(GROUP[domain])
    if location == room and R.random() < 0.7:
        return Ref(acc, nom, pl, ents)          # "Mach das Licht an"
    prep = ROOM_BY_NAME[room][1]
    return Ref(f"{acc} {prep}", f"{nom} {prep}", pl, ents)


# --------------------------------------------------------------------------
# message builders
# --------------------------------------------------------------------------
CALL_ID = iter(range(10**9))


def tool_call(name, args):
    cid = f"call_{next(CALL_ID) % 100000:05d}"
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": S.dumps(args)}}]}, cid


def tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def action(service, ents, data=None):
    a = {"service": service, "entity_id": [e.eid for e in ents]}
    if data:
        a["data"] = data
    return a


def voice(text: str) -> str:
    """Make user text look like speech-to-text output sometimes."""
    r = R.random()
    if r < 0.25:
        text = text.rstrip("?.!")
    elif r < 0.35:
        text = text.rstrip("?.!").lower()
    if R.random() < 0.05:
        text = R.choice(["Hey, ", "Okay, ", "Du, ", "Bitte "]) + text[0].lower() + text[1:]
    return text


def sein(pl):
    return "sind" if pl else "ist"


def confirm(ref: Ref, adj: str, part: str) -> str:
    """adj: state word ('an'), part: participle ('eingeschaltet')."""
    return R.choice([
        f"{cap(ref.nom)} {sein(ref.plural)} jetzt {adj}.",
        f"{cap(ref.nom)} {sein(ref.plural)} {adj}.",
        f"Ich habe {ref.acc} {part}.",
        f"Erledigt, {ref.nom} {sein(ref.plural)} {adj}.",
        f"Okay, {ref.nom} {sein(ref.plural)} jetzt {adj}.",
        "Erledigt.",
    ])


# --------------------------------------------------------------------------
# command intents: each returns (user_text, [actions], confirmation)
# --------------------------------------------------------------------------
def pick_target(h: House, domain, location=None, group_ok=True):
    """Return a Ref for one entity or a room group of `domain`."""
    ents = h.of(domain)
    if not ents:
        return None
    if group_ok and domain in GROUP and R.random() < 0.35:
        rooms = [r for r in h.rooms if h.of(domain, r)]
        if rooms:
            room = location if location in rooms and R.random() < 0.6 else R.choice(rooms)
            return ref_room_group(domain, room, h.of(domain, room), location)
    return ref_single(h, R.choice(ents))


def i_light_onoff(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    on = R.random() < 0.5
    if on:
        user = R.choice([
            "Mach {r} an", "Schalte {r} ein", "Schalt {r} an", "{r} an",
            "Bitte {r} einschalten", "Kannst du {r} anmachen?",
            "Mach bitte {r} an", "{r} einschalten", "Würdest du {r} einschalten?",
            "Ich brauche Licht, mach {r} an", "Schalte bitte {r} an.",
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref.ents)],
                confirm(ref, "an", "eingeschaltet"))
    user = R.choice([
        "Mach {r} aus", "Schalte {r} aus", "Schalt {r} aus", "{r} aus",
        "Bitte {r} ausschalten", "Kannst du {r} ausmachen?", "Mach bitte {r} aus.",
        "{r} ausschalten", "Würdest du {r} ausschalten?",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref.ents)],
            confirm(ref, "aus", "ausgeschaltet"))


def i_light_room_phrase(h, loc):
    """'Licht im Bad an' / 'Es ist zu dunkel im Bad'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    prep = ROOM_BY_NAME[room][1]
    ref = Ref(f"das Licht {prep}", f"das Licht {prep}", False, ents)
    if R.random() < 0.5:
        user = R.choice([f"Licht {prep} an", f"Mach {prep} das Licht an",
                         f"Es ist zu dunkel {prep}", f"{prep} Licht an bitte",
                         f"Licht an {prep}"])
        return user, [action("light.turn_on", ents)], confirm(ref, "an", "eingeschaltet")
    user = R.choice([f"Licht {prep} aus", f"Mach {prep} das Licht aus",
                     f"Licht aus {prep}", f"{prep} das Licht ausmachen"])
    return user, [action("light.turn_off", ents)], confirm(ref, "aus", "ausgeschaltet")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        user = R.choice([
            "Dimme {r} auf {p} Prozent", "Stell {r} auf {p} %",
            "{r} auf {p} Prozent", "Mach {r} auf {p} Prozent",
            "Setz {r} auf {p} Prozent Helligkeit",
        ]).format(r=ref.acc, p=p)
        conf = R.choice([f"{cap(ref.nom)} {sein(ref.plural)} jetzt auf {p} Prozent.",
                         f"Ich habe {ref.acc} auf {p} Prozent gestellt.", "Erledigt."])
        return user, [action("light.turn_on", ref.ents, {"brightness_pct": p})], conf
    up = r < 0.8
    user = R.choice(["Mach {r} heller", "{r} heller", "Etwas heller {r} bitte"]
                    if up else
                    ["Mach {r} dunkler", "Dimme {r}", "{r} etwas dunkler"]).format(r=ref.acc)
    step = 20 if up else -20
    conf = (f"{cap(ref.nom)} {sein(ref.plural)} jetzt "
            f"{'heller' if up else 'dunkler'}.") if R.random() < 0.7 else "Erledigt."
    return user, [action("light.turn_on", ref.ents, {"brightness_step_pct": step})], conf


def i_light_color(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    if R.random() < 0.75:
        de, en = R.choice(list(COLORS.items()))
        data = {"color_name": en}
    else:
        de, k = R.choice(list(COLOR_TEMPS.items()))
        data = {"color_temp_kelvin": k}
    user = R.choice(["Mach {r} {c}", "Stell {r} auf {c}", "Färbe {r} {c}",
                     "{r} in {c} bitte", "Kannst du {r} {c} machen?",
                     "Ich hätte gern {r} in {c}"]).format(r=ref.acc, c=de)
    leucht = "leuchten" if ref.plural else "leuchtet"
    conf = R.choice([f"{cap(ref.nom)} {leucht} jetzt {de}.",
                     f"Ich habe {ref.acc} auf {de} gestellt.", "Erledigt."])
    return user, [action("light.turn_on", ref.ents, data)], conf


def i_switch(h, loc):
    ents = h.of("switch") + h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = R.random() < 0.5
    svc = f"{e.domain}.turn_{'on' if on else 'off'}"
    if on:
        user = R.choice(["Schalte {r} ein", "Mach {r} an", "{r} an", "{r} einschalten",
                         "Kannst du {r} anschalten?", "Starte {r}"])
        conf = confirm(ref, "an", "eingeschaltet")
    else:
        user = R.choice(["Schalte {r} aus", "Mach {r} aus", "{r} aus", "{r} ausschalten",
                         "Kannst du {r} abschalten?"])
        conf = confirm(ref, "aus", "ausgeschaltet")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    user = R.choice(["Stell {r} auf {p} Prozent", "{r} auf {p} %",
                     "Lass {r} mit {p} Prozent laufen"]).format(r=ref.acc, p=p)
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {sein(ref.plural)} auf {p} Prozent.", "Erledigt."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Öffne {r}", "Mach {r} auf", "Fahr {r} hoch", "{r} hoch",
                         "{r} öffnen", "Zieh {r} hoch", "Kannst du {r} öffnen?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "offen", "geöffnet"))
    if r < 0.8:
        user = R.choice(["Schließe {r}", "Mach {r} zu", "Fahr {r} runter", "{r} runter",
                         "{r} schließen", "Lass {r} herunter", "Kannst du {r} schließen?"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "geschlossen", "geschlossen"))
    if r < 0.88:
        user = R.choice(["Stopp {r}", "Halte {r} an", "{r} stopp", "Stoppe {r}"])
        return (user.format(r=ref.acc), [action("cover.stop_cover", ref.ents)],
                R.choice(["Gestoppt.", f"Ich habe {ref.acc} angehalten.", "Erledigt."]))
    p, phrase = R.choice([(50, "halb"), (30, "auf 30 Prozent"), (70, "auf 70 Prozent"),
                          (20, "auf 20 Prozent"), (80, "auf 80 Prozent"),
                          (50, "auf 50 Prozent"), (10, "auf 10 Prozent")])
    if phrase == "halb":
        user = R.choice(["Mach {r} halb zu", "Fahr {r} halb runter", "{r} halb"]).format(r=ref.acc)
    else:
        user = R.choice(["Fahr {r} {ph}", "Stell {r} {ph}", "{r} {ph}"]).format(r=ref.acc, ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} {sein(ref.plural)} jetzt auf {p} Prozent.", "Erledigt."])


def i_climate(h, loc):
    ents = h.of("climate")
    if not ents:
        return None
    rooms = sorted({e.area for e in ents if e.area})
    r = R.random()
    if r < 0.75:
        t = R.choice([17, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 24, 25])
        if R.random() < 0.4 and rooms:
            room = loc if loc in rooms and R.random() < 0.5 else R.choice(rooms)
            tgt = h.of("climate", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Stell die Heizung {prep} auf {{t}} Grad",
                             f"Heizung {prep} auf {{t}} Grad",
                             f"Mach es {prep} {{t}} Grad warm",
                             f"Ich möchte {prep} {{t}} Grad"]).format(t=de_num(t))
            nom = f"die Heizung {prep}"
            pl = False
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Stell {r} auf {t} Grad", "{r} auf {t} Grad",
                             "Setze {r} auf {t} Grad", "Dreh {r} auf {t} Grad"]
                            ).format(r=ref.acc, t=de_num(t))
            nom, pl = ref.nom, ref.plural
        conf = R.choice([f"{cap(nom)} {sein(pl)} jetzt auf {de_num(t)} Grad eingestellt.",
                         f"Ich habe {nom} auf {de_num(t)} Grad gestellt."
                         if not nom.startswith("der ") else "Erledigt.",
                         f"Erledigt, {de_num(t)} Grad."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Schalte {r} ein", "Mach {r} an", "{r} an"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "an", "eingeschaltet")
    user = R.choice(["Schalte {r} aus", "Mach {r} aus", "{r} aus"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "aus", "ausgeschaltet")


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc == e.name and R.random() < 0.7:
        ref = Ref(f"{ART_ACC[e.gender]} {e.name}", f"{ART_NOM[e.gender]} {e.name}",
                  False, [e])
    if R.random() < 0.55:
        user = R.choice(["Schließ {r} ab", "Sperre {r} ab", "Verriegle {r}",
                         "Ist {r} zu? Schließ sie bitte ab", "{r} abschließen"])
        user = user.replace("Schließ sie", "Schließ " + ("ihn" if e.gender == "m" else "sie" if e.gender == "f" else "es")) \
            if "sie bitte" in user else user
        return (user.format(r=ref.acc), [action("lock.lock", [e])],
                confirm(ref, "abgeschlossen", "abgeschlossen"))
    user = R.choice(["Schließ {r} auf", "Entriegle {r}", "Sperr {r} auf", "{r} aufschließen",
                     "Öffne das Schloss von {n}"])
    return (user.format(r=ref.acc, n=e.name), [action("lock.unlock", [e])],
            confirm(ref, "aufgeschlossen", "aufgeschlossen"))


def i_alarm(h, loc):
    ents = h.of("alarm_control_panel")
    if not ents:
        return None
    e = ents[0]
    code = None
    if R.random() < 0.12:
        code = "".join(R.choices(string.digits, k=4))
    r = R.random()
    if r < 0.3:
        svc, user = "alarm_control_panel.alarm_arm_away", R.choice([
            "Schalte die Alarmanlage scharf", "Aktiviere die Alarmanlage",
            "Ich gehe jetzt, mach die Alarmanlage scharf", "Alarmanlage an",
            "Alarmanlage auf abwesend"])
        conf = R.choice(["Die Alarmanlage ist scharf geschaltet.",
                         "Die Alarmanlage ist jetzt aktiv (abwesend)."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Alarmanlage auf Zuhause", "Schalte die Alarmanlage auf anwesend",
            "Aktiviere die Alarmanlage im Modus Zuhause"])
        conf = "Die Alarmanlage ist im Modus Zuhause scharf."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Alarmanlage für die Nacht scharf schalten", "Nachtmodus für die Alarmanlage",
            "Aktiviere die Alarmanlage im Nachtmodus"])
        conf = "Die Alarmanlage ist im Nachtmodus scharf."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Schalte die Alarmanlage aus", "Alarmanlage unscharf", "Deaktiviere die Alarmanlage",
            "Ich bin zu Hause, mach die Alarmanlage aus", "Alarmanlage aus"])
        conf = R.choice(["Die Alarmanlage ist unscharf geschaltet.",
                         "Die Alarmanlage ist jetzt deaktiviert."])
    if code:
        user += f", Code {code}"
    return user, [action(svc, [e], {"code": code} if code else None)], conf


def i_media(h, loc):
    ents = h.of("media_player")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    r = R.random()
    opts = [
        ("media_player.media_pause", ["Pausiere {r}", "Pause auf {n}", "{r} pausieren",
                                      "Halte die Wiedergabe auf {n} an"], None,
         "Pausiert."),
        ("media_player.media_play", ["Spiel weiter auf {n}", "Setz die Wiedergabe auf {n} fort",
                                     "Play auf {n}"], None, "Die Wiedergabe läuft weiter."),
        ("media_player.media_next_track", ["Nächster Titel auf {n}", "Spring auf {n} zum nächsten Lied",
                                           "Nächstes Lied auf {n}"], None, "Nächster Titel."),
        ("media_player.volume_up", ["Mach {r} lauter", "{r} lauter"], None, "Lauter gestellt."),
        ("media_player.volume_down", ["Mach {r} leiser", "{r} leiser"], None, "Leiser gestellt."),
        ("media_player.turn_off", ["Schalte {r} aus", "Mach {r} aus", "{r} aus"], None,
         confirm(ref, "aus", "ausgeschaltet")),
        ("media_player.turn_on", ["Schalte {r} ein", "Mach {r} an"], None,
         confirm(ref, "an", "eingeschaltet")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Stell die Lautstärke von {n} auf {v} Prozent",
                         "Lautstärke {n} auf {v} Prozent",
                         "Mach {r} auf {v} Prozent Lautstärke"]).format(r=ref.acc, n=e.name, v=v)
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"Die Lautstärke ist jetzt auf {v} Prozent."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, n=e.name), [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = f"{ART_ACC[e.gender]} {e.name}"
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Starte {n}", f"Lass {n} saugen", f"{e.name} starten",
                         "Staubsaugen bitte", f"Schick {n} los"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{cap(ART_NOM[e.gender])} {e.name} legt los.", "Der Sauger startet."])
    if r < 0.8:
        user = R.choice([f"Schick {n} zur Ladestation", f"{e.name} zurück zur Basis",
                         f"Schick {n} nach Hause"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{cap(ART_NOM[e.gender])} {e.name} fährt zurück zur Ladestation."
    user = R.choice([f"Pausiere {n}", f"{e.name} anhalten", f"Halte {n} an"])
    return user, [action("vacuum.pause", [e])], f"{cap(ART_NOM[e.gender])} {e.name} ist pausiert."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Aktiviere die Szene {e.name}", f"Szene {e.name}",
                         f"{e.name} aktivieren", f"Starte die Szene {e.name}",
                         f"Mach {e.name}"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"Die Szene {e.name} ist aktiv.", f"Szene {e.name} aktiviert.", "Erledigt."])
    if e.domain == "script":
        user = R.choice([f"Führe {e.name} aus", f"Starte {e.name}", f"{e.name} starten",
                         f"Skript {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} läuft.", f"Ich habe {e.name} gestartet.", "Erledigt."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain)
    if len(ents) < 2:
        return None
    acc, nom = R.choice(ALL_GROUP[domain])
    ref = Ref(acc, nom, not acc.startswith("überall"), ents)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Mach {acc} aus", f"Schalte {acc} aus", f"{cap(acc)} aus",
                             "Licht überall aus"])
            return user, [action("light.turn_off", ents)], R.choice(
                [f"{nom} {sein(ref.plural)} aus.", "Alle Lichter sind aus.", "Erledigt."])
        user = R.choice([f"Mach {acc} an", f"Schalte {acc} ein", f"{cap(acc)} an"])
        return user, [action("light.turn_on", ents)], R.choice(
            [f"{nom} {sein(ref.plural)} an.", "Alle Lichter sind an.", "Erledigt."])
    if R.random() < 0.6:
        user = R.choice([f"Schließe {acc}", f"Mach {acc} zu", f"{cap(acc)} runter",
                         f"Fahr {acc} runter"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{nom} sind geschlossen.", "Erledigt, alles zu."])
    user = R.choice([f"Öffne {acc}", f"Mach {acc} auf", f"{cap(acc)} hoch", f"Fahr {acc} hoch"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{nom} sind offen.", "Erledigt, alles offen."])


COMMANDS = [
    (i_light_onoff, 14), (i_light_room_phrase, 6), (i_light_brightness, 6),
    (i_light_color, 4), (i_switch, 6), (i_fan_pct, 1), (i_cover, 9),
    (i_climate, 7), (i_lock, 3), (i_alarm, 3), (i_media, 4), (i_vacuum, 2),
    (i_scene, 4), (i_all, 4),
]


def pick_command(h, loc):
    for _ in range(30):
        fn = R.choices([c[0] for c in COMMANDS], [c[1] for c in COMMANDS])[0]
        out = fn(h, loc)
        if out:
            return out
    return None


# --------------------------------------------------------------------------
# sample categories
# --------------------------------------------------------------------------
def base_msgs(h, loc):
    return [{"role": "system", "content": h.system_prompt(loc)}]


def s_command(h, loc):
    out = pick_command(h, loc)
    if not out:
        return None
    user, actions, conf = out
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, S.control_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    if ub.startswith(("Kannst", "Würdest", "Ich ", "Bitte", "Es ist", "Hey")) or \
            ua.startswith(("Kannst", "Würdest", "Es ist")):
        return None
    # lowercase the leading verb of the second command ("... und mach ...")
    first = ub.split(" ", 1)
    if first[0] in ("Mach", "Schalte", "Schalt", "Stell", "Öffne", "Schließe", "Fahr",
                    "Dimme", "Starte", "Aktiviere", "Pausiere", "Spiel", "Setze",
                    "Setz", "Lass", "Färbe", "Schließ", "Sperre", "Verriegle",
                    "Entriegle", "Halte", "Stopp", "Stoppe", "Dreh", "Zieh", "Führe",
                    "Schick", "Deaktiviere"):
        ub2 = first[0].lower() + (" " + first[1] if len(first) > 1 else "")
    else:
        ub2 = ub
    user = f"{ua} und {ub2}"
    actions = a[1] + b[1]
    conf = R.choice(["Erledigt.", "Beides erledigt.",
                     f"{a[2].rstrip('.')} und {b[2][0].lower() + b[2][1:]}"
                     if a[2] != "Erledigt." and b[2] != "Erledigt." else "Erledigt."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, S.control_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def _state_word(e: Ent) -> str:
    d, s = e.domain, e.state
    if d in ("light", "switch", "fan", "input_boolean"):
        return "an" if s == "on" else "aus"
    if d == "cover":
        return "offen" if s == "open" else "geschlossen"
    if d == "lock":
        return "abgeschlossen" if s == "locked" else "nicht abgeschlossen"
    if d == "binary_sensor":
        return "offen" if s == "on" else "geschlossen"
    if d == "media_player":
        return {"playing": "spielt gerade", "paused": "pausiert",
                "off": "aus", "idle": "an, spielt aber nichts"}[s]
    return s


def s_question(h, loc):
    """State question -> get_state -> grounded German answer."""
    kinds = []
    if h.of("sensor"):
        kinds += ["temp"] * 3
    if h.of("climate"):
        kinds += ["climate"] * 2
    kinds += ["onoff"] * 3
    if len(h.of("light")) >= 2:
        kinds += ["which_lights"] * 2
    if h.of("binary_sensor"):
        kinds += ["windows"] * 2
    if h.of("lock"):
        kinds += ["lock"] * 2
    if h.of("alarm_control_panel"):
        kinds += ["alarm"]
    if [e for e in h.of("media_player") if e.state == "playing"]:
        kinds += ["media"]
    if [e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"]:
        kinds += ["humidity"]
    if [e for e in h.of("sensor") if e.attrs.get("device_class") == "power"]:
        kinds += ["power"]
    kind = R.choice(kinds)
    must = []
    if kind == "temp":
        temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"]
        if not temps:
            return None
        e = R.choice(temps)
        val = de_num(float(e.state))
        if e.area is None:
            user = R.choice(["Wie warm ist es draußen?", "Wie kalt ist es draußen?",
                             "Wie ist die Außentemperatur?", "Wie viel Grad hat es draußen?"])
            ans = R.choice([f"Draußen sind es {val} Grad.", f"Die Außentemperatur liegt bei {val} Grad."])
        else:
            prep = ROOM_BY_NAME[e.area][1]
            user = R.choice([f"Wie warm ist es {prep}?", f"Welche Temperatur hat es {prep}?",
                             f"Wie viel Grad sind es {prep}?", f"Temperatur {prep}?"])
            ans = R.choice([f"{cap(prep)} sind es {val} Grad.",
                            f"Die Temperatur {prep} liegt bei {val} Grad."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        prep = ROOM_BY_NAME[e.area][1] if e.area else ""
        user = R.choice([f"Wie hoch ist die Luftfeuchtigkeit {prep}?",
                         f"Wie feucht ist es {prep}?"]).replace("  ", " ")
        ans = f"Die Luftfeuchtigkeit {prep} liegt bei {e.state} Prozent.".replace("  ", " ")
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"Wie viel Strom verbraucht {e.name} gerade?",
                         f"Was zeigt {e.name} an?", f"Wie hoch ist {e.name}?"])
        val = de_num(float(e.state))
        ans = f"{e.name} zeigt gerade {val} Watt."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"Auf wie viel Grad ist {ref.nom} eingestellt?",
                             f"Welche Solltemperatur hat {ref.nom}?"])
            val = de_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {sein(ref.plural)} auf {val} Grad eingestellt."
        else:
            if e.area:
                prep = ROOM_BY_NAME[e.area][1]
                user = R.choice([f"Wie warm ist es {prep}?", f"Wie viel Grad sind es {prep}?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None   # ambiguous: sensor and thermostat in same room
            else:
                user = f"Welche Temperatur misst {ref.nom}?"
                prep = ""
            val = de_num(e.attrs["current_temperature"])
            ans = (f"{cap(prep)} sind es {val} Grad." if prep
                   else f"{cap(ref.nom)} misst {val} Grad.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        e = R.choice(cands)
        ref = ref_single(h, e)
        if e.domain == "cover":
            user = R.choice([f"Ist {ref.nom} offen?", f"Sind {ref.nom} unten?"
                             if ref.plural else f"Ist {ref.nom} geschlossen?"])
        else:
            user = R.choice([f"Ist {ref.nom} an?", f"Ist {ref.nom} noch an?",
                             f"Ist {ref.nom} ausgeschaltet?"]) \
                if not ref.plural else f"Sind {ref.nom} an?"
        word = _state_word(e)
        ans = f"{cap(ref.nom)} {sein(ref.plural)} {word}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            ans = f"{cap(ref.nom)} {sein(ref.plural)} an, mit {pct} Prozent Helligkeit."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Welche Lichter sind {prep} an?", f"Brennt {prep} noch Licht?"])
        else:
            ents = h.of("light")
            user = R.choice(["Welche Lichter sind an?", "Brennt irgendwo noch Licht?",
                             "Sind noch Lampen an?", "Welche Lampen sind eingeschaltet?"])
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "Gerade ist kein Licht an." if not room else \
                f"{cap(ROOM_BY_NAME[room][1])} ist gerade kein Licht an."
            must = ["kein"]
        elif len(on) == 1:
            ans = f"Nur {on[0].name} ist an."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"An sind: {', '.join(names[:-1])} und {names[-1]}."
            must = names
        ids = ents
    elif kind == "windows":
        ents = h.of("binary_sensor")
        user = R.choice(["Sind alle Fenster zu?", "Ist noch ein Fenster offen?",
                         "Welche Fenster sind offen?", "Sind die Fenster geschlossen?"])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans, must = "Alle Fenster sind geschlossen.", ["geschlossen"]
        else:
            names = [e.name for e in opn]
            ans = ("Offen ist: " + names[0] + "." if len(names) == 1 else
                   "Offen sind: " + ", ".join(names[:-1]) + " und " + names[-1] + ".")
            if R.random() < 0.5:
                ans = "Nein. " + ans
            must = names
        ids = ents
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = f"{ART_NOM[e.gender]} {e.name}"
        user = R.choice([f"Ist {nom} abgeschlossen?", f"Ist {nom} zu?",
                         f"Hab ich {ART_ACC[e.gender]} {e.name} abgeschlossen?"])
        word = _state_word(e)
        ans = (f"Ja, {nom} ist abgeschlossen." if e.state == "locked"
               else f"Nein, {nom} ist nicht abgeschlossen.")
        ids, must = [e], [word]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        user = R.choice(["Ist die Alarmanlage scharf?", "Ist die Alarmanlage an?",
                         "Welchen Status hat die Alarmanlage?"])
        txt = {"disarmed": "Die Alarmanlage ist unscharf.",
               "armed_away": "Die Alarmanlage ist scharf (abwesend).",
               "armed_home": "Die Alarmanlage ist im Modus Zuhause scharf.",
               "armed_night": "Die Alarmanlage ist im Nachtmodus scharf."}[e.state]
        ans, ids, must = txt, [e], ["unscharf" if e.state == "disarmed" else "scharf"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"Was läuft gerade auf {e.name}?", f"Was spielt {e.name}?"])
        t = e.attrs.get("media_title", "")
        a = e.attrs.get("media_artist")
        ans = f"Auf {e.name} läuft gerade {t}" + (f" von {a}." if a else ".")
        ids, must = [e], [t]
    else:
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [e.eid for e in ids]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of(ids)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": must}


def s_relative_climate(h, loc):
    """'Mach es im Bad wärmer' -> get_state -> set_temperature target +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    prep = ROOM_BY_NAME[e.area][1]
    up = R.random() < 0.6
    user = R.choice([f"Mach es {prep} etwas wärmer", f"Mir ist kalt {prep}",
                     f"Heizung {prep} ein Grad höher"] if up else
                    [f"Mach es {prep} etwas kühler", f"Es ist zu warm {prep}",
                     f"Heizung {prep} ein Grad runter"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, S.control_result()),
             {"role": "assistant",
              "content": f"Ich habe die Heizung {prep} von {de_num(e.attrs['temperature'])} "
                         f"auf {de_num(new)} Grad gestellt."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


def s_unknown(h, loc):
    """User asks for a device or room that does not exist -> no tool call."""
    r = R.random()
    if r < 0.5:
        dev, g = R.choice(UNKNOWN_DEVICES)
        if any(dev.lower() in e.name.lower() for e in h.ents):
            return None
        acc = f"{ART_ACC[g]} {dev}"
        user = R.choice([f"Schalte {acc} ein", f"Mach {acc} an", f"Schalte {acc} aus",
                         f"Ist {ART_NOM[g]} {dev} an?", f"Starte {acc}"])
        ans = R.choice([f"Ein Gerät namens {dev} finde ich nicht in deinem Smart Home.",
                        f"Ich kenne kein Gerät {dev}. Das kann ich leider nicht steuern.",
                        f"{dev} gibt es in der Geräteliste nicht."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != "Badezimmer"
                   and not (r_[0] == "Bad" and "Badezimmer" in h.rooms)]
        room = R.choice(missing)
        prep = ROOM_BY_NAME[room][1]
        what = R.choice([("das Licht", "kein Licht", "das", "an", "aus"),
                         ("die Rollläden", "keine Rollläden", "die", "hoch", "zu"),
                         ("die Heizung", "keine Heizung", "die", "an", "aus")])
        user = R.choice([f"Mach {what[0]} {prep} {what[3]}",
                         f"Mach {what[0]} {prep} {what[4]}",
                         f"{cap(what[0].split()[1])} {prep} {what[4]}"])
        ans = R.choice([f"{cap(prep)} finde ich {what[1]}, {what[2]} ich steuern kann.",
                        f"Ich habe {prep} {what[1]} in der Geräteliste gefunden."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


CHAT = [
    (["Danke", "Danke dir", "Vielen Dank", "Super, danke"],
     ["Gern geschehen.", "Gerne.", "Kein Problem."]),
    (["Hallo", "Hi", "Guten Morgen", "Guten Abend"],
     ["Hallo, wie kann ich helfen?", "Hallo. Was soll ich tun?"]),
    (["Was kannst du?", "Wobei kannst du mir helfen?", "Was kannst du alles steuern?"],
     ["Ich kann Lichter, Rollläden, Heizungen und weitere Geräte in deinem Zuhause steuern "
      "und dir sagen, welchen Zustand sie haben oder welche Werte die Sensoren messen."]),
    (["Wer bist du?"], ["Ich bin dein Sprachassistent für das Smart Home."]),
    (["Erzähl mir einen Witz"],
     ["Warum hat die Lampe keine Freunde? Sie ist immer so schnell ausgebrannt.",
      "Was sagt der Rollladen zum Fenster? Ich hab dich im Blick."]),
    (["Wie wird das Wetter morgen?"],
     ["Eine Wettervorhersage habe ich leider nicht. Ich kann dir aber sagen, was deine "
      "Sensoren gerade messen."]),
    (["Gute Nacht"], ["Gute Nacht. Soll ich noch etwas ausschalten?", "Gute Nacht."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == "Wie wird das Wetter morgen?" and h.of("weather"):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


def s_error(h, loc):
    out = pick_command(h, loc)
    if not out:
        return None
    user, actions, _ = out
    eid = actions[0]["entity_id"][0]
    name = next(e.name for e in h.ents if e.eid == eid)
    err = R.choice([f"Entity {eid} is unavailable",
                    f"Timeout while calling {actions[0]['service']} for {eid}",
                    f"{eid}: device did not respond"])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, S.control_result([err])),
             {"role": "assistant", "content": R.choice([
                 f"Das hat leider nicht funktioniert. {name} reagiert gerade nicht.",
                 f"Leider ist ein Fehler aufgetreten: {name} ist nicht erreichbar.",
                 f"{name} ist gerade nicht erreichbar, deshalb konnte ich den Befehl nicht ausführen."])}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def s_followup(h, loc):
    """Two-turn: command, then 'und wieder aus' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    p1, p2 = ROOM_BY_NAME[r1][1], ROOM_BY_NAME[r2][1]
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Mach das Licht {p1} an",
                                                            f"Licht {p1} an"]))}]
    a1 = [action("light.turn_on", e1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, S.control_result()),
             {"role": "assistant", "content": f"Das Licht {p1} ist an."}]
    k = R.random()
    if k < 0.35:
        u2, a2, ans = R.choice(["Und wieder aus", "Mach es wieder aus", "Doch wieder aus"]), \
            [action("light.turn_off", e1)], f"Das Licht {p1} ist wieder aus."
    elif k < 0.7:
        u2, a2, ans = R.choice([f"Und {p2}?", f"{cap(p2)} auch", f"Und {p2} bitte auch"]), \
            [action("light.turn_on", e2)], f"Das Licht {p2} ist auch an."
    else:
        u2, a2, ans = R.choice(["Etwas dunkler", "Ein bisschen heller", "Auf 30 Prozent"]), \
            None, None
        if "dunkler" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20})]
            ans = f"Das Licht {p1} ist jetzt dunkler."
        elif "heller" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20})]
            ans = f"Das Licht {p1} ist jetzt heller."
        else:
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30})]
            ans = f"Das Licht {p1} ist jetzt auf 30 Prozent."
    msgs.append({"role": "user", "content": voice(u2)})
    c2, cid2 = tool_call("control", {"actions": a2})
    msgs += [c2, tool_msg(cid2, S.control_result()), {"role": "assistant", "content": ans}]
    return msgs, {"cat": "followup", "gold_last": [["control", {"actions": a2}]],
                  "gold": [["control", {"actions": a1}]]}


CATEGORIES = [
    (s_command, 46), (s_compound, 8), (s_question, 24), (s_relative_climate, 2),
    (s_unknown, 7), (s_chat, 3), (s_error, 2), (s_followup, 8),
]


def make_sample():
    size = R.choice([R.randint(8, 18), R.randint(18, 35), R.randint(18, 35),
                     R.randint(35, 60)])
    h = House(size)
    loc = R.choice(h.rooms) if R.random() < 0.3 else None
    for _ in range(20):
        fn = R.choices([c[0] for c in CATEGORIES], [c[1] for c in CATEGORIES])[0]
        out = fn(h, loc)
        if out:
            msgs, meta = out
            meta["location"] = loc
            return {"messages": msgs, "meta": meta}
    return None


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2])
    R.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    rows, seen = [], set()
    while len(rows) < n:
        s = make_sample()
        if not s:
            continue
        key = s["messages"][1]["content"] + str(s["meta"].get("gold"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = {}
    for r in rows:
        cats[r["meta"]["cat"]] = cats.get(r["meta"]["cat"], 0) + 1
    print(f"{len(rows)} samples -> {out}: {cats}")


if __name__ == "__main__":
    main()
