"""Generate the Dutch SFT dataset for Home Assistant tool calling.

Port of src/gen/de/core.py: the same cases, tool calls and gold metadata,
with Dutch rooms, device names and sentences. Every sample uses a freshly
randomised house (rooms, device names, entity_ids, states), so the model
learns to map names from the device list to entity_ids instead of memorising
one house. Prompt and tool schema come from
custom_components/ha_llm_agent/llm_schema.py (LANGUAGES["nl"]).

Each JSONL row: {"messages": [...], "meta": {"cat": ..., "gold": [...],
"answer_contains": [...]}}. "meta" is ignored by training and used by
src/eval_model.py.

This module holds the house model and the device intents; src/gen/nl/assist.py
adds the Assist built-ins (music, timers, lists, weather, time, broadcasts)
through POPULATE_HOOKS, COMMANDS and CASE_HOOKS.

Dutch has no grammatical case, so a Ref carries the same phrase in `acc` and
`nom`; what Dutch needs instead is the article (de/het), the verb agreement
(is/zijn, staat/staan) and separable verbs ("Doe het licht aan").

Usage: python src/gen_dataset.py --lang nl OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

# repository: custom_components/ha_llm_agent/llm_schema.py; Hugging Face
# dataset: llm_schema.py next to gen_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

LANG = "nl"
# answer length of the current sample (llm_schema.VERBOSITY), set in make_sample
VERBOSITY = "medium"
VERBOSITY_WEIGHTS = [("short", 2), ("medium", 4), ("long", 2), ("detailed", 2)]
R = random.Random()
# extensions (assist.py): fn(house) after populating, fn() -> extra cases
POPULATE_HOOKS: list = []
CASE_HOOKS: list = []
# fn(house, action, who) -> wording for describe() or None (assist services)
DESCRIBE_HOOKS: list = []
# set by the case loop in main(): forces one intent / question kind / turn type
FORCE: dict = {}
# the house of the sample being built (join_commands keeps device names capitalised)
CURRENT: list = []

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, preposition phrase, compound prefix, slugs
ROOMS = [
    ("Woonkamer", "in de woonkamer", "Woonkamer", ["woonkamer", "living_room", "wk"]),
    ("Keuken", "in de keuken", "Keuken", ["keuken", "kitchen"]),
    ("Slaapkamer", "in de slaapkamer", "Slaapkamer", ["slaapkamer", "bedroom"]),
    ("Badkamer", "in de badkamer", "Badkamer", ["badkamer", "bathroom"]),
    ("Doucheruimte", "in de doucheruimte", "Douche", ["douche", "shower_room"]),
    ("Kinderkamer", "in de kinderkamer", "Kinderkamer", ["kinderkamer", "kids_room"]),
    ("Kantoor", "in het kantoor", "Kantoor", ["kantoor", "office"]),
    ("Werkkamer", "in de werkkamer", "Werkkamer", ["werkkamer", "study"]),
    ("Gang", "in de gang", "Gang", ["gang", "hallway", "hal"]),
    ("Eetkamer", "in de eetkamer", "Eetkamer", ["eetkamer", "dining_room"]),
    ("Logeerkamer", "in de logeerkamer", "Logeerkamer", ["logeerkamer", "guest_room"]),
    ("Kelder", "in de kelder", "Kelder", ["kelder", "basement"]),
    ("Garage", "in de garage", "Garage", ["garage"]),
    ("Tuin", "in de tuin", "Tuin", ["tuin", "garden"]),
    ("Terras", "op het terras", "Terras", ["terras", "terrace"]),
    ("Balkon", "op het balkon", "Balkon", ["balkon", "balcony"]),
    ("Zolder", "op zolder", "Zolder", ["zolder", "attic"]),
    ("Washok", "in het washok", "Washok", ["washok", "laundry"]),
    ("Toilet", "op het toilet", "Toilet", ["toilet", "wc"]),
    ("Inloopkast", "in de inloopkast", "Inloopkast", ["inloopkast", "dressing_room"]),
    ("Entree", "in de entree", "Entree", ["entree", "entrance"]),
    ("Overloop", "op de overloop", "Overloop", ["overloop", "landing"]),
    ("Buiten", "buiten", "Buiten", ["buiten", "outdoor"]),
    ("Berging", "in de berging", "Berging", ["berging", "storage"]),
    ("Stookruimte", "in de stookruimte", "Stookruimte", ["stookruimte", "boiler_room"]),
    ("Wasruimte", "in de wasruimte", "Wasruimte", ["wasruimte", "utility"]),
    ("Oprit", "op de oprit", "Oprit", ["oprit", "driveway"]),
    ("Bijkeuken", "in de bijkeuken", "Bijkeuken", ["bijkeuken", "utility_room"]),
    ("Schuur", "in de schuur", "Schuur", ["schuur", "shed"]),
    ("Carport", "onder de carport", "Carport", ["carport"]),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
# the German area names of ROOMS, in the same order. Rooms are sorted by them
# (room_order), so a choice from a sorted room list picks the same position as
# the German generator and both datasets keep the same structure.
DE_ROOMS = ["Wohnzimmer", "Küche", "Schlafzimmer", "Bad", "Badezimmer", "Kinderzimmer",
            "Büro", "Arbeitszimmer", "Flur", "Esszimmer", "Gästezimmer", "Keller",
            "Garage", "Garten", "Terrasse", "Balkon", "Dachboden", "Waschküche",
            "Gäste-WC", "Ankleide", "Eingang", "Treppenhaus", "Außenbereich",
            "Abstellraum", "Heizungsraum", "Waschraum", "Einfahrt",
            "Hauswirtschaftsraum", "Werkstatt", "Carport"]
_DE_RANK = {r[0]: DE_ROOMS[i] for i, r in enumerate(ROOMS)}


def room_order(rooms):
    """Sort room names like the German generator sorts its room names."""
    return sorted(rooms, key=lambda r: _DE_RANK[r])
OUTDOOR = ("Tuin", "Terras", "Balkon", "Buiten", "Oprit", "Carport")
UTILITY = ("Berging", "Stookruimte", "Wasruimte", "Bijkeuken",
           "Schuur", "Kelder", "Zolder", "Garage", "Washok")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garage",)]
NO_GROUP_COVERS = ("Zonnescherm", "Garagedeur")
# rooms that often have a window and a balcony or patio door with covers
TWO_COVER_ROOMS = ("Woonkamer", "Eetkamer", "Keuken", "Slaapkamer", "Werkkamer",
                   "Kinderkamer", "Kantoor")

# device type: (label, article de/het or pl, slugs, compound-able)
TYPES = {
    "light": [
        ("Plafondlamp", "de", ["plafondlamp", "ceiling_lamp"], False),
        ("Licht", "het", ["licht", "light"], True),
        ("Plafondverlichting", "de", ["plafond", "ceiling"], False),
        ("Lamp", "de", ["lamp", "lamp"], True),
        ("Staande lamp", "de", ["staande_lamp", "floor_lamp"], False),
        ("Leeslamp", "de", ["leeslamp", "reading_lamp"], False),
        ("Ledstrip", "de", ["ledstrip", "led_strip"], False),
        ("Spots", "pl", ["spots", "spotlights"], False),
        ("Wandlamp", "de", ["wandlamp", "wall_lamp"], False),
        ("Lichtslinger", "de", ["lichtslinger", "fairy_lights"], False),
        ("Hanglamp", "de", ["hanglamp", "pendant"], False),
        ("Nachtlampje", "het", ["nachtlampje", "bedside"], False),
        ("Spiegelverlichting", "de", ["spiegel", "mirror_light"], False),
    ],
    "switch": [
        ("Stopcontact", "het", ["stopcontact", "plug", "socket"], False),
        ("Koffiemachine", "de", ["koffiemachine", "coffee"], False),
        ("Kachel", "de", ["kachel", "heater"], False),
        ("Ontvochtiger", "de", ["ontvochtiger", "dehumidifier"], False),
        ("Vijverpomp", "de", ["vijverpomp", "pond_pump"], False),
        ("Beregening", "de", ["beregening", "irrigation"], False),
        ("Kerstboom", "de", ["kerstboom", "xmas_tree"], False),
        ("Printer", "de", ["printer", "printer"], False),
        ("Waterkoker", "de", ["waterkoker", "kettle"], False),
        ("Monitor", "de", ["monitor", "screen"], False),
    ],
    "cover": [
        ("Rolluik", "het", ["rolluik", "shutter"], True),
        ("Rolluiken", "pl", ["rolluiken", "shutters"], False),
        ("Jaloezie", "de", ["jaloezie", "blinds"], False),
        ("Rolgordijn", "het", ["rolgordijn", "blind"], False),
        ("Zonnescherm", "het", ["zonnescherm", "awning"], False),
    ],
    "climate": [
        ("Verwarming", "de", ["verwarming", "heating", "thermostat"], True),
        ("Thermostaat", "de", ["thermostaat", "trv"], False),
        ("Airco", "de", ["airco", "aircon"], False),
        ("Vloerverwarming", "de", ["vloerverwarming", "floor_heating"], False),
    ],
    "fan": [
        ("Ventilator", "de", ["ventilator", "fan"], False),
        ("Plafondventilator", "de", ["plafondventilator", "ceiling_fan"], False),
        ("Ventilatie", "de", ["ventilatie", "ventilation"], False),
    ],
    "media_player": [
        ("TV", "de", ["tv", "televisie"], False),
        ("Speaker", "de", ["speaker", "sonos", "luidspreker"], False),
        ("Soundbar", "de", ["soundbar"], False),
        ("Radio", "de", ["radio"], False),
    ],
}

LOCKS = [("Voordeur", "de", ["voordeur", "front_door"]),
         ("Kelderdeur", "de", ["kelderdeur", "basement_door"]),
         ("Terrasdeur", "de", ["terrasdeur", "patio_door"]),
         ("Tuinpoort", "de", ["tuinpoort", "garden_gate"]),
         ("Achterdeur", "de", ["achterdeur", "back_door"])]
VACUUMS = [("Robotstofzuiger", "de", ["robotstofzuiger", "roborock", "robot"]),
           ("Stofzuiger", "de", ["stofzuiger", "vacuum"])]
SCENES = ["Filmavond", "Welterusten", "Koken", "Ontspannen", "Feest", "Lezen",
          "Wakker worden", "Avondeten", "Romantisch", "Fel", "Concentratie"]
SCRIPTS = ["Alles uit", "Vakantiemodus", "Bedtijdroutine", "Koffie zetten",
           "Huis verlaten", "Thuiskomen"]
BUTTONS = [("Garagedeur", "de", ["garagedeur_knop", "garage_door_button"]),
           ("Deurbel", "de", ["deurbel", "doorbell_button"])]

# spoken colour -> Home Assistant colour name
COLORS = {"rood": "red", "groen": "green", "blauw": "blue", "geel": "yellow",
          "oranje": "orange", "paars": "purple", "roze": "pink", "wit": "white",
          "turquoise": "turquoise", "violet": "violet"}
COLOR_TEMPS = {"warm wit": 2700, "koud wit": 6000, "neutraal wit": 4000}

# brand-named devices: user says the generic noun, the name is the product
BRAND_MEDIA = [("LG webOS Smart TV", "TV", "de"), ("Samsung TV", "TV", "de"),
               ("Sony Bravia", "TV", "de"), ("Philips Ambilight", "TV", "de"),
               ("Sonos One", "Speaker", "de"), ("Echo Dot", "Speaker", "de"),
               ("HomePod", "Speaker", "de"), ("Bose Soundbar", "Soundbar", "de")]
# voice satellites expose a media_player that is no TV
def voice_media_name():
    hx = "".join(R.choice("0123456789abcdef") for _ in range(6))
    return R.choice([f"Home Assistant Voice {hx} Media Player", f"Home Assistant Voice {hx[:6]}",
                     f"ESPHome Voice {hx[:4]} Media Player", "Echo Show", "Nest Mini",
                     f"Atom Echo {hx[:6]} Media Player"])


BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Robotmaaier", "Grasrobot", "Gardena Sileno", "Worx Landroid"]
PLANTS = ["Tomaten", "Pepers", "Basilicum", "Peterselie", "Aardbeien", "Monstera",
          "Ficus", "Komkommers", "Citroenboom"]
PLURAL_PLANTS = {"Tomaten", "Pepers", "Aardbeien", "Komkommers"}
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "PC Daan", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Warm water Boost", "Hydrofoorpomp", "Aquarium Pomp"]
# appliances: a status sensor and a power switch with nearly the same name
# ("Droger" + "Droger Power"): commands go to the switch, questions to the
# sensor
APPLIANCES = [("Droger", "de", "droger"), ("Wasmachine", "de", "wasmachine"),
              ("Afwasmachine", "de", "afwasmachine"),
              ("Vaatwasser", "de", "vaatwasser"),
              ("3D-printer", "de", "printer_3d"), ("Wasdroger", "de", "dryer")]
# raw states as integrations report them (Home Connect, Miele, SmartThings, ...)
APP_RAW = {"running": ["running", "Bezig", "Actief", "drying", "in_use", "run"],
           "finished": ["finished", "Programma beëindigd", "Klaar", "end", "program_ended"],
           "off": ["off", "Uit"], "idle": ["idle", "Gereed", "standby", "programmed",
                                          "waiting_to_start"],
           "paused": ["pause", "paused", "Gepauzeerd"],
           "offline": ["not_connected", "disconnected", "Offline"]}
# (phrase after the subject, key word); the first word is the finite verb
APP_ANSWER = {"running": ("draait nog", "draait"), "finished": ("is klaar", "klaar"),
              "off": ("is uit", "uit"), "idle": ("staat klaar, maar draait nu niet",
                                                 "draait nu niet"),
              "paused": ("is gepauzeerd", "gepauzeerd"),
              "offline": ("is nu niet verbonden", "niet verbonden")}
VACUUM_BUTTONS = [("Volledige reiniging", "de volledige reiniging"),
                  ("Stofbak legen", "het legen van de stofbak"),
                  ("Mop wassen", "het wassen van de mop")]
TOGGLES = ["Vakantie", "Gastenmodus", "Feestmodus", "Slaapmodus", "Schoonmaakmodus"]

UNKNOWN_DEVICES = [("Sauna", "de"), ("Jacuzzi", "de"), ("Zwembadpomp", "de"),
                   ("Open haard", "de"), ("Aquarium", "het"), ("Grasmaaier", "de"),
                   ("Fontein", "de"), ("Wasdroger", "de"),
                   ("Vaatwasser", "de"), ("Oven", "de"), ("Airconditioner", "de"),
                   ("Garagedeur", "de"), ("Poortverlichting", "de")]

ART = {"de": "de", "het": "het", "pl": "de"}
# relative and demonstrative pronoun ("geen licht dat ...", "dan die")
REL = {"de": "die", "het": "dat", "pl": "die"}
# personal pronoun for a thing ("Is hij nu aan?", "Doe hem op slot")
PRON = {"de": ("hij", "hem"), "het": ("het", "het"), "pl": ("ze", "ze")}


def nl_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}".replace(".", ",")


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def low(s: str) -> str:
    """Lower the first letter: 'Keukenlamp' -> 'keukenlamp', '3D-printer' stays."""
    return s[:1].lower() + s[1:]


def is_(pl):
    return "zijn" if pl else "is"


def staat(pl):
    return "staan" if pl else "staat"


def ascii_slug(s: str) -> str:
    s = s.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    for a, b in (("ë", "e"), ("é", "e"), ("è", "e"), ("ï", "i"), ("ö", "o"), ("ü", "u"),
                 ("&", "en")):
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "Plafondlamp"
    gender: str = "het"      # article: de, het, pl
    state: str = "off"
    attrs: dict = field(default_factory=dict)
    hidden: str = ""         # type label that the name no longer shows

    @property
    def plural(self) -> bool:
        return self.gender == "pl"


class House:
    def __init__(self, size: int):
        self.ents: list[Ent] = []
        self.used_ids: set[str] = set()
        self.used_names: set[str] = set()
        CURRENT[:] = [self]
        self.cryptic = R.random() < 0.3       # some houses use vendor ids
        # some houses: all devices of a domain share a long vendor prefix and
        # differ only in the last hex digits (Homematic, Zigbee, Shelly)
        self.shared = {}
        for dom in ("climate", "light", "cover", "switch", "binary_sensor"):
            if R.random() < 0.18:
                self.shared[dom] = R.choice([
                    "hmip_etrv_b_2_r4m_00a3c5e7f91", "hmip_swdm_2_00b4d6f8a2c3e",
                    "shellydimmer2_3c8a1fe0", "0x00158d000", "zigbee2mqtt_0x84fd27fffe",
                    "lumi_lumi_sensor_magnet_aq2_00158d", "tz3000_ji4araar_ts0001_"])
        self.english_ids = R.random() < 0.4
        n_rooms = max(2, min(len(ROOMS) - 4, size // 4 + R.randint(-1, 2)))
        names = R.sample([r[0] for r in ROOMS if r[0] != "Doucheruimte"], n_rooms)
        if "Badkamer" in names and R.random() < 0.4:
            names[names.index("Badkamer")] = "Doucheruimte"
        self.rooms = names
        self.appliances: list[tuple[str, str, Ent, Ent]] = []
        self._populate(size)
        for hook in POPULATE_HOOKS:
            hook(self)

    # ---- id / name helpers ----
    def _slug(self, room: str | None, type_slugs: list[str], domain: str) -> str:
        rslugs = ROOM_BY_NAME[room][3] if room else ["huis", "home"]
        if domain in self.shared and R.random() < 0.9:
            tail = "".join(R.choices("0123456789abcdef", k=R.choice([2, 3])))
            base = self.shared[domain] + tail
        elif self.cryptic and R.random() < 0.6:
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
                opts += [ROOM_BY_NAME[room][2] + label.lower()]   # "Keukenlamp"
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

    def add_voice_media(self, room):
        name = voice_media_name()
        if name in self.used_names:
            return None
        return self.add("media_player", room, "Speaker", "de",
                        [name.lower().replace(" ", "_"), "voice_media_player"], name=name,
                        state=R.choice(["idle", "idle", "off"]),
                        attrs={"volume_level": R.choice([0.3, 0.5, 0.7])})

    def _populate(self, size: int):
        for room in self.rooms:
            outdoor = room in OUTDOOR
            # lights: 0-3 per room; outdoor/utility rooms often have none
            n_lights = R.choice([0, 0, 1, 1, 2] if room in OUTDOOR + UTILITY
                                else [0, 1, 1, 1, 2, 2, 3])
            room_lights = []
            for lt in R.sample(TYPES["light"], n_lights):
                on = R.random() < 0.4
                room_lights.append(self.add(
                    "light", room, lt[0], lt[1], lt[2], lt[3],
                    state="on" if on else "off",
                    attrs={"brightness": R.randint(20, 255)} if on else {}))
            # real installs often name the main light just like the room
            if room_lights and R.random() < 0.25 and room not in self.used_names:
                # the name no longer says what kind of light it is, so no
                # sample may call it "de plafondlamp in de keuken"
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Links", "Rechts", "Midden"], ["Plafond", "Wand", "Hoek"],
                                      ["Eiland", "Spots", "Bar"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        if f"{room} {side}" not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = f"{room} {side}"
                            lt.hidden, lt.label = lt.label, ""
                            self.used_names.add(lt.name)
            # many lamps carry a name without their room ("Sterrenhemel",
            # "Licht Dressoir"); only the area column ties them to the room
            for lt in room_lights:
                if lt.name != room and R.random() < 0.3:
                    free = [n for n in FREE_LIGHT_NAMES if n not in self.used_names]
                    if free:
                        self.used_names.discard(lt.name)
                        lt.name = R.choice(free)
                        lt.hidden, lt.label = lt.label or lt.hidden, ""
                        self.used_names.add(lt.name)
            if not outdoor and room not in ("Gang", "Overloop", "Toilet") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                if room in TWO_COVER_ROOMS and R.random() < 0.35:
                    # window and balcony door: "de rolluiken in de eetkamer"
                    # means both
                    word = R.choice(["Rolluik", "Screen", "Rolgordijn", "Jaloezie"])
                    gender = {"Jaloezie": "de", "Screen": "de"}.get(word, "het")
                    door = R.choice(["Balkondeur", "Terrasdeur", "Deur", "Tuindeur"])
                    for part, slug in (("Raam", "raam"), (door, "balkondeur")):
                        name = R.choice([f"{word} {part} {room}", f"{room} {word} {part}"])
                        if name in self.used_names:
                            continue
                        p2 = R.choice([pos, pos, 0, 100])
                        self.add("cover", room, word, gender,
                                 [f"{ct[2][0]}_{slug}", f"{ct[2][1]}_{slug}"], name=name,
                                 state="closed" if p2 == 0 else "open",
                                 attrs={"current_position": p2})
                else:
                    self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                             state="closed" if pos == 0 else "open",
                             attrs={"current_position": pos})
            if room in ("Terras", "Balkon", "Buiten") and R.random() < 0.5:
                self.add("cover", room, "Zonnescherm", "het", ["zonnescherm", "awning"],
                         state=R.choice(["open", "closed"]),
                         attrs={"current_position": R.choice([0, 100])})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:3] if room != "Badkamer" else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, "Temperatuur", "de",
                         ["temperatuur", "temperature"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "Luchtvochtigheid", "de",
                         ["luchtvochtigheid", "humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "Raam", "het",
                         ["raam", "window"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Woonkamer", "Slaapkamer", "Keuken", "Kantoor",
                        "Kinderkamer") and R.random() < 0.4:
                mt = R.choice(TYPES["media_player"])
                brand = None
                if R.random() < 0.5:
                    brand = R.choice(BRAND_MEDIA)
                    mt = (brand[1], brand[2], [brand[0].lower().replace(" ", "_")
                                               .replace("-", "_")])
                playing = R.random() < 0.4
                attrs = {"volume_level": R.choice([0.2, 0.3, 0.35, 0.5])}
                if playing:
                    attrs.update(media_title=R.choice(
                        ["NOS Journaal", "Bohemian Rhapsody", "Zij gelooft in mij",
                         "Wie is de Mol?", "Sesamstraat", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "André Hazes", ""]))
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if brand and brand[0] not in self.used_names else None,
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""})
            if not outdoor and R.random() < 0.12:
                self.add_voice_media(room)
            if room in ("Slaapkamer", "Kantoor", "Woonkamer", "Zolder") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "Vermogen", "het", ["vermogen", "power"],
                         name=self._name(room, R.choice(["Stopcontact Vermogen", "Vermogen"]),
                                         False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            area = R.choice(self.rooms + [None])
            self.add("alarm_control_panel", None, "Alarmsysteem", "het",
                     ["alarmsysteem", "alarm"], name="Alarmsysteem",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Gang", "Entree", None]) if lk[0] == "Voordeur" else None
                room = room if room in self.rooms else None
                self.add("lock", room, lk[0], lk[1], lk[2], name=lk[0],
                         state=R.choice(["locked", "locked", "unlocked"]))
        if R.random() < 0.4:
            vt = R.choice(VACUUMS)
            vname = R.choice(BRAND_VACUUM) if R.random() < 0.6 else vt[0]
            self.add("vacuum", R.choice(self.rooms + [None, None]), vt[0], vt[1],
                     [vname.lower().replace(" ", "_")], name=vname,
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.25:
            mname = R.choice(MOWERS)
            room = "Tuin" if "Tuin" in self.rooms else None
            self.add("lawn_mower", room, "Robotmaaier", "de",
                     [ascii_slug(mname)], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = ["Beregening", "Waterkraan"]
            if room:
                names += [f"Beregening {room}", f"{room} Planten", f"Planten {room}"]
                if room == "Tuin":
                    names.append("Tuinkraan")
            self.add("valve", room, "Beregening", "de", ["beregening", "irrigation"],
                     name=R.choice(names), state=R.choice(["open", "closed", "closed"]))
        if R.random() < 0.6:
            for sc in R.sample(SCENES, R.randint(1, 4)):
                self.add("scene", None, sc, "de", [ascii_slug(sc)],
                         name=sc, state="2026-10-30T19:12:00+00:00")
        if R.random() < 0.35:
            for sc in R.sample(SCRIPTS, R.randint(1, 2)):
                self.add("script", None, sc, "het", [ascii_slug(sc)], name=sc, state="off")
        out_t = round(R.uniform(-6, 31), 1)
        has_out = R.random() < 0.5
        if has_out:
            room, names = None, ["Buitentemperatuur", "Temperatuur Buiten",
                                 "Weerstation Temperatuur"]
            orooms = [r for r in self.rooms if r in OUTDOOR]
            if orooms and R.random() < 0.4:
                # weather stations sit on the balcony or in the garden
                room = R.choice(orooms)
                names = [f"Buitentemperatuur {room}", f"Temperatuur {room}"]
            self.add("sensor", room, "Buitentemperatuur", "de",
                     ["buitentemperatuur", "outdoor_temperature"],
                     name=R.choice(names), state=f"{out_t:.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        # heat pumps report their own outdoor sensor next to the climate entity
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            room = "Stookruimte" if "Stookruimte" in self.rooms \
                and R.random() < 0.5 else None
            self.add("sensor", room, "Buitentemperatuur", "de",
                     ["warmtepomp_outdoor_temperature", "thuis_outdoor_temperature"],
                     name=R.choice(["Buitentemperatuur Huis", "Buitentemperatuur Verwarming",
                                    "Warmtepomp Buitentemperatuur"]),
                     state=f"{out_t + R.choice([-0.8, -0.4, 0.3, 0.6]):.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
            has_out = True
        # weather integrations add forecast and derived values; none of them
        # says how warm it is outside right now
        if R.random() < (0.55 if has_out else 0.2):
            for name, label, when, slugs in R.sample(WEATHER_SENSORS, R.randint(1, 3)):
                if name in self.used_names:
                    continue
                if label == "Verwachting":
                    val = out_t + R.uniform(-4, 6) if when == "vandaag" else R.uniform(-6, 31)
                else:
                    val = out_t - R.uniform(0.5, 7)
                self.add("sensor", None, label, "de",
                         slugs, name=name,
                         state=f"{val:.1f}",
                         attrs={"device_class": "temperature", "unit_of_measurement": "°C",
                                "_when": when})
        if "Garage" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garage", "Garagedeur", "de",
                     ["garagedeur", "garage_door"], name="Garagedeur",
                     state=R.choice(["open", "closed", "closed"]),
                     attrs={"current_position": 0})
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(n.startswith(app) for n in self.used_names):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Keuken", "Wasruimte", "Bijkeuken", "Badkamer", "Doucheruimte",
                    "Kelder", "Berging", "Kantoor")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "Status", "de", [slug, slug + "_status"],
                                  name=R.choice([app, app, f"{app} Status"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind})
                power = self.add("switch", room, "Stopcontact", "het",
                                 [slug + "_power", slug + "_stopcontact"],
                                 name=R.choice([f"{app} Power", f"{app} Stopcontact",
                                                f"{app} Stroom"]),
                                 state="off" if kind == "off" and R.random() < 0.5 else "on")
                self.appliances.append((app, g, sensor, power))
        vac = [e for e in self.ents if e.domain == "vacuum"]
        if vac and R.random() < 0.45:
            v = vac[0]
            for label, _ in R.sample(VACUUM_BUTTONS, R.choice([1, 2])):
                self.add("button", v.area, label, "de",
                         [v.eid.split(".")[1] + "_" + label.lower().replace(" ", "_")],
                         name=f"{v.name} {label}", state="unknown")
        valve_rooms = {e.area for e in self.ents if e.domain == "valve"}
        if R.random() < (0.5 if valve_rooms else 0.15):
            free = [r for r in self.rooms if r in OUTDOOR and r not in valve_rooms]
            # many installs leave the irrigation button without an area; its
            # name ("Beregening Tuin") then is the only hint where it waters
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Beregening Tuin", "Tuinberegening"] if room == "Tuin"
                         else [f"Beregening {room}"])
            elif "Tuin" in valve_rooms:
                names = ["Beregening starten"]
            else:
                names = ["Beregening starten", "Beregening Tuin", "Tuinberegening"]
            self.add("button", room, "Beregening", "de", ["beregening_tuin",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # door contacts: balcony/patio doors in living rooms, the front door
        # in the hall; names often add "Contact" while people say only "deur"
        if R.random() < 0.5:
            suffix = R.choice(["", " Contact"])
            for room in [r for r in self.rooms if r in DOOR_ROOMS]:
                if R.random() < 0.55:
                    kind, slug = R.choice([("Balkon", "balkondeur"), ("Terras", "terrasdeur")])
                    name = R.choice([f"{kind}deur {room}{suffix}", f"{room} {kind}deur{suffix}"])
                    if name in self.used_names:
                        continue
                    self.add("binary_sensor", room, f"{kind}deur", "de", [slug, "door"],
                             name=name, state=R.choice(["off", "off", "on"]),
                             attrs={"device_class": "door"})
            if R.random() < 0.7:
                base = R.choice(["Huisdeur", "Entreedeur", "Huisdeur"])
                name = base + suffix
                if name in self.used_names:
                    name = f"{name} Sensor"
                hall = next((r for r in ("Gang", "Entree") if r in self.rooms), None)
                self.add("binary_sensor", hall, base, "de", ["huisdeur_state", "front_door_contact"],
                         name=name, state=R.choice(["off", "off", "on"]),
                         attrs={"device_class": "door"})
        # electric car: charge level and range next to each other
        if R.random() < 0.3:
            car, art, cslug = R.choice(CARS)
            self.add("sensor", None, "Laadniveau", "het", ["battery_hv_state_of_charge", f"{cslug}_soc"],
                     name=f"{car} Laadniveau", state=str(R.randint(10, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
            self.add("sensor", None, "Bereik", "het",
                     [f"{cslug}_range_ev_remaining_range", f"{cslug}_bereik"],
                     name=f"{car} Bereik", state=str(R.randint(40, 450)),
                     attrs={"device_class": "distance", "unit_of_measurement": "km"})
        # heating systems expose their modes as switches next to the climate
        # entity ("Verwarming Huis Afwezigheidsmodus")
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            mode, slug = R.choice(HEAT_MODES)
            base = R.choice(["Verwarming", "Verwarming Huis", "Thermostaat", "Thuis"])
            room = "Stookruimte" if "Stookruimte" in self.rooms \
                and R.random() < 0.5 else None
            if f"{base} {mode}" not in self.used_names:
                self.add("switch", room, mode, "de", [f"verwarming_{slug}", f"thuis_{slug}"],
                         name=f"{base} {mode}", state=R.choice(["on", "off", "off"]))
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            e = self.add("light", None, "__group__", "pl", ["alle_lampen", "all_lights"],
                         name=R.choice(["Alle lampen", "Alle lichten", "Licht Huis"]),
                         state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            self.add("cover", None, "__group__", "pl", ["alle_rolluiken", "all_covers"],
                     name=R.choice(["Alle rolluiken", "Alle Rolluiken", "Rolluiken Huis"]),
                     state=R.choice(["open", "closed"]))
        # read-only distractors and odd names; fill up to the target size
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "Huis"
        if k == 0 and room:
            self.add("binary_sensor", room, "Aanwezigheid", "de", ["presence"],
                     name=R.choice([f"Aanwezigheid {room}", f"{room} Presence",
                                    f"Bezetting {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "Beweging", "de", ["motion"],
                     name=R.choice([f"{rname} Motion", f"Beweging {rname}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant = R.choice(PLANTS)
            self.add("sensor", room, "Bodemvocht", "het", ["plant_sensor", "moisture"],
                     name=f"{plant} Bodemvocht", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "Accu", "de", ["battery"],
                     name=R.choice(["UPS Accu", "Auto Accu", "Rookmelder Batterij",
                                    "Telefoon Accu", f"{rname} Sensor Batterij"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "Printer", "de", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, unit = R.choice([("Windsnelheid", "km/h"), ("Windstoten", "km/h"),
                                 ("Neerslag", "mm/h"), ("Luchtdruk", "hPa"),
                                 ("Regenton Inhoud", "L")])
            self.add("sensor", room, nm, "de", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": unit})
        elif k == 6:
            self.add("sensor", room, "Status", "de", ["status"],
                     name=R.choice(["UPS Status", "Droger", "Wasmachine Status",
                                    "Afwasmachine Programma"]),
                     state=R.choice(["idle", "OL", "running", "off"]))
        elif k == 7:
            nm = R.choice(TOGGLES)
            if nm not in self.used_names:
                self.add("input_boolean", None, nm, "de", [nm.lower()], name=nm,
                         state=R.choice(["on", "off"]))
        elif k == 8:
            nm = R.choice(JUNK_SWITCHES)
            if nm not in self.used_names:
                self.add("switch", room, nm, "de", [ascii_slug(nm)], name=nm,
                         state=R.choice(["on", "off"]))
        else:
            self.add("sensor", room, "Temperatuur", "de", ["temperature"],
                     name=R.choice([f"{rname} Temperatuur", f"Temperatuur {rname}",
                                    f"{rname} Thermometer"]),
                     state=str(round(R.uniform(15, 26), 1)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})

    # ---- queries ----
    def of(self, domain, room="*", plain=False):
        """Entities of a domain (never group entities). plain=True also drops
        covers that are not window covers (awning, garage door)."""
        return [e for e in self.ents if e.domain == domain
                and e.label != "__group__"
                and not (plain and e.label in NO_GROUP_COVERS)
                and (room == "*" or e.area == room)]

    def group(self, domain):
        return next((e for e in self.ents if e.domain == domain
                     and e.label == "__group__"), None)

    def system_prompt(self, location=None):
        return S.build_system_prompt(
            [(e.eid, e.name, e.area) for e in self.ents], location, LANG, VERBOSITY)

    def state_of(self, ents):
        # unreachable entities report no attribute values in Home Assistant
        return S.dumps([S.format_state(e.eid, e.name, e.state,
                                       {} if e.state == "unavailable" else e.attrs, LANG)
                        for e in ents])


# --------------------------------------------------------------------------
# references: how the user names a device or a group of devices
# --------------------------------------------------------------------------
@dataclass
class Ref:
    acc: str      # phrase used in commands ("het licht in de badkamer")
    nom: str      # phrase for confirmations (Dutch: the same words)
    plural: bool
    ents: list
    area: str | None = None   # set for "het licht in de badkamer": control by area


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    # "het licht in de hal" / "de lamp in de hal" mean every light of the
    # room, so a single lamp only gets these words when it is the room's only light
    generic = e.domain == "light" and e.label in ("Licht", "Lamp") \
        and len(h.of("light", e.area)) > 1
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1 \
            and not generic:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 and e.name != e.label:
        opts += ["type_only"]           # "de tv" for "LG webOS Smart TV"
    choice = R.choice(opts)
    if choice == "type_only":
        p = f"{ART[e.gender]} {e.label.lower()}"
        return Ref(p, p, e.plural, [e])
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        art = R.random() < 0.35 and e.label and e.name.endswith(e.label)
        if art:
            return Ref(f"{ART[e.gender]} {name}", f"{ART[e.gender]} {e.name}",
                       e.plural, [e])
        return Ref(name, e.name, e.plural, [e])
    prep = ROOM_BY_NAME[e.area][1]
    p = f"{ART[e.gender]} {e.label.lower()} {prep}"
    return Ref(p, p, e.plural, [e])


# group nouns per domain: (phrase, phrase, plural)
GROUP = {
    "light": [("het licht", "het licht", False), ("de lampen", "de lampen", True),
              ("de lichten", "de lichten", True), ("de verlichting", "de verlichting", False)],
    "cover": [("de rolluiken", "de rolluiken", True), ("de rolgordijnen", "de rolgordijnen", True),
              ("de jaloezieën", "de jaloezieën", True)],
    "climate": [("de verwarming", "de verwarming", False),
                ("de thermostaten", "de thermostaten", True)],
    "media_player": [("de muziek", "de muziek", False)],
}
ALL_GROUP = {
    "light": [("alle lampen", "Alle lampen"), ("alle lichten", "Alle lichten"),
              ("overal het licht", "Het licht in het hele huis")],
    "cover": [("alle rolluiken", "Alle rolluiken"), ("alle rolgordijnen", "Alle rolgordijnen")],
    "climate": [("alle thermostaten", "Alle thermostaten")],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    acc, nom, pl = R.choice(GROUP[domain])
    area = room if domain in S.AREA_DOMAINS else None
    if location == room and R.random() < 0.7:
        return Ref(acc, nom, pl, ents, area)    # "Doe het licht aan"
    prep = ROOM_BY_NAME[room][1]
    return Ref(f"{acc} {prep}", f"{nom} {prep}", pl, ents, area)


# --------------------------------------------------------------------------
# message builders
# --------------------------------------------------------------------------
CALL_ID = iter(range(10**9))


def tool_call(name, args):
    cid = f"call_{next(CALL_ID) % 100000:05d}"
    if name == "control":
        LAST_CONTROL[:] = args["actions"]
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": S.dumps(wire(name, args))}}]}, cid


def tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def action(service, ents, data=None, area=None):
    """ents may be a Ref; its area (room group of lights) carries over.
    Area actions keep the resolved entity_id internally (state tracking,
    evaluation); tool_call() sends only the area to the model."""
    if isinstance(ents, Ref):
        area = area or ents.area
        ents = ents.ents
    a = {"service": service}
    if area is not None:
        a["area"] = area
    a["entity_id"] = [e.eid for e in ents]
    if data:
        a["data"] = data
    return a


def wire(name, args):
    """What the model writes: area actions without their resolved entity_id."""
    if name == "control":
        return {"actions": [{k: v for k, v in a.items()
                             if not (k == "entity_id" and "area" in a)}
                            for a in args["actions"]]}
    if name == "get_state" and args.get("area"):
        return {"domain": args.get("domain", "light"), "area": args["area"]}
    return args


LAST_CONTROL: list = []


def ok_result(actions=None):
    """Successful control result; area actions report what they switched."""
    actions = LAST_CONTROL if actions is None else actions
    if any("area" in a for a in actions):
        return S.control_result(None, [i for a in actions for i in a["entity_id"]])
    return S.control_result()


# imperatives that start a command; lowered when a sentence part moves
# behind another one ("... en doe ...")
IMPERATIVES = ("Doe", "Zet", "Schakel", "Open", "Sluit", "Start", "Pauzeer", "Activeer",
               "Deactiveer", "Dim", "Speel", "Laat", "Draai", "Maak", "Stop", "Stel",
               "Kleur", "Vergrendel", "Ontgrendel", "Rol", "Haal", "Stuur", "Druk",
               "Voer", "Geef", "Besproei", "Hervat", "Ga", "Herhaal", "Vink", "Streep",
               "Verwijder", "Schrijf", "Voeg", "Noteer", "Beëindig", "Trek")


def voice(text: str) -> str:
    """Make user text look like speech-to-text output sometimes."""
    r = R.random()
    if r < 0.25:
        text = text.rstrip("?.!")
    elif r < 0.35:
        text = text.rstrip("?.!").lower()
    if R.random() < 0.05:
        pre = R.choice(["Hé, ", "Oké, ", "Zeg, ", "+"])
        if pre == "+" and text.split(" ", 1)[0] in IMPERATIVES and not text.endswith("?"):
            text = text.rstrip(".!") + ", alsjeblieft"
        else:
            pre = "Hoi, " if pre == "+" else pre
            # speech-to-text output in lower case lowers the opener too
            text = (pre.lower() if text == text.lower() else pre) + text[0].lower() + text[1:]
    return text


def confirm(ref: Ref, adj: str, part: str) -> str:
    """adj: state word ('aan'), part: participle ('aangezet')."""
    return R.choice([
        f"{cap(ref.nom)} {is_(ref.plural)} nu {adj}.",
        f"{cap(ref.nom)} {is_(ref.plural)} {adj}.",
        f"Ik heb {ref.acc} {part}.",
        f"Gedaan, {ref.nom} {is_(ref.plural)} {adj}.",
        f"Oké, {ref.nom} {is_(ref.plural)} nu {adj}.",
        "Gedaan.",
    ])


# --------------------------------------------------------------------------
# command intents: each returns (user_text, [actions], confirmation)
# --------------------------------------------------------------------------
def pick_target(h: House, domain, location=None, group_ok=True):
    """Return a Ref for one entity or a room group of `domain`."""
    plain = domain == "cover"
    ents = h.of(domain, plain=plain)
    if not ents:
        return None
    if group_ok and domain in GROUP and R.random() < 0.35:
        rooms = [r for r in h.rooms if h.of(domain, r, plain=plain)]
        if rooms:
            room = location if location in rooms and R.random() < 0.6 else R.choice(rooms)
            return ref_room_group(domain, room, h.of(domain, room, plain=plain), location)
    return ref_single(h, R.choice(ents))


def i_light_onoff(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    on = R.random() < 0.5
    if on:
        user = R.choice([
            "Doe {r} aan", "Zet {r} aan", "Schakel {r} in", "{r} aan",
            "Graag {r} aan", "Kun je {r} aandoen?",
            "Doe {r} even aan", "{r} aanzetten", "Wil je {r} aanzetten?",
            "Ik heb licht nodig, doe {r} aan", "Zet {r} even aan.",
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref)],
                confirm(ref, "aan", "aangezet"))
    user = R.choice([
        "Doe {r} uit", "Zet {r} uit", "Schakel {r} uit", "{r} uit",
        "Graag {r} uit", "Kun je {r} uitdoen?", "Doe {r} even uit.",
        "{r} uitzetten", "Wil je {r} uitzetten?",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref)],
            confirm(ref, "uit", "uitgezet"))


def i_light_room_phrase(h, loc):
    """'Licht in de badkamer aan' / 'Het is te donker in de badkamer'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    prep = ROOM_BY_NAME[room][1]
    ref = Ref(f"het licht {prep}", f"het licht {prep}", False, ents, room)
    if R.random() < 0.5:
        user = R.choice([f"Licht {prep} aan", f"Doe {prep} het licht aan",
                         f"Het is te donker {prep}", f"{cap(prep)} het licht aan graag",
                         f"Licht aan {prep}"])
        return user, [action("light.turn_on", ref)], confirm(ref, "aan", "aangezet")
    user = R.choice([f"Licht {prep} uit", f"Doe {prep} het licht uit",
                     f"Licht uit {prep}", f"{cap(prep)} het licht uitdoen"])
    return user, [action("light.turn_off", ref)], confirm(ref, "uit", "uitgezet")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        user = R.choice([
            "Dim {r} naar {p} procent", "Zet {r} op {p}%",
            "{r} op {p} procent", "Doe {r} op {p} procent",
            "Zet {r} op {p} procent helderheid",
        ]).format(r=ref.acc, p=p)
        conf = R.choice([f"{cap(ref.nom)} {staat(ref.plural)} nu op {p} procent.",
                         f"Ik heb {ref.acc} op {p} procent gezet.", "Gedaan."])
        return user, [action("light.turn_on", ref, {"brightness_pct": p})], conf
    up = r < 0.8
    user = R.choice(["Maak {r} feller", "{r} feller", "{r} wat feller graag"]
                    if up else
                    ["Maak {r} wat donkerder", "Dim {r}", "{r} wat minder fel"]).format(r=ref.acc)
    step = 20 if up else -20
    conf = (f"{cap(ref.nom)} {is_(ref.plural)} nu "
            f"{'feller' if up else 'gedimd'}.") if R.random() < 0.7 else "Gedaan."
    return user, [action("light.turn_on", ref, {"brightness_step_pct": step})], conf


def i_light_color(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    if R.random() < 0.75:
        nl, en = R.choice(list(COLORS.items()))
        data = {"color_name": en}
    else:
        nl, k = R.choice(list(COLOR_TEMPS.items()))
        data = {"color_temp_kelvin": k}
    user = R.choice(["Maak {r} {c}", "Zet {r} op {c}", "Kleur {r} {c}",
                     "{r} op {c} graag", "Kun je {r} {c} maken?",
                     "Ik wil {r} in het {c}"]).format(r=ref.acc, c=nl)
    brandt = "branden" if ref.plural else "brandt"
    conf = R.choice([f"{cap(ref.nom)} {brandt} nu {nl}.",
                     f"Ik heb {ref.acc} op {nl} gezet.", "Gedaan."])
    return user, [action("light.turn_on", ref, data)], conf


def i_switch(h, loc):
    ents = h.of("switch") + h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = R.random() < 0.5
    svc = f"{e.domain}.turn_{'on' if on else 'off'}"
    if on:
        user = R.choice(["Schakel {r} in", "Zet {r} aan", "{r} aan", "{r} aanzetten",
                         "Kun je {r} aanzetten?", "Start {r}"])
        conf = confirm(ref, "aan", "ingeschakeld")
    else:
        user = R.choice(["Schakel {r} uit", "Zet {r} uit", "{r} uit", "{r} uitzetten",
                         "Kun je {r} uitschakelen?"])
        conf = confirm(ref, "uit", "uitgeschakeld")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    user = R.choice(["Zet {r} op {p} procent", "{r} op {p}%",
                     "Laat {r} op {p} procent draaien"]).format(r=ref.acc, p=p)
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {staat(ref.plural)} op {p} procent.", "Gedaan."])


def i_awning(h, loc):
    """Zonnescherm: uitrollen = open_cover, inrollen = close_cover."""
    ents = [e for e in h.of("cover") if e.label == "Zonnescherm"]
    if not ents:
        return None
    e = R.choice(ents)
    n = "het zonnescherm" if len(ents) == 1 and R.random() < 0.7 else e.name
    if R.random() < 0.5:
        user = R.choice([f"Rol {n} uit", f"{cap(n)} uitrollen", f"Draai {n} uit",
                         f"{cap(n)} uit", "We hebben schaduw nodig, rol het zonnescherm uit"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["Het zonnescherm rolt uit.", "Gedaan, het zonnescherm is uitgerold."])
    user = R.choice([f"Rol {n} in", f"{cap(n)} inrollen", f"Haal {n} in",
                     f"{cap(n)} in", "Het gaat regenen, rol het zonnescherm in"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["Het zonnescherm rolt in.", "Gedaan, het zonnescherm is ingerold."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["de robotmaaier", "de grasmaaier", e.name])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Start {n}", f"Laat {n} maaien", "Gras maaien graag",
                         f"Stuur {n} op pad"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{e.name} maait nu."
    if r < 0.8:
        user = R.choice([f"Stuur {n} naar het laadstation", f"{cap(n)} naar huis",
                         f"Haal {n} binnen"])
        return user, [action("lawn_mower.dock", [e])], f"{e.name} rijdt naar het laadstation."
    user = R.choice([f"Pauzeer {n}", f"Zet {n} even stil"])
    return user, [action("lawn_mower.pause", [e])], f"{e.name} is gepauzeerd."


AREA_PLANTS = {"Balkon": "de balkonplanten", "Terras": "de terrasplanten",
               "Tuin": "de tuin"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == "Beregening"]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Start de beregening", "Geef de planten water"]
    if e.area:
        prep = ROOM_BY_NAME[e.area][1]
        out += [f"Start de beregening {prep}", f"Geef de planten {prep} water",
                f"Besproei de planten {prep}"]
        if e.area in AREA_PLANTS:
            out.append(f"Geef {AREA_PLANTS[e.area]} water")
    elif "tuin" in e.name.lower():
        out += ["Geef de tuin water", "Start de tuinberegening",
                "Besproei de tuin"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name] if several else [e.name, "de beregening", "het ventiel"])
    if R.random() < 0.5:
        user = R.choice([f"Open {n}", f"Zet {n} aan", f"Doe {n} open"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"{e.name} is geopend."
    user = R.choice([f"Sluit {n}", f"Zet {n} uit", f"Doe {n} dicht",
                     "Stop de beregening"])
    return user, [action("valve.close_valve", [e])], f"{e.name} is gesloten."


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    # "Vakantie" is also called "Vakantiemodus"; it stays an input_boolean
    mode = n if n.endswith("modus") else f"{n}modus"
    if mode != n and any(x.name == mode for x in h.ents):
        mode = n          # a script "Vakantiemodus" exists: only the plain name is clear
    m = low(mode)
    if on:
        user = R.choice([f"Activeer {n}", f"{n} aan", f"Zet {n} aan",
                         f"Activeer de {m}", f"{mode} aan", f"Zet de {m} aan",
                         f"Schakel de {m} in"])
        return user, [action("input_boolean.turn_on", [e])], f"{n} is geactiveerd."
    user = R.choice([f"Deactiveer {n}", f"{n} uit", f"Zet {n} uit",
                     f"Deactiveer de {m}", f"{mode} uit", f"Zet de {m} uit"])
    return user, [action("input_boolean.turn_off", [e])], f"{n} is gedeactiveerd."


def i_appliance(h, loc):
    """'Zet de droger uit' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = f"{ART[g]} {low(app)}"
    if R.random() < 0.5:
        user = R.choice([f"Zet {n} uit", f"Doe {n} uit", f"{app} uit",
                         f"Schakel {n} uit", f"Kun je {n} uitzetten?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"{power.name} is uit.", f"Ik heb {power.name} uitgezet."])
    user = R.choice([f"Zet {n} aan", f"Doe {n} aan", f"{app} aan",
                     f"Schakel {n} in"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"{power.name} is aan.", f"Ik heb {power.name} aangezet."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        what = vac[e.label]
        user = R.choice([f"Start {what}", f"Start {what} bij de robotstofzuiger",
                         f"Begin met {what}", f"{e.name} starten"])
    elif e.label == "Beregening":
        user = R.choice(irrigation_phrases(h, e) + [f"Druk op {e.name}"])
    else:
        user = R.choice([f"Druk op {e.name}", f"Activeer {e.name}"])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} is geactiveerd.", f"Ik heb {e.name} gestart.", "Gedaan."])


# lamp names without a room (the names of the acceptance house are left out
# on purpose, so the acceptance test measures generalisation)
FREE_LIGHT_NAMES = ["Sterrenhemel", "Zoutlamp", "Lavalamp", "Schilderijlamp", "Vitrinekast",
                    "Hue Go", "Nanoleaf", "Hue Play", "Wolkenlamp", "Maanlamp",
                    "Lichtlijst", "LED Fotolijst", "Kerstster", "Bollamp",
                    "Hoeklamp", "Lamp Bank", "Licht Dressoir", "Licht Bar",
                    "Licht Bed", "Licht Kast", "Lamp Raam", "Vloerlamp",
                    "Hanglamp Tafel", "Lichtboog", "Neonbord", "Govee Strip", "Tafellamp",
                    "Licht Ladekast", "Licht Spiegel", "Leeslampje", "Lamp Hoek",
                    "Bureaulamp"]

DOOR_ROOMS = ("Woonkamer", "Eetkamer", "Keuken", "Slaapkamer", "Werkkamer")
# (name, article, slug)
CARS = [("Enyaq", "de", "enyaq_iv80"), ("ID.3", "de", "id3"), ("Model 3", "de", "model_3"),
        ("Zoe", "de", "zoe"), ("Enyaq", "de", "enyaq"), ("e-Golf", "de", "egolf"),
        ("EQA", "de", "eqa")]

WEATHER_SENSORS = [  # (name, label, day, slugs)
    ("Verwachting temperatuur vandaag", "Verwachting", "vandaag",
     ["verwachting_temperatuur_vandaag", "forecast_temperature_today"]),
    ("Verwachting temperatuur morgen", "Verwachting", "morgen",
     ["verwachting_temperatuur_morgen", "forecast_temperature_tomorrow"]),
    ("Maximumtemperatuur vandaag", "Verwachting", "vandaag",
     ["maximumtemperatuur_vandaag", "forecast_high_today"]),
    ("Maximumtemperatuur morgen", "Verwachting", "morgen",
     ["maximumtemperatuur_morgen", "forecast_high_tomorrow"]),
    ("Gevoelstemperatuur", "Weerwaarde", None, ["gevoelstemperatuur", "feels_like_temperature"]),
    ("Dauwpunt", "Weerwaarde", None, ["dauwpunt", "dew_point"]),
]

HEAT_MODES = [("Afwezigheidsmodus", "away_mode"), ("Eco-modus", "eco_mode"),
              ("Vorstbeveiliging", "frost_protection")]


def i_heating_mode(h, loc):
    """'Zet de afwezigheidsmodus van de verwarming aan' -> the mode switch,
    not the climate entity."""
    modes = [e for e in h.ents if e.domain == "switch"
             and e.label in [m for m, _ in HEAT_MODES]]
    if not modes or not h.of("climate"):
        return None
    e = R.choice(modes)
    m = low(e.label)
    on = R.random() < 0.6
    if on:
        user = R.choice([f"Activeer de {m} van de verwarming", f"Zet de verwarming op {m}",
                         f"Verwarming op {m}", f"{e.label} aan", f"Zet de {m} aan",
                         f"Zet de verwarming in de {m}" if m.endswith("modus")
                         else f"Schakel de {m} in", f"Schakel de {m} in"])
    else:
        user = R.choice([f"Deactiveer de {m} van de verwarming", f"{e.label} uit",
                         f"Zet de {m} uit", f"Schakel de {m} van de verwarming uit",
                         f"Zet de {m} van de verwarming uit"])
    adj, part = ("aan", "ingeschakeld") if on else ("uit", "uitgeschakeld")
    return (user, [action("switch.turn_on" if on else "switch.turn_off", [e])],
            R.choice([f"{e.name} is nu {adj}.", f"Ik heb {e.name} {part}."]))


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Geef de balkonplanten water" -> balcony valve, not the garden button)."""
    devs = irrigation(h)
    if len(devs) < 2:
        return None
    e = R.choice(devs)
    phrases = irrigation_phrases(h, e)
    if not phrases:
        return None
    user = R.choice(phrases)
    if e.domain == "valve":
        return user, [action("valve.open_valve", [e])], R.choice(
            [f"{e.name} is geopend.", f"Ik heb {e.name} geopend."])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} is geactiveerd.", f"Ik heb {e.name} gestart."])


def i_irrigation_pair(h, loc):
    """A plant valve on the balcony or patio next to an area-less garden
    button, as many real installs have it. "Geef de balkonplanten water"
    opens the valve, "Geef de tuin water" presses the button."""
    rooms = [r for r in h.rooms if r in ("Balkon", "Terras")]
    if not rooms or irrigation(h):
        return None
    room = R.choice(rooms)
    valve = h.add("valve", room, "Beregening", "de", ["planten", "beregening"],
                  name=R.choice([f"{room} Planten", f"Planten {room}",
                                 f"Beregening {room}", "Waterkraan"]),
                  state=R.choice(["open", "closed", "closed"]))
    button = h.add("button", None, "Beregening", "de", ["beregening_tuin"],
                   name=R.choice(["Beregening Tuin", "Tuinberegening"]),
                   state="unknown")
    if R.random() < 0.6:
        prep = ROOM_BY_NAME[room][1]
        user = R.choice([f"Geef {AREA_PLANTS[room]} water", f"Besproei de planten {prep}",
                         f"Start de beregening {prep}", f"Geef de planten {prep} water"])
        return user, [action("valve.open_valve", [valve])], R.choice(
            [f"{valve.name} is geopend.", f"Ik heb {valve.name} geopend."])
    user = R.choice(["Geef de tuin water", "Start de tuinberegening",
                     "Besproei de tuin", "Start de beregening in de tuin"])
    return user, [action("button.press", [button])], R.choice(
        [f"{button.name} is geactiveerd.", f"Ik heb {button.name} gestart."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Open {r}", "Doe {r} open", "Doe {r} omhoog", "{r} omhoog",
                         "{r} openen", "Trek {r} omhoog", "Kun je {r} openen?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "open", "geopend"))
    if r < 0.8:
        user = R.choice(["Sluit {r}", "Doe {r} dicht", "Doe {r} omlaag", "{r} omlaag",
                         "{r} sluiten", "Laat {r} zakken", "Kun je {r} sluiten?"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "dicht", "gesloten"))
    if r < 0.88:
        user = R.choice(["Stop {r}", "{r} stilzetten", "{r} stoppen", "Stop {r} even"])
        return (user.format(r=ref.acc), [action("cover.stop_cover", ref.ents)],
                R.choice(["Gestopt.", f"Ik heb {ref.acc} stilgezet.", "Gedaan."]))
    p, phrase = R.choice([(50, "half"), (30, "op 30 procent"), (70, "op 70 procent"),
                          (20, "op 20 procent"), (80, "op 80 procent"),
                          (50, "op 50 procent"), (10, "op 10 procent")])
    if phrase == "half":
        user = R.choice(["Doe {r} half dicht", "Doe {r} half omlaag", "{r} half"]).format(r=ref.acc)
    else:
        user = R.choice(["Zet {r} {ph}", "Doe {r} {ph}", "{r} {ph}"]).format(r=ref.acc, ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} {staat(ref.plural)} nu op {p} procent.", "Gedaan."])


def i_climate(h, loc):
    ents = h.of("climate")
    if not ents:
        return None
    rooms = room_order({e.area for e in ents if e.area})
    r = R.random()
    if r < 0.75:
        t = R.choice([17, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 24, 25])
        if R.random() < 0.4 and rooms:
            room = loc if loc in rooms and R.random() < 0.5 else R.choice(rooms)
            tgt = h.of("climate", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Zet de verwarming {prep} op {{t}} graden",
                             f"Verwarming {prep} op {{t}} graden",
                             f"Maak het {prep} {{t}} graden",
                             f"Ik wil {prep} {{t}} graden"]).format(t=nl_num(t))
            nom = f"de verwarming {prep}"
            pl = False
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Zet {r} op {t} graden", "{r} op {t} graden",
                             "Stel {r} in op {t} graden", "Draai {r} naar {t} graden"]
                            ).format(r=ref.acc, t=nl_num(t))
            nom, pl = ref.nom, ref.plural
        conf = R.choice([f"{cap(nom)} {staat(pl)} nu op {nl_num(t)} graden.",
                         f"Ik heb {nom} op {nl_num(t)} graden gezet.",
                         f"Gedaan, {nl_num(t)} graden."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Zet {r} aan", "Doe {r} aan", "{r} aan"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "aan", "aangezet")
    user = R.choice(["Zet {r} uit", "Doe {r} uit", "{r} uit"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "uit", "uitgezet")


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc == e.name and R.random() < 0.7:
        p = f"{ART[e.gender]} {low(e.name)}"
        ref = Ref(p, p, False, [e])
    if ref.acc == f"{ART[e.gender]} {e.name}":
        p = f"{ART[e.gender]} {low(e.name)}"
        ref = Ref(p, p, False, [e])
    obj = PRON[e.gender][1]
    if R.random() < 0.55:
        user = R.choice(["Doe {r} op slot", "Sluit {r} af", "Vergrendel {r}",
                         "Is {r} dicht? Doe " + obj + " dan op slot", "{r} op slot doen"])
        return (user.format(r=ref.acc), [action("lock.lock", [e])],
                confirm(ref, "op slot", "op slot gedaan"))
    user = R.choice(["Doe {r} van het slot", "Ontgrendel {r}", "Haal {r} van het slot",
                     "{r} ontgrendelen", "Open het slot van {n}"])
    return (user.format(r=ref.acc, n=f"{ART[e.gender]} {low(e.name)}"), [action("lock.unlock", [e])],
            confirm(ref, "van het slot", "van het slot gehaald"))


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
            "Schakel het alarm in", "Activeer het alarmsysteem",
            "Ik ga weg, zet het alarm aan", "Alarm aan",
            "Alarm op afwezig"])
        conf = R.choice(["Het alarmsysteem is ingeschakeld.",
                         "Het alarmsysteem staat nu aan (afwezig)."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Alarm op thuis", "Zet het alarm op aanwezig",
            "Activeer het alarm in de thuismodus"])
        conf = "Het alarmsysteem is ingeschakeld in de thuismodus."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Zet het alarm aan voor de nacht", "Nachtmodus voor het alarm",
            "Activeer het alarm in de nachtmodus"])
        conf = "Het alarmsysteem is ingeschakeld in de nachtmodus."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Zet het alarm uit", "Alarm uitschakelen", "Deactiveer het alarmsysteem",
            "Ik ben thuis, zet het alarm uit", "Alarm uit"])
        conf = R.choice(["Het alarmsysteem is uitgeschakeld.",
                         "Het alarmsysteem is nu gedeactiveerd."])
    if code:
        user += f", code {code}"
    return user, [action(svc, [e], {"code": code} if code else None)], conf


def i_media(h, loc):
    ents = h.of("media_player")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    r = R.random()
    opts = [
        ("media_player.media_pause", ["Pauzeer {r}", "Pauze op {n}", "{r} pauzeren",
                                      "Afspelen op {n} pauzeren"], None,
         "Gepauzeerd."),
        ("media_player.media_play", ["Speel verder op {n}", "Hervat het afspelen op {n}",
                                     "Play op {n}"], None, "Het afspelen gaat verder."),
        ("media_player.media_next_track", ["Volgend nummer op {n}",
                                           "Ga op {n} naar het volgende nummer",
                                           "Volgende liedje op {n}"], None, "Volgend nummer."),
        ("media_player.volume_up", ["Zet {r} harder", "{r} harder"], None, "Harder gezet."),
        ("media_player.volume_down", ["Zet {r} zachter", "{r} zachter"], None, "Zachter gezet."),
        ("media_player.turn_off", ["Zet {r} uit", "Doe {r} uit", "{r} uit"], None,
         confirm(ref, "uit", "uitgezet")),
        ("media_player.turn_on", ["Zet {r} aan", "Doe {r} aan"], None,
         confirm(ref, "aan", "aangezet")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Zet het volume van {n} op {v} procent",
                         "Volume {n} op {v} procent",
                         "Zet {r} op {v} procent volume"]).format(r=ref.acc, n=e.name, v=v)
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"Het volume staat nu op {v} procent."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, n=e.name), [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = R.choice([f"{ART[e.gender]} {low(e.name)}" if e.name == e.label else e.name,
                  "de stofzuiger", "de robotstofzuiger", "de stofzuiger"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Start {n}", f"Laat {n} stofzuigen", f"{cap(n)} starten",
                         "Stofzuigen graag", f"Stuur {n} op pad"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{e.name} gaat aan de slag.", "De stofzuiger start."])
    if r < 0.8:
        user = R.choice([f"Stuur {n} naar het laadstation", f"{cap(n)} terug naar de basis",
                         f"Stuur {n} naar huis"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{e.name} rijdt terug naar het laadstation."
    user = R.choice([f"Pauzeer {n}", f"{cap(n)} pauzeren", f"{cap(n)} even stilzetten"])
    return user, [action("vacuum.pause", [e])], f"{e.name} is gepauzeerd."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Activeer de scène {e.name}", f"Scène {e.name}",
                         f"{e.name} activeren", f"Start de scène {e.name}",
                         f"Zet scène {e.name} aan"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"De scène {e.name} is actief.", f"Scène {e.name} geactiveerd.", "Gedaan."])
    if e.domain == "script":
        user = R.choice([f"Voer {e.name} uit", f"Start {e.name}", f"{e.name} starten",
                         f"Script {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} loopt.", f"Ik heb {e.name} gestart.", "Gedaan."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    # a group entity ("Alle lampen") replaces the list of members
    if (grp := h.group(domain)) is not None:
        ents = [grp]
    acc, nom = R.choice(ALL_GROUP[domain])
    whole = S.ALL_AREAS if domain in S.AREA_DOMAINS and h.group(domain) is None else None
    ref = Ref(acc, nom, not acc.startswith("overal"), ents, whole)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Doe {acc} uit", f"Zet {acc} uit", f"{cap(acc)} uit",
                             "Overal het licht uit"])
            return user, [action("light.turn_off", ref)], R.choice(
                [f"{nom} {is_(ref.plural)} uit.", "Alle lampen zijn uit.", "Gedaan."])
        user = R.choice([f"Doe {acc} aan", f"Zet {acc} aan", f"{cap(acc)} aan"])
        return user, [action("light.turn_on", ref)], R.choice(
            [f"{nom} {is_(ref.plural)} aan.", "Alle lampen zijn aan.", "Gedaan."])
    if R.random() < 0.6:
        user = R.choice([f"Sluit {acc}", f"Doe {acc} dicht", f"{cap(acc)} omlaag",
                         f"Doe {acc} omlaag"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{nom} zijn dicht.", "Gedaan, alles is dicht."])
    user = R.choice([f"Open {acc}", f"Doe {acc} open", f"{cap(acc)} omhoog", f"Doe {acc} omhoog"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{nom} zijn open.", "Gedaan, alles is open."])


COMMANDS = [
    (i_light_onoff, 14), (i_light_room_phrase, 6), (i_light_brightness, 6),
    (i_light_color, 4), (i_switch, 6), (i_fan_pct, 1), (i_cover, 9),
    (i_climate, 7), (i_lock, 3), (i_alarm, 3), (i_media, 5), (i_vacuum, 3),
    (i_scene, 4), (i_all, 6), (i_awning, 2), (i_mower, 2), (i_valve, 1),
    (i_toggle, 2), (i_appliance, 3), (i_button, 2), (i_irrigation, 12), (i_irrigation_pair, 2),
    (i_heating_mode, 4),
]


def pick_command(h, loc):
    forced = FORCE.get("intent")
    for _ in range(30):
        fn = forced or R.choices([c[0] for c in COMMANDS], [c[1] for c in COMMANDS])[0]
        out = fn(h, loc)
        if out:
            # a sentence starts with a capital, also "de lamp aan"
            return (cap(out[0]),) + tuple(out[1:])
        continue
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
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def lower_start(text):
    """Lower the first letter unless the text starts with a device name
    ("Keukenlamp aan" keeps its capital, "Doe ..." becomes "doe ...")."""
    names = [e.name for e in CURRENT[0].ents] if CURRENT else []
    if any(text.startswith(n) for n in names if n):
        return text
    return low(text)


def join_commands(a, b):
    """Two command intents -> one sentence 'X en y', or None."""
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    if ub.startswith(("Kun je", "Wil je", "Ik ", "Graag", "Het is", "We ")) or \
            ua.startswith(("Kun je", "Wil je", "Het is")):
        return None
    # lowercase the start of the second command ("... en doe ...")
    return f"{ua} en {lower_start(ub)}"


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    actions = a[1] + b[1]
    conf = R.choice(["Gedaan.", "Allebei gedaan.",
                     f"{a[2].rstrip('.')} en {lower_start(b[2])}"
                     if a[2] != "Gedaan." and b[2] != "Gedaan." else "Gedaan."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def _state_word(e: Ent) -> str:
    d, s = e.domain, e.state
    if s == "unavailable":
        return "niet bereikbaar"
    if d in ("light", "switch", "fan", "input_boolean"):
        return "aan" if s == "on" else "uit"
    if d == "cover":
        return "open" if s == "open" else "dicht"
    if d == "lock":
        return "op slot" if s == "locked" else "niet op slot"
    if d == "binary_sensor":
        return "open" if s == "on" else "dicht"
    if d == "media_player":
        return {"playing": "aan het afspelen", "paused": "gepauzeerd",
                "off": "uit", "idle": "aan, maar speelt niets"}[s]
    return s


def s_question(h, loc):
    """State question -> get_state -> grounded Dutch answer."""
    kinds = []
    temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"
             and e.label not in ("Buitentemperatuur", "Verwachting")]
    if temps:
        kinds += ["temp"] * 2
    outs = [e for e in h.of("sensor") if e.label == "Buitentemperatuur"]
    if outs:
        kinds += ["outdoor"] * 2
    fcs = [e for e in h.of("sensor") if e.label == "Verwachting"]
    if fcs:
        kinds += ["forecast"]
    if h.of("climate"):
        kinds += ["climate"] * 2
    kinds += ["onoff"] * 3
    if len(h.of("light")) >= 2:
        kinds += ["which_lights"] * 2
    if [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "window"]:
        kinds += ["windows"] * 2
    if [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]:
        kinds += ["door"] * 3
    if [e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"]:
        kinds += ["car"] * 2
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
    if [e for e in h.of("sensor") if e.attrs.get("device_class") in ("moisture", "battery")]:
        kinds += ["generic"] * 2
    kind = R.choice(kinds)
    q_area = None
    if FORCE.get("qkind"):
        if FORCE["qkind"] not in kinds:
            return None
        kind = FORCE["qkind"]
    must = []
    if kind == "outdoor":
        # every outdoor sensor counts; forecasts and felt temperature do not
        user = R.choice(["Hoe warm is het buiten?", "Hoe koud is het buiten?",
                         "Wat is de buitentemperatuur?", "Hoeveel graden is het buiten?",
                         "Wat voor temperatuur is het buiten?"])
        vals = [nl_num(float(e.state)) for e in outs]
        if len(outs) == 1:
            ans = R.choice([f"Buiten is het {vals[0]} graden.",
                            f"De buitentemperatuur is {vals[0]} graden."])
        else:
            ans = (f"{outs[0].name} meet {vals[0]} graden, "
                   f"{outs[1].name} {vals[1]} graden.")
        ids, must = outs, vals
    elif kind == "forecast":
        if h.of("weather"):
            return None       # the weather entity answers with condition and course
        when = R.choice(fcs).attrs["_when"]
        ids = [e for e in fcs if e.attrs["_when"] == when]
        vals = [nl_num(float(e.state)) for e in ids]
        user = R.choice([f"Hoe warm wordt het {when}?", f"Hoeveel graden wordt het {when}?",
                         f"Wat zegt de verwachting voor {when}?"])
        if len(ids) == 1:
            ans = R.choice([f"Volgens de verwachting wordt het {when} {vals[0]} graden.",
                            f"{cap(when)} wordt het volgens de verwachting {vals[0]} graden."])
        else:
            ans = f"Voor {when} meldt {ids[0].name} {vals[0]} graden en {ids[1].name} {vals[1]} graden."
        must = vals
    elif kind == "temp":
        e = R.choice(temps)
        val = nl_num(float(e.state))
        if e.area is None or e.area in OUTDOOR or e.label != "Temperatuur":
            # a sensor without area is not automatically outside, and a printer
            # bed or a freezer does not tell how warm the room is
            user = R.choice([f"Wat geeft {e.name} aan?", f"Hoeveel graden meet {e.name}?"])
            ans = f"{e.name} meet {val} graden."
        else:
            prep = ROOM_BY_NAME[e.area][1]
            user = R.choice([f"Hoe warm is het {prep}?", f"Wat is de temperatuur {prep}?",
                             f"Hoeveel graden is het {prep}?", f"Temperatuur {prep}?"])
            ans = R.choice([f"{cap(prep)} is het {val} graden.",
                            f"De temperatuur {prep} is {val} graden."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        prep = ROOM_BY_NAME[e.area][1] if e.area else ""
        user = R.choice([f"Hoe hoog is de luchtvochtigheid {prep}?",
                         f"Hoe vochtig is het {prep}?"]).replace(" ?", "?")
        ans = f"De luchtvochtigheid {prep} is {e.state} procent.".replace("  ", " ")
        ids, must = [e], [e.state]
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = e.name.replace(" Bodemvocht", "")
            pl = low(plant)
            moet = "Moeten" if plant in PLURAL_PLANTS else "Moet"
            user = R.choice([f"Hoe vochtig is de grond bij de {pl}?",
                             f"{moet} de {pl} water hebben?",
                             f"Wat is de bodemvochtigheid van de {pl}?"])
            ans = f"De bodemvochtigheid van de {pl} is {e.state} procent."
        else:
            user = R.choice([f"Hoe vol is {e.name}?", f"Hoe staat het met {e.name}?",
                             f"Wat geeft {e.name} aan?"])
            ans = f"{e.name} staat op {e.state} procent."
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"Hoeveel stroom verbruikt {e.name} nu?",
                         f"Wat geeft {e.name} aan?", f"Hoe hoog is {e.name}?"])
        val = nl_num(float(e.state))
        ans = f"{e.name} geeft nu {val} watt aan."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"Op hoeveel graden staat {ref.nom}?",
                             f"Wat is de ingestelde temperatuur van {ref.nom}?"])
            val = nl_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {staat(ref.plural)} op {val} graden."
        else:
            if e.area:
                prep = ROOM_BY_NAME[e.area][1]
                user = R.choice([f"Hoe warm is het {prep}?", f"Hoeveel graden is het {prep}?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None   # ambiguous: sensor and thermostat in same room
            else:
                user = f"Welke temperatuur meet {ref.nom}?"
                prep = ""
            if "current_temperature" not in e.attrs:
                return None   # thermostat without its own sensor
            val = nl_num(e.attrs["current_temperature"])
            ans = (f"{cap(prep)} is het {val} graden." if prep
                   else f"{cap(ref.nom)} meet {val} graden.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        if e.domain == "cover":
            user, asked = R.choice([(f"{cap(is_(ref.plural))} {ref.nom} open?", "open"),
                                    (f"Zijn {ref.nom} omlaag?", "dicht")
                                    if ref.plural else
                                    (f"Is {ref.nom} gesloten?", "dicht"),
                                    (f"{cap(is_(ref.plural))} {ref.nom} dicht?", "dicht")])
        else:
            user, asked = R.choice([(f"Is {ref.nom} aan?", "aan"),
                                    (f"Is {ref.nom} nog aan?", "aan"),
                                    (f"Is {ref.nom} uitgeschakeld?", "uit"),
                                    (f"Is {ref.nom} uit?", "uit")]) \
                if not ref.plural else R.choice([(f"Zijn {ref.nom} aan?", "aan"),
                                                 (f"Zijn {ref.nom} uit?", "uit")])
        word = _state_word(e)
        ans = f"{cap(ref.nom)} {is_(ref.plural)} {word}."
        if word in ("aan", "uit", "open", "dicht") and R.random() < 0.8:
            ans = ("Ja, " if word == asked else "Nee, ") + ref.nom + \
                f" {is_(ref.plural)} {word}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"aan": "Ja, ", "uit": "Nee, "}[asked]
            ans = f"{yes}{ref.nom} {is_(ref.plural)} aan, op {pct} procent helderheid."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Welke lampen zijn er {prep} aan?", f"Brandt er {prep} nog licht?"])
        else:
            ents = h.of("light")
            user = R.choice(["Welke lampen zijn aan?", "Brandt er ergens nog licht?",
                             "Zijn er nog lampen aan?", "Welke lampen staan aan?"])
        q_area = room or S.ALL_AREAS
        if any(e.state == "unavailable" for e in ents):
            return None
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "Er is nu nergens licht aan." if not room else \
                f"{cap(ROOM_BY_NAME[room][1])} is nu geen licht aan."
            must = ["nergens" if not room else "geen"]
        elif len(on) == 1:
            ans = f"Alleen {on[0].name} is aan."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"Deze lampen zijn aan: {', '.join(names[:-1])} en {names[-1]}."
            must = names
        ids = ents
    elif kind == "windows":
        # only window sensors - presence and motion sensors are binary_sensors too
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") == "window"]
        if not ents:
            return None
        user, yes_if_open = R.choice([("Zijn alle ramen dicht?", False),
                                      ("Staat er nog een raam open?", True),
                                      ("Welke ramen staan open?", None),
                                      ("Zijn de ramen gesloten?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "Nee, alle ramen zijn dicht.",
                   False: "Ja, alle ramen zijn dicht.",
                   None: "Alle ramen zijn dicht."}[yes_if_open]
            must = ["dicht"]
        else:
            names = [e.name for e in opn]
            ans = ("Dit raam staat open: " + names[0] + "." if len(names) == 1 else
                   "Deze ramen staan open: " + ", ".join(names[:-1]) + " en " + names[-1] + ".")
            if yes_if_open is not None:
                ans = ("Ja. " if yes_if_open else "Nee. ") + ans
            must = names
        ids = ents
    elif kind == "door":
        # "Staat de balkondeur in de woonkamer open?" -> that contact, not the front door
        doors = [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]
        e = R.choice(doors)
        same = [d for d in doors if d.label == e.label]
        if e.area and (len(same) > 1 or R.random() < 0.5):
            what = f"de {low(e.label)} {ROOM_BY_NAME[e.area][1]}"
        elif len(same) == 1:
            what = f"de {low(e.label)}"
        else:
            return None
        if e.state == "unavailable":
            return None
        opn = e.state == "on"
        user, ask_open = R.choice([(f"Is {what} open?", True), (f"Is {what} dicht?", False),
                                   (f"Is {what} gesloten?", False),
                                   (f"Staat {what} open?", True)])
        word = "open" if opn else "dicht"
        ans = f"{'Ja' if opn == ask_open else 'Nee'}, {e.name} is {word}."
        ids, must = [e], [word]
    elif kind == "car":
        rng = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"])
        car = rng.name.rsplit(" ", 1)[0]
        art = next(a for c, a, _ in CARS if c == car)
        soc = next((e for e in h.of("sensor") if e.name == f"{car} Laadniveau"), None)
        if soc is not None and R.random() < 0.45:
            e = soc
            user = R.choice([f"Wat is het laadniveau van {art} {car}?",
                             f"Hoe vol is {art} {car}?",
                             f"Is {art} {car} al opgeladen?"])
            ans = f"{cap(art)} {car} is voor {e.state} procent opgeladen."
        else:
            e = rng
            user = R.choice([f"Hoe ver komt {art} {car} nog?",
                             f"Hoeveel bereik heeft {art} {car} nog?",
                             f"Wat is het bereik van {art} {car}?"])
            ans = f"{cap(art)} {car} kan nog {e.state} kilometer rijden."
        ids, must = [e], [e.state]
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = f"{ART[e.gender]} {low(e.name)}"
        user = R.choice([f"Is {nom} op slot?", f"Zit {nom} op slot?",
                         f"Heb ik {nom} op slot gedaan?"])
        word = _state_word(e)
        ans = (f"Ja, {nom} is op slot." if e.state == "locked"
               else f"Nee, {nom} is niet op slot.")
        ids, must = [e], [word]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        if e.state == "unavailable":
            return None   # unreachable devices: see _error_question
        user = R.choice(["Staat het alarm aan?", "Is het alarm ingeschakeld?",
                         "Wat is de status van het alarm?"])
        txt = {"disarmed": "Het alarmsysteem is uitgeschakeld.",
               "armed_away": "Het alarmsysteem is ingeschakeld (afwezig).",
               "armed_home": "Het alarmsysteem is ingeschakeld in de thuismodus.",
               "armed_night": "Het alarmsysteem is ingeschakeld in de nachtmodus."}[e.state]
        ans, ids, must = txt, [e], ["uitgeschakeld" if e.state == "disarmed" else "ingeschakeld"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"Wat speelt er nu op {e.name}?", f"Wat speelt {e.name}?"])
        t = e.attrs.get("media_title", "")
        a = e.attrs.get("media_artist")
        ans = f"Op {e.name} speelt nu {t}" + (f" van {a}." if a else ".")
        ids, must = [e], [t]
    else:
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [e.eid for e in ids]}
    if q_area:
        args = {"domain": "light", "area": q_area, **args}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of(ids)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": must}


def s_appliance_question(h, loc):
    """'Is de droger klaar?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = f"{ART[g]} {low(app)}"
    user = R.choice([f"Is {nom} klaar?", f"Draait {nom} nog?", f"Wat doet {nom}?",
                     f"Is {nom} al klaar?", f"Wat is de status van {nom}?"])
    phrase, key = APP_ANSWER[kind]
    verb, rest = phrase.split(" ", 1)
    ans = R.choice([f"{cap(nom)} {phrase}.", f"Volgens Home Assistant {verb} {nom} {rest}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [key]}


def s_relative_climate(h, loc):
    """'Maak het in de badkamer wat warmer' -> get_state -> set_temperature target +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    prep = ROOM_BY_NAME[e.area][1]
    up = R.random() < 0.6
    user = R.choice([f"Maak het {prep} wat warmer", f"Ik heb het koud {prep}",
                     f"Verwarming {prep} een graad hoger"] if up else
                    [f"Maak het {prep} wat koeler", f"Het is te warm {prep}",
                     f"Verwarming {prep} een graad lager"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, ok_result()),
             {"role": "assistant",
              "content": f"Ik heb de verwarming {prep} van {nl_num(e.attrs['temperature'])} "
                         f"naar {nl_num(new)} graden gezet."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


MISSING_TYPES = [
    # domain, phrase, "geen ...", relative pronoun, verb on, verb off,
    # "wel ..." (the kind exists elsewhere), services (on, off), state words
    # (on, off), label filter
    ("light", "het licht", "geen licht", "dat", "aan", "uit", "wel licht",
     ("light.turn_on", "light.turn_off"), ("aan", "uit"), None),
    ("light", "de lamp", "geen lamp", "die", "aan", "uit", "wel een lamp",
     ("light.turn_on", "light.turn_off"), ("aan", "uit"), None),
    ("cover", "de rolluiken", "geen rolluiken", "die", "omhoog", "omlaag", "wel rolluiken",
     ("cover.open_cover", "cover.close_cover"), ("open", "dicht"), "plain"),
    ("climate", "de verwarming", "geen verwarming", "die", "aan", "uit", "wel een verwarming",
     ("climate.turn_on", "climate.turn_off"), ("aan", "uit"), None),
    ("media_player", "de tv", "geen tv", "die", "aan", "uit",
     "wel een tv", ("media_player.turn_on", "media_player.turn_off"), ("aan", "uit"),
     "TV"),
    ("fan", "de ventilator", "geen ventilator", "die", "aan", "uit", "wel een ventilator",
     ("fan.turn_on", "fan.turn_off"), ("aan", "uit"), None),
]


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    mt = FORCE.get("mtype")
    dom, phrase, geen, rel, on, off, wel, svcs, words, filt = R.choice(
        [m for m in MISSING_TYPES if m[0] == mt] if mt not in (None, "other") else
        [m for m in MISSING_TYPES if m[0] != "media_player"] if mt == "other" else MISSING_TYPES)

    def match(e):
        if filt == "plain":
            return e.label not in NO_GROUP_COVERS
        return filt is None or e.label == filt

    ents = [e for e in h.of(dom) if match(e)]
    rooms = [r for r in h.rooms if not any(e.area == r for e in ents)]
    if not rooms:
        return None
    room = R.choice(rooms)
    prep = ROOM_BY_NAME[room][1]
    # the room often holds a same-domain device of another kind (speaker, voice
    # satellite); it is no TV and must not stand in for the room in the answer
    if dom == "media_player" and R.random() < 0.6 and \
            not any(e.area == room for e in h.of(dom)):
        if R.random() < 0.6:
            h.add_voice_media(room)
        else:
            mt = R.choice([t for t in TYPES["media_player"] if t[0] != "TV"])
            h.add("media_player", room, mt[0], mt[1], mt[2], state=R.choice(["idle", "off"]))
    turn_on = R.random() < 0.5
    verb = on if turn_on else off
    noun = phrase.split()[1]
    zijn, staan = ("zijn", "staan") if dom == "cover" else ("is", "staat")
    user = R.choice([f"Doe {phrase} {prep} {verb}",
                     f"Zet {phrase} {prep} {verb}",
                     f"{cap(noun)} {prep} {verb}", f"Doe {prep} {phrase} {verb}"])
    others = [e for e in ents if e.area and e.area != room]
    # suggest the other device only if there are one or two candidates
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        ans = R.choice([f"{cap(prep)} {zijn} {geen}. {cap(oprep)} {zijn} {wel}: {alt.name}.",
                        f"{cap(prep)} vind ik {geen}. Bedoel je {alt.name}{where}?",
                        f"{cap(prep)} {staan} {geen} in de apparatenlijst. {cap(oprep)} wel: {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"{cap(prep)} vind ik {geen} {rel} ik kan bedienen.",
                        f"{cap(prep)} {staan} {geen} in de apparatenlijst.",
                        f"Ik heb {prep} {geen} gevonden {rel} ik kan schakelen."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        # user accepts the device in the other room -> now it is switched
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        that = REL[alt.gender]
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Dan {that} {oprep}", f"Ja, {that} bedoel ik", f"Oké, dan {that}",
             f"Ja, {alt.name}", f"Dan maar {oprep}"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        word = words[0] if turn_on else words[1]
        msgs += [call, tool_msg(cid, ok_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"{alt.name}{where} {is_(alt.plural)} nu {word}.",
                      f"Gedaan, {alt.name}{where} {is_(alt.plural)} {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        # user insists on the room, sometimes twice -> still no invented action
        pushes = ["Jawel, doe het aan!", "Ja, zet het aan!", "Nee joh, dat is er wel",
                  "Doe het toch maar", f"Jawel, {prep}!", f"Nee, ik bedoel {prep}",
                  "Is dat niet zo?", "Jawel, daar is er een", "Doe het gewoon"]
        refusals = [
            f"{cap(prep)} {staan} {geen} in je apparatenlijst. Controleer of het apparaat "
            f"beschikbaar is gemaakt voor Assist.",
            f"Ik zie {prep} echt {geen} {rel} ik kan bedienen.",
            f"{cap(prep)} {staan} {geen} in mijn apparatenlijst, daarom kan ik daar niets schakelen."]
        p1, p2 = R.sample(pushes, 2)
        a1, a2 = R.sample(refusals, 2)
        msgs += [{"role": "user", "content": voice(p1)}, {"role": "assistant", "content": a1}]
        if R.random() < 0.45:
            msgs += [{"role": "user", "content": voice(p2)}, {"role": "assistant", "content": a2}]
        cat = "unknown"
    else:
        cat = "unknown"
    meta = {"cat": cat, "gold": gold}
    if cat == "followup":
        meta["gold_last"] = gold
    return msgs, meta


def s_unknown(h, loc):
    """User asks for a device or room that does not exist -> no tool call."""
    forced = FORCE.get("unknown")
    if not forced and R.random() < 0.45:
        return s_missing_type(h, loc)
    r = R.random()
    if forced:
        r = 0.0 if forced == "device" else 0.9
    if r < 0.5:
        dev, g = R.choice(UNKNOWN_DEVICES)
        labels = " ".join(f"{e.name} {e.label} {e.hidden}" for e in h.ents).lower()
        if dev.lower() in labels or (dev == "Grasmaaier" and h.of("lawn_mower")):
            return None
        acc = f"{ART[g]} {dev.lower()}"
        user = R.choice([f"Zet {acc} aan", f"Doe {acc} aan", f"Zet {acc} uit",
                         f"Is {acc} aan?", f"Start {acc}"])
        ans = R.choice([f"Een apparaat met de naam {dev} vind ik niet in je slimme huis.",
                        f"Ik ken geen apparaat dat {dev} heet. Dat kan ik helaas niet bedienen.",
                        f"{dev} staat niet in de apparatenlijst."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != "Doucheruimte"
                   and not (r_[0] == "Badkamer" and "Doucheruimte" in h.rooms)]
        room = R.choice(missing)
        prep = ROOM_BY_NAME[room][1]
        what = R.choice([("het licht", "geen licht", "dat", "aan", "uit"),
                         ("de rolluiken", "geen rolluiken", "die", "omhoog", "dicht"),
                         ("de verwarming", "geen verwarming", "die", "aan", "uit")])
        user = R.choice([f"Doe {what[0]} {prep} {what[3]}",
                         f"Doe {what[0]} {prep} {what[4]}",
                         f"{cap(what[0].split()[1])} {prep} {what[4]}"])
        ans = R.choice([f"{cap(prep)} vind ik {what[1]} {what[2]} ik kan bedienen.",
                        f"Ik heb {prep} {what[1]} gevonden in de apparatenlijst."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


WEATHER_CHAT = "Wat voor weer wordt het morgen?"
CHAT = [
    (["Bedankt", "Dank je", "Dank je wel", "Top, bedankt"],
     ["Graag gedaan.", "Geen dank.", "Geen probleem."]),
    (["Hallo", "Hoi", "Goedemorgen", "Goedenavond"],
     ["Hallo, waarmee kan ik je helpen?", "Hoi. Wat kan ik voor je doen?"]),
    (["Wat kun je?", "Waarmee kun je me helpen?", "Wat kun je allemaal bedienen?"],
     ["Ik kan lampen, rolluiken, de verwarming en andere apparaten in je huis bedienen "
      "en je vertellen in welke toestand ze zijn of wat de sensoren meten. "
      "Daarnaast speel ik muziek, zet ik timers, houd ik je lijsten bij, vertel ik je "
      "hoe laat het is en wat voor weer het wordt en doe ik omroepberichten.",
      "Ik bedien je apparaten, speel muziek, zet timers, zet dingen op je "
      "boodschappenlijst, vertel je de tijd en het weer en doe omroepberichten in huis."]),
    (["Wie ben jij?"], ["Ik ben je spraakassistent voor je slimme huis."]),
    (["Vertel eens een mop"],
     ["Wat zegt de ene lamp tegen de andere? Zullen we samen uitgaan?",
      "Waarom is het rolluik nooit gestrest? Het laat alles gewoon langzaam zakken."]),
    ([WEATHER_CHAT],
     ["Een weersverwachting heb ik helaas niet. Ik kan je wel vertellen wat je "
      "sensoren nu meten."]),
    (["Welterusten"], ["Welterusten. Zal ik nog iets uitzetten?", "Welterusten."]),
    # HassNevermind: nothing to do
    (["Laat maar", "Annuleren", "Laat maar zitten", "Niks", "Maakt niet uit", "Ach, niks",
      "Toch niet", "Is al geregeld", "Vergeet het maar", "Nee, niks",
      "Ik zei niks", "Is goed zo", "Nee, laat maar"],
     ["Oké.", "Prima.", "Oké, dan niet."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == WEATHER_CHAT and (
            h.of("weather") or any(e.label == "Verwachting" for e in h.of("sensor"))):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


PLURAL_WORDS = {"lampen", "lichten", "rolluiken", "rolgordijnen", "jaloezieën", "spots",
                "screens", "verwarmingen", "thermostaten", "stopcontacten"}


def is_plural_name(name: str) -> bool:
    """'Alle lampen', 'Spots Woonkamer', 'Rolluiken Eetkamer' take a plural verb."""
    words = name.lower().split()
    return bool(words) and (words[0] == "alle" or any(w in PLURAL_WORDS for w in words))


def ist(name: str) -> str:
    return "zijn" if is_plural_name(name) else "is"


def reageert(name: str) -> str:
    return "reageren" if is_plural_name(name) else "reageert"


def _unreachable(name):
    return R.choice([f"Dat is helaas niet gelukt. {name} {reageert(name)} nu niet.",
                     f"Er is helaas een fout opgetreden: {name} {ist(name)} niet bereikbaar.",
                     f"{name} {ist(name)} nu niet bereikbaar, daarom kon ik de opdracht niet uitvoeren."])


def _single_command(h, loc):
    """A command intent that targets exactly one entity."""
    for _ in range(10):
        out = pick_command(h, loc)
        if out and len(out[1]) == 1 and len(out[1][0]["entity_id"]) == 1:
            return out
    return None


def s_error(h, loc):
    """The device does not respond: the answer reports it, never success."""
    r = R.random()
    if r < 0.3:
        return _error_partial(h, loc)
    if r < 0.45:
        return _error_question(h, loc)
    if R.random() < 0.35:
        out = _error_room(h, loc)
        if out:
            return out
    return s_error_single(h, loc, r)


def s_error_single(h, loc, r=None):
    """One device does not respond; sometimes a retry or a doubt follows."""
    if r is None:
        r = R.uniform(0.45, 1.0)
    out = _single_command(h, loc) if r < 0.65 else pick_command(h, loc)
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
             {"role": "assistant", "content": _unreachable(name)}]
    meta = {"cat": "error", "gold": [["control", {"actions": actions}]]}
    if r >= 0.65 or len(actions) != 1:
        return msgs, meta
    # second turn after the failure: retry or doubt, the device is still down
    if R.random() < 0.5:
        msgs.append({"role": "user", "content": voice(R.choice(
            ["Probeer het nog eens", "Nog een keer graag", "Probeer het opnieuw",
             "Doe het nog eens"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": R.choice([
                     f"{name} {ist(name)} nog steeds niet bereikbaar. Controleer of het apparaat stroom en verbinding heeft.",
                     f"Ook de tweede poging is niet gelukt, {name} {reageert(name)} niet."])}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        msgs.append({"role": "user", "content": voice(R.choice(
            ["Is het nu aan?", "Nee joh, toch niet?", "Is het gelukt?", "En, is het aan?"]
            if onoff else ["Is het gelukt?", "En, heeft het gewerkt?", "Klopt dat?"]))})
        e = next(x for x in h.ents if x.eid == eid)
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": R.choice([
                     f"Nee. {name} {ist(name)} niet bereikbaar, daarom kon ik het niet schakelen.",
                     f"Volgens Home Assistant {ist(name)} {name} nu niet bereikbaar. Er is niets geschakeld."])}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def _error_room(h, loc):
    """Some lights of a room are unreachable, the others work. Questions and
    commands about the room report both groups by name."""
    rooms = [r for r in h.rooms if len([e for e in h.of("light", r)
                                         if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    # rooms with a light named like the room are the hard case ("Keuken" is
    # down, "Keuken Links" is not), so they come up more often
    named_rooms = [r for r in rooms if any(e.name == r for e in h.of("light", r))]
    # forced variant: question about a room whose light named like the room is
    # down next to others; the model tended to drop the room-named light
    named_q = FORCE.get("eroom") == "named_q"
    if named_q and not named_rooms:
        return None
    room = R.choice(named_rooms) if named_rooms and (named_q or R.random() < 0.6) else R.choice(rooms)
    ents = [e for e in h.of("light", room) if e.label != "__group__"]
    k = len(ents) if R.random() < 0.15 else R.randint(1, len(ents) - 1)
    down = R.sample(ents, k)
    named = [e for e in ents if e.name == room]
    if named and named[0] not in down and (named_q or R.random() < 0.6):
        down[0] = named[0]
    for e in down:
        e.state, e.attrs = "unavailable", {}
    up = [e for e in ents if e not in down]
    prep = ROOM_BY_NAME[room][1]
    ids = [e.eid for e in ents]
    nd = _names(h, [e.eid for e in down])
    down_txt = f"{nd} {is_(len(down) > 1)} nu niet bereikbaar."
    msgs = base_msgs(h, loc)
    if named_q or R.random() < 0.5:
        # question
        user, kind = R.choice([(f"Is het licht {prep} aan?", "yn"),
                               (f"Brandt er {prep} nog licht?", "yn"),
                               (f"Welke lampen zijn er {prep} aan?", "which"),
                               (f"Zijn de lampen {prep} uit?", "yn_off")])
        on = [e for e in up if e.state == "on"]
        off = [e for e in up if e.state != "on"]
        parts = []
        if on:
            parts.append(f"{_names(h, [e.eid for e in on])} {is_(len(on) > 1)} aan.")
        if off and (not on or R.random() < 0.5):
            parts.append(f"{_names(h, [e.eid for e in off])} {is_(len(off) > 1)} uit.")
        if not up:
            ans = f"{cap(down_txt)} Of daar licht brandt, kan ik daarom niet zeggen."
        else:
            if kind == "yn":
                parts[0] = ("Ja, " if on else "Nee, ") + parts[0]
            elif kind == "yn_off":
                # same wording as the "aan?" question, only the yes/no flips;
                # a separate phrase made the model mix up both questions
                parts[0] = ("Nee, " if on else "Ja, ") + parts[0]
            ans = " ".join(parts) + " " + down_txt
        args = {"domain": "light", "area": room, "entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "error", "gold": [["get_state", args]]}
    # command for the whole room: the reachable lights switch
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    user = R.choice([f"Doe het licht {prep} aan", f"Licht {prep} aan",
                     f"Zet het licht {prep} aan"] if on_cmd else
                    [f"Doe het licht {prep} uit", f"Licht {prep} uit",
                     f"Zet het licht {prep} uit"])
    actions = [action(svc, ents, area=room)]
    done = [e.eid for e in up]
    word = "aangezet" if on_cmd else "uitgezet"
    if up:
        did = _names(h, done)
        ans = R.choice([
            f"Ik heb {did} {word}. {down_txt}",
            f"Dat is maar deels gelukt: {did} {is_(len(up) > 1)} {'aan' if on_cmd else 'uit'}, maar {nd} {is_(len(down) > 1)} nu niet bereikbaar.",
            f"{down_txt} De rest heb ik {word}: {did}.",
        ])
    else:
        ans = f"{down_txt} Daarom kon ik het licht {prep} niet schakelen."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, S.control_result(
                 [f"Entity {', '.join(e.eid for e in down)} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _named_room_light(h, loc):
    """A light carries the room's name ("Keuken") next to other lights there.
    "het licht in de keuken" means all lights of the room; only the bare name
    ("Doe Keuken aan") means that single lamp."""
    rooms = [r for r in h.rooms
             if any(e.name == r for e in h.of("light", r))
             and len([e for e in h.of("light", r) if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = [e for e in h.of("light", room) if e.label != "__group__"]
    named = next(e for e in ents if e.name == room)
    prep = ROOM_BY_NAME[room][1]
    msgs = base_msgs(h, loc)
    r = R.random()
    if r < 0.3:
        user, kind = R.choice([(f"Is het licht {prep} aan?", "yn"),
                               (f"Brandt er {prep} licht?", "yn"),
                               (f"Is er {prep} nog licht aan?", "yn"),
                               (f"Welke lampen zijn er {prep} aan?", "which")])
        on = [e for e in ents if e.state == "on"]
        off = [e for e in ents if e.state != "on"]
        if on:
            ans = f"{_names(h, [e.eid for e in on])} {is_(len(on) > 1)} aan."
            if off and R.random() < 0.6:
                ans += f" {_names(h, [e.eid for e in off])} {is_(len(off) > 1)} uit."
            if kind == "yn":
                ans = "Ja, " + ans
        elif kind == "yn":
            ans = R.choice([f"Nee, {_names(h, [e.eid for e in off])} {is_(len(off) > 1)} uit.",
                            f"Nee, {prep} is geen licht aan."])
        else:
            ans = f"{cap(prep)} is nu geen licht aan."
        args = {"domain": "light", "area": room, "entity_ids": [e.eid for e in ents]}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        must = [e.name for e in on] if on else ["nee" if kind == "yn" else "geen"]
        return msgs, {"cat": "question", "gold": [["get_state", args]],
                      "answer_contains": must}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    adj, part = ("aan", "aangezet") if on_cmd else ("uit", "uitgezet")
    if r < 0.8:
        # the room: every light in it
        user = R.choice([f"Doe het licht {prep} aan", f"Zet het licht {prep} aan",
                         f"Licht {prep} aan", f"Doe {prep} het licht aan",
                         f"Kun je het licht {prep} aandoen?"] if on_cmd else
                        [f"Doe het licht {prep} uit", f"Zet het licht {prep} uit",
                         f"Licht {prep} uit", f"Doe {prep} het licht uit",
                         f"Kun je het licht {prep} uitdoen?"])
        targets = Ref(f"het licht {prep}", f"het licht {prep}", False, ents, room)
        conf = confirm(targets, adj, part)
    else:
        # the bare name: only the lamp called like the room
        user = R.choice([f"Doe {room} aan", f"Zet {room} aan", f"{room} aan",
                         f"Zet de lamp {room} aan"] if on_cmd else
                        [f"Doe {room} uit", f"Zet {room} uit", f"{room} uit",
                         f"Zet de lamp {room} uit"])
        targets = [named]
        conf = confirm(Ref(room, room, False, targets), adj, part)
    actions = [action(svc, targets)]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, ok_result()), {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_no_areas(h, loc):
    """Home Assistant without any areas: a room cannot be resolved. The model
    says so and explains how to set up areas; device names and "alle
    lampen" keep working."""
    lights = h.of("light")
    rooms = [r for r in h.rooms if h.of("light", r) or h.of("cover", r, plain=True)]
    if len(lights) < 2 or not rooms:
        return None
    room = R.choice(rooms)
    prep = ROOM_BY_NAME[room][1]
    kind = "light" if h.of("light", room) and (not h.of("cover", room, plain=True)
                                              or R.random() < 0.7) else "cover"
    named = [e for e in h.of(kind, room, plain=kind == "cover")
             if room.lower() in e.name.lower()]
    for e in h.ents:
        e.area = None
    msgs = base_msgs(h, None)
    r = R.random()
    if r < 0.6:
        if kind == "light":
            user = R.choice([f"Doe het licht {prep} aan", f"Licht {prep} uit",
                             f"Is het licht {prep} aan?", f"Welke lampen zijn er {prep} aan?",
                             f"Zet het licht {prep} aan", f"Doe {prep} het licht uit"])
            what = "lampen"
        else:
            user = R.choice([f"Doe de rolluiken {prep} omlaag", f"Rolluiken {prep} omhoog",
                             f"Zijn de rolluiken {prep} dicht?", f"Sluit de rolluiken {prep}"])
            what = "rolluiken"
        ans = R.choice([
            f"In je Home Assistant zijn geen ruimtes ingesteld, daarom weet ik niet welke "
            f"{what} {prep} zijn. Maak in de instellingen onder “Ruimtes, labels en zones” "
            f"de ruimtes aan en wijs de apparaten toe.",
            f"Ik kan {prep} niets toewijzen, omdat er in Home Assistant geen ruimtes zijn "
            f"aangemaakt. Stel in de instellingen onder “Ruimtes, labels en zones” de "
            f"ruimtes in en wijs de apparaten toe."])
        if named:
            ans += f" Tot die tijd kun je een apparaat ook bij naam noemen, bijvoorbeeld “{named[0].name}”."
        msgs += [{"role": "user", "content": voice(user)}, {"role": "assistant", "content": ans}]
        return msgs, {"cat": "unknown", "gold": [], "no_history": True}
    on = R.random() < 0.5
    svc = "light.turn_on" if on else "light.turn_off"
    adj, part = ("aan", "aangezet") if on else ("uit", "uitgezet")
    if r < 0.85 or h.group("light") is not None:
        # a device named directly still works
        e = R.choice(lights)
        user = R.choice([f"Zet {e.name} aan", f"Doe {e.name} aan", f"{e.name} aan"] if on
                        else [f"Zet {e.name} uit", f"Doe {e.name} uit", f"{e.name} uit"])
        actions = [action(svc, [e])]
        conf = R.choice([f"{e.name} is nu {adj}.", f"Ik heb {e.name} {part}."])
    else:
        # the whole house needs no areas
        user = R.choice(["Doe alle lampen aan", "Zet alle lichten aan"] if on
                        else ["Doe alle lampen uit", "Overal het licht uit"])
        actions = [action(svc, lights, area=S.ALL_AREAS)]
        conf = f"Alle lampen zijn {adj}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]],
                  "no_history": True}


def partial_answer(h, ok_actions, failed_name, conf=None):
    """Both outcomes, always: what worked and what did not."""
    if any(len(a["entity_id"]) > 3 for a in ok_actions):
        # long lists: the normal confirmation ("Alle lampen zijn aan.") is clearer
        if conf and conf != "Gedaan.":
            return R.choice([f"{conf} Maar {failed_name} {ist(failed_name)} nu niet bereikbaar.",
                             f"{failed_name} {ist(failed_name)} nu niet bereikbaar. Verder geldt: {conf}"])
        n = sum(len(a["entity_id"]) for a in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = f"{n} apparaten {PARTICIPLE.get(svc, 'geschakeld')}"
    else:
        did = describe_all(h, ok_actions)
    v = ist(failed_name)
    return R.choice([
        f"Ik heb {did}. Maar {failed_name} {v} nu niet bereikbaar.",
        f"Dat is maar deels gelukt: ik heb {did}, maar {failed_name} {v} nu niet bereikbaar.",
        f"{failed_name} {v} nu niet bereikbaar. De rest is gelukt: ik heb {did}.",
    ])


def _error_partial(h, loc):
    """Two commands, one device is down: report both outcomes."""
    a = _single_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    eid = a[1][0]["entity_id"][0]
    name = next(e.name for e in h.ents if e.eid == eid)
    actions = a[1] + b[1]
    done = [i for x in b[1] for i in x["entity_id"]]
    ans = partial_answer(h, b[1], name, b[2])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, S.control_result([f"Entity {eid} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _error_question(h, loc):
    """State question about a device that is unavailable."""
    ents = [e for e in h.ents if e.domain in ("light", "switch", "cover", "lock",
                                               "media_player", "climate", "fan")
            and e.label != "__group__"]
    if not ents:
        return None
    e = R.choice(ents)
    user = R.choice([f"Is {e.name} aan?", f"Wat is de status van {e.name}?",
                     f"Wat doet {e.name}?", f"Is {e.name} ingeschakeld?"])
    args = {"entity_ids": [e.eid]}
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": R.choice([
                 f"{e.name} {ist(e.name)} nu niet bereikbaar, daarom weet ik niet in welke toestand het apparaat is.",
                 f"Dat kan ik niet zeggen: {e.name} {ist(e.name)} nu niet bereikbaar."])}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


def s_followup(h, loc):
    """Two-turn: command, then 'en weer uit' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    p1, p2 = ROOM_BY_NAME[r1][1], ROOM_BY_NAME[r2][1]
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Doe het licht {p1} aan",
                                                            f"Licht {p1} aan"]))}]
    a1 = [action("light.turn_on", e1, area=r1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": f"Het licht {p1} is aan."}]
    k = R.random()
    if k < 0.15:
        # user doubts the result -> check the real state instead of guessing
        u2 = R.choice(["Nee hoor, het is niet aan", "Is het echt aan?", "Klopt dat?",
                       "Bij mij is het nog donker"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        q = {"domain": "light", "area": r1, "entity_ids": [e.eid for e in e1]}
        c2, cid2 = tool_call("get_state", q)
        ans = (f"Volgens Home Assistant is het licht {p1} aan." if really_on else
               f"Volgens Home Assistant is het licht {p1} nog uit. Zal ik het opnieuw aanzetten?")
        if u2 == "Is het echt aan?":
            ans = ("Ja. " if really_on else "Nee. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", q]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["En weer uit", "Doe het weer uit", "Toch maar weer uit"]), \
            [action("light.turn_off", e1, area=r1)], f"Het licht {p1} is weer uit."
    elif k < 0.72:
        u2, a2, ans = R.choice([f"En {p2}?", f"{cap(p2)} ook", f"En {p2} ook graag"]), \
            [action("light.turn_on", e2, area=r2)], f"Het licht {p2} is ook aan."
    else:
        kind, u2 = R.choice([("down", "Iets minder fel"), ("up", "Een beetje feller"),
                             ("pct", "Op 30 procent")])
        if kind == "down":
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20}, area=r1)]
            ans = f"Het licht {p1} is nu gedimd."
        elif kind == "up":
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20}, area=r1)]
            ans = f"Het licht {p1} is nu feller."
        else:
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30}, area=r1)]
            ans = f"Het licht {p1} staat nu op 30 procent."
    msgs.append({"role": "user", "content": voice(u2)})
    c2, cid2 = tool_call("control", {"actions": a2})
    msgs += [c2, tool_msg(cid2, ok_result()), {"role": "assistant", "content": ans}]
    return msgs, {"cat": "followup", "gold_last": [["control", {"actions": a2}]],
                  "gold": [["control", {"actions": a1}]]}


# --------------------------------------------------------------------------
# multi-turn conversations: later turns refer to earlier ones
# --------------------------------------------------------------------------
INVERSE = {"turn_on": "turn_off", "turn_off": "turn_on", "open_cover": "close_cover",
           "close_cover": "open_cover", "lock": "unlock", "unlock": "lock",
           "media_play": "media_pause", "media_pause": "media_play",
           "open_valve": "close_valve", "close_valve": "open_valve",
           "start": "return_to_base", "start_mowing": "dock"}
# Dutch puts the participle at the end: "Ik heb de lamp aangezet"
PARTICIPLE = {"turn_on": "aangezet", "turn_off": "uitgezet",
              "open_cover": "geopend", "close_cover": "gesloten",
              "stop_cover": "stilgezet", "lock": "op slot gedaan",
              "unlock": "van het slot gehaald",
              "media_play": "hervat", "media_pause": "gepauzeerd", "media_stop": "gestopt",
              "media_next_track": "naar het volgende nummer gezet",
              "media_previous_track": "naar het vorige nummer gezet",
              "volume_up": "harder gezet", "volume_down": "zachter gezet",
              "start": "gestart", "pause": "gepauzeerd",
              "return_to_base": "naar het laadstation gestuurd",
              "start_mowing": "gestart", "dock": "naar het laadstation gestuurd",
              "open_valve": "geopend", "close_valve": "gesloten", "press": "geactiveerd",
              "alarm_arm_home": "ingeschakeld", "alarm_arm_away": "ingeschakeld",
              "alarm_arm_night": "in de nachtmodus ingeschakeld",
              "alarm_disarm": "uitgeschakeld"}
DOMAIN_NOUN = {"light": ("Welke lamp", "Welke lampen"),
               "cover": ("Welk rolluik", "Welke rolluiken"),
               "climate": ("Welke verwarming", "Welke verwarmingen"),
               "switch": ("Welk apparaat", "Welke apparaten"),
               "media_player": ("Welke speaker", "Welke apparaten"),
               "fan": ("Welke ventilator", "Welke ventilatoren")}


def apply_actions(h, actions):
    """Change the simulated states, so later questions see the result."""
    by_id = {e.eid: e for e in h.ents}
    for a in actions:
        dom, svc = a["service"].split(".", 1)
        data = a.get("data") or {}
        for eid in a["entity_id"]:
            e = by_id.get(eid)
            if e is None or e.state == "unavailable":
                continue
            if svc == "turn_on":
                e.state = "on" if dom not in ("media_player", "climate") else (
                    "idle" if dom == "media_player" else "heat")
                if dom == "light":
                    b = e.attrs.get("brightness", 180)
                    if "brightness_pct" in data:
                        b = round(data["brightness_pct"] * 2.55)
                    elif "brightness_step_pct" in data:
                        b = max(3, min(255, b + round(data["brightness_step_pct"] * 2.55)))
                    e.attrs["brightness"] = b
            elif svc == "turn_off":
                e.state = "off"
                e.attrs.pop("brightness", None)
            elif svc in ("open_cover", "close_cover", "set_cover_position"):
                pos = data.get("position", 100 if svc == "open_cover" else 0)
                e.state = "open" if pos > 0 else "closed"
                e.attrs["current_position"] = pos
            elif svc == "set_temperature":
                e.attrs["temperature"] = data.get("temperature")
            elif svc in ("lock", "unlock"):
                e.state = "locked" if svc == "lock" else "unlocked"
            elif svc in ("media_play", "media_pause", "media_stop"):
                e.state = {"media_play": "playing", "media_pause": "paused",
                           "media_stop": "idle"}[svc]
            elif svc in ("start", "start_mowing"):
                e.state = "cleaning" if dom == "vacuum" else "mowing"
            elif svc in ("return_to_base", "dock"):
                e.state = "returning"
            elif svc in ("open_valve", "close_valve"):
                e.state = "open" if svc == "open_valve" else "closed"
            elif svc == "set_percentage":
                e.state = "on"
                e.attrs["percentage"] = data.get("percentage")


def _names(h, ids):
    by_id = {e.eid: e.name for e in h.ents}
    ns = [by_id[i] for i in ids]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " en " + ns[-1]


def describe(h, a, by_name=False):
    """'Plafondlamp Gang aangezet' for one action (object + participle)."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if a.get("area") == S.ALL_AREAS and not by_name:
        who = "alle lampen"
    elif a.get("area") and not by_name:
        who = f"het licht {ROOM_BY_NAME[a['area']][1]}"
    else:
        who = _names(h, a["entity_id"])
    for hook in DESCRIBE_HOOKS:
        out = hook(h, a, who)
        if out is not None:
            return out
    if "brightness_pct" in data:
        return f"{who} op {data['brightness_pct']} procent gezet"
    if "brightness_step_pct" in data:
        return f"{who} {'feller gezet' if data['brightness_step_pct'] > 0 else 'gedimd'}"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"de kleur van {who} veranderd"
    if svc == "set_temperature":
        return f"{who} op {nl_num(data['temperature'])} graden gezet"
    if svc == "set_cover_position":
        return f"{who} op {data['position']} procent gezet"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"{who} op {data.get('percentage')} procent gezet"
    if svc == "volume_set":
        return f"het volume van {who} op {round(data['volume_level'] * 100)} procent gezet"
    if dom == "scene":
        return f"de scène {who} geactiveerd"
    if dom == "script":
        return f"{who} gestart"
    return f"{who} {PARTICIPLE.get(svc, 'geschakeld')}"


def merge_actions(actions):
    """Consecutive list items for the same list become one action, so the
    answer says 'melk en eieren op de boodschappenlijst gezet'."""
    out = []
    for a in actions:
        prev = out[-1] if out else None
        if prev and a["service"].startswith("todo.") and prev["service"] == a["service"] \
                and prev["entity_id"] == a["entity_id"] and "item" in (a.get("data") or {}):
            prev["data"]["items"] = prev["data"].get("items", [prev["data"]["item"]]) + [a["data"]["item"]]
            continue
        out.append({**a, "data": dict(a.get("data") or {})} if a.get("data") else dict(a))
    return out


def describe_all(h, actions):
    """'X aangezet, Y gezet en Z gesloten' for several actions."""
    parts = [describe(h, a) for a in merge_actions(actions)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " en " + parts[-1]


class Conv:
    """Builds one conversation turn by turn and tracks what really happened."""

    def __init__(self, h, loc):
        self.h, self.loc = h, loc
        self.msgs = base_msgs(h, loc)
        self.done: list[dict] = []        # successful actions, in order
        self.last: list[dict] | None = None
        self.gold_last: list = []

    def user(self, text):
        self.msgs.append({"role": "user", "content": voice(text)})

    def control(self, actions, answer, errors=None, done=None):
        call, cid = tool_call("control", {"actions": actions})
        result = S.control_result(errors, done) if errors else ok_result(actions)
        self.msgs += [call, tool_msg(cid, result),
                      {"role": "assistant", "content": answer}]
        ok = [a for a in actions if not errors or
              all(i in (done or []) for i in a["entity_id"])]
        apply_actions(self.h, ok)
        self.done += ok
        self.last = ok or None
        self.gold_last = [["control", {"actions": actions}]]

    def get_state(self, ents, answer, area=None):
        args = {"entity_ids": [e.eid for e in ents]}
        if area is not None:
            args = {"domain": "light", "area": area, **args}
        call, cid = tool_call("get_state", args)
        self.msgs += [call, tool_msg(cid, self.h.state_of(ents)),
                      {"role": "assistant", "content": answer}]
        self.gold_last = [["get_state", args]]

    def say(self, answer):
        self.msgs.append({"role": "assistant", "content": answer})
        self.gold_last = []


def turn_command(c):
    out = pick_command(c.h, c.loc)
    if not out:
        return False
    user, actions, conf = out
    c.user(user)
    c.control(actions, conf)
    return True


def turn_question(c):
    out = s_question(c.h, c.loc)
    if not out:
        return False
    msgs = out[0][1:]
    # s_question answers from the attributes; after a failure earlier in the
    # conversation the entity may be unreachable, which it does not handle
    if any(m["role"] == "tool" and '"state": "unavailable"' in m["content"] for m in msgs):
        return False
    c.msgs += msgs
    c.gold_last = out[1]["gold"]
    return True


def turn_chat(c):
    users, answers = R.choice(CHAT[:4])
    c.user(R.choice(users))
    c.say(R.choice(answers))
    return True


def turn_error(c):
    out = _single_command(c.h, c.loc)
    if not out:
        return False
    if R.random() < 0.4:
        # partial success inside a conversation: the done part counts as done
        b = pick_command(c.h, c.loc)
        user = join_commands(out, b) if b else None
        eid = out[1][0]["entity_id"][0]
        if user and eid not in [i for x in b[1] for i in x["entity_id"]]:
            e = next(x for x in c.h.ents if x.eid == eid)
            e.state = "unavailable"
            done = [i for x in b[1] for i in x["entity_id"]]
            c.user(user)
            c.control(out[1] + b[1], partial_answer(c.h, b[1], e.name, b[2]),
                      errors=[f"Entity {eid} is unavailable"], done=done)
            return True
    user, actions, _ = out
    eid = actions[0]["entity_id"][0]
    e = next(x for x in c.h.ents if x.eid == eid)
    e.state = "unavailable"
    c.user(user)
    c.control(actions, _unreachable(e.name), errors=[f"Entity {eid} is unavailable"])
    c.last = None
    return True


def turn_unknown(c):
    out = s_missing_type(c.h, c.loc) if R.random() < 0.5 else s_unknown(c.h, c.loc)
    if not out or len(out[0]) != 3:      # only the single-turn variant here
        return False
    c.msgs += out[0][1:]
    c.gold_last = []
    return True


def turn_which(c):
    """'Welke lamp heb je uitgezet?' -> answered from the history."""
    if not c.last:
        return False
    a = c.last[-1]
    dom, svc = a["service"].split(".", 1)
    ids = a["entity_id"]
    if dom in DOMAIN_NOUN and svc in PARTICIPLE and not a.get("data"):
        sg, pl = DOMAIN_NOUN[dom]
        q = (pl if len(ids) > 1 else sg) + f" heb je {PARTICIPLE[svc]}?"
        user = R.choice([q, q, "Wat heb je net gedaan?", "Welk apparaat was dat?"])
    else:
        user = R.choice(["Wat heb je net gedaan?", "Wat heb je precies gedaan?"])
    c.user(user)
    # the lamps an area stood for are known from "done" in the tool result
    parts = [describe(c.h, x, by_name=True) for x in c.last]
    c.say(R.choice(["Ik heb ", "Net heb ik "]) + " en ".join(parts) + ".")
    return True


def turn_summary(c):
    """'Wat heb je allemaal gedaan?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    # "Welke verwarmingen heb je aangepast?" after the bathroom and "En in de
    # keuken ook": every action on that device type, not only the last one
    by_dom = {}
    for x in c.done:
        by_dom.setdefault(x["service"].split(".")[0], []).append(x)
    multi = [d for d, xs in by_dom.items() if len(xs) >= 2 and d in SUMMARY_ASK]
    if multi and R.random() < 0.6:
        d = R.choice(multi)
        c.user(R.choice(SUMMARY_ASK[d]))
        parts = [describe(c.h, x) for x in by_dom[d]]
    else:
        c.user(R.choice(["Wat heb je allemaal gedaan?", "Wat heb je tot nu toe geschakeld?",
                         "Vat eens samen wat je gedaan hebt"]))
        parts = [describe(c.h, x) for x in c.done]
    c.say("Ik heb " + ", ".join(parts[:-1]) + " en " + parts[-1] + ".")
    return True


SUMMARY_ASK = {
    "climate": ["Welke verwarmingen heb je aangepast?", "Welke thermostaten heb je veranderd?",
                "Wat heb je aan de verwarming veranderd?", "Welke verwarmingen heb je ingesteld?"],
    "light": ["Welke lampen heb je geschakeld?", "Welke lichten heb je geschakeld?",
              "Wat heb je met de lampen gedaan?"],
    "cover": ["Welke rolluiken heb je bewogen?", "Welke rolgordijnen heb je bediend?",
              "Wat heb je met de rolluiken gedaan?"],
}


def turn_undo(c):
    if not c.last:
        return False
    inv = []
    for a in c.last:
        dom, svc = a["service"].split(".", 1)
        if svc not in INVERSE or a.get("data"):
            return False
        inv.append({"service": f"{dom}.{INVERSE[svc]}",
                    **({"area": a["area"]} if "area" in a else {}),
                    "entity_id": a["entity_id"]})
    c.user(R.choice(["Maak dat ongedaan", "Toch niet", "En weer terug",
                     "Draai dat terug"] +
                    (["En weer uit", "Doe het weer uit"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    (["En weer aan"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "Oké, ik heb " + describe_all(c.h, inv) + ".")
    return True


def turn_correct(c):
    """'Nee, ik bedoelde in de slaapkamer' -> undo + the right device."""
    if not c.last or len(c.last) != 1:
        return False
    a = c.last[0]
    dom, svc = a["service"].split(".", 1)
    if svc not in ("turn_on", "turn_off", "open_cover", "close_cover") or a.get("data") \
            or dom not in ("light", "cover", "switch"):
        return False
    # undo everything the last action switched, not only its first entity
    cur_ids = list(a["entity_id"])
    cur_areas = {e.area for e in c.h.ents if e.eid in cur_ids}
    others = [e for e in c.h.of(dom, plain=dom == "cover")
              if e.area and e.area not in cur_areas and e.state != "unavailable"]
    if not others:
        return False
    o = R.choice(others)
    prep = ROOM_BY_NAME[o.area][1]
    if R.random() < 0.6:
        # "ik bedoelde in de eetkamer": every device of that kind in the room
        c.user(R.choice([f"Nee, ik bedoelde {prep}", f"Fout, {prep}",
                         f"Nee, {prep}", f"Ik bedoelde {prep}"]))
        new_ids = [e.eid for e in c.h.of(dom, o.area, plain=dom == "cover")
                   if e.state != "unavailable"]
        by_room = len(new_ids) == len(c.h.of(dom, o.area, plain=dom == "cover"))
    else:
        c.user(R.choice([f"Niet die, maar {o.name}", f"Ik bedoelde {o.name}"]))
        new_ids = [o.eid]
        by_room = False
    undo = {"service": f"{dom}.{INVERSE[svc]}", "entity_id": cur_ids}
    if "area" in a:
        undo = {"service": undo["service"], "area": a["area"], "entity_id": cur_ids}
    new = {"service": a["service"], "entity_id": new_ids}
    if dom in S.AREA_DOMAINS and by_room:
        new = {"service": a["service"], "area": o.area, "entity_id": new_ids}
    acts = [undo, new]
    c.control(acts, f"Oké, ik heb {describe(c.h, acts[0])} en {describe(c.h, acts[1])}.")
    return True


def turn_correct_recap(c):
    """Correction followed by "Wat heb je allemaal gedaan?": the recap lists
    exactly the executed actions."""
    if not turn_correct(c):
        return False
    if R.random() < 0.7:
        return turn_summary(c)
    return turn_which(c)


def turn_also(c):
    """'In de badkamer ook' -> same service on the same device type in another room."""
    if not c.last or len(c.last) != 1:
        return False
    a = c.last[0]
    dom = a["service"].split(".")[0]
    if dom not in ("light", "cover", "climate"):
        return False
    first = next(e for e in c.h.ents if e.eid == a["entity_id"][0])
    rooms = [r for r in c.h.rooms if r != first.area and
             [e for e in c.h.of(dom, r, plain=dom == "cover") if e.state != "unavailable"]]
    if not rooms:
        return False
    r = R.choice(rooms)
    prep = ROOM_BY_NAME[r][1]
    ents = [e for e in c.h.of(dom, r, plain=dom == "cover") if e.state != "unavailable"]
    c.user(R.choice([f"En {prep} ook", f"{cap(prep)} ook", f"Hetzelfde {prep}"]))
    act = {"service": a["service"],
           **({"area": r} if dom in S.AREA_DOMAINS and len(ents) == len(c.h.of(dom, r)) else {}),
           "entity_id": [e.eid for e in ents]}
    if a.get("data"):
        act["data"] = a["data"]
    c.control([act], R.choice(["Oké, ik heb ", "Gedaan, ik heb "])
              + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["Iets feller", "Nog feller", "Maak het feller"]), {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["Iets minder fel", "Nog wat zachter", "Dim het wat"]), \
            {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = f"Op {p} procent", {"brightness_pct": p}
    act = {"service": "light.turn_on",
           **({"area": c.last[0]["area"]} if "area" in c.last[0] else {}),
           "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "Oké, ik heb " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'Is hij nu aan?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    area = c.last[-1].get("area")
    ents = [e for e in c.h.ents if e.eid in ids]
    if not ents or ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    many = len(ents) > 1 or ents[0].plural or is_plural_name(ents[0].name)
    pron = "ze" if many else PRON[ents[0].gender][0]
    verb = "Zijn" if many else "Is"
    pairs = {"cover": [("open", "open"), ("dicht", "dicht"), ("omlaag", "dicht"),
                       ("omhoog", "open")],
             "lock": [("op slot", "op slot"), ("open", "niet op slot")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("aan", "aan"), ("uit", "uit")]))
    question = f"{verb} {pron} nu {word_q}?"
    words = {_state_word(e) for e in ents}
    states = ", ".join(f"{e.name} {'zijn' if e.plural else ist(e.name)} {_state_word(e)}"
                       for e in ents)
    if R.random() < 0.6 and len(words) == 1 and words <= {
            "aan", "uit", "open", "dicht", "op slot", "niet op slot"}:
        c.user(question)
        word = words.pop()
        yes = "Ja" if word == asked else "Nee"
        c.get_state(ents, f"{yes}, volgens Home Assistant: {states}.", area)
    else:
        c.user(R.choice(["Is het gelukt?", "Klopt dat echt?", "Kun je dat controleren?"]))
        c.get_state(ents, f"Volgens Home Assistant: {states}.", area)
    return True


OPENERS = [(turn_command, 6), (turn_question, 2), (turn_chat, 2), (turn_error, 1),
           (turn_unknown, 1)]
NEXT = [(turn_command, 4), (turn_question, 2), (turn_which, 4), (turn_undo, 3),
        (turn_correct, 2), (turn_also, 2), (turn_brightness, 1), (turn_check, 2),
        (turn_summary, 2), (turn_error, 1), (turn_unknown, 1), (turn_chat, 1)]


def s_conversation(h, loc):
    """3-6 turns; the loss covers every assistant turn, the eval checks the last."""
    c = Conv(h, loc)
    n = R.choice([3, 3, 4, 4, 5, 6])
    pool = OPENERS
    turns = 0
    forced = FORCE.get("conv_turn")
    done = forced is None
    for _ in range(n * 6):
        if not done and turns >= 1 and forced(c):
            # the turn type this sample stands for, as early as it fits
            done = True
            turns += 1
            if turns >= n:
                break
            continue
        if not done and turns >= n - 1:
            continue          # keep the last slot for the forced turn
        fn = R.choices([p[0] for p in pool], [p[1] for p in pool])[0]
        if fn(c):
            turns += 1
            pool = NEXT
            if turns >= n:
                break
    if turns < 3 or not done:
        return None
    return c.msgs, {"cat": "conversation", "gold": [], "gold_last": c.gold_last}


CATEGORIES = [
    (s_command, 46), (s_compound, 8), (s_question, 24), (s_relative_climate, 2),
    (s_unknown, 9), (s_chat, 3), (s_error, 5), (s_followup, 8),
    (s_missing_type, 6), (s_appliance_question, 3), (s_conversation, 12),
    (_error_room, 3), (_named_room_light, 5),
]


def history_prefix(h, loc):
    """An earlier exchange in the same conversation (small talk or a command)."""
    if R.random() < 0.4:
        users, answers = R.choice(CHAT)
        return [{"role": "user", "content": voice(R.choice(users))},
                {"role": "assistant", "content": R.choice(answers)}]
    out = s_command(h, loc)
    return out[0][1:] if out else []


QKINDS = ["temp", "outdoor", "forecast", "climate", "onoff", "which_lights", "windows", "door", "car", "lock",
          "alarm", "media", "humidity", "power", "generic"]


def build_cases():
    """Every case gets the same share of the dataset, rare ones included:
    one case per command intent, per question kind, per error/unknown
    variant and per conversation turn type."""
    cases = [(f"command:{fn.__name__}", s_command, {"intent": fn}) for fn, _ in COMMANDS]
    cases += [(f"question:{k}", s_question, {"qkind": k}) for k in QKINDS]
    cases += [("compound", s_compound, {}), ("relative_climate", s_relative_climate, {}),
              ("unknown:device", s_unknown, {"unknown": "device"}),
              ("unknown:room", s_unknown, {"unknown": "room"}),
              ("chat", s_chat, {}),
              ("appliance_question", s_appliance_question, {}),
              ("followup", s_followup, {}), ("error:single", s_error_single, {}),
              ("error:partial", _error_partial, {}), ("error:question", _error_question, {}),
              ("error:room", _error_room, {}),
              ("error:room_named_q", _error_room, {"eroom": "named_q"}),
              ("named_room_light", _named_room_light, {}),
              ("no_areas", s_no_areas, {})]
    # one mixed case for the missing device kinds plus the TV case (one case
    # per kind made the refusal share too large)
    cases += [("unknown:missing_type", s_missing_type, {"mtype": "other"}),
              ("unknown:missing_media_player", s_missing_type, {"mtype": "media_player"})]
    seen = set()
    for fn, _ in NEXT:
        if fn not in seen:
            seen.add(fn)
            cases.append((f"conversation:{fn.__name__}", s_conversation, {"conv_turn": fn}))
    # "Licht in de keuken aan" -> "Nee, ik bedoelde in de eetkamer": undo every
    # kitchen light, switch every dining room light
    cases.append(("conversation:room_correct", s_conversation,
                  {"conv_turn": turn_correct, "intent": i_light_room_phrase}))
    cases.append(("conversation:correct_recap", s_conversation,
                  {"conv_turn": turn_correct_recap, "intent": i_light_room_phrase}))
    for hook in CASE_HOOKS:
        cases += hook()
    return cases


SHORT_OK = ["Gedaan.", "Oké.", "Klaar.", "Is gebeurd."]
BARE = {"Gedaan.", "Oké.", "Klaar.", "Is gebeurd.", "Gepauzeerd.", "Harder gezet.",
        "Zachter gezet.", "Volgend nummer.", "Gestopt.", "Vorig nummer."}
DESCRIBED = {"turn_on", "turn_off", "open_cover", "close_cover", "stop_cover", "lock", "unlock",
             "media_play", "media_pause", "media_stop", "media_next_track",
             "media_previous_track", "volume_up", "volume_down", "start", "pause",
             "return_to_base", "start_mowing", "dock", "open_valve", "close_valve", "press",
             "set_temperature", "set_cover_position", "set_percentage", "volume_set",
             # assist services, worded by DESCRIBE_HOOKS
             "add_item", "complete_item", "remove_item", "search_and_play", "shuffle_set",
             "repeat_set", "volume_mute", "set_humidity", "set_valve_position"}


def apply_verbosity(h, msgs):
    """Answers after a successful control call follow the answer length:
    short -> a word or two, long/detailed -> never a bare 'Gedaan.'."""
    if VERBOSITY == "medium":
        return
    for i in range(2, len(msgs)):
        m, res, call = msgs[i], msgs[i - 1], msgs[i - 2]
        if m["role"] != "assistant" or m.get("tool_calls") or res["role"] != "tool" \
                or not call.get("tool_calls") or \
                call["tool_calls"][0]["function"]["name"] != "control":
            continue
        try:
            result = json.loads(res["content"])
        except ValueError:
            continue
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        if VERBOSITY == "short":
            if R.random() < 0.7:
                m["content"] = R.choice(SHORT_OK)
            continue
        if m["content"] not in BARE:
            continue
        acts = json.loads(call["tool_calls"][0]["function"]["arguments"])["actions"]
        if not all(a["service"].split(".", 1)[1] in DESCRIBED for a in acts):
            continue
        for a in acts:
            a.setdefault("entity_id", [])
        m["content"] = "Ik heb " + describe_all(h, acts) + "."


# "1 graden" -> "1 graad" (also -1), not "21 graden" or "0,1 graden"
ONE_DEGREE = re.compile(r"(?<![\d,])(-?1) graden")


def make_sample(case):
    global VERBOSITY
    name, fn, force = case
    for _ in range(20):
        VERBOSITY = R.choices([v for v, _ in VERBOSITY_WEIGHTS],
                              [w for _, w in VERBOSITY_WEIGHTS])[0]
        size = R.choice([R.randint(8, 18), R.randint(18, 35), R.randint(25, 45),
                         R.randint(35, 60), R.randint(60, 110)])
        h = House(size)
        loc = R.choice(h.rooms) if R.random() < 0.25 else None
        FORCE.clear()
        FORCE.update(force)
        try:
            out = fn(h, loc)
        finally:
            FORCE.clear()
        if out:
            msgs, meta = out
            if meta["cat"] not in ("followup", "conversation") and \
                    not meta.pop("no_history", False) and R.random() < 0.2:
                msgs = msgs[:1] + history_prefix(h, loc) + msgs[1:]
            apply_verbosity(h, msgs)
            for m in msgs:
                if m["role"] == "assistant" and m.get("content"):
                    m["content"] = ONE_DEGREE.sub(r"\1 graad", m["content"])
            meta["location"] = loc
            meta["case"] = name
            meta["verbosity"] = VERBOSITY
            meta["groups"] = [e.eid for e in h.ents if e.label == "__group__"]
            return {"locale": LANG, "messages": msgs, "meta": meta}
    return None


def main(argv):
    """argv: OUT.jsonl N [SEED]. N may be "per_case:K" for K samples per case."""
    out = Path(argv[0])
    R.seed(int(argv[2]) if len(argv) > 2 else 1)
    cases = build_cases()
    n = len(cases) * int(argv[1].split(":")[1]) if argv[1].startswith("per_case:") \
        else int(argv[1])
    target = {c[0]: n // len(cases) + (1 if i < n % len(cases) else 0)
              for i, c in enumerate(cases)}
    count = {c[0]: 0 for c in cases}
    fails = {c[0]: 0 for c in cases}
    exhausted = set()
    rows, seen = [], set()
    while len(rows) < n:
        open_ = [c for c in cases if c[0] not in exhausted]
        if not open_:
            break
        active = [c for c in open_ if count[c[0]] < target[c[0]]]
        if not active:
            # an exhausted case left quota behind: share it among the others
            for c in open_:
                target[c[0]] += 1
            continue
        low_ = min(count[c[0]] for c in active)
        case = R.choice([c for c in active if count[c[0]] == low_])
        s = make_sample(case)
        key = None
        if s:
            last_user = [m["content"] for m in s["messages"] if m["role"] == "user"][-1]
            key = last_user + str(s["meta"].get("gold")) + str(s["meta"].get("gold_last"))
        if not s or key in seen:
            fails[case[0]] += 1
            if fails[case[0]] >= 300:      # no new distinct samples left
                exhausted.add(case[0])
            continue
        fails[case[0]] = 0
        seen.add(key)
        rows.append(s)
        count[case[0]] += 1
    R.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = {}
    for r in rows:
        cats[r["meta"]["cat"]] = cats.get(r["meta"]["cat"], 0) + 1
    print(f"{len(rows)} samples -> {out}: {cats}")
    print(f"{len(cases)} cases, {min(count.values())}-{max(count.values())} samples each"
          + (f"; exhausted: {sorted(exhausted)}" if exhausted else ""))
