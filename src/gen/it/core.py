"""Generate the Italian SFT dataset for Home Assistant tool calling.

Port of src/gen/de/core.py: the same cases, categories, tool calls and gold
metadata, with Italian rooms, device names and sentences. Every sample uses a
freshly randomised house (rooms, device names, entity_ids, states), so the
model learns to map names from the device list to entity_ids instead of
memorising one house. Prompt and tool schema come from
custom_components/ha_llm_agent/llm_schema.py (LANGUAGES["it"]).

Italian needs articles and agreement everywhere ("la luce è accesa", "i
faretti sono accesi"), so every device carries two grammatical genders: the one
of its type label ("la plafoniera") and the one of its name ("Cielo stellato"
is masculine). Codes: "m", "f" (singular), "mp", "fp" (plural).

Each JSONL row: {"messages": [...], "meta": {"cat": ..., "gold": [...],
"answer_contains": [...]}}.

Usage: python src/gen_dataset.py --lang it OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

LANG = "it"
VERBOSITY = "medium"
VERBOSITY_WEIGHTS = [("short", 2), ("medium", 4), ("long", 2), ("detailed", 2)]
R = random.Random()
POPULATE_HOOKS: list = []
CASE_HOOKS: list = []
FORCE: dict = {}


# --------------------------------------------------------------------------
# grammar: articles, articulated prepositions, agreement, numbers
# --------------------------------------------------------------------------
VOWELS = "aeiouàèéìíòóùAEIOUÀÈÉÌÒÙ"


def _lo(word: str) -> bool:
    """Masculine words that take 'lo'/'gli' (s+consonant, z, x, y, gn, ps, pn)."""
    w = word.lower()
    return (len(w) > 1 and w[0] == "s" and w[1] not in VOWELS) or w[:1] in ("z", "x", "y") \
        or w[:2] in ("gn", "ps", "pn")


PLURAL_WORDS = {"luci", "lampade", "tapparelle", "persiane", "veneziane", "faretti", "lucine",
                "piante", "termostati", "prese"}


def is_plural_name(name: str) -> bool:
    """'Tutte le luci', 'Faretti Cucina', 'Luci casa' take a plural verb."""
    words = name.lower().split()
    return bool(words) and (words[0] in ("tutte", "tutti") or words[0] in PLURAL_WORDS)


def gp(code: str) -> tuple[str, bool]:
    """'fp' -> ('f', True)."""
    return code[0], code.endswith("p")


def art(word: str, code: str) -> str:
    """Definite article incl. trailing space ('il ', 'l'', 'gli ')."""
    if code == "inf":
        return ""
    g, pl = gp(code)
    v = word[:1] in VOWELS
    if g == "m" and not pl:
        return "l'" if v else "lo " if _lo(word) else "il "
    if g == "m":
        return "gli " if v or _lo(word) else "i "
    if not pl:
        return "l'" if v else "la "
    return "le "


def the(word: str, code: str) -> str:
    return art(word, code) + word


_PA = {"di": "de", "a": "a", "da": "da", "in": "ne", "su": "su"}
_TAIL = {"il": "l ", "lo": "llo ", "l'": "ll'", "la": "lla ", "i": "i ", "gli": "gli ",
         "le": "lle "}


def pa(prep: str, word: str, code: str) -> str:
    """Articulated preposition + word: pa('di', 'cucina', 'f') -> 'della cucina'."""
    a = art(word, code).strip()
    if not a:
        return f"{prep} {word}"
    return _PA[prep] + _TAIL[a] + word


INVARIABLE = {"blu", "viola", "rosa", "lilla", "beige"}


def _agree1(w: str, g: str, pl: bool) -> str:
    if w in INVARIABLE:
        return w
    if w.endswith("co") and pl:
        return w[:-2] + ("chi" if g == "m" else "che")
    if w.endswith("o"):
        return w[:-1] + ("i" if pl and g == "m" else "e" if pl else "a" if g == "f" else "o")
    if w.endswith("e") and pl:
        return w[:-1] + "i"
    return w


def agree(phrase: str, code: str, all_words: bool = False) -> str:
    """'acceso' for 'fp' -> 'accese'; only the first word unless all_words."""
    g, pl = gp(code)
    words = phrase.split(" ")
    for i in range(len(words) if all_words else 1):
        words[i] = _agree1(words[i], g, pl)
    return " ".join(words)


def essere(pl: bool) -> str:
    return "sono" if pl else "è"


def c_e(pl: bool) -> str:
    return "ci sono" if pl else "c'è"


def clitic(code: str) -> str:
    return {"m": "lo", "f": "la", "mp": "li", "fp": "le"}.get(code, "lo")


def num_word(n) -> str:
    """A word that sounds like the start of the number, for elision:
    'all'80 percento', 'allo 0', 'al 50'."""
    s = str(n).lstrip("-")
    if s[:1] == "8" or s in ("1", "11"):
        return "otto"
    if s == "0":
        return "zero"
    return "cento"


def pnum(prep: str, n, unit: str = "percento") -> str:
    """pnum('a', 80) -> "all'80 percento"; pnum('di', 50) -> 'del 50 percento'."""
    w = num_word(n)
    head = pa(prep, w, "m")[:-len(w)]
    sep = "" if unit == "%" else " "
    return f"{head}{n}{sep}{unit}".rstrip()


def it_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}".replace(".", ",")


def gradi(v) -> str:
    v = str(v)
    return f"{v} grado" if v in ("1", "-1") else f"{v} gradi"


def ci_sono_gradi(v) -> str:
    v = str(v)
    return f"c'è {v} grado" if v in ("1", "-1") else f"ci sono {v} gradi"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def join_it(items, word="e"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, preposition phrase, gender, slugs
ROOMS = [
    ("Soggiorno", "in soggiorno", "m", ["soggiorno", "living_room", "salotto"]),
    ("Cucina", "in cucina", "f", ["cucina", "kitchen"]),
    ("Camera da letto", "in camera da letto", "f", ["camera_da_letto", "bedroom", "camera"]),
    ("Bagno", "in bagno", "m", ["bagno", "bathroom"]),
    ("Bagno principale", "nel bagno principale", "m", ["bagno_principale", "main_bathroom"]),
    ("Cameretta", "nella cameretta", "f", ["cameretta", "kids_room"]),
    ("Ufficio", "in ufficio", "m", ["ufficio", "office"]),
    ("Studio", "nello studio", "m", ["studio", "study"]),
    ("Corridoio", "nel corridoio", "m", ["corridoio", "hallway"]),
    ("Sala da pranzo", "in sala da pranzo", "f", ["sala_da_pranzo", "dining_room"]),
    ("Camera degli ospiti", "nella camera degli ospiti", "f", ["camera_ospiti", "guest_room"]),
    ("Cantina", "in cantina", "f", ["cantina", "basement"]),
    ("Garage", "in garage", "m", ["garage"]),
    ("Giardino", "in giardino", "m", ["giardino", "garden"]),
    ("Terrazzo", "in terrazzo", "m", ["terrazzo", "terrace"]),
    ("Balcone", "sul balcone", "m", ["balcone", "balcony"]),
    ("Soffitta", "in soffitta", "f", ["soffitta", "attic"]),
    ("Lavanderia", "in lavanderia", "f", ["lavanderia", "laundry"]),
    ("Bagno ospiti", "nel bagno ospiti", "m", ["bagno_ospiti", "wc"]),
    ("Cabina armadio", "nella cabina armadio", "f", ["cabina_armadio", "dressing_room"]),
    ("Ingresso", "all'ingresso", "m", ["ingresso", "entrance"]),
    ("Scale", "sulle scale", "fp", ["scale", "stairs"]),
    ("Esterno", "all'esterno", "m", ["esterno", "outdoor"]),
    ("Ripostiglio", "nel ripostiglio", "m", ["ripostiglio", "storage"]),
    ("Locale caldaia", "nel locale caldaia", "m", ["locale_caldaia", "boiler_room"]),
    ("Locale tecnico", "nel locale tecnico", "m", ["locale_tecnico", "utility"]),
    ("Vialetto", "nel vialetto", "m", ["vialetto", "driveway"]),
    ("Taverna", "in taverna", "f", ["taverna", "utility_room"]),
    ("Laboratorio", "in laboratorio", "m", ["laboratorio", "workshop"]),
    ("Tettoia", "sotto la tettoia", "f", ["tettoia", "carport"]),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
BATH, BATH2 = "Bagno", "Bagno principale"
OUTDOOR = ("Giardino", "Terrazzo", "Balcone", "Esterno", "Vialetto", "Tettoia")
UTILITY = ("Ripostiglio", "Locale caldaia", "Locale tecnico", "Taverna",
           "Laboratorio", "Cantina", "Soffitta", "Garage", "Lavanderia")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garage",)]
NO_GROUP_COVERS = ("tenda da sole", "portone del garage")
TWO_COVER_ROOMS = ("Soggiorno", "Sala da pranzo", "Cucina", "Camera da letto", "Studio",
                   "Cameretta", "Ufficio")


def room_di(room: str) -> str:
    """'del soggiorno', 'della cucina'."""
    return pa("di", room.lower(), ROOM_BY_NAME[room][2])


# device type: (label, gender, slugs, compound-able); labels are lowercase
TYPES = {
    "light": [
        ("plafoniera", "f", ["plafoniera", "ceiling"], False),
        ("luce", "f", ["luce", "light"], True),
        ("lampadario", "m", ["lampadario", "chandelier"], False),
        ("lampada", "f", ["lampada", "lamp"], True),
        ("lampada da terra", "f", ["piantana", "floor_lamp"], False),
        ("lampada da lettura", "f", ["lampada_lettura", "reading_lamp"], False),
        ("striscia LED", "f", ["striscia_led", "led_strip"], False),
        ("faretti", "mp", ["faretti", "spotlights"], False),
        ("applique", "f", ["applique", "wall_lamp"], False),
        ("lucine", "fp", ["lucine", "fairy_lights"], False),
        ("lampada a sospensione", "f", ["sospensione", "pendant"], False),
        ("lampada da comodino", "f", ["comodino", "bedside"], False),
        ("luce specchio", "f", ["specchio", "mirror_light"], False),
    ],
    "switch": [
        ("presa", "f", ["presa", "plug", "socket"], False),
        ("macchina del caffè", "f", ["macchina_caffe", "coffee"], False),
        ("termoventilatore", "m", ["termoventilatore", "heater"], False),
        ("deumidificatore", "m", ["deumidificatore", "dehumidifier"], False),
        ("pompa del laghetto", "f", ["pompa_laghetto", "pond_pump"], False),
        ("irrigazione", "f", ["irrigazione", "irrigation"], False),
        ("albero di Natale", "m", ["albero_natale", "xmas_tree"], False),
        ("stampante", "f", ["stampante", "printer"], False),
        ("bollitore", "m", ["bollitore", "kettle"], False),
        ("monitor", "m", ["monitor", "screen"], False),
    ],
    "cover": [
        ("tapparella", "f", ["tapparella", "shutter"], True),
        ("tapparelle", "fp", ["tapparelle", "shutters"], False),
        ("veneziana", "f", ["veneziana", "blinds"], False),
        ("persiana", "f", ["persiana", "blind"], False),
        ("tenda da sole", "f", ["tenda_sole", "awning"], False),
    ],
    "climate": [
        ("riscaldamento", "m", ["riscaldamento", "heating", "thermostat"], True),
        ("termostato", "m", ["termostato", "trv"], False),
        ("condizionatore", "m", ["condizionatore", "aircon"], False),
        ("riscaldamento a pavimento", "m", ["pavimento", "floor_heating"], False),
    ],
    "fan": [
        ("ventilatore", "m", ["ventilatore", "fan"], False),
        ("ventilatore a soffitto", "m", ["ventilatore_soffitto", "ceiling_fan"], False),
        ("ventilazione", "f", ["ventilazione", "ventilation"], False),
    ],
    "media_player": [
        ("televisore", "m", ["tv", "televisore"], False),
        ("altoparlante", "m", ["speaker", "sonos", "altoparlante"], False),
        ("soundbar", "f", ["soundbar"], False),
        ("radio", "f", ["radio"], False),
    ],
}

LOCKS = [("Porta d'ingresso", "f", ["porta_ingresso", "front_door"]),
         ("Porta della cantina", "f", ["porta_cantina", "basement_door"]),
         ("Portafinestra", "f", ["portafinestra", "patio_door"]),
         ("Cancello", "m", ["cancello", "garden_gate"]),
         ("Porta sul retro", "f", ["porta_retro", "back_door"])]
FRONT_DOOR = "Porta d'ingresso"
VACUUMS = [("robot aspirapolvere", "m", ["robot_aspirapolvere", "roborock", "robot"]),
           ("aspirapolvere", "m", ["aspirapolvere", "vacuum"])]
SCENES = ["Serata film", "Buonanotte", "Cucinare", "Relax", "Festa", "Lettura",
          "Risveglio", "Cena", "Romantica", "Luce piena", "Concentrazione"]
SCRIPTS = ["Spegni tutto", "Modalità vacanza", "Routine della buonanotte", "Prepara il caffè",
           "Esco di casa", "Rientro a casa"]

COLORS = {"rosso": "red", "verde": "green", "blu": "blue", "giallo": "yellow",
          "arancione": "orange", "viola": "purple", "rosa": "pink", "bianco": "white",
          "turchese": "turquoise", "lilla": "violet"}
COLOR_TEMPS = {"bianco caldo": 2700, "bianco freddo": 6000, "bianco neutro": 4000}

# brand-named devices: (name, generic label, label gender, name gender)
BRAND_MEDIA = [("LG webOS Smart TV", "televisore", "m", "f"),
               ("Samsung TV", "televisore", "m", "f"),
               ("Sony Bravia", "televisore", "m", "m"),
               ("Philips Ambilight", "televisore", "m", "m"),
               ("Sonos One", "altoparlante", "m", "m"), ("Echo Dot", "altoparlante", "m", "m"),
               ("HomePod", "altoparlante", "m", "m"), ("Bose Soundbar", "soundbar", "f", "f")]


def voice_media_name():
    hx = "".join(R.choice("0123456789abcdef") for _ in range(6))
    return R.choice([f"Home Assistant Voice {hx} Media Player", f"Home Assistant Voice {hx[:6]}",
                     f"ESPHome Voice {hx[:4]} Media Player", "Echo Show", "Nest Mini",
                     f"Atom Echo {hx[:6]} Media Player"])


BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Robot tagliaerba", "Gardena Sileno", "Worx Landroid", "Tosaerba"]
# plant, gender (for "il basilico", "i pomodori")
PLANTS = [("Pomodori", "mp"), ("Peperoncino", "m"), ("Basilico", "m"), ("Prezzemolo", "m"),
          ("Fragole", "fp"), ("Monstera", "f"), ("Ficus", "m"), ("Cetrioli", "mp"),
          ("Limone", "m"), ("Rosmarino", "m"), ("Orchidea", "f")]
PLANT_G = dict(PLANTS)
MOISTURE = "Umidità terreno"
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "PC Marco", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Boost acqua calda", "Autoclave", "Pompa acquario"]
# appliances: a status sensor and a power switch with nearly the same name
APPLIANCES = [("Asciugatrice", "f", "asciugatrice"), ("Lavatrice", "f", "lavatrice"),
              ("Lavastoviglie", "f", "lavastoviglie"), ("Forno", "m", "forno"),
              ("Stampante 3D", "f", "stampante_3d"), ("Asciugabiancheria", "f", "dryer")]
APP_RAW = {"running": ["running", "In funzione", "In corso", "drying", "in_use", "run"],
           "finished": ["finished", "Programma terminato", "Fine", "end", "program_ended"],
           "off": ["off", "Spento"], "idle": ["idle", "Pronto", "standby", "programmed",
                                             "waiting_to_start"],
           "paused": ["pause", "paused", "In pausa"],
           "offline": ["not_connected", "disconnected", "Offline"]}


def app_answer(kind, code):
    """-> (predicate, key the answer must contain)."""
    return {"running": ("è ancora in funzione", "in funzione"),
            "finished": ("ha finito", "finito"),
            "off": (f"è {agree('spento', code)}", agree("spento", code)),
            "idle": (f"è {agree('pronto', code)}, ma al momento non è in funzione",
                     "non è in funzione"),
            "paused": ("è in pausa", "in pausa"),
            "offline": (f"al momento non è {agree('connesso', code)}",
                        f"non è {agree('connesso', code)}")}[kind]


# label, phrase for "Avvia ..."
VACUUM_BUTTONS = [("pulizia completa", "la pulizia completa"), ("svuotamento", "lo svuotamento"),
                  ("lavaggio mop", "il lavaggio del mop")]
TOGGLES = ["Vacanza", "Modalità ospiti", "Modalità festa", "Modalità notte", "Modalità pulizie"]

UNKNOWN_DEVICES = [("sauna", "f"), ("idromassaggio", "m"), ("pompa della piscina", "f"),
                   ("camino", "m"), ("acquario", "m"), ("tosaerba", "m"),
                   ("fontana", "f"), ("asciugatrice", "f"),
                   ("lavastoviglie", "f"), ("forno", "m"), ("climatizzatore portatile", "m"),
                   ("portone del garage", "m"), ("luci del cancello", "fp")]

OUT_LABEL = "temperatura esterna"
FC_LABEL = "previsione"
TEMP_LABEL = "temperatura"
IRR_LABEL = "irrigazione"
AWNING = "tenda da sole"
TV_LABEL = "televisore"


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "plafoniera"
    gender: str = "f"        # gender of the label
    state: str = "off"
    attrs: dict = field(default_factory=dict)
    hidden: str = ""         # type label that the name no longer shows
    ngender: str = ""        # gender of the name ("Cielo stellato": m)

    @property
    def plural(self) -> bool:
        return self.gender.endswith("p")

    @property
    def ng(self) -> str:
        """Gender of the name; group names ("Tutte le luci", "Luci casa") are plural."""
        code = self.ngender or self.gender
        if is_plural_name(self.name) and not code.endswith("p"):
            code = code[0] + "p"
        return code


class House:
    def __init__(self, size: int):
        self.ents: list[Ent] = []
        self.used_ids: set[str] = set()
        self.used_names: set[str] = set()
        self.cryptic = R.random() < 0.3
        self.shared = {}
        for dom in ("climate", "light", "cover", "switch", "binary_sensor"):
            if R.random() < 0.18:
                self.shared[dom] = R.choice([
                    "hmip_etrv_b_2_r4m_00a3c5e7f91", "hmip_swdm_2_00b4d6f8a2c3e",
                    "shellydimmer2_3c8a1fe0", "0x00158d000", "zigbee2mqtt_0x84fd27fffe",
                    "lumi_lumi_sensor_magnet_aq2_00158d", "tz3000_ji4araar_ts0001_"])
        self.english_ids = R.random() < 0.4
        n_rooms = max(2, min(len(ROOMS) - 4, size // 4 + R.randint(-1, 2)))
        names = R.sample([r[0] for r in ROOMS if r[0] != BATH2], n_rooms)
        if BATH in names and R.random() < 0.4:
            names[names.index(BATH)] = BATH2
        self.rooms = names
        self.appliances: list[tuple[str, str, Ent, Ent]] = []
        self._populate(size)
        for hook in POPULATE_HOOKS:
            hook(self)

    # ---- id / name helpers ----
    def _slug(self, room: str | None, type_slugs: list[str], domain: str) -> str:
        rslugs = ROOM_BY_NAME[room][3] if room else ["casa", "home"]
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
            base = R.choice([f"{ts}_{rs}", f"{ts}_{rs}", f"{rs}_{ts}"])
        eid = f"{domain}.{base}"
        i = 2
        while eid in self.used_ids:
            eid = f"{domain}.{base}_{i}"
            i += 1
        self.used_ids.add(eid)
        return eid

    def _name(self, room: str | None, label: str, compound: bool) -> str:
        """Italian installs mostly name devices 'Luce Cucina', some put the
        room first ('Cucina Luce') to sort the list by room."""
        lab = cap(label)
        if room:
            opts = [f"{lab} {room}", f"{lab} {room}", f"{lab} {room}", f"{room} {lab}"]
            if compound:
                opts.append(f"{lab} {room.lower()}")
        else:
            opts = [lab]
        name = R.choice(opts)
        i = 2
        base = name
        while name in self.used_names:
            name = f"{base} {i}"
            i += 1
        self.used_names.add(name)
        return name

    def add(self, domain, room, label, gender, slugs, compound=False,
            name=None, state="off", attrs=None, ngender=None):
        eid = self._slug(room, slugs, domain)
        if name is None:
            name = self._name(room, label, compound)
        else:
            self.used_names.add(name)
        e = Ent(eid, name, room, domain, label, gender, state, attrs or {},
                ngender=ngender or "")
        self.ents.append(e)
        return e

    def add_voice_media(self, room):
        name = voice_media_name()
        if name in self.used_names:
            return None
        return self.add("media_player", room, "altoparlante", "m",
                        [name.lower().replace(" ", "_"), "voice_media_player"], name=name,
                        state=R.choice(["idle", "idle", "off"]),
                        attrs={"volume_level": R.choice([0.3, 0.5, 0.7])})

    def _populate(self, size: int):
        for room in self.rooms:
            outdoor = room in OUTDOOR
            rg = ROOM_BY_NAME[room][2]
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
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                room_lights[0].ngender = rg
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Sinistra", "Destra", "Centro"],
                                      ["Soffitto", "Parete", "Angolo"],
                                      ["Isola", "Faretti", "Bancone"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        if f"{room} {side}" not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = f"{room} {side}"
                            lt.hidden, lt.label = lt.label, ""
                            lt.ngender = rg
                            self.used_names.add(lt.name)
            # many lamps carry a name without their room ("Cielo stellato")
            for lt in room_lights:
                if lt.name != room and R.random() < 0.3:
                    free = [n for n in FREE_LIGHT_NAMES if n[0] not in self.used_names]
                    if free:
                        self.used_names.discard(lt.name)
                        lt.name, lt.ngender = R.choice(free)
                        lt.hidden, lt.label = lt.label or lt.hidden, ""
                        self.used_names.add(lt.name)
            if not outdoor and room not in ("Corridoio", "Scale", "Bagno ospiti") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                if room in TWO_COVER_ROOMS and R.random() < 0.35:
                    # window and balcony door: "le tapparelle in sala da pranzo"
                    # means both
                    word = R.choice(["Tapparella", "Avvolgibile", "Persiana", "Veneziana"])
                    gender = "m" if word == "Avvolgibile" else "f"
                    door = R.choice(["Portafinestra", "Porta finestra", "Balcone", "Porta"])
                    for part, slug in (("Finestra", "finestra"), (door, "portafinestra")):
                        name = R.choice([f"{word} {part} {room}", f"{word} {room} {part}"])
                        if name in self.used_names:
                            continue
                        p2 = R.choice([pos, pos, 0, 100])
                        self.add("cover", room, word.lower(), gender,
                                 [f"{ct[2][0]}_{slug}", f"{ct[2][1]}_{slug}"], name=name,
                                 state="closed" if p2 == 0 else "open",
                                 attrs={"current_position": p2})
                else:
                    self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                             state="closed" if pos == 0 else "open",
                             attrs={"current_position": pos})
            if room in ("Terrazzo", "Balcone", "Esterno") and R.random() < 0.5:
                self.add("cover", room, AWNING, "f", ["tenda_sole", "awning"],
                         state=R.choice(["open", "closed"]),
                         attrs={"current_position": R.choice([0, 100])})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:3] if room != BATH else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, TEMP_LABEL, "f",
                         ["temperatura", "temperature"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "umidità", "f",
                         ["umidita", "humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "finestra", "f",
                         ["finestra", "window"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Soggiorno", "Camera da letto", "Cucina", "Ufficio",
                        "Cameretta") and R.random() < 0.4:
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
                        ["Tg1", "Bohemian Rhapsody", "Albachiara",
                         "Il commissario Montalbano", "Che tempo che fa", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "Vasco Rossi", ""]))
                use_brand = brand and brand[0] not in self.used_names
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if use_brand else None,
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""},
                         ngender=brand[3] if use_brand else None)
            if not outdoor and R.random() < 0.12:
                self.add_voice_media(room)
            if room in ("Camera da letto", "Ufficio", "Soggiorno", "Soffitta") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "consumo", "m", ["consumo", "power"],
                         name=self._name(room, R.choice(["Presa consumo", "Consumo"]), False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            area = R.choice(self.rooms + [None])
            self.add("alarm_control_panel", None, "allarme", "m",
                     ["allarme", "alarm"], name="Allarme",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Corridoio", "Ingresso", None]) if lk[0] == FRONT_DOOR else None
                room = room if room in self.rooms else None
                self.add("lock", room, lk[0].lower(), lk[1], lk[2], name=lk[0],
                         state=R.choice(["locked", "locked", "unlocked"]))
        if R.random() < 0.4:
            vt = R.choice(VACUUMS)
            vname = R.choice(BRAND_VACUUM) if R.random() < 0.6 else cap(vt[0])
            self.add("vacuum", R.choice(self.rooms + [None, None]), vt[0], vt[1],
                     [vname.lower().replace(" ", "_")], name=vname,
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.25:
            mname = R.choice(MOWERS)
            room = "Giardino" if "Giardino" in self.rooms else None
            self.add("lawn_mower", room, "robot tagliaerba", "m",
                     [mname.lower().replace(" ", "_")], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = ["Irrigazione", "Valvola acqua"]
            if room:
                names += [f"Irrigazione {room}", f"Piante {room}", f"{room} Piante"]
                if room == "Giardino":
                    names.append("Valvola giardino")
            name = R.choice(names)
            self.add("valve", room, IRR_LABEL, "f", ["irrigazione", "irrigation"],
                     name=name, state=R.choice(["open", "closed", "closed"]),
                     ngender="fp" if "Piante" in name else "f")
        if R.random() < 0.6:
            for sc in R.sample(SCENES, R.randint(1, 4)):
                self.add("scene", None, sc, "f", [slugify(sc)],
                         name=sc, state="2026-10-30T19:12:00+00:00")
        if R.random() < 0.35:
            for sc in R.sample(SCRIPTS, R.randint(1, 2)):
                self.add("script", None, sc, "m", [slugify(sc)], name=sc, state="off")
        out_t = round(R.uniform(-6, 31), 1)
        has_out = R.random() < 0.5
        if has_out:
            room, names = None, ["Temperatura esterna", "Temperatura Esterno",
                                 "Stazione meteo Temperatura"]
            orooms = [r for r in self.rooms if r in OUTDOOR]
            if orooms and R.random() < 0.4:
                room = R.choice(orooms)
                names = [f"Temperatura esterna {room}", f"Temperatura {room}"]
            self.add("sensor", room, OUT_LABEL, "f",
                     ["temperatura_esterna", "outdoor_temperature"],
                     name=R.choice(names), state=f"{out_t:.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            room = "Locale caldaia" if "Locale caldaia" in self.rooms and R.random() < 0.5 else None
            self.add("sensor", room, OUT_LABEL, "f",
                     ["caldaia_outdoor_temperature", "casa_outdoor_temperature"],
                     name=R.choice(["Temperatura esterna Casa", "Temperatura esterna Caldaia",
                                    "Caldaia Temperatura esterna"]),
                     state=f"{out_t + R.choice([-0.8, -0.4, 0.3, 0.6]):.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
            has_out = True
        if R.random() < (0.55 if has_out else 0.2):
            for name, label, when, slugs in R.sample(WEATHER_SENSORS, R.randint(1, 3)):
                if name in self.used_names:
                    continue
                if label == FC_LABEL:
                    val = out_t + R.uniform(-4, 6) if when == "oggi" else R.uniform(-6, 31)
                else:
                    val = out_t - R.uniform(0.5, 7)
                self.add("sensor", None, label, "f",
                         slugs, name=name,
                         state=f"{val:.1f}",
                         attrs={"device_class": "temperature", "unit_of_measurement": "°C",
                                "_when": when})
        if "Garage" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garage", "portone del garage", "m",
                     ["portone_garage", "garage_door"], name="Portone garage",
                     state=R.choice(["open", "closed", "closed"]),
                     attrs={"current_position": 0})
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(app in n for n in self.used_names):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Cucina", "Lavanderia", "Locale tecnico", "Bagno", "Bagno principale",
                    "Cantina", "Ripostiglio", "Ufficio", "Taverna")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "stato", "m", [slug, slug + "_stato"],
                                  name=R.choice([app, app, f"{app} Stato"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind},
                                  ngender=g)
                power = self.add("switch", room, "presa", "f",
                                 [slug + "_power", slug + "_presa"],
                                 name=R.choice([f"{app} Power", f"{app} Presa",
                                                f"Presa {app}"]),
                                 state="off" if kind == "off" and R.random() < 0.5 else "on",
                                 ngender=g)
                self.appliances.append((app, g, sensor, power))
        vac = [e for e in self.ents if e.domain == "vacuum"]
        if vac and R.random() < 0.45:
            v = vac[0]
            for label, _ in R.sample(VACUUM_BUTTONS, R.choice([1, 2])):
                self.add("button", v.area, label, "f",
                         [v.eid.split(".")[1] + "_" + label.replace(" ", "_")],
                         name=f"{v.name} {cap(label)}", state="unknown",
                         ngender="m" if label != "pulizia completa" else "f")
        valve_rooms = {e.area for e in self.ents if e.domain == "valve"}
        if R.random() < (0.5 if valve_rooms else 0.15):
            free = [r for r in self.rooms if r in OUTDOOR and r not in valve_rooms]
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Irrigazione Giardino", "Irrigazione giardino"] if room == "Giardino"
                         else [f"Irrigazione {room}"])
            elif "Giardino" in valve_rooms:
                names = ["Avvia irrigazione"]
            else:
                names = ["Avvia irrigazione", "Irrigazione Giardino", "Irrigazione giardino"]
            self.add("button", room, IRR_LABEL, "f", ["irrigazione_giardino",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # door contacts: balcony doors in living rooms, the front door in the
        # hall; names often write "Porta finestra" while people say
        # "portafinestra"
        if R.random() < 0.5:
            split = R.random() < 0.5
            for room in [r for r in self.rooms if r in DOOR_ROOMS]:
                if R.random() < 0.55:
                    label, slug = R.choice([("portafinestra", "portafinestra"),
                                            ("porta del balcone", "porta_balcone")])
                    shown = "Porta finestra" if split and label == "portafinestra" else cap(label)
                    name = R.choice([f"{shown} {room}", f"{room} {shown}"])
                    if name in self.used_names:
                        continue
                    self.add("binary_sensor", room, label, "f", [slug, "door"],
                             name=name, state=R.choice(["off", "off", "on"]),
                             attrs={"device_class": "door"})
            if R.random() < 0.7:
                base = R.choice(["porta d'ingresso", "porta di casa", "porta d'ingresso"])
                name = {"porta d'ingresso": "Porta ingresso", "porta di casa": "Porta casa"}[base] \
                    if split else cap(base)
                if name in self.used_names:
                    name = f"{name} Contatto"
                hall = next((r for r in ("Corridoio", "Ingresso") if r in self.rooms), None)
                self.add("binary_sensor", hall, base, "f", ["porta_ingresso_state",
                                                             "front_door_contact"],
                         name=name, state=R.choice(["off", "off", "on"]),
                         attrs={"device_class": "door"})
        # electric car: charge level and range next to each other
        if R.random() < 0.3:
            car, cslug = R.choice(CARS)
            self.add("sensor", None, "batteria", "f", ["battery_hv_state_of_charge",
                                                       f"{cslug}_soc"],
                     name=f"{car} Batteria", state=str(R.randint(10, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
            self.add("sensor", None, "autonomia", "f",
                     [f"{cslug}_range_ev_remaining_range", f"{cslug}_autonomia"],
                     name=f"{car} Autonomia", state=str(R.randint(40, 450)),
                     attrs={"device_class": "distance", "unit_of_measurement": "km"})
        # heating systems expose their modes as switches next to the climate entity
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            mode, slug, mg = R.choice(HEAT_MODES)
            base = R.choice(["Riscaldamento", "Riscaldamento Casa", "Termostato", "Caldaia"])
            room = "Locale caldaia" if "Locale caldaia" in self.rooms and R.random() < 0.5 else None
            if f"{base} {cap(mode)}" not in self.used_names:
                self.add("switch", room, mode, mg, [f"riscaldamento_{slug}", f"casa_{slug}"],
                         name=f"{base} {cap(mode)}", state=R.choice(["on", "off", "off"]),
                         ngender="f" if base == "Caldaia" else "m")
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            self.add("light", None, "__group__", "fp", ["tutte_le_luci", "all_lights"],
                     name=R.choice(["Tutte le luci", "Tutte le lampade", "Luci casa"]),
                     state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            self.add("cover", None, "__group__", "fp", ["tutte_le_tapparelle", "all_covers"],
                     name=R.choice(["Tutte le tapparelle", "Tutte le persiane",
                                    "Tapparelle casa"]),
                     state=R.choice(["open", "closed"]))
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "Casa"
        if k == 0 and room:
            self.add("binary_sensor", room, "presenza", "f", ["presence"],
                     name=R.choice([f"Presenza {room}", f"{room} Presence",
                                    f"Occupazione {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "movimento", "m", ["motion"],
                     name=R.choice([f"{rname} Motion", f"Movimento {rname}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant, pg = R.choice(PLANTS)
            self.add("sensor", room, "umidità terreno", "f", ["plant_sensor", "moisture"],
                     name=f"{plant} {MOISTURE}", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "batteria", "f", ["battery"],
                     name=R.choice(["Batteria UPS", "Batteria auto", "Batteria rilevatore fumo",
                                    "Batteria telefono", f"Batteria sensore {rname}"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "stampante", "f", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, unit = R.choice([("Velocità vento", "km/h"), ("Raffiche di vento", "km/h"),
                                 ("Pioggia", "mm/h"), ("Pressione atmosferica", "hPa"),
                                 ("Serbatoio acqua livello", "L")])
            self.add("sensor", room, nm.lower(), "f", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": unit})
        elif k == 6:
            self.add("sensor", room, "stato", "m", ["status"],
                     name=R.choice(["UPS Stato", "Asciugatrice", "Lavatrice Stato",
                                    "Lavastoviglie Programma"]),
                     state=R.choice(["idle", "OL", "running", "off"]))
        elif k == 7:
            nm = R.choice(TOGGLES)
            if nm not in self.used_names:
                self.add("input_boolean", None, nm, "f", [slugify(nm)], name=nm,
                         state=R.choice(["on", "off"]))
        elif k == 8:
            nm = R.choice(JUNK_SWITCHES)
            if nm not in self.used_names:
                # vendor names are no type label: never "il Pompa acquario in cucina"
                self.add("switch", room, "", "m", [slugify(nm)], name=nm,
                         state=R.choice(["on", "off"]),
                         ngender="f" if nm.startswith("Pompa") else "m")
        else:
            self.add("sensor", room, TEMP_LABEL, "f", ["temperature"],
                     name=R.choice([f"{rname} Temperatura", f"Temperatura {rname}",
                                    f"{rname} Termometro"]),
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
        return S.dumps([S.format_state(e.eid, e.name, e.state,
                                       {} if e.state == "unavailable" else e.attrs, LANG)
                        for e in ents])


# device names that are plain nouns: answers say "l'aspirapolvere"
COMMON_NAMES = {"Aspirapolvere", "Robot aspirapolvere", "Robot tagliaerba", "Tosaerba",
                "Irrigazione", "Valvola acqua", "Valvola giardino", "Allarme",
                "Porta d'ingresso", "Porta della cantina", "Portafinestra", "Cancello",
                "Porta sul retro"}


def noun_name(e: Ent) -> str:
    return the(lower_first(e.name), e.ng) if e.name in COMMON_NAMES else e.name


def slugify(s: str) -> str:
    out = s.lower()
    for a, b in (("à", "a"), ("è", "e"), ("é", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u"),
                 ("'", "_"), ("-", "_"), (".", ""), (" ", "_")):
        out = out.replace(a, b)
    return out


# --------------------------------------------------------------------------
# references: how the user names a device or a group of devices
# --------------------------------------------------------------------------
@dataclass
class Ref:
    bare: str            # "luce in cucina", "Luce Cucina", "tutte le luci"
    code: str            # gender/number for the article and agreement
    ents: list
    area: str | None = None   # set for "la luce in cucina": control by area
    art: bool = True          # bare carries no article yet

    @property
    def acc(self) -> str:
        return the(self.bare, self.code) if self.art else self.bare

    nom = acc

    @property
    def plural(self) -> bool:
        return self.code.endswith("p")

    def p(self, prep: str) -> str:
        """'della luce in cucina', 'di Luce Cucina'."""
        return pa(prep, self.bare, self.code) if self.art else f"{prep} {self.bare}"

    def adj(self, word: str) -> str:
        return agree(word, self.code)


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    # "la luce in corridoio" means every light of the room, so a single lamp
    # only gets these words when it is the room's only light
    generic = e.domain == "light" and e.label in ("luce", "lampada") \
        and len(h.of("light", e.area)) > 1
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1 \
            and not generic:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 \
            and e.name.lower() != e.label.lower():
        opts += ["type_only"]           # "il televisore" for "LG webOS Smart TV"
    choice = R.choice(opts)
    if choice == "type_only":
        return Ref(e.label, e.gender, [e])
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        with_art = R.random() < 0.45 and e.label and e.name.lower().startswith(e.label.lower())
        if with_art and e.name.lower() == e.label.lower():
            name = e.label          # "la portafinestra", not "la Portafinestra"
        return Ref(name, e.ng, [e], art=bool(with_art))
    prep = ROOM_BY_NAME[e.area][1]
    return Ref(f"{e.label} {prep}", e.gender, [e])


# group nouns per domain: (noun, gender code)
GROUP = {
    "light": [("luce", "f"), ("luci", "fp"), ("lampade", "fp"), ("illuminazione", "f")],
    "cover": [("tapparelle", "fp"), ("persiane", "fp"), ("veneziane", "fp")],
    "climate": [("riscaldamento", "m"), ("termostati", "mp")],
    "media_player": [("musica", "f")],
}
# (phrase, gender code, has article inside the phrase)
ALL_GROUP = {
    "light": [("tutte le luci", "fp", False), ("tutte le lampade", "fp", False),
              ("luce in tutta la casa", "f", True)],
    "cover": [("tutte le tapparelle", "fp", False), ("tutte le persiane", "fp", False)],
    "climate": [("tutti i termostati", "mp", False)],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    noun, code = R.choice(GROUP[domain])
    area = room if domain in S.AREA_DOMAINS else None
    if location == room and R.random() < 0.7:
        return Ref(noun, code, ents, area)    # "Accendi la luce"
    prep = ROOM_BY_NAME[room][1]
    return Ref(f"{noun} {prep}", code, ents, area)


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
    if name == "control":
        return {"actions": [{k: v for k, v in a.items()
                             if not (k == "entity_id" and "area" in a)}
                            for a in args["actions"]]}
    if name == "get_state" and args.get("area"):
        return {"domain": args.get("domain", "light"), "area": args["area"]}
    return args


LAST_CONTROL: list = []


def ok_result(actions=None):
    actions = LAST_CONTROL if actions is None else actions
    if any("area" in a for a in actions):
        return S.control_result(None, [i for a in actions for i in a["entity_id"]])
    return S.control_result()


def voice(text: str) -> str:
    """Make user text look like speech-to-text output sometimes."""
    r = R.random()
    if r < 0.25:
        text = text.rstrip("?.!")
    elif r < 0.35:
        text = text.rstrip("?.!").lower()
    if R.random() < 0.05:
        text = R.choice(["Ehi, ", "Ok, ", "Senti, ", "Allora, ", "Per favore "]) + \
            text[0].lower() + text[1:]
    return text


# state -> (adjective for "è ...", participle for "Ho ... X")
STATE_WORDS = {"on": ("acceso", "acceso"), "off": ("spento", "spento"),
               "open": ("aperto", "aperto"), "closed": ("chiuso", "chiuso"),
               "locked": ("chiuso a chiave", "chiuso a chiave"),
               "unlocked": ("sbloccato", "sbloccato")}


def confirm(ref: Ref, state: str) -> str:
    adj, part = STATE_WORDS[state]
    a = ref.adj(adj)
    e = essere(ref.plural)
    return R.choice([
        f"{cap(ref.nom)} ora {e} {a}.",
        f"{cap(ref.nom)} {e} {a}.",
        f"Ho {part} {ref.acc}.",
        f"Fatto, {ref.nom} {e} {a}.",
        f"Ok, ora {ref.nom} {e} {a}.",
        "Fatto.",
    ])


# --------------------------------------------------------------------------
# command intents: each returns (user_text, [actions], confirmation)
# --------------------------------------------------------------------------
def pick_target(h: House, domain, location=None, group_ok=True):
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
            "Accendi {r}", "Accendi {r}, per favore", "Mi accendi {r}?",
            "Puoi accendere {r}?", "Per favore accendi {r}", "Potresti accendere {r}?",
            "Accendi subito {r}", "Ho bisogno di luce, accendi {r}", "Accendi un attimo {r}",
            "Dai, accendi {r}", "Vorrei {r} " + ref.adj("acceso"),
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref)], confirm(ref, "on"))
    user = R.choice([
        "Spegni {r}", "Spegni {r}, per favore", "Mi spegni {r}?", "Puoi spegnere {r}?",
        "Per favore spegni {r}", "Potresti spegnere {r}?", "Spegni pure {r}",
        "Dai, spegni {r}", "Spegnimi {r}",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref)], confirm(ref, "off"))


def i_light_room_phrase(h, loc):
    """'Luce in bagno' / 'In bagno è troppo buio'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    prep = ROOM_BY_NAME[room][1]
    plural = R.random() < 0.3
    ref = Ref(f"{'luci' if plural else 'luce'} {prep}", "fp" if plural else "f", ents, room)
    noun = "le luci" if plural else "la luce"
    if R.random() < 0.5:
        user = R.choice([f"Accendi {noun} {prep}", f"{cap(prep)} accendi {noun}",
                         f"{cap(prep)} è troppo buio" if not plural else f"Accendi {noun} {prep}",
                         f"{cap(noun.split()[1])} {prep}, per favore",
                         f"Mi accendi {noun} {prep}?"])
        return user, [action("light.turn_on", ref)], confirm(ref, "on")
    user = R.choice([f"Spegni {noun} {prep}", f"{cap(prep)} spegni {noun}",
                     f"Mi spegni {noun} {prep}?", f"{cap(noun.split()[1])} {prep} spent"
                     + ("e" if plural else "a")])
    return user, [action("light.turn_off", ref)], confirm(ref, "off")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        al = pnum("a", p)
        user = R.choice([
            f"Regola {{r}} {al}", f"Imposta {{r}} {pnum('a', p, '%')}",
            f"{{R}} {al}", f"Metti {{r}} {al}",
            f"Porta la luminosità {{di}} {al}",
        ]).format(r=ref.acc, R=cap(ref.acc), di=ref.p("di"))
        conf = R.choice([f"{cap(ref.nom)} ora {essere(ref.plural)} {al}.",
                         f"Ho impostato {ref.acc} {al}.", "Fatto."])
        return user, [action("light.turn_on", ref, {"brightness_pct": p})], conf
    up = r < 0.8
    if up:
        user = R.choice(["Alza la luminosità {di}", "Rendi {r} più " + ref.adj("luminoso"),
                         "Un po' più di luce {in_}, per favore" if ref.area else
                         "Aumenta la luminosità {di}"])
    else:
        user = R.choice(["Abbassa {r}", "Abbassa la luminosità {di}",
                         "Rendi {r} meno " + ref.adj("luminoso")])
    user = user.format(r=ref.acc, di=ref.p("di"),
                       in_=ROOM_BY_NAME[ref.area][1] if ref.area else "")
    step = 20 if up else -20
    conf = (f"{cap(ref.nom)} ora {essere(ref.plural)} "
            f"{'più' if up else 'meno'} {ref.adj('luminoso')}.") if R.random() < 0.7 else "Fatto."
    return user, [action("light.turn_on", ref, {"brightness_step_pct": step})], conf


def i_light_color(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    if R.random() < 0.75:
        word, en = R.choice(list(COLORS.items()))
        data = {"color_name": en}
    else:
        word, k = R.choice(list(COLOR_TEMPS.items()))
        data = {"color_temp_kelvin": k}
    sul = pa("su", word, "m")
    adj = agree(word, ref.code, all_words=True)
    user = R.choice(["Metti {r} " + sul, "Imposta {r} " + sul, "Colora {r} di " + word,
                     "Fai {r} " + adj, "Puoi fare {r} " + adj + "?",
                     "Vorrei {r} " + adj]).format(r=ref.acc)
    conf = R.choice([f"Ora {ref.nom} {essere(ref.plural)} {adj}.",
                     f"Ho impostato {ref.acc} {sul}.", "Fatto."])
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
        user = R.choice(["Accendi {r}", "Attiva {r}", "Mi accendi {r}?", "Accendi {r}, grazie",
                         "Puoi accendere {r}?", "Avvia {r}"])
        conf = confirm(ref, "on")
    else:
        user = R.choice(["Spegni {r}", "Disattiva {r}", "Spegni {r}, grazie", "Stacca {r}",
                         "Puoi spegnere {r}?"])
        conf = confirm(ref, "off")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    al = pnum("a", p)
    user = R.choice([f"Imposta {{r}} {al}", f"{{R}} {pnum('a', p, '%')}",
                     f"Fai andare {{r}} {al}"]).format(r=ref.acc, R=cap(ref.acc))
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {essere(ref.plural)} {al}.", "Fatto."])


def i_awning(h, loc):
    """Tenda da sole: aprire/estendere = open_cover, chiudere/ritirare = close_cover."""
    ents = [e for e in h.of("cover") if e.label == AWNING]
    if not ents:
        return None
    e = R.choice(ents)
    n = "la tenda da sole" if len(ents) == 1 and R.random() < 0.7 else e.name
    if R.random() < 0.5:
        user = R.choice([f"Apri {n}", f"Estendi {n}", f"Srotola {n}",
                         f"Tira fuori {n}", "Ci serve ombra, apri la tenda da sole"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["La tenda da sole si sta aprendo.", "Fatto, la tenda da sole è aperta."])
    user = R.choice([f"Chiudi {n}", f"Ritira {n}", f"Riavvolgi {n}",
                     f"Tira dentro {n}", "Piove, ritira la tenda da sole"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["La tenda da sole si sta chiudendo.", "Fatto, la tenda da sole è chiusa."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["il robot tagliaerba", "il tosaerba", e.name])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Avvia {n}", f"Fai partire {n}", "Taglia l'erba, per favore",
                         f"Manda {n} a tagliare l'erba"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{cap(noun_name(e))} ora taglia l'erba."
    if r < 0.8:
        user = R.choice([f"Manda {n} alla base", f"Riporta {n} alla base di ricarica",
                         f"Fai rientrare {n}"])
        return user, [action("lawn_mower.dock", [e])], f"{cap(noun_name(e))} torna alla base di ricarica."
    user = R.choice([f"Metti in pausa {n}", f"Ferma {n}"])
    return user, [action("lawn_mower.pause", [e])], f"{cap(noun_name(e))} è in pausa."


AREA_PLANTS = {"Balcone": "le piante del balcone", "Terrazzo": "le piante del terrazzo",
               "Giardino": "il giardino"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == IRR_LABEL]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Avvia l'irrigazione", "Innaffia le piante"]
    if e.area:
        prep = ROOM_BY_NAME[e.area][1]
        out += [f"Avvia l'irrigazione {prep}", f"Innaffia le piante {prep}",
                f"Bagna le piante {prep}"]
        if e.area in AREA_PLANTS:
            out.append(f"Innaffia {AREA_PLANTS[e.area]}")
    elif "giardino" in e.name.lower():
        out += ["Innaffia il giardino", "Avvia l'irrigazione del giardino",
                "Bagna il giardino"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name] if several else [e.name, "l'irrigazione", "la valvola"])
    if R.random() < 0.5:
        user = R.choice([f"Apri {n}", f"Accendi {n}", f"Attiva {n}"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"Ho aperto {noun_name(e)}."
    user = R.choice([f"Chiudi {n}", f"Spegni {n}", f"Disattiva {n}",
                     "Ferma l'irrigazione"])
    return user, [action("valve.close_valve", [e])], f"Ho chiuso {noun_name(e)}."


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    # "Vacanza" is also called "modalità vacanza"; it stays an input_boolean
    mode = n if n.startswith("Modalità") else f"Modalità {n.lower()}"
    if mode != n and any(x.name.lower() == mode.lower() for x in h.ents):
        mode = n          # a script "Modalità vacanza" exists: only the plain name is clear
    m = lower_first(mode)
    nn = the(lower_first(n), "f") if n.startswith("Modalità") else n
    if on:
        user = R.choice([f"Attiva {n}", f"Accendi {n}", f"Attiva la {m}",
                         f"Accendi la {m}", f"Passa alla {m}", f"Metti la {m}"]
                        if mode != n else
                        [f"Attiva {n}", f"Accendi {n}", f"Attiva la {m}", f"Accendi la {m}",
                         f"Passa alla {m}"])
        return user, [action("input_boolean.turn_on", [e])], R.choice(
            [f"{cap(nn)} è attivata.", f"Ho attivato {nn}."])
    user = R.choice([f"Disattiva {n}", f"Spegni {n}", f"Disattiva la {m}",
                     f"Spegni la {m}", f"Esci dalla {m}", f"Basta con la {m}"])
    return user, [action("input_boolean.turn_off", [e])], R.choice(
        [f"{cap(nn)} è disattivata.", f"Ho disattivato {nn}."])


def i_appliance(h, loc):
    """'Spegni l'asciugatrice' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = the(app.lower(), g)
    if R.random() < 0.5:
        user = R.choice([f"Spegni {n}", f"Stacca {n}", f"Spegni {app.lower()}",
                         f"Togli la corrente {pa('a', app.lower(), g)}",
                         f"Puoi spegnere {n}?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"{power.name} ora è {agree('spento', g)}.", f"Ho spento {power.name}."])
    user = R.choice([f"Accendi {n}", f"Attiva {n}", f"Accendi {app.lower()}",
                     f"Mi accendi {n}?"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"{power.name} ora è {agree('acceso', g)}.", f"Ho acceso {power.name}."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        what = vac[e.label]
        user = R.choice([f"Avvia {what}", f"Avvia {what} del robot", f"Fai {what}",
                         f"Avvia {e.name}"])
    elif e.label == IRR_LABEL:
        user = R.choice(irrigation_phrases(h, e) + [f"Premi {e.name}"])
    else:
        user = R.choice([f"Premi {e.name}", f"Attiva {e.name}"])
    return user, [action("button.press", [e])], R.choice(
        [f"Ho attivato {e.name}.", f"Ho avviato {e.name}.", "Fatto."])


# lamp names without a room (the names of the acceptance house are left out
# on purpose, so the acceptance test measures generalisation)
FREE_LIGHT_NAMES = [("Cielo stellato", "m"), ("Lampada di sale", "f"), ("Lampada lava", "f"),
                    ("Luce quadro", "f"), ("Vetrina", "f"), ("Hue Go", "f"), ("Nanoleaf", "m"),
                    ("Hue Play", "f"), ("Lampada nuvola", "f"), ("Lampada luna", "f"),
                    ("Barra luminosa", "f"), ("Cornice LED", "f"), ("Stella di Natale", "f"),
                    ("Stella di carta", "f"), ("Sfera luminosa", "f"), ("Lampada angolo", "f"),
                    ("Lampada divano", "f"), ("Luce credenza", "f"), ("Luce bar", "f"),
                    ("Luce letto", "f"), ("Luce armadio", "f"), ("Luce finestra", "f"),
                    ("Piantana", "f"), ("Lampada sospesa", "f"), ("Arco luminoso", "m"),
                    ("Insegna neon", "f"), ("Govee Strip", "f"), ("Lampada da tavolo", "f"),
                    ("Luce comò", "f"), ("Luce specchio", "f"), ("Luce lettura", "f"),
                    ("Lumetto", "m")]

DOOR_ROOMS = ("Soggiorno", "Sala da pranzo", "Cucina", "Camera da letto", "Studio")
# (name, slug); cars are feminine in Italian ("la Zoe", "l'Enyaq")
CARS = [("Enyaq", "enyaq_iv80"), ("ID.3", "id3"), ("Model 3", "model_3"),
        ("Zoe", "zoe"), ("Enyaq", "enyaq"), ("e-Golf", "egolf"), ("EQA", "eqa"),
        ("500e", "fiat_500e")]

WEATHER_SENSORS = [  # (name, label, day, slugs)
    ("Temperatura prevista oggi", FC_LABEL, "oggi",
     ["temperatura_prevista_oggi", "forecast_temperature_today"]),
    ("Temperatura prevista domani", FC_LABEL, "domani",
     ["temperatura_prevista_domani", "forecast_temperature_tomorrow"]),
    ("Temperatura massima oggi", FC_LABEL, "oggi",
     ["temperatura_massima_oggi", "forecast_high_today"]),
    ("Temperatura massima domani", FC_LABEL, "domani",
     ["temperatura_massima_domani", "forecast_high_tomorrow"]),
    ("Temperatura percepita", "valore meteo", None, ["temperatura_percepita",
                                                     "feels_like_temperature"]),
    ("Punto di rugiada", "valore meteo", None, ["punto_di_rugiada", "dew_point"]),
]

# (label, slug, gender)
HEAT_MODES = [("modalità assenza", "away_mode", "f"), ("modalità eco", "eco_mode", "f"),
              ("antigelo", "frost_protection", "m")]


def i_heating_mode(h, loc):
    """'Attiva la modalità assenza del riscaldamento' -> the mode switch, not
    the climate entity."""
    modes = [e for e in h.ents if e.domain == "switch"
             and e.label in [m for m, _, _ in HEAT_MODES]]
    if not modes or not h.of("climate"):
        return None
    e = R.choice(modes)
    m = e.label
    tm = the(m, e.gender)
    on = R.random() < 0.6
    if on:
        user = R.choice([f"Attiva {tm} del riscaldamento", f"Metti il riscaldamento in {m}",
                         f"Riscaldamento in {m}", f"Attiva {tm}", f"Accendi {tm}",
                         f"Imposta il riscaldamento in {m}", f"Passa il riscaldamento in {m}"])
    else:
        user = R.choice([f"Disattiva {tm} del riscaldamento", f"Disattiva {tm}",
                         f"Spegni {tm}", f"Togli {tm} del riscaldamento",
                         f"Esci dalla {m}" if e.gender == "f" else f"Togli {tm}"])
    verb = "attivato" if on else "disattivato"
    return (user, [action("switch.turn_on" if on else "switch.turn_off", [e])],
            R.choice([f"Ho {verb} {e.name}.",
                      f"Fatto, {tm} è {agree(verb, e.gender)}."]))


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Innaffia le piante del balcone" -> balcony valve, not the garden button)."""
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
            [f"Ho aperto {noun_name(e)}.", f"Fatto, ho aperto {noun_name(e)}."])
    return user, [action("button.press", [e])], R.choice(
        [f"Ho attivato {e.name}.", f"Ho avviato {e.name}."])


def i_irrigation_pair(h, loc):
    """A plant valve on the balcony or terrace next to an area-less garden
    button. "Innaffia le piante del balcone" opens the valve, "Innaffia il
    giardino" presses the button."""
    rooms = [r for r in h.rooms if r in ("Balcone", "Terrazzo")]
    if not rooms or irrigation(h):
        return None
    room = R.choice(rooms)
    vname = R.choice([f"{room} Piante", f"Piante {room}", f"Irrigazione {room}",
                      "Valvola acqua"])
    valve = h.add("valve", room, IRR_LABEL, "f", ["piante", "irrigazione"],
                  name=vname, state=R.choice(["open", "closed", "closed"]),
                  ngender="fp" if "Piante" in vname else "f")
    button = h.add("button", None, IRR_LABEL, "f", ["irrigazione_giardino"],
                   name=R.choice(["Irrigazione Giardino", "Irrigazione giardino"]),
                   state="unknown")
    if R.random() < 0.6:
        prep = ROOM_BY_NAME[room][1]
        user = R.choice([f"Innaffia {AREA_PLANTS[room]}", f"Bagna le piante {prep}",
                         f"Avvia l'irrigazione {prep}", f"Innaffia le piante {prep}"])
        return user, [action("valve.open_valve", [valve])], R.choice(
            [f"Ho aperto {noun_name(valve)}.", f"Fatto, ho aperto {noun_name(valve)}."])
    user = R.choice(["Innaffia il giardino", "Avvia l'irrigazione del giardino",
                     "Bagna il giardino", "Avvia l'irrigazione in giardino"])
    return user, [action("button.press", [button])], R.choice(
        [f"Ho attivato {button.name}.", f"Ho avviato {button.name}."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Apri {r}", "Alza {r}", "Tira su {r}", "Solleva {r}",
                         "Puoi alzare {r}?", "Alza {r}, per favore", "Puoi aprire {r}?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "open"))
    if r < 0.8:
        user = R.choice(["Chiudi {r}", "Abbassa {r}", "Tira giù {r}", "Abbassa del tutto {r}",
                         "Puoi abbassare {r}?", "Chiudi {r}, per favore", "Puoi chiudere {r}?"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "closed"))
    if r < 0.88:
        user = R.choice(["Ferma {r}", "Stop {a}", "Ferma subito {r}", "Arresta {r}"])
        return (user.format(r=ref.acc, a=ref.p("a")), [action("cover.stop_cover", ref.ents)],
                R.choice([f"{cap(ref.nom)} {'si sono fermate' if ref.plural else 'si è fermata'}."
                          if ref.code in ("f", "fp") else "Fatto.",
                          f"Ho fermato {ref.acc}.", "Fatto."]))
    p, phrase = R.choice([(50, "a metà"), (30, pnum("a", 30)), (70, pnum("a", 70)),
                          (20, pnum("a", 20)), (80, pnum("a", 80)),
                          (50, pnum("a", 50)), (10, pnum("a", 10))])
    if phrase == "a metà":
        user = R.choice(["Abbassa {r} a metà", "Chiudi {r} a metà", "{R} a metà"]
                        ).format(r=ref.acc, R=cap(ref.acc))
    else:
        user = R.choice(["Porta {r} {ph}", "Imposta {r} {ph}", "{R} {ph}"]
                        ).format(r=ref.acc, R=cap(ref.acc), ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} ora {essere(ref.plural)} {pnum('a', p)}.", "Fatto."])


def i_climate(h, loc):
    ents = h.of("climate")
    if not ents:
        return None
    rooms = sorted({e.area for e in ents if e.area})
    r = R.random()
    if r < 0.75:
        t = R.choice([17, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 24, 25])
        tn = it_num(t)
        if R.random() < 0.4 and rooms:
            room = loc if loc in rooms and R.random() < 0.5 else R.choice(rooms)
            tgt = h.of("climate", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Imposta il riscaldamento {prep} a {tn} gradi",
                             f"Riscaldamento {prep} a {tn} gradi",
                             f"Porta la temperatura {prep} a {tn} gradi",
                             f"Voglio {tn} gradi {prep}",
                             f"{cap(prep)} metti {tn} gradi"])
            nom, code = f"il riscaldamento {prep}", "m"
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Imposta {r} a {t} gradi", "{R} a {t} gradi",
                             "Regola {r} su {t} gradi", "Metti {r} a {t} gradi"]
                            ).format(r=ref.acc, R=cap(ref.acc), t=tn)
            nom, code = ref.nom, ref.code
        conf = R.choice([f"{cap(nom)} ora {essere(code.endswith('p'))} "
                         f"{agree('impostato', code)} a {tn} gradi.",
                         f"Ho impostato {nom} a {tn} gradi.",
                         f"Fatto, {tn} gradi."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Accendi {r}", "Mi accendi {r}?", "Attiva {r}"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "on")
    user = R.choice(["Spegni {r}", "Mi spegni {r}?", "Disattiva {r}"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "off")


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc == e.name:
        ref = Ref(lower_first(e.name), e.gender, [e])
    if R.random() < 0.55:
        user = R.choice(["Chiudi a chiave {r}", "Blocca {r}", "Chiudi {r} a chiave",
                         "{R} è " + ref.adj("aperto") + "? Chiudi" + clitic(ref.code)
                         + " a chiave", "Puoi chiudere a chiave {r}?"])
        return (user.format(r=ref.acc, R=cap(ref.acc)), [action("lock.lock", [e])],
                confirm(ref, "locked"))
    user = R.choice(["Sblocca {r}", "Apri {r}", "Puoi sbloccare {r}?", "Sblocca {r}, per favore",
                     "Apri la serratura {di}"])
    return (user.format(r=ref.acc, di=ref.p("di")), [action("lock.unlock", [e])],
            confirm(ref, "unlocked"))


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
            "Inserisci l'allarme", "Attiva l'allarme",
            "Sto uscendo, inserisci l'allarme", "Allarme on",
            "Allarme in modalità fuori casa"])
        conf = R.choice(["L'allarme è inserito.",
                         "L'allarme ora è attivo (fuori casa)."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Allarme in modalità casa", "Inserisci l'allarme in modalità presenza",
            "Attiva l'allarme in modalità casa"])
        conf = "L'allarme è inserito in modalità casa."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Inserisci l'allarme per la notte", "Modalità notte per l'allarme",
            "Attiva l'allarme in modalità notte"])
        conf = "L'allarme è inserito in modalità notte."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Disinserisci l'allarme", "Disattiva l'allarme", "Spegni l'allarme",
            "Sono a casa, togli l'allarme", "Allarme off"])
        conf = R.choice(["L'allarme è disinserito.",
                         "L'allarme ora è disattivato."])
    if code:
        user += f", codice {code}"
    return user, [action(svc, [e], {"code": code} if code else None)], conf


def i_media(h, loc):
    ents = h.of("media_player")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    r = R.random()
    opts = [
        ("media_player.media_pause", ["Metti in pausa {r}", "Pausa su {n}",
                                      "Ferma un attimo {r}",
                                      "Metti in pausa la riproduzione su {n}"], None,
         "In pausa."),
        ("media_player.media_play", ["Riprendi la riproduzione su {n}",
                                     "Continua a riprodurre su {n}", "Play su {n}"], None,
         "La riproduzione riprende."),
        ("media_player.media_next_track", ["Brano successivo su {n}",
                                           "Passa alla canzone successiva su {n}",
                                           "Prossima canzone su {n}"], None, "Brano successivo."),
        ("media_player.volume_up", ["Alza il volume {di}", "Alza {r}"], None, "Volume alzato."),
        ("media_player.volume_down", ["Abbassa il volume {di}", "Abbassa {r}"], None,
         "Volume abbassato."),
        ("media_player.turn_off", ["Spegni {r}", "Mi spegni {r}?", "Spegni {r}, grazie"], None,
         confirm(ref, "off")),
        ("media_player.turn_on", ["Accendi {r}", "Mi accendi {r}?"], None,
         confirm(ref, "on")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Imposta il volume {di} {al}", "Volume di {n} {al}",
                         "Metti {r} {al} di volume"]).format(
            r=ref.acc, n=e.name, di=ref.p("di"), al=pnum("a", v))
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"Il volume ora è {pnum('a', v)}."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, n=e.name, di=ref.p("di")), \
        [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = R.choice([the(e.label, e.gender) if e.name.lower() == e.label else e.name,
                  "l'aspirapolvere", "il robot aspirapolvere", "il robot"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Avvia {n}", f"Fai partire {n}", f"Manda {n} a pulire",
                         "Passa l'aspirapolvere, per favore", "Avvia la pulizia"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{cap(noun_name(e))} si mette al lavoro.", "L'aspirapolvere parte."])
    if r < 0.8:
        user = R.choice([f"Manda {n} alla base", f"Riporta {n} alla base di ricarica",
                         f"Rimanda {n} a casa"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{cap(noun_name(e))} torna alla base di ricarica."
    user = R.choice([f"Metti in pausa {n}", f"Ferma {n}", f"Ferma un attimo {n}"])
    return user, [action("vacuum.pause", [e])], f"{cap(noun_name(e))} è in pausa."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Attiva la scena {e.name}", f"Scena {e.name}",
                         f"Attiva {e.name}", f"Avvia la scena {e.name}",
                         f"Metti la scena {e.name}"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"La scena {e.name} è attiva.", f"Scena {e.name} attivata.", "Fatto."])
    if e.domain == "script":
        user = R.choice([f"Esegui {e.name}", f"Avvia {e.name}", f"Fai partire {e.name}",
                         f"Script {e.name}", f"Esegui lo script {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} è in esecuzione.", f"Ho avviato {e.name}.", "Fatto."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    if (grp := h.group(domain)) is not None:
        ents = [grp]
    phrase, code, with_art = R.choice(ALL_GROUP[domain])
    whole = S.ALL_AREAS if domain in S.AREA_DOMAINS and h.group(domain) is None else None
    ref = Ref(phrase, code, ents, whole, art=with_art)
    acc, nom = ref.acc, cap(ref.nom)
    E = essere(ref.plural)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Spegni {acc}", f"Puoi spegnere {acc}?", f"Spegni {acc}, grazie",
                             "Spegni la luce ovunque", "Spegni le luci in tutta la casa"])
            return user, [action("light.turn_off", ref)], R.choice(
                [f"{nom} {E} {ref.adj('spento')}.", "Tutte le luci sono spente.", "Fatto."])
        user = R.choice([f"Accendi {acc}", f"Puoi accendere {acc}?", f"Accendi {acc}, grazie"])
        return user, [action("light.turn_on", ref)], R.choice(
            [f"{nom} {E} {ref.adj('acceso')}.", "Tutte le luci sono accese.", "Fatto."])
    if R.random() < 0.6:
        user = R.choice([f"Chiudi {acc}", f"Abbassa {acc}", f"Tira giù {acc}",
                         f"Puoi abbassare {acc}?"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{nom} sono chiuse.", "Fatto, tutto chiuso."])
    user = R.choice([f"Apri {acc}", f"Alza {acc}", f"Tira su {acc}", f"Puoi alzare {acc}?"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{nom} sono aperte.", "Fatto, tutto aperto."])


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
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


# imperatives that start a command; behind "e" they lose their capital
VERBS = ("Accendi", "Spegni", "Apri", "Chiudi", "Alza", "Abbassa", "Imposta", "Metti",
         "Avvia", "Attiva", "Disattiva", "Regola", "Porta", "Ferma", "Fai", "Riproduci",
         "Tira", "Solleva", "Arresta", "Stop", "Blocca", "Sblocca", "Colora", "Rendi",
         "Estendi", "Ritira", "Srotola", "Riavvolgi", "Manda", "Riporta", "Rimanda",
         "Esegui", "Premi", "Innaffia", "Bagna", "Passa", "Togli", "Stacca", "Inserisci",
         "Disinserisci", "Aggiungi", "Scrivi", "Segna", "Inserisci", "Spunta", "Depenna",
         "Cancella", "Rimuovi", "Ripeti", "Riprendi", "Continua", "Dai", "Aumenta",
         "Accendimi", "Spegnimi", "Sintonizza", "Voglio", "Vorrei")
POLITE = ("Puoi", "Potresti", "Mi ", "Per favore", "Ho ", "Ci serve", "Sto ", "Sono ",
          "Piove", "Ehi", "Dobbiamo", "Devo", "È ", "Dai,")


def join_commands(a, b):
    """Two command intents -> one sentence 'X e y', or None."""
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    for tail in (", per favore", ", grazie"):
        ua, ub = ua.removesuffix(tail), ub.removesuffix(tail)
    if ub.startswith(POLITE) or ua.startswith(POLITE) or "buio" in ua or "buio" in ub:
        return None
    first = ub.split(" ", 1)
    if first[0] in VERBS or first[0] in ("In", "Nel", "Nella", "Nello", "Sul", "Sulla",
                                         "Sulle", "All'esterno", "All'ingresso", "Sotto"):
        ub = lower_first(ub)
    elif first[0][:1].isupper() and first[0] not in ("Luce", "Luci"):
        return None      # a device name in front: "Tapparella Cucina a metà"
    else:
        ub = lower_first(ub)
    e = "ed" if ub[:1] in "eE" else "e"
    return f"{ua} {e} {ub}"


def _join_conf(c1, c2):
    for pre in ("Ok, ora ", "Ok, ", "Fatto, "):
        if c2.startswith(pre):
            c2 = c2[len(pre):]
            break
    if c2.split(" ", 1)[0] in ("La", "Il", "Lo", "I", "Gli", "Le", "Ho", "Ora") or \
            c2.startswith("L'"):
        c2 = lower_first(c2)
    return f"{c1.rstrip('.')} e {c2}"


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    actions = a[1] + b[1]
    conf = R.choice(["Fatto.", "Fatto, tutte e due le cose.",
                     _join_conf(a[2], b[2])
                     if a[2] != "Fatto." and b[2] != "Fatto." else "Fatto."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def state_key(e: Ent) -> str:
    d, s = e.domain, e.state
    if s == "unavailable":
        return "unavailable"
    if d in ("light", "switch", "fan", "input_boolean", "humidifier"):
        return "on" if s == "on" else "off"
    if d in ("cover", "binary_sensor", "valve"):
        if d == "binary_sensor":
            return "open" if s == "on" else "closed"
        return "open" if s == "open" else "closed"
    if d == "lock":
        return "locked" if s == "locked" else "unlocked"
    if d == "media_player":
        return {"playing": "playing", "paused": "paused", "off": "off",
                "idle": "idle"}.get(s, s)
    return s


def state_pred(e: Ent, code: str | None = None) -> str:
    """'è accesa', 'sono chiuse', 'non è chiusa a chiave', 'sta riproducendo'."""
    code = code or e.ng
    pl = code.endswith("p")
    E = essere(pl)
    k = state_key(e)
    if k == "unavailable":
        return f"non {E} {'raggiungibili' if pl else 'raggiungibile'}"
    if k in ("on", "off", "open", "closed"):
        return f"{E} {agree(STATE_WORDS[k][0], code)}"
    if k == "locked":
        return f"{E} {agree('chiuso', code)} a chiave"
    if k == "unlocked":
        return f"non {E} {agree('chiuso', code)} a chiave"
    if k == "playing":
        return "stanno riproducendo" if pl else "sta riproducendo"
    if k == "paused":
        return f"{E} in pausa"
    if k == "idle":
        return f"{E} {agree('acceso', code)}, ma non " + \
            ("stanno" if pl else "sta") + " riproducendo niente"
    return f"{E} {e.state}"


def state_word(e: Ent, code: str | None = None) -> str:
    """The adjective of the state ('accesa'), for answer_contains."""
    code = code or e.ng
    k = state_key(e)
    if k in STATE_WORDS:
        return agree(STATE_WORDS[k][0], code)
    return state_pred(e, code)


def names_code(h, ids) -> str:
    """Gender of a list of names: one name keeps its own, several are plural,
    masculine as soon as one of them is."""
    by_id = {e.eid: e for e in h.ents}
    ents = [by_id[i] for i in ids]
    if len(ents) == 1:
        return ents[0].ng
    return "mp" if any(e.ng[0] == "m" for e in ents) else "fp"


def s_question(h, loc):
    """State question -> get_state -> grounded Italian answer."""
    kinds = []
    temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"
             and e.label not in (OUT_LABEL, FC_LABEL)]
    if temps:
        kinds += ["temp"] * 2
    outs = [e for e in h.of("sensor") if e.label == OUT_LABEL]
    if outs:
        kinds += ["outdoor"] * 2
    fcs = [e for e in h.of("sensor") if e.label == FC_LABEL]
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
        user = R.choice(["Che temperatura c'è fuori?", "Quanto fa freddo fuori?",
                         "Qual è la temperatura esterna?", "Quanti gradi ci sono fuori?",
                         "Fa caldo fuori?", "Quanti gradi fa fuori?"])
        vals = [it_num(float(e.state)) for e in outs]
        if len(outs) == 1:
            ans = R.choice([f"Fuori {ci_sono_gradi(vals[0])}.",
                            f"La temperatura esterna è di {gradi(vals[0])}."])
        else:
            ans = (f"{outs[0].name} misura {gradi(vals[0])}, "
                   f"{outs[1].name} {gradi(vals[1])}.")
        ids, must = outs, vals
    elif kind == "forecast":
        if h.of("weather"):
            return None
        when = R.choice(fcs).attrs["_when"]
        ids = [e for e in fcs if e.attrs["_when"] == when]
        vals = [it_num(float(e.state)) for e in ids]
        user = R.choice([f"Quanto farà caldo {when}?", f"Quanti gradi sono previsti {when}?",
                         f"Cosa dicono le previsioni per {when}?",
                         f"Che temperatura ci sarà {when}?"])
        if len(ids) == 1:
            ans = R.choice([f"Secondo le previsioni {when} ci saranno {gradi(vals[0])}.",
                            f"{cap(when)} sono previsti {gradi(vals[0])}."])
        else:
            ans = (f"Per {when} {ids[0].name} indica {gradi(vals[0])} e "
                   f"{ids[1].name} {gradi(vals[1])}.")
        must = vals
    elif kind == "temp":
        e = R.choice(temps)
        val = it_num(float(e.state))
        if e.area is None or e.area in OUTDOOR or e.label != TEMP_LABEL:
            user = R.choice([f"Cosa indica {e.name}?", f"Quanti gradi misura {e.name}?",
                             f"Quanto segna {e.name}?"])
            ans = f"{e.name} misura {gradi(val)}."
        else:
            prep = ROOM_BY_NAME[e.area][1]
            user = R.choice([f"Che temperatura c'è {prep}?", f"Quanto fa caldo {prep}?",
                             f"Quanti gradi ci sono {prep}?", f"Temperatura {prep}?",
                             f"{cap(prep)} quanti gradi ci sono?"])
            ans = R.choice([f"{cap(prep)} {ci_sono_gradi(val)}.",
                            f"La temperatura {prep} è di {gradi(val)}."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        prep = ROOM_BY_NAME[e.area][1] if e.area else ""
        user = R.choice([f"Quanta umidità c'è {prep}?", f"Com'è l'umidità {prep}?",
                         f"Qual è l'umidità {prep}?"]).replace(" ?", "?")
        ans = f"L'umidità {prep} è {pnum('di', e.state)}.".replace("  ", " ")
        ids, must = [e], [e.state]
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = e.name.replace(f" {MOISTURE}", "")
            code = PLANT_G.get(plant, "m")
            pl = plant.lower()
            di = pa("di", pl, code)
            user = R.choice([f"Quanto è umida la terra {di}?",
                             f"Bisogna annaffiare {the(pl, code)}?",
                             f"Com'è l'umidità del terreno {di}?"])
            ans = f"L'umidità del terreno {di} è {pnum('di', e.state)}."
        else:
            user = R.choice([f"Cosa indica {e.name}?", f"A che livello è {e.name}?",
                             f"Quanto segna {e.name}?"])
            ans = f"{e.name} è {pnum('a', e.state)}."
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"Quanta corrente consuma {e.name} adesso?",
                         f"Cosa indica {e.name}?", f"Quanto segna {e.name}?"])
        val = it_num(float(e.state))
        ans = f"{e.name} indica al momento {val} watt."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"Su quanti gradi è {ref.adj('impostato')} {ref.nom}?",
                             f"Qual è la temperatura impostata {ref.p('di')}?",
                             f"A che temperatura è {ref.adj('regolato')} {ref.nom}?"])
            val = it_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {essere(ref.plural)} {ref.adj('impostato')} a {gradi(val)}."
        else:
            if e.area:
                prep = ROOM_BY_NAME[e.area][1]
                user = R.choice([f"Che temperatura c'è {prep}?",
                                 f"Quanti gradi ci sono {prep}?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None
            else:
                user = f"Che temperatura misura {ref.nom}?"
                prep = ""
            if "current_temperature" not in e.attrs:
                return None
            val = it_num(e.attrs["current_temperature"])
            ans = (f"{cap(prep)} {ci_sono_gradi(val)}." if prep
                   else f"{cap(ref.nom)} misura {gradi(val)}.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        nom, A = ref.nom, ref.adj
        E = essere(ref.plural)
        if e.domain == "cover":
            user, asked = R.choice([(f"{cap(nom)} {E} {A('aperto')}?", "open"),
                                    (f"{cap(E)} {A('aperto')} {nom}?", "open"),
                                    (f"{cap(nom)} {E} {A('chiuso')}?", "closed"),
                                    (f"{cap(nom)} {E} {A('abbassato')}?", "closed")])
        else:
            user, asked = R.choice([(f"{cap(nom)} {E} {A('acceso')}?", "on"),
                                    (f"{cap(nom)} {E} ancora {A('acceso')}?", "on"),
                                    (f"{cap(E)} {A('acceso')} {nom}?", "on"),
                                    (f"{cap(nom)} {E} {A('spento')}?", "off"),
                                    (f"{cap(E)} {A('spento')} {nom}?", "off")])
        k = state_key(e)
        word = state_word(e, ref.code)
        pred = state_pred(e, ref.code)
        ans = f"{cap(nom)} {pred}."
        if k in ("on", "off", "open", "closed") and R.random() < 0.8:
            ans = ("Sì, " if k == asked else "No, ") + nom + f" {pred}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"on": "Sì, ", "off": "No, "}[asked]
            ans = f"{yes}{nom} {E} {A('acceso')}, con la luminosità {pnum('a', pct)}."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Quali luci sono accese {prep}?",
                             f"{cap(prep)} c'è ancora una luce accesa?"])
        else:
            ents = h.of("light")
            user = R.choice(["Quali luci sono accese?", "C'è ancora qualche luce accesa?",
                             "Ci sono ancora lampade accese?", "Quali lampade sono accese?",
                             "È rimasta accesa qualche luce?"])
        q_area = room or S.ALL_AREAS
        if any(e.state == "unavailable" for e in ents):
            return None
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "Al momento non c'è nessuna luce accesa." if not room else \
                f"{cap(ROOM_BY_NAME[room][1])} al momento non c'è nessuna luce accesa."
            must = ["nessuna"]
        elif len(on) == 1:
            ans = f"È {agree('acceso', on[0].ng)} solo {on[0].name}."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"Sono accese: {', '.join(names[:-1])} e {names[-1]}."
            must = names
        ids = ents
    elif kind == "windows":
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") == "window"]
        if not ents:
            return None
        user, yes_if_open = R.choice([("Le finestre sono tutte chiuse?", False),
                                      ("C'è ancora una finestra aperta?", True),
                                      ("Quali finestre sono aperte?", None),
                                      ("Sono chiuse le finestre?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "No, tutte le finestre sono chiuse.",
                   False: "Sì, tutte le finestre sono chiuse.",
                   None: "Tutte le finestre sono chiuse."}[yes_if_open]
            must = ["chiuse"]
        else:
            names = [e.name for e in opn]
            ans = ("È aperta: " + names[0] + "." if len(names) == 1 else
                   "Sono aperte: " + ", ".join(names[:-1]) + " e " + names[-1] + ".")
            if yes_if_open is not None:
                ans = ("Sì. " if yes_if_open else "No. ") + ans
            must = names
        ids = ents
    elif kind == "door":
        doors = [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]
        e = R.choice(doors)
        same = [d for d in doors if d.label == e.label]
        if e.area and (len(same) > 1 or R.random() < 0.5):
            what = f"{the(e.label, 'f')} {ROOM_BY_NAME[e.area][1]}"
        elif len(same) == 1:
            what = the(e.label, "f")
        else:
            return None
        if e.state == "unavailable":
            return None
        opn = e.state == "on"
        user, ask_open = R.choice([(f"{cap(what)} è aperta?", True),
                                   (f"{cap(what)} è chiusa?", False),
                                   (f"È chiusa {what}?", False),
                                   (f"È rimasta aperta {what}?", True)])
        word = agree("aperto" if opn else "chiuso", e.ng)
        ans = f"{'Sì' if opn == ask_open else 'No'}, {e.name} è {word}."
        ids, must = [e], [word]
    elif kind == "car":
        rng = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"])
        car = rng.name.rsplit(" ", 1)[0]
        tc = the(car, "f")
        soc = next((e for e in h.of("sensor") if e.name == f"{car} Batteria"), None)
        if soc is not None and R.random() < 0.45:
            e = soc
            user = R.choice([f"Quanto è carica {tc}?", f"A che livello di carica è {tc}?",
                             f"{cap(tc)} è già carica?"])
            ans = f"{cap(tc)} è carica {pnum('a', e.state)}."
        else:
            e = rng
            user = R.choice([f"Quanti chilometri può ancora fare {tc}?",
                             f"Quanta autonomia ha ancora {tc}?",
                             f"Che autonomia ha ancora {tc}?"])
            ans = f"{cap(tc)} ha ancora {e.state} chilometri di autonomia."
        ids, must = [e], [e.state]
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = the(lower_first(e.name), e.gender)
        A = lambda w: agree(w, e.gender)  # noqa: E731
        user = R.choice([f"{cap(nom)} è {A('chiuso')} a chiave?", f"{cap(nom)} è {A('chiuso')}?",
                         f"Ho chiuso a chiave {nom}?"])
        pred = state_pred(e, e.gender)
        ans = (f"Sì, {nom} {pred}." if e.state == "locked" else f"No, {nom} {pred}.")
        ids, must = [e], [pred]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        if e.state == "unavailable":
            return None
        user = R.choice(["L'allarme è inserito?", "L'allarme è attivo?",
                         "In che stato è l'allarme?"])
        txt = {"disarmed": "L'allarme è disinserito.",
               "armed_away": "L'allarme è inserito (fuori casa).",
               "armed_home": "L'allarme è inserito in modalità casa.",
               "armed_night": "L'allarme è inserito in modalità notte."}[e.state]
        ans, ids, must = txt, [e], ["disinserito" if e.state == "disarmed" else "inserito"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"Cosa sta suonando su {e.name}?", f"Cosa c'è su {e.name}?",
                         f"Cosa riproduce {e.name}?"])
        t = e.attrs.get("media_title", "")
        a = e.attrs.get("media_artist")
        ans = f"Su {e.name} al momento c'è {t}" + (f" di {a}." if a else ".") if t else \
            f"Su {e.name} è in riproduzione qualcosa, ma Home Assistant non indica il titolo."
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
    """'La lavatrice ha finito?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = the(app.lower(), g)
    user = R.choice([f"{cap(nom)} ha finito?", f"{cap(nom)} è ancora in funzione?",
                     f"Cosa sta facendo {nom}?", f"{cap(nom)} ha già finito?",
                     f"Qual è lo stato {pa('di', app.lower(), g)}?"])
    phrase, key = app_answer(kind, g)
    ans = R.choice([f"{cap(nom)} {phrase}.", f"Secondo Home Assistant {nom} {phrase}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [key]}


def s_relative_climate(h, loc):
    """'Alza un po' la temperatura in bagno' -> get_state -> set_temperature +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    prep = ROOM_BY_NAME[e.area][1]
    up = R.random() < 0.6
    user = R.choice([f"Alza un po' la temperatura {prep}", f"{cap(prep)} ho freddo",
                     f"Alza il riscaldamento {prep} di un grado",
                     f"Riscaldamento {prep} un grado in più"] if up else
                    [f"Abbassa un po' la temperatura {prep}", f"{cap(prep)} fa troppo caldo",
                     f"Abbassa il riscaldamento {prep} di un grado",
                     f"Riscaldamento {prep} un grado in meno"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, ok_result()),
             {"role": "assistant",
              "content": f"Ho portato il riscaldamento {prep} da {it_num(e.attrs['temperature'])} "
                         f"a {gradi(it_num(new))}."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


MISSING_TYPES = [
    # domain, phrase, gender, "nessun ...", "Un ... c'è", verb on, verb off,
    # services (on, off), states (on, off), label filter
    ("light", "luce", "f", "nessuna luce", "Una luce c'è", "Accendi", "Spegni",
     ("light.turn_on", "light.turn_off"), ("on", "off"), None),
    ("light", "lampada", "f", "nessuna lampada", "Una lampada c'è", "Accendi", "Spegni",
     ("light.turn_on", "light.turn_off"), ("on", "off"), None),
    ("cover", "tapparelle", "fp", "nessuna tapparella", "Le tapparelle ci sono", "Alza",
     "Abbassa", ("cover.open_cover", "cover.close_cover"), ("open", "closed"), "plain"),
    ("climate", "riscaldamento", "m", "nessun riscaldamento", "Un riscaldamento c'è",
     "Accendi", "Spegni", ("climate.turn_on", "climate.turn_off"), ("on", "off"), None),
    ("media_player", "televisore", "m", "nessun televisore", "Un televisore c'è",
     "Accendi", "Spegni", ("media_player.turn_on", "media_player.turn_off"), ("on", "off"),
     TV_LABEL),
    ("fan", "ventilatore", "m", "nessun ventilatore", "Un ventilatore c'è", "Accendi",
     "Spegni", ("fan.turn_on", "fan.turn_off"), ("on", "off"), None),
]
THAT = {"m": "quello", "f": "quella", "mp": "quelli", "fp": "quelle"}


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    mt = FORCE.get("mtype")
    dom, noun, g, none, exists, von, voff, svcs, states, filt = R.choice(
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
    if dom == "media_player" and R.random() < 0.6 and \
            not any(e.area == room for e in h.of(dom)):
        if R.random() < 0.6:
            h.add_voice_media(room)
        else:
            mt = R.choice([t for t in TYPES["media_player"] if t[0] != TV_LABEL])
            h.add("media_player", room, mt[0], mt[1], mt[2], state=R.choice(["idle", "off"]))
    turn_on = R.random() < 0.5
    verb = von if turn_on else voff
    phrase = the(noun, g)
    user = R.choice([f"{verb} {phrase} {prep}", f"{cap(prep)} {lower_first(verb)} {phrase}",
                     f"Mi {lower_first(verb)} {phrase} {prep}?",
                     f"{verb} {phrase} {prep}, per favore"])
    others = [e for e in ents if e.area and e.area != room]
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        ans = R.choice([f"{cap(prep)} non c'è {none}. {exists} {oprep}: {alt.name}.",
                        f"{cap(prep)} non trovo {none}. Intendi {alt.name}{where}?",
                        f"{cap(prep)} non ho {none} nella lista dei dispositivi. "
                        f"{cap(oprep)} {c_e(alt.ng.endswith('p'))} {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"{cap(prep)} non trovo {none} che io possa controllare.",
                        f"{cap(prep)} non c'è {none} nella lista dei dispositivi.",
                        f"{cap(prep)} non ho trovato {none} da comandare."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        that = THAT[alt.gender if alt.label else g]
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Allora {that} {oprep}", f"Sì, intendo {that}", f"Ok, allora {that}",
             f"Sì, {alt.name}", f"Va bene, allora {oprep}"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        st = states[0] if turn_on else states[1]
        word = agree(STATE_WORDS[st][0], alt.ng)
        E = essere(alt.ng.endswith("p"))
        msgs += [call, tool_msg(cid, ok_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"{alt.name}{where} ora {E} {word}.",
                      f"Fatto, {alt.name}{where} {E} {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        pushes = ["Dai, fallo lo stesso!", f"Sì, {lower_first(verb)}!", "Ma c'è!",
                  "Fallo comunque", f"Sì, {prep}!", f"No, intendo {prep}",
                  "Ma come no?", "Sì che c'è", "Fallo e basta"]
        refusals = [
            f"{cap(prep)} nella tua lista dei dispositivi non c'è {none}. Verifica che il "
            f"dispositivo sia esposto ad Assist.",
            f"{cap(prep)} davvero non vedo {none} che io possa controllare.",
            f"{cap(prep)} non c'è {none} nella mia lista dei dispositivi, quindi lì non posso "
            f"comandare niente."]
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
        if dev.lower() in labels or (dev == "tosaerba" and h.of("lawn_mower")):
            return None
        td = the(dev, g)
        pl = g.endswith("p")
        user = R.choice([f"Accendi {td}", f"Spegni {td}", f"Mi accendi {td}?",
                         f"{cap(td)} {essere(pl)} {agree('acceso', g)}?", f"Avvia {td}"])
        ans = R.choice([f"Non trovo nessun dispositivo chiamato {dev} nella tua casa.",
                        f"Non conosco nessun dispositivo «{dev}». Purtroppo non posso "
                        f"controllarlo.",
                        f"{cap(td)} non {c_e(pl)} nella lista dei dispositivi."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != BATH2
                   and not (r_[0] == BATH and BATH2 in h.rooms)]
        room = R.choice(missing)
        prep = ROOM_BY_NAME[room][1]
        what = R.choice([("la luce", "nessuna luce", "Accendi", "Spegni"),
                         ("le tapparelle", "nessuna tapparella", "Alza", "Abbassa"),
                         ("il riscaldamento", "nessun riscaldamento", "Accendi", "Spegni")])
        user = R.choice([f"{what[2]} {what[0]} {prep}",
                         f"{what[3]} {what[0]} {prep}",
                         f"{cap(prep)} {lower_first(what[3])} {what[0]}"])
        ans = R.choice([f"{cap(prep)} non trovo {what[1]} che io possa controllare.",
                        f"{cap(prep)} non ho trovato {what[1]} nella lista dei dispositivi."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


WEATHER_CHAT = "Che tempo farà domani?"
CHAT = [
    (["Grazie", "Grazie mille", "Ti ringrazio", "Perfetto, grazie"],
     ["Prego.", "Di niente.", "Figurati."]),
    (["Ciao", "Ehi", "Buongiorno", "Buonasera"],
     ["Ciao, come posso aiutarti?", "Ciao. Cosa posso fare per te?"]),
    (["Cosa sai fare?", "In cosa puoi aiutarmi?", "Cosa puoi controllare?"],
     ["Posso comandare luci, tapparelle, riscaldamento e altri dispositivi di casa tua e "
      "dirti in che stato sono o cosa misurano i sensori. Inoltre metto la musica, imposto "
      "timer, gestisco le tue liste, ti dico l'ora e il meteo e faccio annunci in casa.",
      "Comando i tuoi dispositivi, metto la musica, imposto timer, aggiungo cose alla lista "
      "della spesa, ti dico l'ora e il meteo e faccio annunci in tutta la casa."]),
    (["Chi sei?"], ["Sono il tuo assistente vocale per la casa intelligente."]),
    (["Raccontami una barzelletta"],
     ["Perché la lampadina è andata dallo psicologo? Perché si sentiva sempre spenta.",
      "Cosa dice la tapparella alla finestra? Ti tengo d'occhio."]),
    ([WEATHER_CHAT],
     ["Purtroppo non ho le previsioni del tempo. Posso però dirti cosa misurano adesso i "
      "tuoi sensori."]),
    (["Buonanotte"], ["Buonanotte. Vuoi che spenga ancora qualcosa?", "Buonanotte."]),
    # HassNevermind: nothing to do
    (["Lascia perdere", "Annulla", "Niente", "Fa niente", "Lascia stare", "Ah, niente",
      "Non importa", "Lascia perdere, dai", "No, niente", "Non ho detto niente",
      "Va bene così", "No, lascia stare", "Niente, scusa"],
     ["Ok.", "Va bene.", "Ok, allora niente."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == WEATHER_CHAT and (
            h.of("weather") or any(e.label == FC_LABEL for e in h.of("sensor"))):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


def _unreachable(e: Ent):
    pl = e.ng.endswith("p")
    n = noun_name(e)
    return R.choice([
        f"Purtroppo non ha funzionato: {n} al momento non {'rispondono' if pl else 'risponde'}.",
        f"Si è verificato un errore: {n} non {essere(pl)} "
        f"{'raggiungibili' if pl else 'raggiungibile'}.",
        f"{cap(n)} al momento non {essere(pl)} {'raggiungibili' if pl else 'raggiungibile'}, "
        f"quindi non ho potuto eseguire il comando."])


def _single_command(h, loc):
    for _ in range(10):
        out = pick_command(h, loc)
        if out and len(out[1]) == 1 and len(out[1][0]["entity_id"]) == 1:
            return out
    return None


def s_error(h, loc):
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
    e = next(x for x in h.ents if x.eid == eid)
    name = e.name
    pl = e.ng.endswith("p")
    err = R.choice([f"Entity {eid} is unavailable",
                    f"Timeout while calling {actions[0]['service']} for {eid}",
                    f"{eid}: device did not respond"])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, S.control_result([err])),
             {"role": "assistant", "content": _unreachable(e)}]
    meta = {"cat": "error", "gold": [["control", {"actions": actions}]]}
    if r >= 0.65 or len(actions) != 1:
        return msgs, meta
    raggi = "raggiungibili" if pl else "raggiungibile"
    if R.random() < 0.5:
        msgs.append({"role": "user", "content": voice(R.choice(
            ["Riprova", "Prova di nuovo", "Riprova per favore", "Prova ancora una volta"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": R.choice([
                     f"{name} non {essere(pl)} ancora {raggi}. Controlla che il dispositivo "
                     f"sia alimentato e connesso alla rete.",
                     f"Anche il secondo tentativo non ha funzionato, {name} non "
                     f"{'rispondono' if pl else 'risponde'}."])}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        word = agree("aperto" if actions[0]["service"].endswith(("open_cover", "open_valve"))
                     else "acceso", e.ng)
        E = essere(pl)
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Adesso {E} {word}?", f"Non {E} {word}, vero?", "Ha funzionato?",
             f"Allora, {E} {word}?"]
            if onoff else ["Ha funzionato?", "Allora, ha funzionato?", "È vero?"]))})
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": R.choice([
                     f"No. {name} non {E} {raggi}, quindi non ho potuto comandar"
                     f"{clitic(e.ng)}.",
                     f"Secondo Home Assistant {name} al momento non {E} {raggi}. "
                     f"Non è stato comandato niente."])}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def _not_reachable(h, ids):
    nd = _names(h, ids)
    pl = len(ids) > 1 or names_code(h, ids).endswith("p")
    return nd, pl, f"{nd} al momento non {essere(pl)} {'raggiungibili' if pl else 'raggiungibile'}."


def _error_room(h, loc):
    """Some lights of a room are unreachable, the others work."""
    rooms = [r for r in h.rooms if len([e for e in h.of("light", r)
                                         if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    named_rooms = [r for r in rooms if any(e.name == r for e in h.of("light", r))]
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
    nd, dpl, down_txt = _not_reachable(h, [e.eid for e in down])
    msgs = base_msgs(h, loc)
    if named_q or R.random() < 0.5:
        user, kind = R.choice([(f"La luce {prep} è accesa?", "yn"),
                               (f"{cap(prep)} c'è ancora una luce accesa?", "yn"),
                               (f"Quali luci sono accese {prep}?", "which"),
                               (f"Le lampade {prep} sono spente?", "yn_off")])
        on = [e for e in up if e.state == "on"]
        off = [e for e in up if e.state != "on"]
        parts = []
        if on:
            c = names_code(h, [e.eid for e in on]) if len(on) == 1 else \
                ("mp" if any(e.ng[0] == "m" for e in on) else "fp")
            parts.append(f"{_names(h, [e.eid for e in on])} {essere(len(on) > 1 or c.endswith('p'))} "
                         f"{agree('acceso', c)}.")
        if off and (not on or R.random() < 0.5):
            c = names_code(h, [e.eid for e in off])
            parts.append(f"{_names(h, [e.eid for e in off])} {essere(c.endswith('p'))} "
                         f"{agree('spento', c)}.")
        if not up:
            ans = f"{cap(down_txt)} Per questo non posso dire se lì c'è una luce accesa."
        else:
            if kind == "yn":
                parts[0] = ("Sì, " if on else "No, ") + parts[0]
            elif kind == "yn_off":
                parts[0] = ("No, " if on else "Sì, ") + parts[0]
            ans = " ".join(parts) + " " + down_txt
        args = {"domain": "light", "area": room, "entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "error", "gold": [["get_state", args]]}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    user = R.choice([f"Accendi la luce {prep}", f"Accendi le luci {prep}",
                     f"Mi accendi la luce {prep}?"] if on_cmd else
                    [f"Spegni la luce {prep}", f"Spegni le luci {prep}",
                     f"Mi spegni la luce {prep}?"])
    actions = [action(svc, ents, area=room)]
    done = [e.eid for e in up]
    part = "acceso" if on_cmd else "spento"
    if up:
        did = _names(h, done)
        c = names_code(h, done)
        upl = c.endswith("p")
        ans = R.choice([
            f"Ho {part} {did}. {down_txt}",
            f"Ha funzionato solo in parte: {did} {essere(upl)} {agree(part, c)}, ma {nd} al "
            f"momento non {essere(dpl)} {'raggiungibili' if dpl else 'raggiungibile'}.",
            f"{down_txt} Il resto l'ho {part}: {did}.",
        ])
    else:
        ans = f"{down_txt} Per questo non ho potuto {'accendere' if on_cmd else 'spegnere'} " \
              f"la luce {prep}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, S.control_result(
                 [f"Entity {', '.join(e.eid for e in down)} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _named_room_light(h, loc):
    """A light carries the room's name ("Cucina") next to other lights there.
    "la luce in cucina" means all lights of the room; only the bare name
    ("Accendi Cucina") means that single lamp."""
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
        user, kind = R.choice([(f"La luce {prep} è accesa?", "yn"),
                               (f"{cap(prep)} c'è una luce accesa?", "yn"),
                               (f"{cap(prep)} c'è ancora la luce accesa?", "yn"),
                               (f"Quali luci sono accese {prep}?", "which")])
        on = [e for e in ents if e.state == "on"]
        off = [e for e in ents if e.state != "on"]

        def said(group, word):
            c = names_code(h, [e.eid for e in group])
            return f"{_names(h, [e.eid for e in group])} {essere(c.endswith('p'))} " \
                   f"{agree(word, c)}."
        if on:
            ans = said(on, "acceso")
            if off and R.random() < 0.6:
                ans += " " + said(off, "spento")
            if kind == "yn":
                ans = "Sì, " + ans
        elif kind == "yn":
            ans = R.choice([f"No, {said(off, 'spento')}",
                            f"No, {prep} non c'è nessuna luce accesa."])
        else:
            ans = f"{cap(prep)} al momento non c'è nessuna luce accesa."
        args = {"domain": "light", "area": room, "entity_ids": [e.eid for e in ents]}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        must = [e.name for e in on] if on else ["No" if kind == "yn" else "nessuna"]
        return msgs, {"cat": "question", "gold": [["get_state", args]],
                      "answer_contains": must}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    st = "on" if on_cmd else "off"
    if r < 0.8:
        plural = R.random() < 0.3
        noun = "le luci" if plural else "la luce"
        user = R.choice([f"Accendi {noun} {prep}", f"{cap(prep)} accendi {noun}",
                         f"Mi accendi {noun} {prep}?", f"Puoi accendere {noun} {prep}?"]
                        if on_cmd else
                        [f"Spegni {noun} {prep}", f"{cap(prep)} spegni {noun}",
                         f"Mi spegni {noun} {prep}?", f"Puoi spegnere {noun} {prep}?"])
        targets = Ref(f"{noun.split()[1]} {prep}", "fp" if plural else "f", ents, room)
        conf = confirm(targets, st)
    else:
        user = R.choice([f"Accendi {room}", f"Puoi accendere {room}?", f"Accendi {room}, grazie",
                         f"Accendi la lampada {room}"] if on_cmd else
                        [f"Spegni {room}", f"Puoi spegnere {room}?", f"Spegni {room}, grazie",
                         f"Spegni la lampada {room}"])
        targets = [named]
        conf = confirm(Ref(room, named.ng, targets, art=False), st)
    actions = [action(svc, targets)]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, ok_result()), {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_no_areas(h, loc):
    """Home Assistant without any areas: a room cannot be resolved."""
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
            user = R.choice([f"Accendi la luce {prep}", f"Spegni la luce {prep}",
                             f"La luce {prep} è accesa?", f"Quali luci sono accese {prep}?",
                             f"Accendi le luci {prep}", f"{cap(prep)} spegni la luce"])
            what = "luci"
        else:
            user = R.choice([f"Abbassa le tapparelle {prep}", f"Alza le tapparelle {prep}",
                             f"Le tapparelle {prep} sono chiuse?", f"Chiudi le tapparelle {prep}"])
            what = "tapparelle"
        ans = R.choice([
            f"Nel tuo Home Assistant non ci sono aree configurate, quindi non so quali {what} "
            f"si trovano {prep}. Crea le stanze nelle impostazioni, alla voce «Aree, etichette "
            f"e zone», e assegna loro i dispositivi.",
            f"Non riesco ad abbinare niente a «{prep}», perché in Home Assistant non ci sono "
            f"aree. Configura le stanze nelle impostazioni, alla voce «Aree, etichette e zone», "
            f"e assegna i dispositivi."])
        if named:
            ans += f" Nel frattempo puoi anche chiamare un dispositivo per nome, per esempio " \
                   f"«{named[0].name}»."
        msgs += [{"role": "user", "content": voice(user)}, {"role": "assistant", "content": ans}]
        return msgs, {"cat": "unknown", "gold": [], "no_history": True}
    on = R.random() < 0.5
    svc = "light.turn_on" if on else "light.turn_off"
    part = "acceso" if on else "spento"
    if r < 0.85 or h.group("light") is not None:
        e = R.choice(lights)
        user = R.choice([f"Accendi {e.name}", f"Mi accendi {e.name}?", f"Accendi {e.name}, grazie"]
                        if on else
                        [f"Spegni {e.name}", f"Mi spegni {e.name}?", f"Spegni {e.name}, grazie"])
        actions = [action(svc, [e])]
        conf = R.choice([f"{e.name} ora {essere(e.ng.endswith('p'))} {agree(part, e.ng)}.",
                         f"Ho {part} {e.name}."])
    else:
        user = R.choice(["Accendi tutte le luci", "Accendi tutte le lampade"] if on
                        else ["Spegni tutte le luci", "Spegni la luce ovunque"])
        actions = [action(svc, lights, area=S.ALL_AREAS)]
        conf = f"Tutte le luci sono {agree(part, 'fp')}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]],
                  "no_history": True}


def partial_answer(h, ok_actions, failed: Ent, conf=None):
    """Both outcomes, always: what worked and what did not."""
    pl = failed.ng.endswith("p")
    fn = noun_name(failed)
    down = f"{fn} al momento non {essere(pl)} {'raggiungibili' if pl else 'raggiungibile'}"
    if any(len(a["entity_id"]) > 3 for a in ok_actions):
        if conf and conf != "Fatto.":
            return R.choice([f"{conf} Però {down}.",
                             f"{cap(down)}. Per il resto: {conf}"])
        n = sum(len(a["entity_id"]) for a in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = PARTICIPLE.get(svc, "azionato {w}").format(
            w=f"{n} dispositivi", di=f"di {n} dispositivi", su=f"su {n} dispositivi",
            a=f"a {n} dispositivi", in_=f"in {n} dispositivi", da=f"da {n} dispositivi")
    else:
        did = describe_all(h, ok_actions)
    return R.choice([
        f"Ho {did}. Però {down}.",
        f"Ha funzionato solo in parte: ho {did}, ma {down}.",
        f"{cap(down)}. Il resto l'ho fatto: ho {did}.",
    ])


def _error_partial(h, loc):
    a = _single_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    eid = a[1][0]["entity_id"][0]
    failed = next(e for e in h.ents if e.eid == eid)
    actions = a[1] + b[1]
    done = [i for x in b[1] for i in x["entity_id"]]
    ans = partial_answer(h, b[1], failed, b[2])
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
    pl = e.ng.endswith("p")
    E = essere(pl)
    user = R.choice([f"{e.name} {E} {agree('acceso', e.ng)}?", f"Qual è lo stato di {e.name}?",
                     f"Cosa fa {e.name}?", f"{cap(E)} {agree('acceso', e.ng)} {e.name}?"])
    args = {"entity_ids": [e.eid]}
    raggi = "raggiungibili" if pl else "raggiungibile"
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": R.choice([
                 f"{e.name} al momento non {E} {raggi}, quindi non conosco "
                 f"{'il loro' if pl else 'il suo'} stato attuale.",
                 f"Non posso dirlo: {e.name} al momento non {E} {raggi}."])}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


def s_followup(h, loc):
    """Two-turn: command, then 'e ora spegnila' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    p1, p2 = ROOM_BY_NAME[r1][1], ROOM_BY_NAME[r2][1]
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Accendi la luce {p1}",
                                                            f"Mi accendi la luce {p1}?"]))}]
    a1 = [action("light.turn_on", e1, area=r1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": f"La luce {p1} è accesa."}]
    k = R.random()
    if k < 0.15:
        u2 = R.choice(["No, non è vero", "È davvero accesa?", "Sicuro?",
                       "Qui è ancora buio"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        q = {"domain": "light", "area": r1, "entity_ids": [e.eid for e in e1]}
        c2, cid2 = tool_call("get_state", q)
        ans = (f"Secondo Home Assistant la luce {p1} è accesa." if really_on else
               f"Secondo Home Assistant la luce {p1} è ancora spenta. Vuoi che provi ad "
               f"accenderla di nuovo?")
        if u2 == "È davvero accesa?":
            ans = ("Sì. " if really_on else "No. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", q]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["E ora spegnila", "Rispegnila", "Anzi, spegnila di nuovo"]), \
            [action("light.turn_off", e1, area=r1)], f"La luce {p1} è di nuovo spenta."
    elif k < 0.72:
        u2, a2, ans = R.choice([f"E {p2}?", f"Anche {p2}", f"E anche {p2}, per favore"]), \
            [action("light.turn_on", e2, area=r2)], f"Anche la luce {p2} è accesa."
    else:
        u2, a2, ans = R.choice(["Un po' meno luminosa", "Un po' più luminosa",
                                "Al 30 percento"]), None, None
        if "meno" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20}, area=r1)]
            ans = f"Ora la luce {p1} è meno luminosa."
        elif "più" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20}, area=r1)]
            ans = f"Ora la luce {p1} è più luminosa."
        else:
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30}, area=r1)]
            ans = f"Ora la luce {p1} è al 30 percento."
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
# "Ho <...>": {w} the devices, {di}/{su}/{a}/{in_}/{da} with a preposition
PARTICIPLE = {"turn_on": "acceso {w}", "turn_off": "spento {w}",
              "open_cover": "aperto {w}", "close_cover": "chiuso {w}",
              "stop_cover": "fermato {w}", "lock": "chiuso a chiave {w}",
              "unlock": "sbloccato {w}",
              "media_play": "ripreso la riproduzione {su}", "media_pause": "messo in pausa {w}",
              "media_stop": "fermato la riproduzione {su}",
              "media_next_track": "messo il brano successivo {su}",
              "media_previous_track": "messo il brano precedente {su}",
              "volume_up": "alzato il volume {di}", "volume_down": "abbassato il volume {di}",
              "start": "avviato {w}", "pause": "messo in pausa {w}",
              "return_to_base": "rimandato {w} alla base",
              "start_mowing": "avviato {w}", "dock": "rimandato {w} alla base",
              "open_valve": "aperto {w}", "close_valve": "chiuso {w}", "press": "attivato {w}",
              "alarm_arm_home": "inserito {w} in modalità casa", "alarm_arm_away": "inserito {w}",
              "alarm_arm_night": "inserito {w} in modalità notte",
              "alarm_disarm": "disinserito {w}",
}
# participle for "Quale luce hai ...?"
PART_Q = {"turn_on": "acceso", "turn_off": "spento", "open_cover": "aperto",
          "close_cover": "chiuso", "stop_cover": "fermato", "media_pause": "messo in pausa",
          "media_play": "fatto ripartire", "volume_up": "alzato", "volume_down": "abbassato"}
DOMAIN_NOUN = {"light": ("Quale luce", "Quali luci"),
               "cover": ("Quale tapparella", "Quali tapparelle"),
               "climate": ("Quale termostato", "Quali termostati"),
               "switch": ("Quale dispositivo", "Quali dispositivi"),
               "media_player": ("Quale altoparlante", "Quali dispositivi"),
               "fan": ("Quale ventilatore", "Quali ventilatori")}
# filled by assist.py: to-do list name -> "della spesa"; item -> gender code;
# describe() wording for the assist services: fn(h, action, who) -> str | None
DESCRIBE_HOOKS: list = []
LIST_TAILS: dict = {}
ITEM_G: dict = {}


def the_item(item: str, quote=True) -> str:
    code = ITEM_G.get(item, "m")
    if code == "inf":
        return f"«{item}»" if quote else item
    return the(item, code)


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
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " e " + ns[-1]


def describe(h, a, by_name=False):
    """'acceso Luce Corridoio' for one action (after 'Ho')."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if a.get("area") == S.ALL_AREAS and not by_name:
        who = Ref("tutte le luci", "fp", [], art=False)
    elif a.get("area") and not by_name:
        who = Ref(f"luce {ROOM_BY_NAME[a['area']][1]}", "f", [])
    elif dom == "todo" and len(a["entity_id"]) == 1:
        name = next(e.name for e in h.ents if e.eid == a["entity_id"][0])
        who = Ref(f"lista {LIST_TAILS[name]}", "f", []) if name in LIST_TAILS \
            else Ref(name, "f", [], art=False)
    elif len(a["entity_id"]) == 1:
        e = next(x for x in h.ents if x.eid == a["entity_id"][0])
        who = Ref(lower_first(e.name), e.ng, []) if e.name in COMMON_NAMES \
            else Ref(e.name, e.ng, [], art=False)
    else:
        ids = a["entity_id"]
        who = Ref(_names(h, ids), names_code(h, ids), [], art=False)
    w = who.acc
    for hook in DESCRIBE_HOOKS:
        out = hook(h, a, who)
        if out is not None:
            return out
    if "brightness_pct" in data:
        return f"impostato {w} {pnum('a', data['brightness_pct'])}"
    if "brightness_step_pct" in data:
        return f"{'aumentato' if data['brightness_step_pct'] > 0 else 'abbassato'} " \
               f"la luminosità {who.p('di')}"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"cambiato il colore {who.p('di')}"
    if svc == "set_temperature":
        return f"impostato {w} a {gradi(it_num(data['temperature']))}"
    if svc == "set_cover_position":
        return f"portato {w} {pnum('a', data['position'])}"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"impostato {w} {pnum('a', data.get('percentage'))}"
    if svc == "volume_set":
        return f"impostato il volume {who.p('di')} {pnum('a', round(data['volume_level'] * 100))}"
    if dom == "scene":
        return f"attivato la scena {w}"
    if dom == "script":
        return f"avviato {w}"
    return PARTICIPLE.get(svc, "azionato {w}").format(
        w=w, di=who.p("di"), su=who.p("su"), a=who.p("a"), in_=who.p("in"), da=who.p("da"))


def merge_actions(actions):
    """Consecutive items for the same list become one action, so the answer
    says 'aggiunto latte e uova alla lista della spesa'."""
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
    """'acceso X, impostato Y e aggiunto Z' for several actions (after 'Ho')."""
    parts = [describe(h, a) for a in merge_actions(actions)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " e " + parts[-1]


class Conv:
    """Builds one conversation turn by turn and tracks what really happened."""

    def __init__(self, h, loc):
        self.h, self.loc = h, loc
        self.msgs = base_msgs(h, loc)
        self.done: list[dict] = []
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
        b = pick_command(c.h, c.loc)
        user = join_commands(out, b) if b else None
        eid = out[1][0]["entity_id"][0]
        if user and eid not in [i for x in b[1] for i in x["entity_id"]]:
            e = next(x for x in c.h.ents if x.eid == eid)
            e.state = "unavailable"
            done = [i for x in b[1] for i in x["entity_id"]]
            c.user(user)
            c.control(out[1] + b[1], partial_answer(c.h, b[1], e, b[2]),
                      errors=[f"Entity {eid} is unavailable"], done=done)
            return True
    user, actions, _ = out
    eid = actions[0]["entity_id"][0]
    e = next(x for x in c.h.ents if x.eid == eid)
    e.state = "unavailable"
    c.user(user)
    c.control(actions, _unreachable(e), errors=[f"Entity {eid} is unavailable"])
    c.last = None
    return True


def turn_unknown(c):
    out = s_missing_type(c.h, c.loc) if R.random() < 0.5 else s_unknown(c.h, c.loc)
    if not out or len(out[0]) != 3:
        return False
    c.msgs += out[0][1:]
    c.gold_last = []
    return True


def turn_which(c):
    """'Quale luce hai spento?' -> answered from the history."""
    if not c.last:
        return False
    a = c.last[-1]
    dom, svc = a["service"].split(".", 1)
    ids = a["entity_id"]
    if dom in DOMAIN_NOUN and svc in PART_Q and not a.get("data"):
        sg, pl = DOMAIN_NOUN[dom]
        q = (pl if len(ids) > 1 else sg) + f" hai {PART_Q[svc]}?"
        user = R.choice([q, q, "Cosa hai appena fatto?", "Che dispositivo era?"])
    else:
        user = R.choice(["Cosa hai appena fatto?", "Cosa hai fatto esattamente?"])
    c.user(user)
    parts = [describe(c.h, x, by_name=True) for x in c.last]
    c.say(R.choice(["Ho ", "Poco fa ho "]) + " e ".join(parts) + ".")
    return True


def turn_summary(c):
    """'Cosa hai fatto finora?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    by_dom = {}
    for x in c.done:
        by_dom.setdefault(x["service"].split(".")[0], []).append(x)
    multi = [d for d, xs in by_dom.items() if len(xs) >= 2 and d in SUMMARY_ASK]
    if multi and R.random() < 0.6:
        d = R.choice(multi)
        c.user(R.choice(SUMMARY_ASK[d]))
        parts = [describe(c.h, x) for x in merge_actions(by_dom[d])]
    else:
        c.user(R.choice(["Cosa hai fatto finora?", "Che cosa hai comandato finora?",
                         "Riassumimi cosa hai fatto"]))
        parts = [describe(c.h, x) for x in merge_actions(c.done)]
    c.say("Ho " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " e " + parts[-1]) + ".")
    return True


SUMMARY_ASK = {
    "climate": ["Quali termostati hai cambiato?", "Cosa hai cambiato al riscaldamento?",
                "Quali riscaldamenti hai regolato?", "Quali termostati hai impostato?"],
    "light": ["Quali luci hai comandato?", "Cosa hai fatto con le luci?",
              "Quali lampade hai toccato?"],
    "cover": ["Quali tapparelle hai mosso?", "Quali persiane hai azionato?",
              "Cosa hai fatto con le tapparelle?"],
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
    c.user(R.choice(["Annulla l'ultima cosa", "Rimetti com'era", "Torna come prima",
                     "Annulla quello che hai fatto"] +
                    (["E ora rispegni", "Spegni di nuovo"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    (["Riaccendi"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "Ok, ho " + describe_all(c.h, inv) + ".")
    return True


def turn_correct(c):
    """'No, intendevo in camera da letto' -> undo + the right device."""
    if not c.last or len(c.last) != 1:
        return False
    a = c.last[0]
    dom, svc = a["service"].split(".", 1)
    if svc not in ("turn_on", "turn_off", "open_cover", "close_cover") or a.get("data") \
            or dom not in ("light", "cover", "switch"):
        return False
    cur_ids = list(a["entity_id"])
    cur_areas = {e.area for e in c.h.ents if e.eid in cur_ids}
    others = [e for e in c.h.of(dom, plain=dom == "cover")
              if e.area and e.area not in cur_areas and e.state != "unavailable"]
    if not others:
        return False
    o = R.choice(others)
    prep = ROOM_BY_NAME[o.area][1]
    if R.random() < 0.6:
        c.user(R.choice([f"No, intendevo {prep}", f"Sbagliato, {prep}",
                         f"No, {prep}", f"Intendevo {prep}"]))
        new_ids = [e.eid for e in c.h.of(dom, o.area, plain=dom == "cover")
                   if e.state != "unavailable"]
        by_room = len(new_ids) == len(c.h.of(dom, o.area, plain=dom == "cover"))
    else:
        c.user(R.choice([f"No, intendevo {o.name}", f"Intendevo {o.name}",
                         f"Non quello, {o.name}"]))
        new_ids = [o.eid]
        by_room = False
    undo = {"service": f"{dom}.{INVERSE[svc]}", "entity_id": cur_ids}
    if "area" in a:
        undo = {"service": undo["service"], "area": a["area"], "entity_id": cur_ids}
    new = {"service": a["service"], "entity_id": new_ids}
    if dom in S.AREA_DOMAINS and by_room:
        new = {"service": a["service"], "area": o.area, "entity_id": new_ids}
    acts = [undo, new]
    c.control(acts, f"Ok, ho {describe(c.h, acts[0])} e {describe(c.h, acts[1])}.")
    return True


def turn_correct_recap(c):
    if not turn_correct(c):
        return False
    if R.random() < 0.7:
        return turn_summary(c)
    return turn_which(c)


def turn_also(c):
    """'Anche in bagno' -> same service on the same device type in another room."""
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
    c.user(R.choice([f"Anche {prep}", f"E anche {prep}", f"Lo stesso {prep}"]))
    act = {"service": a["service"],
           **({"area": r} if dom in S.AREA_DOMAINS and len(ents) == len(c.h.of(dom, r)) else {}),
           "entity_id": [e.eid for e in ents]}
    if a.get("data"):
        act["data"] = a["data"]
    c.control([act], R.choice(["Ok, ho ", "Fatto, ho "]) + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["Un po' più luce", "Ancora più luminosa", "Alza la luminosità"]), \
            {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["Abbassa un po'", "Un po' meno luce", "Abbassa la luminosità"]), \
            {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = cap(pnum("a", p)), {"brightness_pct": p}
    act = {"service": "light.turn_on",
           **({"area": c.last[0]["area"]} if "area" in c.last[0] else {}),
           "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "Ok, ho " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'Adesso è accesa?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    area = c.last[-1].get("area")
    ents = [e for e in c.h.ents if e.eid in ids]
    if not ents or ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    code = names_code(c.h, [e.eid for e in ents])
    pl = code.endswith("p")
    pairs = {"cover": [("aperto", "open"), ("chiuso", "closed"), ("abbassato", "closed"),
                       ("alzato", "open")],
             "lock": [("chiuso a chiave", "locked"), ("aperto", "unlocked")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("acceso", "on"), ("spento", "off")]))
    question = R.choice([f"Adesso {essere(pl)} {agree(word_q, code)}?",
                         f"{cap(essere(pl))} {agree(word_q, code)} adesso?"])
    keys = {state_key(e) for e in ents}
    states = ", ".join(f"{noun_name(e)} {state_pred(e)}" for e in ents)
    if R.random() < 0.6 and len(keys) == 1 and keys <= {
            "on", "off", "open", "closed", "locked", "unlocked"}:
        c.user(question)
        k = keys.pop()
        yes = "Sì" if k == asked else "No"
        c.get_state(ents, f"{yes}, secondo Home Assistant: {states}.", area)
    else:
        c.user(R.choice(["Ha funzionato?", "È davvero così?", "Puoi controllare?"]))
        c.get_state(ents, f"Secondo Home Assistant: {states}.", area)
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
            done = True
            turns += 1
            if turns >= n:
                break
            continue
        if not done and turns >= n - 1:
            continue
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
    """Every case gets the same share of the dataset, rare ones included."""
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
    cases += [("unknown:missing_type", s_missing_type, {"mtype": "other"}),
              ("unknown:missing_media_player", s_missing_type, {"mtype": "media_player"})]
    seen = set()
    for fn, _ in NEXT:
        if fn not in seen:
            seen.add(fn)
            cases.append((f"conversation:{fn.__name__}", s_conversation, {"conv_turn": fn}))
    cases.append(("conversation:room_correct", s_conversation,
                  {"conv_turn": turn_correct, "intent": i_light_room_phrase}))
    cases.append(("conversation:correct_recap", s_conversation,
                  {"conv_turn": turn_correct_recap, "intent": i_light_room_phrase}))
    for hook in CASE_HOOKS:
        cases += hook()
    return cases


SHORT_OK = ["Fatto.", "Ok.", "Ecco fatto.", "Eseguito."]
BARE = {"Fatto.", "Ok.", "Ecco fatto.", "Eseguito.", "In pausa.", "Volume alzato.",
        "Volume abbassato.", "Brano successivo.", "Brano precedente."}
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
    short -> a word or two, long/detailed -> never a bare 'Fatto.'."""
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
        m["content"] = "Ho " + describe_all(h, acts) + "."


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
            for c in open_:
                target[c[0]] += 1
            continue
        low = min(count[c[0]] for c in active)
        case = R.choice([c for c in active if count[c[0]] == low])
        s = make_sample(case)
        key = None
        if s:
            last_user = [m["content"] for m in s["messages"] if m["role"] == "user"][-1]
            key = last_user + str(s["meta"].get("gold")) + str(s["meta"].get("gold_last"))
        if not s or key in seen:
            fails[case[0]] += 1
            if fails[case[0]] >= 300:
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
