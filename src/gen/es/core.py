"""Generate the Spanish SFT dataset for Home Assistant tool calling.

Port of src/gen/de/core.py: the same house model, cases, tool calls and
metadata; only the language differs. Every sample uses a freshly randomised
house (rooms, device names, entity_ids, states), so the model learns to map
names from the device list to entity_ids instead of memorising one house.
Prompt and tool schema come from custom_components/ha_llm_agent/llm_schema.py
(LANGUAGES["es"]).

Each JSONL row: {"locale": "es", "messages": [...], "meta": {"cat": ...,
"gold": [...], "answer_contains": [...]}}. "meta" is ignored by training and
used by src/eval_model.py.

Grammar: Spanish has no cases but articles, gender and number agreement
(la luz encendida / las luces encendidas). Every entity carries the
grammatical gender of its name ("m", "f", "mp", "fp"); agr() makes the state
words agree, de() and al() build the contractions del / al.

src/gen/es/assist.py adds the Assist built-ins (music, timers, lists,
weather, time, broadcasts) through POPULATE_HOOKS, COMMANDS and CASE_HOOKS.

Usage: python src/gen_dataset.py --lang es OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import string
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# repository: custom_components/ha_llm_agent/llm_schema.py; Hugging Face
# dataset: llm_schema.py next to gen_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

LANG = "es"
# answer length of the current sample (llm_schema.VERBOSITY), set in make_sample
VERBOSITY = "medium"
VERBOSITY_WEIGHTS = [("short", 2), ("medium", 4), ("long", 2), ("detailed", 2)]
R = random.Random()
# extensions (assist.py): fn(house) after populating, fn() -> extra cases
POPULATE_HOOKS: list = []
CASE_HOOKS: list = []
# set by the case loop in main(): forces one intent / question kind / turn type
FORCE: dict = {}
# fn(house, action, who) -> wording behind "He ..." or None (assist.py)
DESCRIBE_HOOKS: list = []


# --------------------------------------------------------------------------
# grammar helpers
# --------------------------------------------------------------------------
ART = {"m": "el", "f": "la", "mp": "los", "fp": "las"}
UN = {"m": "un", "f": "una", "mp": "unos", "fp": "unas"}
NING = {"m": "ningún", "f": "ninguna", "mp": "ningún", "fp": "ninguna"}
PRON = {"m": "lo", "f": "la", "mp": "los", "fp": "las"}
_ENDING = {"m": "o", "f": "a", "mp": "os", "fp": "as"}


def is_pl(g: str) -> bool:
    return g in ("mp", "fp")


def plural_g(g: str) -> str:
    return g if is_pl(g) else g + "p"


def group_g(gs) -> str:
    """Gender of several things together: feminine only if all are."""
    gs = list(gs)
    if len(gs) == 1:
        return gs[0]
    return "fp" if all(g.startswith("f") for g in gs) else "mp"


def agr(phrase: str, g: str) -> str:
    """'encendido' -> 'encendida' / 'encendidos' / 'encendidas'; agrees the
    first adjective of the phrase ('cerrado con llave', 'no conectado')."""
    words = phrase.split(" ")
    for i, w in enumerate(words):
        if w in ("no", "ya", "muy"):
            continue
        core, tail = (w[:-1], w[-1]) if w.endswith(",") else (w, "")
        if core.endswith("o"):
            words[i] = core[:-1] + _ENDING[g] + tail
        elif is_pl(g) and core.endswith("e"):
            words[i] = core + "s" + tail
        break
    return " ".join(words)


def esta(g) -> str:
    """'está' / 'están'; g may be a gender or a bool (plural)."""
    pl = g if isinstance(g, bool) else is_pl(g)
    return "están" if pl else "está"


def de(phrase: str) -> str:
    """'de' + phrase with the contraction 'del'."""
    if phrase.startswith("el "):
        return "del " + phrase[3:]
    return "de " + phrase


def al(phrase: str) -> str:
    """'a' + phrase with the contraction 'al'."""
    if phrase.startswith("el "):
        return "al " + phrase[3:]
    return "a " + phrase


def es_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}".replace(".", ",")


de_num = es_num        # name used by shared helpers


def cap(s: str) -> str:
    """Capitalise the first letter, also behind an opening '¿' or '¡'."""
    if s[:1] in ("¿", "¡"):
        return s[:1] + s[1:2].upper() + s[2:]
    return s[:1].upper() + s[1:]


def low(s: str) -> str:
    """Lowercase a label inside a sentence ('Lámpara de pie' -> 'lámpara de
    pie'), but keep acronyms ('LED', 'TV')."""
    if len(s) > 1 and s[1].isupper():
        return s
    return s[:1].lower() + s[1:]


def lower_start(s: str) -> str:
    if s[:1] in ("¿", "¡"):
        return s[:1] + s[1:2].lower() + s[2:]
    return s[:1].lower() + s[1:]


def y_word(nxt: str, word: str = "y") -> str:
    """'y' -> 'e' before an i-sound, 'o' -> 'u' before an o-sound."""
    n = nxt.lower()
    if word == "y" and (n.startswith("i") or n.startswith("hi")) and not n.startswith("hie"):
        return "e"
    if word == "o" and n.startswith("o"):
        return "u"
    return word


def join_es(items, word="y"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {y_word(items[-1], word)} " + items[-1]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def slugify(s: str) -> str:
    s = strip_accents(s.lower())
    out = "".join(c if c.isalnum() else "_" for c in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, gender, slugs; ROOMS adds "en el salón" and "del salón"
_ROOMS = [
    ("Salón", "m", ["salon", "living_room"]),
    ("Cocina", "f", ["cocina", "kitchen"]),
    ("Dormitorio", "m", ["dormitorio", "bedroom"]),
    ("Baño", "m", ["bano", "bathroom"]),
    ("Cuarto de baño", "m", ["cuarto_bano", "bathroom"]),
    ("Habitación infantil", "f", ["habitacion_infantil", "kids_room"]),
    ("Oficina", "f", ["oficina", "office"]),
    ("Despacho", "m", ["despacho", "study"]),
    ("Pasillo", "m", ["pasillo", "hallway"]),
    ("Comedor", "m", ["comedor", "dining_room"]),
    ("Habitación de invitados", "f", ["habitacion_invitados", "guest_room"]),
    ("Sótano", "m", ["sotano", "basement"]),
    ("Garaje", "m", ["garaje", "garage"]),
    ("Jardín", "m", ["jardin", "garden"]),
    ("Terraza", "f", ["terraza", "terrace"]),
    ("Balcón", "m", ["balcon", "balcony"]),
    ("Desván", "m", ["desvan", "attic"]),
    ("Lavadero", "m", ["lavadero", "laundry"]),
    ("Aseo", "m", ["aseo", "wc"]),
    ("Vestidor", "m", ["vestidor", "dressing_room"]),
    ("Entrada", "f", ["entrada", "entrance"]),
    ("Escalera", "f", ["escalera", "stairs"]),
    ("Exterior", "m", ["exterior", "outdoor"]),
    ("Trastero", "m", ["trastero", "storage"]),
    ("Sala de calderas", "f", ["sala_calderas", "boiler_room"]),
    ("Cuarto de lavado", "m", ["cuarto_lavado", "utility"]),
    ("Patio", "m", ["patio", "yard"]),
    ("Cuarto de servicio", "m", ["cuarto_servicio", "utility_room"]),
    ("Taller", "m", ["taller", "workshop"]),
    ("Porche", "m", ["porche", "carport"]),
]
# (name, "en el salón", "del salón", slugs, gender)
ROOMS = [(n, f"en {ART[g]} {low(n)}", de(f"{ART[g]} {low(n)}"), sl, g) for n, g, sl in _ROOMS]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
OUTDOOR = ("Jardín", "Terraza", "Balcón", "Exterior", "Patio", "Porche")
UTILITY = ("Trastero", "Sala de calderas", "Cuarto de lavado", "Cuarto de servicio",
           "Taller", "Sótano", "Desván", "Garaje", "Lavadero")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garaje",)]
NO_GROUP_COVERS = ("Toldo", "Puerta del garaje")
# rooms that often have a window and a balcony or patio door with covers
TWO_COVER_ROOMS = ("Salón", "Comedor", "Cocina", "Dormitorio", "Despacho",
                   "Habitación infantil", "Oficina")


def en_room(room):
    return ROOM_BY_NAME[room][1]


def de_room(room):
    return ROOM_BY_NAME[room][2]


def room_phrase(room):
    """'del salón' behind a noun ('la luz del salón')."""
    return ROOM_BY_NAME[room][2]


# device type: (label, gender, slugs, may be named "Lámpara del salón")
TYPES = {
    "light": [
        ("Luz de techo", "f", ["techo", "luz_techo", "ceiling"], False),
        ("Luz", "f", ["luz", "light"], True),
        ("Plafón", "m", ["plafon", "ceiling_lamp"], False),
        ("Lámpara", "f", ["lampara", "lamp"], True),
        ("Lámpara de pie", "f", ["lampara_pie", "floor_lamp"], False),
        ("Lámpara de lectura", "f", ["lampara_lectura", "reading_lamp"], False),
        ("Tira LED", "f", ["tira_led", "led_strip"], False),
        ("Focos", "mp", ["focos", "spotlights"], False),
        ("Aplique", "m", ["aplique", "wall_lamp"], False),
        ("Guirnalda de luces", "f", ["guirnalda", "fairy_lights"], False),
        ("Lámpara colgante", "f", ["lampara_colgante", "pendant"], False),
        ("Lámpara de noche", "f", ["lampara_noche", "bedside"], False),
        ("Luz del espejo", "f", ["espejo", "mirror_light"], False),
    ],
    "switch": [
        ("Enchufe", "m", ["enchufe", "plug", "socket"], False),
        ("Cafetera", "f", ["cafetera", "coffee"], False),
        ("Calefactor", "m", ["calefactor", "heater"], False),
        ("Deshumidificador", "m", ["deshumidificador", "dehumidifier"], False),
        ("Bomba del estanque", "f", ["bomba_estanque", "pond_pump"], False),
        ("Riego", "m", ["riego", "irrigation"], False),
        ("Árbol de Navidad", "m", ["arbol_navidad", "xmas_tree"], False),
        ("Impresora", "f", ["impresora", "printer"], False),
        ("Hervidor", "m", ["hervidor", "kettle"], False),
        ("Monitor", "m", ["monitor", "screen"], False),
    ],
    "cover": [
        ("Persiana", "f", ["persiana", "shutter"], True),
        ("Persianas", "fp", ["persianas", "shutters"], False),
        ("Veneciana", "f", ["veneciana", "blinds"], False),
        ("Estor", "m", ["estor", "blind"], False),
        ("Toldo", "m", ["toldo", "awning"], False),
    ],
    "climate": [
        ("Calefacción", "f", ["calefaccion", "heating", "thermostat"], True),
        ("Termostato", "m", ["termostato", "trv"], False),
        ("Aire acondicionado", "m", ["aire", "aircon"], False),
        ("Suelo radiante", "m", ["suelo_radiante", "floor_heating"], False),
    ],
    "fan": [
        ("Ventilador", "m", ["ventilador", "fan"], False),
        ("Ventilador de techo", "m", ["ventilador_techo", "ceiling_fan"], False),
        ("Ventilación", "f", ["ventilacion", "ventilation"], False),
    ],
    "media_player": [
        ("Televisor", "m", ["tv", "televisor"], False),
        ("Altavoz", "m", ["speaker", "sonos", "altavoz"], False),
        ("Barra de sonido", "f", ["soundbar", "barra_sonido"], False),
        ("Radio", "f", ["radio"], False),
    ],
}

LOCKS = [("Puerta principal", "f", ["puerta_principal", "front_door"]),
         ("Puerta del sótano", "f", ["puerta_sotano", "basement_door"]),
         ("Puerta de la terraza", "f", ["puerta_terraza", "patio_door"]),
         ("Puerta del jardín", "f", ["puerta_jardin", "garden_gate"]),
         ("Puerta trasera", "f", ["puerta_trasera", "back_door"])]
VACUUMS = [("Robot aspirador", "m", ["robot_aspirador", "roborock", "robot"]),
           ("Aspiradora", "f", ["aspiradora", "vacuum"])]
SCENES = ["Noche de cine", "Buenas noches", "Cocinar", "Relax", "Fiesta", "Lectura",
          "Despertar", "Cena", "Romántico", "Luz intensa", "Concentración"]
SCRIPTS = ["Apagar todo", "Modo vacaciones", "Rutina de buenas noches", "Hacer café",
           "Salir de casa", "Llegar a casa"]

COLORS = {"rojo": "red", "verde": "green", "azul": "blue", "amarillo": "yellow",
          "naranja": "orange", "morado": "purple", "rosa": "pink", "blanco": "white",
          "turquesa": "turquoise", "violeta": "violet"}
COLOR_TEMPS = {"blanco cálido": 2700, "blanco frío": 6000, "blanco neutro": 4000}

# brand-named devices: user says the generic noun, the name is the product
BRAND_MEDIA = [("LG webOS Smart TV", "Televisor", "m"), ("Samsung TV", "Televisor", "m"),
               ("Sony Bravia", "Televisor", "m"), ("Philips Ambilight", "Televisor", "m"),
               ("Sonos One", "Altavoz", "m"), ("Echo Dot", "Altavoz", "m"),
               ("HomePod", "Altavoz", "m"), ("Bose Soundbar", "Barra de sonido", "f")]


# voice satellites expose a media_player that is no TV
def voice_media_name():
    hx = "".join(R.choice("0123456789abcdef") for _ in range(6))
    return R.choice([f"Home Assistant Voice {hx} Media Player", f"Home Assistant Voice {hx[:6]}",
                     f"ESPHome Voice {hx[:4]} Media Player", "Echo Show", "Nest Mini",
                     f"Atom Echo {hx[:6]} Media Player"])


BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Robomow", "Gardena Sileno", "Worx Landroid", "Navimow"]
# plant, article
PLANTS = [("Tomates", "los"), ("Guindillas", "las"), ("Albahaca", "la"), ("Perejil", "el"),
          ("Fresas", "las"), ("Monstera", "la"), ("Ficus", "el"), ("Pepinos", "los"),
          ("Limonero", "el")]
PLANT_ART = dict(PLANTS)
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "PC Javi", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Agua caliente Boost", "Grupo de presión", "Bomba acuario"]
# appliances: a status sensor and a power switch with nearly the same name
# ("Secadora" + "Secadora Enchufe"): commands go to the switch, questions to
# the sensor
APPLIANCES = [("Secadora", "f", "secadora"), ("Lavadora", "f", "lavadora"),
              ("Lavavajillas", "m", "lavavajillas"), ("Friegaplatos", "m", "friegaplatos"),
              ("Impresora 3D", "f", "impresora_3d"), ("Robot de cocina", "m", "dryer")]
# raw states as integrations report them (Home Connect, Miele, SmartThings, ...)
APP_RAW = {"running": ["running", "En marcha", "Funcionando", "drying", "in_use", "run"],
           "finished": ["finished", "Programa terminado", "Fin", "end", "program_ended"],
           "off": ["off", "Apagado"], "idle": ["idle", "Listo", "standby", "programmed",
                                               "waiting_to_start"],
           "paused": ["pause", "paused", "En pausa"],
           "offline": ["not_connected", "disconnected", "Offline"]}


def app_answer(kind, g):
    """-> (predicate for the answer, word the answer must contain)."""
    return {"running": ("sigue en marcha", "en marcha"),
            "finished": ("ya ha terminado", "terminado"),
            "off": (f"está {agr('apagado', g)}", agr("apagado", g)),
            "idle": (f"está {agr('preparado', g)}, pero ahora mismo no está en marcha",
                     "no está en marcha"),
            "paused": ("está en pausa", "en pausa"),
            "offline": (f"no está {agr('conectado', g)} ahora mismo",
                        f"no está {agr('conectado', g)}")}[kind]


# button label, what the user starts
VACUUM_BUTTONS = [("Limpieza completa", "la limpieza completa"),
                  ("Vaciado", "el vaciado del depósito"),
                  ("Lavar mopa", "el lavado de la mopa")]
TOGGLES = ["Vacaciones", "Modo invitados", "Modo fiesta", "Modo noche", "Modo limpieza"]

UNKNOWN_DEVICES = [("Sauna", "f"), ("Jacuzzi", "m"), ("Bomba de la piscina", "f"),
                   ("Chimenea", "f"), ("Acuario", "m"), ("Cortacésped", "m"),
                   ("Fuente", "f"), ("Robot de cocina", "m"), ("Friegaplatos", "m"),
                   ("Horno", "m"), ("Climatizador", "m"), ("Puerta del garaje", "f"),
                   ("Iluminación del portón", "f")]


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "Plafón"
    gender: str = "f"        # grammatical gender of the name
    state: str = "off"
    attrs: dict = field(default_factory=dict)
    hidden: str = ""         # type label that the name no longer shows
    voice: bool = False      # media player of a voice satellite

    @property
    def plural(self) -> bool:
        return is_pl(self.gender)


class House:
    def __init__(self, size: int):
        self.ents: list[Ent] = []
        self.used_ids: set[str] = set()
        self.used_names: set[str] = set()
        self.sat_player = None                # speaker of the voice device
        self.sat_loc = None                   # ... and the area of the device
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
        names = R.sample([r[0] for r in ROOMS if r[0] != "Cuarto de baño"], n_rooms)
        if "Baño" in names and R.random() < 0.4:
            names[names.index("Baño")] = "Cuarto de baño"
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
            base = R.choice([f"{ts}_{rs}", f"{rs}_{ts}", f"{ts}_{rs}"])
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
                opts += [f"{label} {de_room(room)}"]
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
        e = self.add("media_player", room, "Altavoz", "m",
                     [name.lower().replace(" ", "_"), "voice_media_player"], name=name,
                     state=R.choice(["idle", "idle", "off"]),
                     attrs={"volume_level": R.choice([0.3, 0.5, 0.7])})
        e.voice = True
        return e

    def _populate(self, size: int):
        for room in self.rooms:
            outdoor = room in OUTDOOR
            rg = ROOM_BY_NAME[room][4]
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
                # sample may call it "la lámpara de la cocina"
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                room_lights[0].gender = rg
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Izquierda", "Derecha", "Centro"], ["Techo", "Pared", "Rincón"],
                                      ["Isla", "Focos", "Barra"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        if f"{room} {side}" not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = f"{room} {side}"
                            lt.hidden, lt.label = lt.label, ""
                            lt.gender = rg
                            self.used_names.add(lt.name)
            # many lamps carry a name without their room ("Cielo estrellado",
            # "Luz aparador"); only the area column ties them to the room
            for lt in room_lights:
                if lt.name != room and R.random() < 0.3:
                    free = [n for n in FREE_LIGHT_NAMES if n[0] not in self.used_names]
                    if free:
                        self.used_names.discard(lt.name)
                        lt.name, lt.gender = R.choice(free)
                        lt.hidden, lt.label = lt.label or lt.hidden, ""
                        self.used_names.add(lt.name)
            if not outdoor and room not in ("Pasillo", "Escalera", "Aseo") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                if room in TWO_COVER_ROOMS and R.random() < 0.35:
                    # window and balcony door: "las persianas del comedor"
                    # means both
                    word = R.choice(["Persiana", "Persiana", "Estor", "Veneciana"])
                    gender = {"Estor": "m"}.get(word, "f")
                    door = R.choice(["Puerta balcón", "Balconera", "Puerta terraza", "Puerta"])
                    for part, slug in (("Ventana", "ventana"), (door, "balconera")):
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
            if room in ("Terraza", "Balcón", "Exterior") and R.random() < 0.5:
                self.add("cover", room, "Toldo", "m", ["toldo", "awning"],
                         state=(st := R.choice(["open", "closed"])),
                         attrs={"current_position": 100 if st == "open" else 0})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:3] if room != "Baño" else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, "Temperatura", "f",
                         ["temperatura", "temperature"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "Humedad", "f",
                         ["humedad", "humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "Ventana", "f",
                         ["ventana", "window"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Salón", "Dormitorio", "Cocina", "Oficina",
                        "Habitación infantil") and R.random() < 0.4:
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
                        ["Telediario", "Bohemian Rhapsody", "Despacito",
                         "La casa de papel", "Pocoyó", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "Rosalía", ""]))
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if brand and brand[0] not in self.used_names else None,
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""})
            if not outdoor and R.random() < 0.12:
                self.add_voice_media(room)
            if room in ("Dormitorio", "Oficina", "Salón", "Desván") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "Potencia", "f", ["potencia", "power"],
                         name=self._name(room, R.choice(["Enchufe Potencia", "Potencia"]), False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            area = R.choice(self.rooms + [None])
            self.add("alarm_control_panel", None, "Alarma", "f",
                     ["alarma", "alarm"], name="Alarma",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Pasillo", "Entrada", None]) if lk[0] == "Puerta principal" else None
                room = room if room in self.rooms else None
                self.add("lock", room, lk[0], lk[1], lk[2], name=lk[0],
                         state=R.choice(["locked", "locked", "unlocked"]))
        if R.random() < 0.4:
            vt = R.choice(VACUUMS)
            vname = R.choice(BRAND_VACUUM) if R.random() < 0.6 else vt[0]
            self.add("vacuum", R.choice(self.rooms + [None, None]), vt[0],
                     vt[1] if vname == vt[0] else "m",
                     [slugify(vname)], name=vname,
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.25:
            mname = R.choice(MOWERS)
            room = "Jardín" if "Jardín" in self.rooms else None
            self.add("lawn_mower", room, "Robot cortacésped", "m",
                     [slugify(mname)], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = ["Riego", "Válvula de agua"]
            if room:
                names += [f"Riego {room}", f"{room} Plantas", f"Plantas {room}"]
                if room == "Jardín":
                    names.append("Válvula jardín")
            self.add("valve", room, "Riego", "m", ["riego", "irrigation"],
                     name=R.choice(names), state=R.choice(["open", "closed", "closed"]))
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
            room, names = None, ["Temperatura exterior", "Exterior Temperatura",
                                 "Estación meteorológica Temperatura"]
            orooms = [r for r in self.rooms if r in OUTDOOR]
            if orooms and R.random() < 0.4:
                # weather stations sit on the balcony or in the garden
                room = R.choice(orooms)
                names = [f"Temperatura exterior {room}", f"Temperatura {room}"]
            self.add("sensor", room, "Temperatura exterior", "f",
                     ["temperatura_exterior", "outdoor_temperature"],
                     name=R.choice(names), state=f"{out_t:.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        # heat pumps report their own outdoor sensor next to the climate entity
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            room = "Sala de calderas" if "Sala de calderas" in self.rooms and R.random() < 0.5 else None
            self.add("sensor", room, "Temperatura exterior", "f",
                     ["calefaccion_outdoor_temperature", "casa_outdoor_temperature"],
                     name=R.choice(["Temperatura exterior Casa", "Temperatura exterior Calefacción",
                                    "Calefacción Temperatura exterior"]),
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
                if label == "Previsión":
                    val = out_t + R.uniform(-4, 6) if when == "hoy" else R.uniform(-6, 31)
                else:
                    val = out_t - R.uniform(0.5, 7)
                self.add("sensor", None, label, "f",
                         slugs, name=name,
                         state=f"{val:.1f}",
                         attrs={"device_class": "temperature", "unit_of_measurement": "°C",
                                "_when": when})
        if "Garaje" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garaje", "Puerta del garaje", "f",
                     ["puerta_garaje", "garage_door"], name="Puerta del garaje",
                     state=(st := R.choice(["open", "closed", "closed"])),
                     attrs={"current_position": 100 if st == "open" else 0})
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(n.startswith(app) for n in sorted(self.used_names)):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Cocina", "Cuarto de lavado", "Cuarto de servicio", "Baño", "Cuarto de baño",
                    "Sótano", "Trastero", "Oficina")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "Estado", "m", [slug, slug + "_status"],
                                  name=R.choice([app, app, f"{app} Estado"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind})
                power = self.add("switch", room, "Enchufe", "m",
                                 [slug + "_power", slug + "_enchufe"],
                                 name=R.choice([f"{app} Enchufe", f"Enchufe {app}",
                                                f"{app} Power"]),
                                 state="off" if kind == "off" and R.random() < 0.5 else "on")
                self.appliances.append((app, g, sensor, power))
        vac = [e for e in self.ents if e.domain == "vacuum"]
        if vac and R.random() < 0.45:
            v = vac[0]
            for label, _ in R.sample(VACUUM_BUTTONS, R.choice([1, 2])):
                self.add("button", v.area, label, "m",
                         [v.eid.split(".")[1] + "_" + slugify(label)],
                         name=f"{v.name} {label}", state="unknown")
        valve_rooms = {e.area for e in self.ents if e.domain == "valve"}
        if R.random() < (0.5 if valve_rooms else 0.15):
            free = [r for r in self.rooms if r in OUTDOOR and r not in valve_rooms]
            # many installs leave the irrigation button without an area; its
            # name ("Riego Jardín") then is the only hint where it waters
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Riego Jardín", "Riego del jardín"] if room == "Jardín"
                         else [f"Riego {room}"])
            elif "Jardín" in valve_rooms:
                names = ["Iniciar riego"]
            else:
                names = ["Iniciar riego", "Riego Jardín", "Riego del jardín"]
            self.add("button", room, "Riego", "m", ["riego_jardin",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # door contacts: balcony/patio doors in living rooms, the front door
        # in the hall; names often say "Puerta balcón" while people say "la
        # puerta del balcón"
        if R.random() < 0.5:
            short = R.choice([False, True])
            for room in [r for r in self.rooms if r in DOOR_ROOMS]:
                if R.random() < 0.55:
                    (label, sname, slug) = R.choice([
                        ("Puerta del balcón", "Puerta balcón", "puerta_balcon"),
                        ("Puerta de la terraza", "Puerta terraza", "puerta_terraza")])
                    base = sname if short else label
                    name = R.choice([f"{base} {room}", f"{room} {base}"])
                    if name in self.used_names:
                        continue
                    self.add("binary_sensor", room, label, "f", [slug, "door"],
                             name=name, state=R.choice(["off", "off", "on"]),
                             attrs={"device_class": "door"})
            if R.random() < 0.7:
                base = R.choice(["Puerta de entrada", "Puerta de casa", "Puerta de entrada"])
                name = base.replace(" de ", " ") if short else base
                if name in self.used_names:
                    name = f"{name} Contacto"
                hall = next((r for r in ("Pasillo", "Entrada") if r in self.rooms), None)
                self.add("binary_sensor", hall, base, "f", ["puerta_entrada_state", "front_door_contact"],
                         name=name, state=R.choice(["off", "off", "on"]),
                         attrs={"device_class": "door"})
        # electric car: charge level and range next to each other
        if R.random() < 0.3:
            car, art, cslug = R.choice(CARS)
            self.add("sensor", None, "Batería", "f", ["battery_hv_state_of_charge", f"{cslug}_soc"],
                     name=f"{car} Batería", state=str(R.randint(10, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
            self.add("sensor", None, "Autonomía", "f",
                     [f"{cslug}_range_ev_remaining_range", f"{cslug}_autonomia"],
                     name=f"{car} Autonomía", state=str(R.randint(40, 450)),
                     attrs={"device_class": "distance", "unit_of_measurement": "km"})
        # heating systems expose their modes as switches next to the climate
        # entity ("Calefacción Casa Modo ausente")
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            mode, slug, _ = R.choice(HEAT_MODES)
            base = R.choice(["Calefacción", "Calefacción Casa", "Termostato", "Casa"])
            room = "Sala de calderas" if "Sala de calderas" in self.rooms and R.random() < 0.5 else None
            if f"{base} {mode}" not in self.used_names:
                self.add("switch", room, mode, "m", [f"calefaccion_{slug}", f"casa_{slug}"],
                         name=f"{base} {mode}", state=R.choice(["on", "off", "off"]))
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            e = self.add("light", None, "__group__", "fp", ["todas_las_luces", "all_lights"],
                         name=R.choice(["Todas las luces", "Todas las lámparas", "Luces casa"]),
                         state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            self.add("cover", None, "__group__", "fp", ["todas_las_persianas", "all_covers"],
                     name=R.choice(["Todas las persianas", "Persianas casa", "Todas las persianas"]),
                     state=R.choice(["open", "closed"]))
        # read-only distractors and odd names; fill up to the target size
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "Casa"
        if k == 0 and room:
            self.add("binary_sensor", room, "Presencia", "f", ["presence"],
                     name=R.choice([f"Presencia {room}", f"{room} Presence",
                                    f"Ocupación {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "Movimiento", "m", ["motion"],
                     name=R.choice([f"{rname} Motion", f"Movimiento {rname}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant = R.choice(PLANTS)[0]
            self.add("sensor", room, "Humedad del suelo", "f", ["plant_sensor", "moisture"],
                     name=f"Humedad suelo {plant}", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "Batería", "f", ["battery"],
                     name=R.choice(["SAI Batería", "Coche Batería", "Detector de humo Batería",
                                    "Móvil Batería", f"{rname} Sensor Batería"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "Impresora", "f", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, unit = R.choice([("Velocidad del viento", "km/h"), ("Rachas de viento", "km/h"),
                                 ("Lluvia", "mm/h"), ("Presión atmosférica", "hPa"),
                                 ("Depósito de agua Nivel", "L")])
            self.add("sensor", room, nm, "f", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": unit})
        elif k == 6:
            self.add("sensor", room, "Estado", "m", ["status"],
                     name=R.choice(["SAI Estado", "Secadora", "Lavadora Estado",
                                    "Lavavajillas Programa"]),
                     state=R.choice(["idle", "OL", "running", "off"]))
        elif k == 7:
            nm = R.choice(TOGGLES)
            if nm not in self.used_names:
                self.add("input_boolean", None, nm, "m", [slugify(nm)], name=nm,
                         state=R.choice(["on", "off"]))
        elif k == 8:
            nm = R.choice(JUNK_SWITCHES)
            if nm not in self.used_names:
                self.add("switch", room, nm, "m", [slugify(nm)], name=nm,
                         state=R.choice(["on", "off"]))
        else:
            self.add("sensor", room, "Temperatura", "f", ["temperature"],
                     name=R.choice([f"{rname} Temperatura", f"Temperatura {rname}",
                                    f"{rname} Termómetro"]),
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
        # the speaker of the voice device is listed with the location only
        sat = self.sat_player if location and self.sat_player is not None \
            and self.sat_loc == location and self.sat_player in self.ents else None
        return S.build_system_prompt(
            [(e.eid, e.name, e.area) for e in self.ents], location, LANG, VERBOSITY,
            player=sat.eid if sat else None)

    def pick_sat_player(self, loc):
        """Half of the voice devices with a location have a speaker of
        their own; it may have to be added to the room."""
        self.sat_player, self.sat_loc = None, None
        if not loc or loc in OUTDOOR or R.random() < 0.5:
            return
        own = [e for e in self.of("media_player", loc) if e.voice]
        e = R.choice(own) if own else self.add_voice_media(loc)
        if e is not None and e.state != "unavailable":
            self.set_sat_player(e, loc)

    def set_sat_player(self, e, loc):
        """A player that wraps the satellite's own (Music Assistant) often
        has no area: then only the prompt line ties it to the device."""
        self.sat_player, self.sat_loc = e, loc
        if R.random() < 0.3:
            e.area = None

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
    acc: str      # phrase in the user's sentence ("la luz del baño")
    nom: str      # phrase for the answer (proper capitalisation of names)
    g: str        # gender and number of the phrase
    ents: list
    area: str | None = None   # set for "la luz del baño": control by area

    @property
    def plural(self) -> bool:
        return is_pl(self.g)


def the(e: Ent) -> str:
    """Name with article when it starts with its type ('la Lámpara Salón')."""
    if e.label and e.label != "__group__" and e.name.lower().startswith(e.label.lower()):
        return f"{ART[e.gender]} {e.name}"
    return e.name


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    # "la luz del pasillo" / "la lámpara del pasillo" mean every light of the
    # room, so a single lamp only gets these words when it is the room's only light
    generic = e.domain == "light" and e.label in ("Luz", "Lámpara") \
        and len(h.of("light", e.area)) > 1
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1 \
            and not generic:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 and e.name != e.label:
        opts += ["type_only"]           # "el televisor" for "LG webOS Smart TV"
    choice = R.choice(opts)
    g = e.gender
    if choice == "type_only":
        t = f"{ART[g]} {low(e.label)}"
        return Ref(t, t, g, [e])
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        art = R.random() < 0.45 and e.label and e.name.lower().startswith(e.label.lower())
        if art:
            return Ref(f"{ART[g]} {name}", f"{ART[g]} {e.name}", g, [e])
        return Ref(name, e.name, g, [e])
    t = f"{ART[g]} {low(e.label)} {room_phrase(e.area)}"
    return Ref(t, t, g, [e])


# group nouns per domain: (phrase, gender)
GROUP = {
    "light": [("la luz", "f"), ("las luces", "fp"), ("las lámparas", "fp"),
              ("la iluminación", "f")],
    "cover": [("las persianas", "fp"), ("los estores", "mp"), ("las venecianas", "fp")],
    "climate": [("la calefacción", "f"), ("los radiadores", "mp")],
    "media_player": [("la música", "f")],
}
ALL_GROUP = {
    "light": [("todas las luces", "fp"), ("todas las lámparas", "fp"),
              ("la luz de toda la casa", "f")],
    "cover": [("todas las persianas", "fp"), ("todos los estores", "mp")],
    "climate": [("todos los radiadores", "mp")],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    phrase, g = R.choice(GROUP[domain])
    area = room if domain in S.AREA_DOMAINS else None
    if location == room and R.random() < 0.7:
        return Ref(phrase, phrase, g, ents, area)    # "Enciende la luz"
    t = f"{phrase} {room_phrase(room)}"
    return Ref(t, t, g, ents, area)


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
        out = {"domain": args.get("domain", "light"), "area": args["area"]}
        if args.get("device_class"):
            out["device_class"] = args["device_class"]
        return out
    return args


LAST_CONTROL: list = []


def ok_result(actions=None):
    """Successful control result; area actions report what they switched."""
    actions = LAST_CONTROL if actions is None else actions
    if any("area" in a for a in actions):
        return S.control_result(None, [i for a in actions for i in a["entity_id"]])
    return S.control_result()


def bare(text: str) -> str:
    """Without the closing punctuation and the opening '¿' / '¡'."""
    text = text.rstrip("?.!")
    if text[:1] in ("¿", "¡"):
        text = text[1:]
    return text


def voice(text: str) -> str:
    """Make user text look like speech-to-text or quickly typed text
    sometimes: no punctuation, lowercase without accents, 'oye', 'por favor'.
    The classes match the German generator, so both take the same random
    draws and find the same duplicates."""
    r = R.random()
    if r < 0.25:
        text = bare(text)
    elif r < 0.35:
        text = strip_accents(bare(text)).lower()
    if R.random() < 0.05:
        text = R.choice(["Oye, ", "Vale, ", "Oye, por favor, ", "Por favor, "]) + lower_start(text)
    return text


def confirm(ref: Ref, adj: str, part: str) -> str:
    """adj: state word in the masculine singular ('encendido'), part:
    participle for 'he ...' ('encendido')."""
    a, v = agr(adj, ref.g), esta(ref.g)
    return R.choice([
        f"{cap(ref.nom)} ya {v} {a}.",
        f"{cap(ref.nom)} {v} {a}.",
        f"He {part} {ref.nom}.",
        f"Listo, {ref.nom} {v} {a}.",
        f"Vale, {ref.nom} ya {v} {a}.",
        "Hecho.",
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
        # same positions as German: questions where German asks, "Por favor"
        # for "Bitte", infinitives where German starts with the device
        user = R.choice([
            "Enciende {r}", "Enciéndeme {r}", "Prende {r}", "Encender {r}",
            "Por favor, enciende {r}", "¿Puedes encender {r}?",
            "Enciende {r}, por favor", "Encender {r} ahora", "¿Podrías encender {r}?",
            "Necesito luz, enciende {r}", "Enciende ya {r}.",
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref)],
                confirm(ref, "encendido", "encendido"))
    user = R.choice([
        "Apaga {r}", "Apágame {r}", "Quita {r}", "Apagar {r}", "Por favor, apaga {r}",
        "¿Puedes apagar {r}?", "Apaga {r}, por favor.", "Apagar {r} ahora",
        "¿Podrías apagar {r}?",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref)],
            confirm(ref, "apagado", "apagado"))


def i_light_room_phrase(h, loc):
    """'Luz del baño' / 'Está muy oscuro en el baño'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    en, dl = en_room(room), de_room(room)
    ref = Ref(f"la luz {dl}", f"la luz {dl}", "f", ents, room)
    if R.random() < 0.5:
        user = R.choice([f"Encender luz {en}", f"Enciende la luz {dl}",
                         f"Está muy oscuro {en}", f"Luz {en}, por favor",
                         f"Encender la luz {dl}"])
        return user, [action("light.turn_on", ref)], confirm(ref, "encendido", "encendido")
    user = R.choice([f"Apagar luz {en}", f"Apaga la luz {dl}",
                     f"Apagar la luz {dl}", f"La luz {dl}, apágala"])
    return user, [action("light.turn_off", ref)], confirm(ref, "apagado", "apagado")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        user = R.choice([
            "Atenúa {r} al {p} %", "Pon {r} al {p} %",
            "{R} al {p} por ciento", "Pon {r} al {p} por ciento",
            "Ajusta el brillo {dr} al {p} %",
        ]).format(r=ref.acc, R=cap(ref.acc), dr=de(ref.acc), p=p)
        conf = R.choice([f"{cap(ref.nom)} ya {esta(ref.g)} al {p} %.",
                         f"He puesto {ref.nom} al {p} %.", "Hecho."])
        return user, [action("light.turn_on", ref, {"brightness_pct": p})], conf
    up = r < 0.8
    user = R.choice(["Sube {r}", "Más brillo para {r}", "Sube un poco el brillo {dr}, por favor"]
                    if up else
                    ["Baja {r}", "Atenúa {r}", "Baja un poco el brillo {dr}"]
                    ).format(r=ref.acc, dr=de(ref.acc))
    step = 20 if up else -20
    conf = (f"He {'subido' if up else 'bajado'} el brillo {de(ref.nom)}."
            if R.random() < 0.7 else "Hecho.")
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
    user = R.choice(["Pon {r} en {c}", "Pon {r} de color {c}", "Cambia {r} a {c}",
                     "{R} en {c}, por favor", "¿Puedes poner {r} en {c}?",
                     "Quiero {r} en {c}"]).format(r=ref.acc, R=cap(ref.acc), c=word)
    conf = R.choice([f"{cap(ref.nom)} ya {esta(ref.g)} en {word}.",
                     f"He puesto {ref.nom} en {word}.", "Hecho."])
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
        user = R.choice(["Enciende {r}", "Pon {r}", "Encender {r}", "Conectar {r}",
                         "¿Puedes encender {r}?", "Arranca {r}"])
        conf = confirm(ref, "encendido", "encendido")
    else:
        user = R.choice(["Apaga {r}", "Desconecta {r}", "Apagar {r}", "Desconectar {r}",
                         "¿Puedes apagar {r}?"])
        conf = confirm(ref, "apagado", "apagado")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    user = R.choice(["Pon {r} al {p} %", "{R} al {p} %",
                     "Deja {r} al {p} por ciento"]).format(r=ref.acc, R=cap(ref.acc), p=p)
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {esta(ref.g)} al {p} %.", "Hecho."])


def i_awning(h, loc):
    """Toldo: sacar = open_cover, recoger = close_cover."""
    ents = [e for e in h.of("cover") if e.label == "Toldo"]
    if not ents:
        return None
    e = R.choice(ents)
    n = "el toldo" if len(ents) == 1 and R.random() < 0.7 else e.name
    if R.random() < 0.5:
        user = R.choice([f"Saca {n}", f"Extiende {n}", f"Abre {n}",
                         f"Sacar {n}", "Necesitamos sombra, saca el toldo"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["El toldo se está extendiendo.", "Hecho, el toldo está extendido."])
    user = R.choice([f"Recoge {n}", f"Cierra {n}", f"Recoger {n}",
                     f"Guarda {n}", "Está lloviendo, recoge el toldo"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["El toldo se está recogiendo.", "Hecho, el toldo está recogido."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["el robot cortacésped", "el cortacésped", e.name])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Inicia {n}", f"Pon a cortar {n}", "Corta el césped, por favor",
                         f"Manda {n} a cortar el césped"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{e.name} ya está cortando el césped."
    if r < 0.8:
        user = R.choice([f"Manda {n} a la base", f"Que vuelva {n} a la base",
                         f"Recoge {n}"])
        return user, [action("lawn_mower.dock", [e])], f"{e.name} vuelve a la base de carga."
    user = R.choice([f"Pausa {n}", f"Para {n}"])
    return user, [action("lawn_mower.pause", [e])], f"{e.name} está en pausa."


AREA_PLANTS = {"Balcón": "las plantas del balcón", "Terraza": "las plantas de la terraza",
               "Jardín": "el jardín"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == "Riego"]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Inicia el riego", "Riega las plantas"]
    if e.area:
        en, dl = en_room(e.area), de_room(e.area)
        out += [f"Inicia el riego {dl}", f"Riega las plantas {dl}",
                f"Riega las plantas {en}"]
        if e.area in AREA_PLANTS:
            out.append(f"Riega {AREA_PLANTS[e.area]}")
    elif "jardín" in e.name.lower():
        out += ["Riega el jardín", "Inicia el riego del jardín",
                "Pon el riego del jardín"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name] if several else [e.name, "el riego", "la válvula"])
    if R.random() < 0.5:
        user = R.choice([f"Abre {n}", f"Enciende {n}", f"Activa {n}"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"He abierto {e.name}."
    user = R.choice([f"Cierra {n}", f"Apaga {n}", f"Corta {n}",
                     "Para el riego"])
    return user, [action("valve.close_valve", [e])], f"He cerrado {e.name}."


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    # "Vacaciones" is also called "modo vacaciones"; it stays an input_boolean
    mode = n if n.startswith("Modo") else f"Modo {low(n)}"
    if mode != n and any(x.name == mode for x in h.ents):
        mode = n          # a script "Modo vacaciones" exists: only the plain name is clear
    m = low(mode)
    if on:
        user = R.choice([f"Activa {n}", f"Activar {low(n)}", f"Enciende {n}",
                         f"Activa el {m}" if mode != n or m.startswith("modo") else f"Activa {n}",
                         f"Pon el {m}" if m.startswith("modo") else f"Activa {n}",
                         f"Enciende el {m}" if m.startswith("modo") else f"Enciende {n}",
                         f"Activar el {m}" if m.startswith("modo") else f"Activar {low(n)}"])
        return user, [action("input_boolean.turn_on", [e])], R.choice(
            [f"He activado {n}.", f"Listo, he activado {n}." if m.startswith("modo")
             else f"He activado {n}."])
    user = R.choice([f"Desactiva {n}", f"Desactivar {low(n)}", f"Apaga {n}",
                     f"Desactiva el {m}" if m.startswith("modo") else f"Desactiva {n}",
                     f"Quita el {m}" if m.startswith("modo") else f"Quita {n}",
                     f"Termina el {m}" if m.startswith("modo") else f"Desactiva {n}"])
    return user, [action("input_boolean.turn_off", [e])], R.choice(
        [f"He desactivado {n}.", f"Listo, he desactivado {n}." if m.startswith("modo")
         else f"He desactivado {n}."])


def i_appliance(h, loc):
    """'Apaga la secadora' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = f"{ART[g]} {low(app)}"
    if R.random() < 0.5:
        user = R.choice([f"Apaga {n}", f"Desconecta {n}", f"Apagar {low(app)}",
                         f"Quita la corriente {de(n)}", f"¿Puedes apagar {n}?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"He apagado {power.name}.", f"Vale, {power.name} ya no tiene corriente."])
    user = R.choice([f"Enciende {n}", f"Conecta {n}", f"Encender {low(app)}",
                     f"Dale corriente {al(n)}"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"He encendido {power.name}.", f"Vale, {power.name} vuelve a tener corriente."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        what = vac[e.label]
        user = R.choice([f"Inicia {what}", f"Inicia {what} del robot aspirador",
                         f"Haz {what}", f"Iniciar {e.name}"])
    elif e.label == "Riego":
        user = R.choice(irrigation_phrases(h, e) + [f"Pulsa {e.name}"])
    else:
        user = R.choice([f"Pulsa {e.name}", f"Activa {e.name}"])
    return user, [action("button.press", [e])], R.choice(
        [f"He pulsado {e.name}.", f"He iniciado {e.name}.", "Hecho."])


# lamp names without a room (the names of the acceptance house are left out
# on purpose, so the acceptance test measures generalisation); (name, gender)
FREE_LIGHT_NAMES = [("Cielo estrellado", "m"), ("Lámpara de sal", "f"), ("Lámpara de lava", "f"),
                    ("Luz cuadro", "f"), ("Vitrina", "f"), ("Hue Go", "f"), ("Nanoleaf", "f"),
                    ("Hue Play", "f"), ("Lámpara nube", "f"), ("Lámpara luna", "f"),
                    ("Regleta LED", "f"), ("Marco LED", "m"), ("Estrella de Navidad", "f"),
                    ("Estrella de papel", "f"), ("Lámpara bola", "f"), ("Lámpara rincón", "f"),
                    ("Lámpara sofá", "f"), ("Luz aparador", "f"), ("Luz barra", "f"),
                    ("Luz cama", "f"), ("Luz armario", "f"), ("Luz ventana", "f"),
                    ("Lámpara grande", "f"), ("Lámpara de techo grande", "f"),
                    ("Arco de luces", "m"), ("Letrero neón", "m"), ("Govee Strip", "f"),
                    ("Lámpara de mesa", "f"), ("Luz cómoda", "f"), ("Luz tocador", "f"),
                    ("Luz de lectura", "f"), ("Lámpara esquina", "f")]

DOOR_ROOMS = ("Salón", "Comedor", "Cocina", "Dormitorio", "Despacho")
# (name, article, slug)
CARS = [("Enyaq", "el", "enyaq_iv80"), ("ID.3", "el", "id3"), ("Model 3", "el", "model_3"),
        ("Zoe", "el", "zoe"), ("Enyaq", "el", "enyaq"), ("e-Golf", "el", "egolf"),
        ("EQA", "el", "eqa")]

WEATHER_SENSORS = [  # (name, label, day, slugs)
    ("Previsión temperatura hoy", "Previsión", "hoy",
     ["prevision_temperatura_hoy", "forecast_temperature_today"]),
    ("Previsión temperatura mañana", "Previsión", "mañana",
     ["prevision_temperatura_manana", "forecast_temperature_tomorrow"]),
    ("Temperatura máxima hoy", "Previsión", "hoy",
     ["temperatura_maxima_hoy", "forecast_high_today"]),
    ("Temperatura máxima mañana", "Previsión", "mañana",
     ["temperatura_maxima_manana", "forecast_high_tomorrow"]),
    ("Sensación térmica", "Valor meteorológico", None, ["sensacion_termica", "feels_like_temperature"]),
    ("Punto de rocío", "Valor meteorológico", None, ["punto_rocio", "dew_point"]),
]

# label in the name, slug, phrase with article
HEAT_MODES = [("Modo ausente", "away_mode", "el modo ausente"),
              ("Modo eco", "eco_mode", "el modo eco"),
              ("Antiheladas", "frost_protection", "la protección antiheladas")]


def i_heating_mode(h, loc):
    """'Activa el modo ausente de la calefacción' -> the mode switch, not
    the climate entity."""
    modes = [e for e in h.ents if e.domain == "switch"
             and e.label in [m for m, _, _ in HEAT_MODES]]
    if not modes or not h.of("climate"):
        return None
    e = R.choice(modes)
    m = next(p for lab, _, p in HEAT_MODES if lab == e.label)
    word = low(e.label)
    on = R.random() < 0.6
    if on:
        user = R.choice([f"Activa {m} de la calefacción", f"Pon la calefacción en {word}",
                         f"Calefacción en {word}", f"Activa {m}", f"Enciende {m}",
                         f"Pon {m}", f"Activar {m}"])
    else:
        user = R.choice([f"Desactiva {m} de la calefacción", f"Desactiva {m}",
                         f"Apaga {m}", f"Quita {m} de la calefacción",
                         f"Desactivar {m}"])
    part = "activado" if on else "desactivado"
    return (user, [action("switch.turn_on" if on else "switch.turn_off", [e])],
            R.choice([f"He {part} {e.name}.", f"Hecho, he {part} {e.name}."]))


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Riega las plantas del balcón" -> balcony valve, not the garden button)."""
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
            [f"He abierto {e.name}.", f"Hecho, {e.name} está en marcha."])
    return user, [action("button.press", [e])], R.choice(
        [f"He pulsado {e.name}.", f"He iniciado {e.name}."])


def i_irrigation_pair(h, loc):
    """A plant valve on the balcony or patio next to an area-less garden
    button, as many real installs have it. "Riega las plantas del balcón"
    opens the valve, "Riega el jardín" presses the button."""
    rooms = [r for r in h.rooms if r in ("Balcón", "Terraza")]
    if not rooms or irrigation(h):
        return None
    room = R.choice(rooms)
    valve = h.add("valve", room, "Riego", "m", ["plantas", "riego"],
                  name=R.choice([f"{room} Plantas", f"Plantas {room}",
                                 f"Riego {room}", "Válvula de agua"]),
                  state=R.choice(["open", "closed", "closed"]))
    button = h.add("button", None, "Riego", "m", ["riego_jardin"],
                   name=R.choice(["Riego Jardín", "Riego del jardín"]),
                   state="unknown")
    if R.random() < 0.6:
        en, dl = en_room(room), de_room(room)
        user = R.choice([f"Riega {AREA_PLANTS[room]}", f"Riega las plantas {en}",
                         f"Inicia el riego {dl}", f"Pon el riego {dl}"])
        return user, [action("valve.open_valve", [valve])], R.choice(
            [f"He abierto {valve.name}.", f"Hecho, {valve.name} está en marcha."])
    user = R.choice(["Riega el jardín", "Inicia el riego del jardín",
                     "Pon el riego del jardín", "Inicia el riego en el jardín"])
    return user, [action("button.press", [button])], R.choice(
        [f"He pulsado {button.name}.", f"He iniciado {button.name}."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Abre {r}", "Sube {r}", "Subir {r}", "Levanta {r}",
                         "Abrir {r}", "Sube del todo {r}", "¿Puedes subir {r}?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "abierto", "abierto"))
    if r < 0.8:
        user = R.choice(["Cierra {r}", "Baja {r}", "Bajar {r}", "Cerrar {r}",
                         "Baja del todo {r}", "¿Puedes bajar {r}?", "¿Puedes cerrar {r}?"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "cerrado", "cerrado"))
    if r < 0.88:
        user = R.choice(["Para {r}", "Detén {r}", "Stop {r}", "Parar {r}"])
        return (user.format(r=ref.acc), [action("cover.stop_cover", ref.ents)],
                R.choice(["Detenido.", f"He detenido {ref.nom}.", "Hecho."]))
    p, phrase = R.choice([(50, "a la mitad"), (30, "al 30 %"), (70, "al 70 %"),
                          (20, "al 20 %"), (80, "al 80 %"),
                          (50, "al 50 %"), (10, "al 10 %")])
    if phrase == "a la mitad":
        user = R.choice(["Baja {r} hasta la mitad", "Pon {r} a la mitad",
                         "{R} a media altura"]).format(r=ref.acc, R=cap(ref.acc))
    else:
        user = R.choice(["Pon {r} {ph}", "Ajusta {r} {ph}", "{R} {ph}"]).format(
            r=ref.acc, R=cap(ref.acc), ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} ya {esta(ref.g)} al {p} %.", "Hecho."])


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
            en, dl = en_room(room), de_room(room)
            rart = f"{ART[ROOM_BY_NAME[room][4]]} {low(room)}"
            user = R.choice([f"Pon la calefacción {dl} a {{t}} grados",
                             f"Calefacción {dl} a {{t}} grados",
                             f"Pon {rart} a {{t}} grados",
                             f"Quiero {{t}} grados {en}"]).format(t=es_num(t))
            nom, g = f"la calefacción {dl}", "f"
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Pon {r} a {t} grados", "{R} a {t} grados",
                             "Ajusta {r} a {t} grados", "Cambia {r} a {t} grados"]
                            ).format(r=ref.acc, R=cap(ref.acc), t=es_num(t))
            nom, g = ref.nom, ref.g
        conf = R.choice([f"{cap(nom)} ya {esta(g)} a {es_num(t)} grados.",
                         f"He puesto {nom} a {es_num(t)} grados.",
                         f"Hecho, {es_num(t)} grados."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Enciende {r}", "Encender {r}", "Pon en marcha {r}"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "encendido", "encendido")
    user = R.choice(["Apaga {r}", "Apagar {r}", "Desconecta {r}"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "apagado", "apagado")


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc in (e.name, e.name.lower()) and (R.random() < 0.7 or ref.acc == e.name.lower()):
        ref = Ref(f"{ART[e.gender]} {low(e.name)}", f"{ART[e.gender]} {e.name}",
                  e.gender, [e])
    if R.random() < 0.55:
        user = R.choice(["Cierra con llave {r}", "Echa la llave {ar}", "Bloquea {r}",
                         "¿Está cerrada {r}? Ciérrala con llave, por favor",
                         "Cerrar con llave {r}"])
        if "Ciérrala" in user:
            user = user.replace("Ciérrala", "Ciérra" + PRON[e.gender]).replace(
                "cerrada", agr("cerrado", e.gender))
        return (user.format(r=ref.acc, ar=al(ref.acc)), [action("lock.lock", [e])],
                confirm(ref, "cerrado con llave", "cerrado con llave"))
    user = R.choice(["Desbloquea {r}", "Quita la llave {ar}", "Abre la cerradura {dn}",
                     "Desbloquear {r}", "Quita el cerrojo {ar}"])
    return (user.format(r=ref.acc, ar=al(ref.acc), dn=de(the(e) if R.random() < 0.6 else e.name)),
            [action("lock.unlock", [e])],
            confirm(ref, "desbloqueado", "desbloqueado"))


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
            "Activa la alarma", "Conecta la alarma",
            "Me voy, activa la alarma", "Activar alarma",
            "Pon la alarma en modo ausente"])
        conf = R.choice(["La alarma está activada.",
                         "La alarma ya está activada (modo ausente)."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Pon la alarma en modo casa", "Activa la alarma en modo en casa",
            "Alarma en modo casa"])
        conf = "La alarma está activada en modo casa."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Activa la alarma para la noche", "Modo noche para la alarma",
            "Pon la alarma en modo noche"])
        conf = "La alarma está activada en modo noche."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Desactiva la alarma", "Desconecta la alarma", "Quita la alarma",
            "Ya estoy en casa, desactiva la alarma", "Apagar alarma"])
        conf = R.choice(["La alarma está desactivada.",
                         "He desactivado la alarma."])
    if code:
        user += f", código {code}"
    return user, [action(svc, [e], {"code": code} if code else None)], conf


def i_media(h, loc):
    ents = h.of("media_player")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    r = R.random()
    opts = [
        ("media_player.media_pause", ["Pausa {r}", "Pon en pausa {r}", "Pausar {r}",
                                      "Para la reproducción en {n}"], None,
         "En pausa."),
        ("media_player.media_play", ["Continúa la reproducción en {n}", "Reanuda {r}",
                                     "Play en {n}", "Sigue reproduciendo en {n}"], None,
         "La reproducción continúa."),
        ("media_player.media_next_track", ["Siguiente canción en {n}",
                                           "Pasa a la siguiente canción en {n}",
                                           "Siguiente tema en {n}"], None, "Siguiente canción."),
        ("media_player.volume_up", ["Sube el volumen {dr}", "Sube {r}"], None, "Volumen subido."),
        ("media_player.volume_down", ["Baja el volumen {dr}", "Baja {r}"], None, "Volumen bajado."),
        ("media_player.turn_off", ["Apaga {r}", "Apagar {r}", "Quita {r}"], None,
         confirm(ref, "apagado", "apagado")),
        ("media_player.turn_on", ["Enciende {r}", "Encender {r}"], None,
         confirm(ref, "encendido", "encendido")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Pon el volumen {dn} al {v} %",
                         "Volumen {dn} al {v} %",
                         "Pon {r} al {v} % de volumen"]).format(
            r=ref.acc, dn=de(the(e) if R.random() < 0.5 else e.name), v=v)
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"El volumen ya está al {v} %."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, dr=de(ref.acc), n=e.name), \
        [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = R.choice([f"{ART[e.gender]} {low(e.name)}" if e.name == e.label else e.name,
                  "el robot aspirador", "la aspiradora", "el robot"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Inicia {n}", f"Pon a aspirar {n}", f"Arranca {n}",
                         "Aspira la casa, por favor", f"Manda {n} a limpiar"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{cap(the(e))} se pone en marcha.", "El robot empieza a limpiar."])
    if r < 0.8:
        user = R.choice([f"Manda {n} a la base", f"Envía {n} a la base de carga",
                         f"Que vuelva {n} a la base"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{cap(the(e))} vuelve a la base de carga."
    user = R.choice([f"Pausa {n}", f"Para {n}", f"Detén {n}"])
    return user, [action("vacuum.pause", [e])], f"{cap(the(e))} está en pausa."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Activa la escena {e.name}", f"Escena {e.name}",
                         f"Activar {e.name}", f"Pon la escena {e.name}",
                         f"Inicia la escena {e.name}"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"La escena {e.name} está activada.", f"Escena {e.name} activada.", "Hecho."])
    if e.domain == "script":
        user = R.choice([f"Ejecuta {e.name}", f"Inicia {e.name}", f"Lanza {e.name}",
                         f"Ejecuta el script {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} está en marcha.", f"He iniciado {e.name}.", "Hecho."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    # a group entity ("Todas las luces") replaces the list of members
    if (grp := h.group(domain)) is not None:
        ents = [grp]
    phrase, g = R.choice(ALL_GROUP[domain])
    whole = S.ALL_AREAS if domain in S.AREA_DOMAINS and h.group(domain) is None else None
    ref = Ref(phrase, phrase, g, ents, whole)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Apaga {phrase}", f"Apagar {phrase}", f"Apágame {phrase}",
                             "Apaga todas las luces de la casa"])
            return user, [action("light.turn_off", ref)], R.choice(
                [f"{cap(phrase)} {esta(g)} {agr('apagado', g)}.",
                 "Todas las luces están apagadas.", "Hecho."])
        user = R.choice([f"Enciende {phrase}", f"Encender {phrase}", f"Enciéndeme {phrase}"])
        return user, [action("light.turn_on", ref)], R.choice(
            [f"{cap(phrase)} {esta(g)} {agr('encendido', g)}.",
             "Todas las luces están encendidas.", "Hecho."])
    if R.random() < 0.6:
        user = R.choice([f"Cierra {phrase}", f"Baja {phrase}", f"Bajar {phrase}",
                         f"Baja del todo {phrase}"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{cap(phrase)} están {agr('cerrado', g)}.", "Hecho, todo cerrado."])
    user = R.choice([f"Abre {phrase}", f"Sube {phrase}", f"Subir {phrase}",
                     f"Sube del todo {phrase}"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{cap(phrase)} están {agr('abierto', g)}.", "Hecho, todo abierto."])


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


# imperatives that may start a second command or follow "En 10 minutos"
VERBS = ("Enciende", "Encender", "Enciéndeme", "Prende", "Apaga", "Apagar", "Apágame",
         "Pon", "Ponme", "Sube", "Subir", "Baja", "Bajar", "Abre", "Abrir", "Cierra",
         "Cerrar", "Para", "Parar", "Detén", "Ajusta", "Cambia", "Activa", "Activar",
         "Desactiva", "Desactivar", "Inicia", "Iniciar", "Arranca", "Pausa", "Pausar",
         "Reanuda", "Reproduce", "Atenúa", "Manda", "Envía", "Saca", "Sacar", "Recoge",
         "Recoger", "Extiende", "Levanta", "Bloquea", "Desbloquea", "Desbloquear",
         "Echa", "Ejecuta", "Lanza", "Pulsa", "Riega", "Conecta", "Desconecta", "Quita",
         "Haz", "Continúa", "Guarda", "Corta", "Dale", "Aspira")
NO_JOIN = ("¿", "Por favor", "Necesito", "Necesitamos", "Está ", "Me ", "Tengo", "Quiero",
           "Ya estoy", "Que ", "Oye", "Vale", "Hace ")


def join_commands(a, b):
    """Two command intents -> one sentence 'X y y', or None."""
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    if ub.startswith(NO_JOIN) or ua.startswith(NO_JOIN) or "," in ub:
        return None
    if ua.endswith(", por favor"):
        ua = ua[:-len(", por favor")]
    # lowercase the leading verb of the second command ("... y apaga ...")
    first = ub.split(" ", 1)
    if first[0] in VERBS:
        ub = first[0].lower() + (" " + first[1] if len(first) > 1 else "")
    return f"{ua} {y_word(ub)} {ub}"


def join_answers(a, b):
    """Two confirmations -> 'La luz está encendida y he bajado la persiana.'"""
    for pre in ("Vale, ", "Listo, ", "Hecho, "):
        if b.startswith(pre):
            b = b[len(pre):]
            break
    if b.split(" ", 1)[0] in ("La", "El", "Las", "Los", "He", "Ya"):
        b = lower_start(b)
    return f"{a.rstrip('.')} {y_word(b)} {b}"


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    actions = a[1] + b[1]
    conf = R.choice(["Hecho.", "Las dos cosas están hechas.",
                     join_answers(a[2], b[2])
                     if a[2] != "Hecho." and b[2] != "Hecho." else "Hecho."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def _state(e: Ent, g=None):
    """-> (state key, predicate 'está encendida', word the answer contains)."""
    g = g or e.gender
    d, s = e.domain, e.state
    v = esta(g)
    if s == "unavailable":
        return "unavailable", f"no {v} {agr('disponible', g)}", "disponible"
    if d in ("light", "switch", "fan", "input_boolean"):
        w = agr("encendido" if s == "on" else "apagado", g)
        return ("on" if s == "on" else "off"), f"{v} {w}", w
    if d in ("cover", "binary_sensor"):
        is_open = s == "open" if d == "cover" else s == "on"
        w = agr("abierto" if is_open else "cerrado", g)
        return ("open" if is_open else "closed"), f"{v} {w}", w
    if d == "lock":
        w = agr("cerrado", g)
        if s == "locked":
            return "locked", f"{v} {w} con llave", f"{w} con llave"
        return "unlocked", f"no {v} {w} con llave", f"no {v} {w}"
    if d == "media_player":
        return {"playing": ("playing", f"{v} reproduciendo algo", "reproduciendo"),
                "paused": ("paused", f"{v} en pausa", "pausa"),
                "off": ("off", f"{v} {agr('apagado', g)}", agr("apagado", g)),
                "idle": ("idle", f"{v} {agr('encendido', g)}, pero no reproduce nada",
                         "no reproduce")}[s]
    return s, f"{v} {s}", s


ON_WORDS = {"on", "off", "open", "closed"}


def s_question(h, loc):
    """State question -> get_state -> grounded Spanish answer."""
    kinds = []
    temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"
             and e.label not in ("Temperatura exterior", "Previsión")]
    if temps:
        kinds += ["temp"] * 2
    outs = [e for e in h.of("sensor") if e.label == "Temperatura exterior"]
    if outs:
        kinds += ["outdoor"] * 2
    fcs = [e for e in h.of("sensor") if e.label == "Previsión"]
    if fcs:
        kinds += ["forecast"]
    if h.of("climate"):
        kinds += ["climate"] * 2
    kinds += ["onoff"] * 3
    if len(h.of("light")) >= 2:
        kinds += ["which_lights"] * 2
    if [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "window"]:
        kinds += ["windows"] * 2
    if len(h.of("cover")) >= 2:
        kinds += ["which_covers"] * 2
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
    # list questions read a whole domain by area (q_area, "*" = whole house)
    # instead of listing entity_ids: in a house with many covers the model
    # otherwise picks only some of them (v16: 2 of 8 covers)
    q_area, q_dom, q_dc = None, "light", None
    if FORCE.get("qkind"):
        if FORCE["qkind"] not in kinds:
            return None
        kind = FORCE["qkind"]
    must = []
    if kind == "outdoor":
        # every outdoor sensor counts; forecasts and felt temperature do not
        user = R.choice(["¿Qué temperatura hace fuera?", "¿Cuánto frío hace fuera?",
                         "¿Cuál es la temperatura exterior?", "¿Cuántos grados hace afuera?",
                         "¿Qué temperatura hace en la calle?", "¿Hace calor fuera?"])
        vals = [es_num(float(e.state)) for e in outs]
        if len(outs) == 1:
            ans = R.choice([f"Fuera hay {vals[0]} grados.",
                            f"La temperatura exterior es de {vals[0]} grados."])
        else:
            ans = (f"{outs[0].name} marca {vals[0]} grados y "
                   f"{outs[1].name}, {vals[1]} grados.")
        ids, must = outs, vals
    elif kind == "forecast":
        if h.of("weather"):
            return None       # the weather entity answers with condition and course
        when = R.choice(fcs).attrs["_when"]
        ids = [e for e in fcs if e.attrs["_when"] == when]
        vals = [es_num(float(e.state)) for e in ids]
        user = R.choice([f"¿Qué temperatura hará {when}?", f"¿Cuántos grados hará {when}?",
                         f"¿Qué dice la previsión para {when}?"])
        if len(ids) == 1:
            ans = R.choice([f"Según la previsión, {when} habrá {vals[0]} grados.",
                            f"{cap(when)} se esperan {vals[0]} grados, según la previsión."])
        else:
            ans = f"Para {when}, {ids[0].name} indica {vals[0]} grados y {ids[1].name}, {vals[1]} grados."
        must = vals
    elif kind == "temp":
        e = R.choice(temps)
        val = es_num(float(e.state))
        if e.area is None or e.area in OUTDOOR or e.label != "Temperatura":
            # a sensor without area is not automatically outside, and a printer
            # bed or a freezer does not tell how warm the room is
            user = R.choice([f"¿Qué marca {e.name}?", f"¿Cuántos grados marca {e.name}?"])
            ans = f"{e.name} marca {val} grados."
        else:
            en, dl = en_room(e.area), de_room(e.area)
            user = R.choice([f"¿Qué temperatura hace {en}?", f"¿Cuántos grados hay {en}?",
                             f"¿Qué temperatura hay {en}?", f"¿Temperatura {dl}?"])
            ans = R.choice([f"{cap(en)} hay {val} grados.",
                            f"La temperatura {en} es de {val} grados."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        en = en_room(e.area) if e.area else ""
        user = R.choice([f"¿Qué humedad hay {en}?",
                         f"¿Cuánta humedad hay {en}?"]).replace(" ?", "?")
        ans = f"La humedad {en} es del {e.state} %.".replace("  ", " ")
        ids, must = [e], [e.state]
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = e.name.replace("Humedad suelo ", "")
            pp = f"{PLANT_ART[plant]} {low(plant)}"
            user = R.choice([f"¿Qué humedad tiene la tierra {de(pp)}?",
                             f"¿Hay que regar {pp}?",
                             f"¿Cómo está la humedad {de(pp)}?"])
            ans = f"La humedad de la tierra {de(pp)} es del {e.state} %."
        else:
            user = R.choice([f"¿A cuánto está {e.name}?", f"¿Cómo está {e.name}?",
                             f"¿Qué marca {e.name}?"])
            ans = f"{e.name} está al {e.state} %."
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"¿Cuánto consume ahora {e.name}?",
                         f"¿Qué marca {e.name}?", f"¿Cuánta potencia marca {e.name}?"])
        val = es_num(float(e.state))
        ans = f"{e.name} marca ahora {val} vatios."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"¿A cuántos grados está {agr('puesto', ref.g)} {ref.nom}?",
                             f"¿Qué temperatura tiene {agr('programado', ref.g)} {ref.nom}?"])
            val = es_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {esta(ref.g)} {agr('programado', ref.g)} a {val} grados."
        else:
            if e.area:
                en = en_room(e.area)
                user = R.choice([f"¿Qué temperatura hace {en}?", f"¿Cuántos grados hay {en}?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None   # ambiguous: sensor and thermostat in same room
            else:
                user = f"¿Qué temperatura mide {ref.nom}?"
                en = ""
            if "current_temperature" not in e.attrs:
                return None   # thermostat without its own sensor
            val = es_num(e.attrs["current_temperature"])
            ans = (f"{cap(en)} hay {val} grados." if en
                   else f"{cap(ref.nom)} mide {val} grados.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        g, v = ref.g, esta(ref.g)
        if e.domain == "cover":
            user, asked = R.choice([(f"¿{cap(v)} {agr('abierto', g)} {ref.nom}?", "open"),
                                    (f"¿Están {agr('bajado', g)} {ref.nom}?", "closed")
                                    if ref.plural else
                                    (f"¿Está {agr('cerrado', g)} {ref.nom}?", "closed"),
                                    (f"¿{cap(v)} {agr('cerrado', g)} {ref.nom}?", "closed")])
        else:
            user, asked = R.choice([(f"¿{cap(v)} {agr('encendido', g)} {ref.nom}?", "on"),
                                    (f"¿{'Siguen' if ref.plural else 'Sigue'} "
                                     f"{agr('encendido', g)} {ref.nom}?", "on"),
                                    (f"¿{cap(v)} {agr('apagado', g)} {ref.nom}?", "off")])
        key, pred, word = _state(e, g)
        ans = f"{cap(ref.nom)} {pred}."
        if key in ON_WORDS and R.random() < 0.8:
            ans = ("Sí, " if key == asked else "No, ") + ref.nom + f" {pred}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"on": "Sí, ", "off": "No, "}[asked]
            ans = f"{yes}{ref.nom} {pred}, con un brillo del {pct} %."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            en = en_room(room)
            user = R.choice([f"¿Qué luces están encendidas {en}?",
                             f"¿Queda alguna luz encendida {en}?"])
        else:
            ents = h.of("light")
            user = R.choice(["¿Qué luces están encendidas?", "¿Queda alguna luz encendida?",
                             "¿Hay lámparas encendidas?", "¿Qué lámparas están encendidas?"])
        q_area = room or S.ALL_AREAS
        if any(e.state == "unavailable" for e in ents):
            return None
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "Ahora mismo no hay ninguna luz encendida." if not room else \
                f"{cap(en_room(room))} no hay ninguna luz encendida ahora mismo."
            must = ["ninguna"]
        elif len(on) == 1:
            ans = f"Solo está encendida {on[0].name}."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"Están encendidas: {join_es(names)}."
            must = names
        ids = ents
    elif kind == "which_covers":
        room = None
        rooms = [r for r in h.rooms if len(h.of("cover", r)) >= 2]
        if rooms and R.random() < 0.35:
            room = R.choice(rooms)
            en = en_room(room)
            user = R.choice([f"¿Qué persianas están abiertas {en}?",
                             f"¿Queda alguna persiana abierta {en}?",
                             f"¿Qué persianas están subidas {en}?"])
        else:
            user = R.choice(["¿Qué persianas están abiertas?", "¿Queda alguna persiana abierta?",
                             "¿Qué persianas están subidas?", "¿Hay persianas abiertas todavía?",
                             "¿Qué persianas están cerradas?", "¿Qué persianas están bajadas?"])
        ents = h.of("cover", room or "*")
        q_area, q_dom = room or S.ALL_AREAS, "cover"
        if any(e.state == "unavailable" for e in ents):
            return None
        want_open = not any(w in user for w in ("cerradas", "bajadas"))
        hits = [e for e in ents if (e.state == "open") == want_open]
        word = "abiertas" if want_open else "cerradas"
        of_room = f" {de_room(room)}" if room else ""
        if not hits:
            other = "cerradas" if want_open else "abiertas"
            ans = R.choice([f"Todas las persianas{of_room} están {other}.",
                            f"Ninguna persiana{of_room} está {word[:-1]}; todas están {other}."])
            must = [other]
        elif len(hits) == len(ents):
            ans = f"Todas las persianas{of_room} están {word}."
            must = [word]
        elif len(hits) == 1:
            e = hits[0]
            adj = agr("abierto" if want_open else "cerrado", e.gender)
            ans = f"Solo {esta(e.gender)} {adj} {e.name}."
            must = [e.name]
        else:
            names = [e.name for e in hits]
            g = "fp" if all(e.gender in ("f", "fp") for e in hits) else "mp"
            ans = f"Están {agr('abierto' if want_open else 'cerrado', g)}: {join_es(names)}."
            must = names
        ids = ents
    elif kind == "windows":
        # only window sensors - presence and motion sensors are binary_sensors too
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") == "window"]
        if not ents:
            return None
        user, yes_if_open = R.choice([("¿Están cerradas todas las ventanas?", False),
                                      ("¿Queda alguna ventana abierta?", True),
                                      ("¿Qué ventanas están abiertas?", None),
                                      ("¿Están cerradas las ventanas?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "No, todas las ventanas están cerradas.",
                   False: "Sí, todas las ventanas están cerradas.",
                   None: "Todas las ventanas están cerradas."}[yes_if_open]
            must = ["cerradas"]
        else:
            names = [e.name for e in opn]
            ans = ("Está abierta: " + names[0] + "." if len(names) == 1 else
                   "Están abiertas: " + join_es(names) + ".")
            if yes_if_open is not None:
                ans = ("Sí. " if yes_if_open else "No. ") + ans
            must = names
        ids = ents
        q_area, q_dom, q_dc = S.ALL_AREAS, "binary_sensor", "window"
    elif kind == "door":
        # "¿Está abierta la puerta del balcón del salón?" -> that contact,
        # not the front door
        doors = [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]
        e = R.choice(doors)
        same = [d for d in doors if d.label == e.label]
        if e.area and (len(same) > 1 or R.random() < 0.5):
            what = f"la {low(e.label)} {room_phrase(e.area)}"
        elif len(same) == 1:
            what = f"la {low(e.label)}"
        else:
            return None
        if e.state == "unavailable":
            return None
        opn = e.state == "on"
        user, ask_open = R.choice([(f"¿Está abierta {what}?", True),
                                   (f"¿Está cerrada {what}?", False),
                                   (f"¿Está bien cerrada {what}?", False),
                                   (f"¿Se ha quedado abierta {what}?", True)])
        word = "abierta" if opn else "cerrada"
        ans = f"{'Sí' if opn == ask_open else 'No'}, {e.name} está {word}."
        ids, must = [e], [word]
    elif kind == "car":
        rng = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"])
        car = rng.name.rsplit(" ", 1)[0]
        art = next(a for c, a, _ in CARS if c == car)
        the_car = f"{art} {car}"
        soc = next((e for e in h.of("sensor") if e.name == f"{car} Batería"), None)
        if soc is not None and R.random() < 0.45:
            e = soc
            user = R.choice([f"¿Cuánta batería tiene {the_car}?",
                             f"¿Qué carga tiene {the_car}?",
                             f"¿Está cargado ya {the_car}?"])
            ans = f"{cap(the_car)} tiene la batería al {e.state} %."
        else:
            e = rng
            user = R.choice([f"¿Cuánta autonomía le queda {al(the_car)}?",
                             f"¿Cuántos kilómetros puede hacer todavía {the_car}?",
                             f"¿Qué autonomía tiene {the_car}?"])
            ans = f"{cap(the_car)} tiene todavía {e.state} kilómetros de autonomía."
        ids, must = [e], [e.state]
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = f"{ART[e.gender]} {e.name}"
        c = agr("cerrado", e.gender)
        user = R.choice([f"¿Está {c} con llave {nom}?", f"¿Está {c} {nom}?",
                         f"¿He cerrado con llave {nom}?"])
        key, pred, word = _state(e)
        ans = (f"Sí, {nom} está {c} con llave." if e.state == "locked"
               else f"No, {nom} no está {c} con llave.")
        ids, must = [e], [word]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        if e.state == "unavailable":
            return None   # unreachable devices: see _error_question
        user = R.choice(["¿Está activada la alarma?", "¿Está puesta la alarma?",
                         "¿Qué estado tiene la alarma?"])
        txt = {"disarmed": "La alarma está desactivada.",
               "armed_away": "La alarma está activada (modo ausente).",
               "armed_home": "La alarma está activada en modo casa.",
               "armed_night": "La alarma está activada en modo noche."}[e.state]
        ans, ids, must = txt, [e], ["desactivada" if e.state == "disarmed" else "activada"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"¿Qué está sonando en {e.name}?", f"¿Qué se reproduce en {e.name}?"])
        t = e.attrs.get("media_title", "")
        a = e.attrs.get("media_artist")
        ans = f"En {e.name} está sonando {t}" + (f", de {a}." if a else ".")
        ids, must = [e], [t]
    else:
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [e.eid for e in ids]}
    if q_area:
        args = {"domain": q_dom, "area": q_area}
        if q_dc:
            args["device_class"] = q_dc
        # the integration returns them in device-list order
        ids = sorted(ids, key=lambda e: S.sort_key((e.eid, e.name, e.area)))
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of(ids)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": must}


COMPOUND_QKINDS = ["temp", "outdoor", "climate", "onoff", "which_lights", "which_covers",
                   "windows", "door", "lock", "alarm", "humidity", "car"]


def s_compound_question(h, loc):
    """Two questions in one sentence ("¿Qué persianas están abiertas y qué
    temperatura hace fuera?"): both are read and answered (v16 dropped the
    second part). Two entity_id lists go into one get_state call, a domain
    query and another question into two calls."""
    saved = FORCE.get("qkind")
    parts = []
    try:
        for k in R.sample(COMPOUND_QKINDS, len(COMPOUND_QKINDS)):
            if len(parts) == 2:
                break
            FORCE["qkind"] = k
            out = s_question(h, loc)
            if out:
                parts.append(out)
    finally:
        if saved is None:
            FORCE.pop("qkind", None)
        else:
            FORCE["qkind"] = saved
    if len(parts) < 2:
        return None
    (m1, meta1), (m2, meta2) = parts
    u1, c1, t1, a1 = m1[-4:]
    u2, c2, t2, a2 = m2[-4:]
    q1 = u1["content"].rstrip("?.! ")
    q2 = u2["content"].strip()
    for pre in ("Oye, por favor, ", "Oye, ", "Vale, ", "Por favor, "):
        q2 = q2.removeprefix(pre)
    q2 = q2.removeprefix("¿")
    q2 = q2[0].lower() + q2[1:]
    # one pair of question marks around the whole sentence, or none
    q2 = q2.rstrip("?.! ") + ("?" if "¿" in q1 else "")
    msgs = m1[:-4] + [{"role": "user", "content": f"{q1} {y_word(q2)} {q2}"}]
    g1, g2 = meta1["gold"][0][1], meta2["gold"][0][1]
    if list(g1) == ["entity_ids"] and list(g2) == ["entity_ids"]:
        ids = list(dict.fromkeys(g1["entity_ids"] + g2["entity_ids"]))
        rows = {r["entity_id"]: r for r in json.loads(t1["content"]) + json.loads(t2["content"])}
        args = {"entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([rows[i] for i in ids]))]
        gold = [["get_state", args]]
    else:
        msgs += [c1, t1, c2, t2]
        gold = [["get_state", g1], ["get_state", g2]]
    msgs += [{"role": "assistant", "content": f"{a1['content']} {a2['content']}"}]
    return msgs, {"cat": "compound", "gold": gold,
                  "answer_contains": meta1["answer_contains"] + meta2["answer_contains"]}


def s_appliance_question(h, loc):
    """'¿Ha terminado la secadora?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = f"{ART[g]} {low(app)}"
    user = R.choice([f"¿Ha terminado {nom}?", f"¿Sigue funcionando {nom}?", f"¿Qué hace {nom}?",
                     f"¿Ya ha acabado {nom}?", f"¿Cuál es el estado {de(nom)}?"])
    phrase, key = app_answer(kind, g)
    ans = R.choice([f"{cap(nom)} {phrase}.", f"Según Home Assistant, {nom} {phrase}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [key]}


def s_relative_climate(h, loc):
    """'Sube un poco la temperatura del baño' -> get_state -> set_temperature
    target +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    en, dl = en_room(e.area), de_room(e.area)
    up = R.random() < 0.6
    user = R.choice([f"Sube un poco la temperatura {en}", f"Tengo frío {en}",
                     f"Sube un grado la calefacción {dl}"] if up else
                    [f"Baja un poco la temperatura {en}", f"Hace demasiado calor {en}",
                     f"Baja un grado la calefacción {dl}"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, ok_result()),
             {"role": "assistant",
              "content": f"He {'subido' if up else 'bajado'} la calefacción {dl} de "
                         f"{es_num(e.attrs['temperature'])} a {es_num(new)} grados."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


MISSING_TYPES = [
    # domain, phrase, gender, "ninguna luz", "una luz", verbs (on, off),
    # infinitives (on, off), services (on, off), state words (on, off), label filter
    ("light", "la luz", "f", "ninguna luz", "una luz", ("Enciende", "Apaga"),
     ("encender", "apagar"), ("light.turn_on", "light.turn_off"),
     ("encendido", "apagado"), None),
    ("light", "la lámpara", "f", "ninguna lámpara", "una lámpara", ("Enciende", "Apaga"),
     ("encender", "apagar"), ("light.turn_on", "light.turn_off"),
     ("encendido", "apagado"), None),
    ("cover", "las persianas", "fp", "ninguna persiana", "persianas", ("Sube", "Baja"),
     ("subir", "bajar"), ("cover.open_cover", "cover.close_cover"),
     ("abierto", "cerrado"), "plain"),
    ("climate", "la calefacción", "f", "ninguna calefacción", "calefacción",
     ("Enciende", "Apaga"), ("encender", "apagar"),
     ("climate.turn_on", "climate.turn_off"), ("encendido", "apagado"), None),
    ("media_player", "la tele", "f", "ninguna tele", "una tele", ("Enciende", "Apaga"),
     ("encender", "apagar"), ("media_player.turn_on", "media_player.turn_off"),
     ("encendido", "apagado"), "Televisor"),
    ("fan", "el ventilador", "m", "ningún ventilador", "un ventilador", ("Enciende", "Apaga"),
     ("encender", "apagar"), ("fan.turn_on", "fan.turn_off"), ("encendido", "apagado"), None),
]


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    mt = FORCE.get("mtype")
    dom, phrase, pg, ning, una, verbs, infs, svcs, words, filt = R.choice(
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
    en, dl = en_room(room), de_room(room)
    # the room often holds a same-domain device of another kind (speaker, voice
    # satellite); it is no TV and must not stand in for the room in the answer
    if dom == "media_player" and R.random() < 0.6 and \
            not any(e.area == room for e in h.of(dom)):
        if R.random() < 0.6:
            h.add_voice_media(room)
        else:
            mt = R.choice([t for t in TYPES["media_player"] if t[0] != "Televisor"])
            h.add("media_player", room, mt[0], mt[1], mt[2], state=R.choice(["idle", "off"]))
    turn_on = R.random() < 0.5
    verb = verbs[0] if turn_on else verbs[1]
    inf = infs[0] if turn_on else infs[1]
    user = R.choice([f"{verb} {phrase} {dl}", f"{verb} {phrase} {en}",
                     f"{cap(inf)} {phrase.split(' ', 1)[1]} {en}",
                     f"¿Puedes {inf} {phrase} {dl}?"])
    others = [e for e in ents if e.area and e.area != room]
    # suggest the other device only if there are one or two candidates
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        oen = en_room(alt.area)
        where = "" if alt.area in alt.name else f" {oen}"
        ans = R.choice([f"{cap(en)} no hay {ning}. Hay {una} {oen}: {alt.name}.",
                        f"{cap(en)} no encuentro {ning}. ¿Te refieres a {alt.name}{where}?",
                        f"No tengo {ning} {en} en la lista de dispositivos. {cap(oen)} está {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"{cap(en)} no encuentro {ning} que pueda controlar.",
                        f"{cap(en)} no hay {ning} en la lista de dispositivos.",
                        f"No he encontrado {ning} {en} que pueda {inf}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        # user accepts the device in the other room -> now it is switched
        oen, odl = en_room(alt.area), de_room(alt.area)
        where = "" if alt.area in alt.name else f" {oen}"
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Pues {ART[alt.gender]} {odl}", "Sí, ese mismo" if alt.gender == "m" else "Sí, esa misma",
             "Vale, pues ese" if alt.gender == "m" else "Vale, pues esa",
             f"Sí, {alt.name}", f"Pues {oen} entonces"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        word = agr(words[0] if turn_on else words[1], alt.gender)
        msgs += [call, tool_msg(cid, ok_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"{alt.name}{where} ya {esta(alt.gender)} {word}.",
                      f"Hecho, {alt.name}{where} {esta(alt.gender)} {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        # user insists on the room, sometimes twice -> still no invented action
        pushes = ["¡Que sí, hazlo!", "Sí, enciéndelo", "Que sí que hay",
                  "Hazlo igualmente", f"¡Sí, {en}!", f"No, digo {en}",
                  "¿Seguro que no?", "Sí que hay una", "Tú hazlo"]
        refusals = [
            f"{cap(en)} no hay {ning} en tu lista de dispositivos. Comprueba si el dispositivo está expuesto a Assist.",
            f"De verdad que no veo {ning} {en} que pueda controlar.",
            f"{cap(en)} no hay {ning} en mi lista de dispositivos, así que no puedo controlar nada allí."]
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
        if dev.lower() in labels or (dev == "Cortacésped" and h.of("lawn_mower")):
            return None
        acc = f"{ART[g]} {low(dev)}"
        user = R.choice([f"Enciende {acc}", f"Pon en marcha {acc}", f"Apaga {acc}",
                         f"¿Está {agr('encendido', g)} {acc}?", f"Arranca {acc}"])
        ans = R.choice([f"No encuentro ningún dispositivo llamado {dev} en tu casa.",
                        f"No conozco ningún dispositivo {dev}, así que no puedo controlarlo.",
                        f"{dev} no aparece en la lista de dispositivos."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != "Cuarto de baño"
                   and not (r_[0] == "Baño" and "Cuarto de baño" in h.rooms)]
        room = R.choice(missing)
        en, dl = en_room(room), de_room(room)
        what = R.choice([("la luz", "ninguna luz", "Enciende", "Apaga"),
                         ("las persianas", "ninguna persiana", "Sube", "Baja"),
                         ("la calefacción", "ninguna calefacción", "Enciende", "Apaga")])
        user = R.choice([f"{what[2]} {what[0]} {dl}",
                         f"{what[3]} {what[0]} {dl}",
                         f"{what[3]} {what[0]} {en}"])
        ans = R.choice([f"{cap(en)} no encuentro {what[1]} que pueda controlar.",
                        f"No he encontrado {what[1]} {en} en la lista de dispositivos."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


WEATHER_CHAT_Q = "¿Qué tiempo hará mañana?"
CHAT = [
    (["Gracias", "Muchas gracias", "Gracias, majo", "Genial, gracias"],
     ["De nada.", "Con mucho gusto.", "No hay de qué."]),
    (["Hola", "Buenas", "Buenos días", "Buenas tardes"],
     ["Hola, ¿en qué te puedo ayudar?", "Hola. ¿Qué necesitas?"]),
    (["¿Qué sabes hacer?", "¿En qué me puedes ayudar?", "¿Qué puedes controlar?"],
     ["Puedo controlar las luces, las persianas, la calefacción y otros dispositivos de tu "
      "casa, y decirte en qué estado están o qué miden los sensores. Además pongo música, "
      "pongo temporizadores, gestiono tus listas, te digo la hora y el tiempo y hago "
      "anuncios por la casa.",
      "Controlo tus dispositivos, pongo música y temporizadores, apunto cosas en tu lista "
      "de la compra, te digo la hora y el tiempo y hago anuncios en casa."]),
    (["¿Quién eres?"], ["Soy tu asistente de voz para el hogar inteligente."]),
    (["Cuéntame un chiste"],
     ["¿Por qué la lámpara no tiene amigos? Porque siempre se funde enseguida.",
      "¿Qué le dice la persiana a la ventana? Tranquila, yo te cubro."]),
    ([WEATHER_CHAT_Q],
     ["Lo siento, no tengo previsión del tiempo. Pero puedo decirte lo que miden ahora "
      "tus sensores."]),
    (["Buenas noches"], ["Buenas noches. ¿Quieres que apague algo más?", "Buenas noches."]),
    # HassNevermind: nothing to do
    (["Olvídalo", "Cancelar", "Déjalo", "Nada", "Da igual", "Ah, nada",
      "Mejor no", "Ya está solucionado", "Olvídalo, de verdad", "No, nada",
      "No he dicho nada", "Está bien, déjalo", "No, déjalo"],
     ["Vale.", "De acuerdo.", "Vale, entonces no."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == WEATHER_CHAT_Q and (
            h.of("weather") or any(e.label == "Previsión" for e in h.of("sensor"))):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


PLURAL_WORDS = {"Luces", "Lámparas", "Persianas", "Estores", "Venecianas", "Focos",
                "Radiadores", "Enchufes"}


def is_plural_name(name: str) -> bool:
    """'Todas las luces', 'Salón Persianas' take a plural verb."""
    words = name.split()
    return bool(words) and (words[0] in ("Todas", "Todos")
                            or any(w in PLURAL_WORDS for w in words))


def nd(name):
    """'X no está disponible' / 'Todas las luces no están disponibles'."""
    if is_plural_name(name):
        return f"{name} no están disponibles"
    return f"{name} no está disponible"


def _unreachable(name):
    pl = is_plural_name(name)
    return R.choice([f"No ha funcionado. {name} no {'responden' if pl else 'responde'} ahora mismo.",
                     f"Ha habido un error: {nd(name)}.",
                     f"{nd(name)} ahora mismo, así que no he podido ejecutar la orden."])


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
            ["Inténtalo otra vez", "Otra vez, por favor", "Prueba de nuevo",
             "Vuelve a intentarlo"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": R.choice([
                     f"{name} {'siguen' if is_plural_name(name) else 'sigue'} sin estar disponible. Comprueba que el dispositivo tenga corriente y conexión.",
                     f"El segundo intento tampoco ha funcionado, {name} no {'responden' if is_plural_name(name) else 'responde'}."])}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        msgs.append({"role": "user", "content": voice(R.choice(
            ["¿Ya funciona?", "¿Seguro que no?", "¿Ha funcionado?", "¿Y bien, ya está?"]
            if onoff else ["¿Ha funcionado?", "¿Y, ha funcionado?", "¿Seguro?"]))})
        e = next(x for x in h.ents if x.eid == eid)
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": R.choice([
                     f"No. {nd(name)}, así que no he podido controlarlo.",
                     f"Según Home Assistant, {nd(name)} ahora mismo. No se ha cambiado nada."])}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def names_pred(h, ids, adj):
    """'Plafón Cocina está encendido' / 'A y B están apagadas'."""
    by_id = {e.eid: e for e in h.ents}
    ents = [by_id[i] for i in ids]
    g = group_g([plural_g(e.gender) if len(ents) > 1 else e.gender for e in ents])
    return f"{_names(h, ids)} {esta(g)} {agr(adj, g)}"


def _down_txt(h, down):
    ids = [e.eid for e in down]
    pl = len(down) > 1
    return f"{_names(h, ids)} no {esta(pl)} {'disponibles' if pl else 'disponible'} ahora mismo."


def _error_room(h, loc):
    """Some lights of a room are unreachable, the others work. Questions and
    commands about the room report both groups by name."""
    rooms = [r for r in h.rooms if len([e for e in h.of("light", r)
                                         if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    # rooms with a light named like the room are the hard case ("Cocina" is
    # down, "Cocina Izquierda" is not), so they come up more often
    named_rooms = [r for r in rooms if any(e.name == r for e in h.of("light", r))]
    # forced variant: question about a room whose light named like the room is
    # down next to others; the model tended to drop the room-named light from
    # the answer
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
    en, dl = en_room(room), de_room(room)
    ids = [e.eid for e in ents]
    nd = _names(h, [e.eid for e in down])
    down_txt = _down_txt(h, down)
    msgs = base_msgs(h, loc)
    if named_q or R.random() < 0.5:
        # question
        user, kind = R.choice([(f"¿Está encendida la luz {dl}?", "yn"),
                               (f"¿Queda alguna luz encendida {en}?", "yn"),
                               (f"¿Qué luces están encendidas {en}?", "which"),
                               (f"¿Están apagadas las lámparas {dl}?", "yn_off")])
        on = [e for e in up if e.state == "on"]
        off = [e for e in up if e.state != "on"]
        parts = []
        if on:
            parts.append(names_pred(h, [e.eid for e in on], "encendido") + ".")
        if off and (not on or R.random() < 0.5):
            parts.append(names_pred(h, [e.eid for e in off], "apagado") + ".")
        if not up:
            ans = f"{cap(down_txt)} Por eso no puedo decirte si hay alguna luz encendida allí."
        else:
            if kind == "yn":
                parts[0] = ("Sí, " if on else "No, ") + parts[0]
            elif kind == "yn_off":
                # same wording as the "encendida?" question, only the yes/no
                # flips; a separate phrase made the model mix up both questions
                parts[0] = ("No, " if on else "Sí, ") + parts[0]
            ans = " ".join(parts) + " " + down_txt
        args = {"domain": "light", "area": room, "entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "error", "gold": [["get_state", args]]}
    # command for the whole room: the reachable lights switch
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    user = R.choice([f"Enciende la luz {dl}", f"Encender luz {en}",
                     f"Enciende las luces {dl}"] if on_cmd else
                    [f"Apaga la luz {dl}", f"Apagar luz {en}",
                     f"Apaga las luces {dl}"])
    actions = [action(svc, ents, area=room)]
    done = [e.eid for e in up]
    word = "encendido" if on_cmd else "apagado"
    if up:
        did = _names(h, done)
        ans = R.choice([
            f"He {word} {did}. {down_txt}",
            f"Solo ha funcionado en parte: {names_pred(h, done, word)}, pero {nd} "
            f"no {esta(len(down) > 1)} {'disponibles' if len(down) > 1 else 'disponible'} ahora mismo.",
            f"{down_txt} El resto lo he {word}: {did}.",
        ])
    else:
        ans = f"{down_txt} Por eso no he podido {'encender' if on_cmd else 'apagar'} la luz {dl}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, S.control_result(
                 [f"Entity {', '.join(e.eid for e in down)} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _named_room_light(h, loc):
    """A light carries the room's name ("Cocina") next to other lights there.
    "la luz de la cocina" means all lights of the room; only the bare name
    ("Enciende Cocina") means that single lamp."""
    rooms = [r for r in h.rooms
             if any(e.name == r for e in h.of("light", r))
             and len([e for e in h.of("light", r) if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = [e for e in h.of("light", room) if e.label != "__group__"]
    named = next(e for e in ents if e.name == room)
    en, dl = en_room(room), de_room(room)
    msgs = base_msgs(h, loc)
    r = R.random()
    if r < 0.3:
        user, kind = R.choice([(f"¿Está encendida la luz {dl}?", "yn"),
                               (f"¿Hay alguna luz encendida {en}?", "yn"),
                               (f"¿Queda luz encendida {en}?", "yn"),
                               (f"¿Qué luces están encendidas {en}?", "which")])
        on = [e for e in ents if e.state == "on"]
        off = [e for e in ents if e.state != "on"]
        if on:
            ans = names_pred(h, [e.eid for e in on], "encendido") + "."
            if off and R.random() < 0.6:
                ans += " " + names_pred(h, [e.eid for e in off], "apagado") + "."
            if kind == "yn":
                ans = "Sí, " + ans
        elif kind == "yn":
            ans = R.choice([f"No, {names_pred(h, [e.eid for e in off], 'apagado')}.",
                            f"No, {en} no hay ninguna luz encendida."])
        else:
            ans = f"{cap(en)} no hay ninguna luz encendida ahora mismo."
        args = {"domain": "light", "area": room, "entity_ids": [e.eid for e in ents]}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        must = [e.name for e in on] if on else ["no" if kind == "yn" else "ninguna"]
        return msgs, {"cat": "question", "gold": [["get_state", args]],
                      "answer_contains": must}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    adj = "encendido" if on_cmd else "apagado"
    if r < 0.8:
        # the room: every light in it
        user = R.choice([f"Enciende la luz {dl}", f"Enciende la luz {en}",
                         f"Encender luz {en}", f"Enciéndeme la luz {dl}",
                         f"¿Puedes encender la luz {dl}?"] if on_cmd else
                        [f"Apaga la luz {dl}", f"Apaga la luz {en}",
                         f"Apagar luz {en}", f"Apágame la luz {dl}",
                         f"¿Puedes apagar la luz {dl}?"])
        targets = Ref(f"la luz {dl}", f"la luz {dl}", "f", ents, room)
        conf = confirm(targets, adj, adj)
    else:
        # the bare name: only the lamp called like the room
        user = R.choice([f"Enciende {room}", f"Encender {room}", f"Enciende la lámpara {room}",
                         f"Enciende la luz {room}"] if on_cmd else
                        [f"Apaga {room}", f"Apagar {room}", f"Apaga la lámpara {room}",
                         f"Apaga la luz {room}"])
        targets = [named]
        conf = confirm(Ref(room, room, named.gender, targets), adj, adj)
    actions = [action(svc, targets)]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, ok_result()), {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_no_areas(h, loc):
    """Home Assistant without any areas: a room cannot be resolved. The model
    says so and explains how to set up areas; device names and "todas las
    luces" keep working."""
    lights = h.of("light")
    rooms = [r for r in h.rooms if h.of("light", r) or h.of("cover", r, plain=True)]
    if len(lights) < 2 or not rooms:
        return None
    room = R.choice(rooms)
    en, dl = en_room(room), de_room(room)
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
            user = R.choice([f"Enciende la luz {dl}", f"Apaga la luz {en}",
                             f"¿Está encendida la luz {dl}?", f"¿Qué luces están encendidas {en}?",
                             f"Encender luz {en}", f"Apágame la luz {dl}"])
            what = "luces"
        else:
            user = R.choice([f"Baja las persianas {dl}", f"Sube las persianas {en}",
                             f"¿Están bajadas las persianas {dl}?", f"Cierra las persianas {dl}"])
            what = "persianas"
        ans = R.choice([
            f"Tu Home Assistant no tiene áreas configuradas, así que no sé qué {what} hay {en}. "
            f"Crea las habitaciones en Ajustes, en «Áreas, etiquetas y zonas», y asígnales "
            f"los dispositivos.",
            f"No puedo saber qué hay {en} porque en Home Assistant no hay áreas creadas. "
            f"Configura las habitaciones en Ajustes, en «Áreas, etiquetas y zonas», y asigna "
            f"allí los dispositivos."])
        if named:
            ans += f" Mientras tanto, puedes llamar a un dispositivo por su nombre, por ejemplo «{named[0].name}»."
        msgs += [{"role": "user", "content": voice(user)}, {"role": "assistant", "content": ans}]
        return msgs, {"cat": "unknown", "gold": [], "no_history": True}
    on = R.random() < 0.5
    svc = "light.turn_on" if on else "light.turn_off"
    adj = "encendido" if on else "apagado"
    if r < 0.85 or h.group("light") is not None:
        # a device named directly still works
        e = R.choice(lights)
        user = R.choice([f"Enciende {e.name}", f"Encender {e.name}", f"Enciende {the(e)}"] if on
                        else [f"Apaga {e.name}", f"Apagar {e.name}", f"Apaga {the(e)}"])
        actions = [action(svc, [e])]
        conf = R.choice([f"{cap(the(e))} ya {esta(e.gender)} {agr(adj, e.gender)}.",
                         f"He {adj} {the(e)}."])
    else:
        # the whole house needs no areas
        user = R.choice(["Enciende todas las luces", "Enciende todas las lámparas"] if on
                        else ["Apaga todas las luces", "Apaga la luz de toda la casa"])
        actions = [action(svc, lights, area=S.ALL_AREAS)]
        conf = f"Todas las luces están {agr(adj, 'fp')}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]],
                  "no_history": True}


def partial_answer(h, ok_actions, failed_name, conf=None):
    """Both outcomes, always: what worked and what did not."""
    if any(len(a["entity_id"]) > 3 for a in ok_actions):
        # long lists: the normal confirmation ("Todas las luces están
        # encendidas.") is clearer
        if conf and conf != "Hecho.":
            return R.choice([f"{conf} Pero {nd(failed_name)} ahora mismo.",
                             f"{nd(failed_name)} ahora mismo. Por lo demás: {conf}"])
        n = sum(len(a["entity_id"]) for a in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = DESC.get(svc, "accionado {w}").format(w=f"{n} dispositivos",
                                                    dw=f"de {n} dispositivos")
    else:
        did = describe_all(h, ok_actions)
    return R.choice([
        f"He {did}. Pero {nd(failed_name)} ahora mismo.",
        f"Solo ha funcionado en parte: he {did}, pero {nd(failed_name)} ahora mismo.",
        f"{nd(failed_name)} ahora mismo. El resto está hecho: he {did}.",
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
    user = R.choice([f"¿Está {agr('encendido', e.gender)} {the(e)}?",
                     f"¿Cuál es el estado {de(the(e))}?",
                     f"¿Qué hace {the(e)}?", f"¿{e.name} está {agr('encendido', e.gender)}?"])
    args = {"entity_ids": [e.eid]}
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": R.choice([
                 f"{nd(e.name)} ahora mismo, así que no sé en qué estado está.",
                 f"No te lo puedo decir: {nd(e.name)} ahora mismo."])}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


def s_followup(h, loc):
    """Two-turn: command, then 'apágala otra vez' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    d1, d2 = de_room(r1), de_room(r2)
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Enciende la luz {d1}",
                                                            f"Encender luz {en_room(r1)}"]))}]
    a1 = [action("light.turn_on", e1, area=r1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": f"La luz {d1} está encendida."}]
    k = R.random()
    if k < 0.15:
        # user doubts the result -> check the real state instead of guessing
        doubt = "¿Seguro que está encendida?"
        u2 = R.choice(["No, no está encendida", doubt, "¿Es verdad?",
                       "Pues aquí sigue oscuro"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        q = {"domain": "light", "area": r1, "entity_ids": [e.eid for e in e1]}
        c2, cid2 = tool_call("get_state", q)
        ans = (f"Según Home Assistant, la luz {d1} está encendida." if really_on else
               f"Según Home Assistant, la luz {d1} sigue apagada. ¿Quieres que la encienda otra vez?")
        if u2 == doubt:
            ans = ("Sí. " if really_on else "No. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", q]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["Y apágala otra vez", "Apágala otra vez", "Mejor apágala"]), \
            [action("light.turn_off", e1, area=r1)], f"La luz {d1} vuelve a estar apagada."
    elif k < 0.72:
        en2 = en_room(r2)
        u2, a2, ans = R.choice([f"¿Y {en2}?", f"{cap(en2)} también", f"Y {en2} también, por favor"]), \
            [action("light.turn_on", e2, area=r2)], f"La luz {d2} también está encendida."
    else:
        kind = R.choice(["down", "up", "pct"])
        u2 = {"down": "Un poco menos de luz", "up": "Un poco más de luz", "pct": "Al 30 %"}[kind]
        if kind == "down":
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20}, area=r1)]
            ans = f"He bajado el brillo de la luz {d1}."
        elif kind == "up":
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20}, area=r1)]
            ans = f"He subido el brillo de la luz {d1}."
        else:
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30}, area=r1)]
            ans = f"La luz {d1} ya está al 30 %."
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
# participle for "¿Qué luz has ...?"
PARTICIPLE = {"turn_on": "encendido", "turn_off": "apagado",
              "open_cover": "abierto", "close_cover": "cerrado",
              "stop_cover": "detenido", "lock": "cerrado con llave", "unlock": "desbloqueado",
              "media_play": "reanudado", "media_pause": "pausado", "media_stop": "detenido",
              "start": "puesto en marcha", "pause": "pausado",
              "return_to_base": "mandado a la base",
              "start_mowing": "puesto en marcha", "dock": "mandado a la base",
              "open_valve": "abierto", "close_valve": "cerrado", "press": "pulsado",
              "alarm_arm_home": "activado", "alarm_arm_away": "activado",
              "alarm_arm_night": "activado", "alarm_disarm": "desactivado"}
# "he ..." with the devices ({w}) or "de" + devices ({dw})
DESC = {"turn_on": "encendido {w}", "turn_off": "apagado {w}",
        "open_cover": "abierto {w}", "close_cover": "cerrado {w}",
        "stop_cover": "detenido {w}", "lock": "cerrado con llave {w}",
        "unlock": "desbloqueado {w}",
        "media_play": "reanudado la reproducción en {w}", "media_pause": "pausado {w}",
        "media_stop": "detenido la reproducción en {w}",
        "media_next_track": "pasado a la siguiente canción en {w}",
        "media_previous_track": "vuelto a la canción anterior en {w}",
        "volume_up": "subido el volumen {dw}", "volume_down": "bajado el volumen {dw}",
        "start": "puesto en marcha {w}", "pause": "pausado {w}",
        "return_to_base": "mandado {w} a la base de carga",
        "start_mowing": "puesto en marcha {w}", "dock": "mandado {w} a la base de carga",
        "open_valve": "abierto {w}", "close_valve": "cerrado {w}", "press": "pulsado {w}",
        "alarm_arm_home": "activado {w} en modo casa", "alarm_arm_away": "activado {w}",
        "alarm_arm_night": "activado {w} en modo noche", "alarm_disarm": "desactivado {w}"}
DOMAIN_NOUN = {"light": ("¿Qué luz", "¿Qué lámparas"),
               "cover": ("¿Qué persiana", "¿Qué persianas"),
               "climate": ("¿Qué calefacción", "¿Qué radiadores"),
               "switch": ("¿Qué aparato", "¿Qué aparatos"),
               "media_player": ("¿Qué altavoz", "¿Qué aparatos"),
               "fan": ("¿Qué ventilador", "¿Qué ventiladores")}


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
    return join_es([by_id[i] for i in ids])


def describe(h, a, by_name=False):
    """'encendido Plafón Pasillo' for one action, behind 'He ...'."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if a.get("area") == S.ALL_AREAS and not by_name:
        who = "todas las luces"
    elif a.get("area") and not by_name:
        who = f"la luz {de_room(a['area'])}"
    else:
        by_id = {e.eid: e for e in h.ents}
        who = join_es([the(by_id[i]) if i in by_id else i for i in a["entity_id"]])
    for fn in DESCRIBE_HOOKS:
        out = fn(h, a, who)
        if out is not None:
            return out
    if "brightness_pct" in data:
        return f"puesto {who} al {data['brightness_pct']} %"
    if "brightness_step_pct" in data:
        return f"{'subido' if data['brightness_step_pct'] > 0 else 'bajado'} el brillo {de(who)}"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"cambiado el color {de(who)}"
    if svc == "set_temperature":
        return f"puesto {who} a {es_num(data['temperature'])} grados"
    if svc == "set_cover_position":
        return f"puesto {who} al {data['position']} %"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"puesto {who} al {data.get('percentage')} %"
    if svc == "volume_set":
        return f"puesto el volumen {de(who)} al {round(data['volume_level'] * 100)} %"
    if dom == "scene":
        return f"activado la escena {who}"
    if dom == "script":
        return f"iniciado {who}"
    return DESC.get(svc, "accionado {w}").format(w=who, dw=de(who))


def merge_actions(actions):
    """Consecutive list items for the same list become one action, so the
    answer says 'he añadido leche y huevos a la lista de la compra'."""
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
    """'encendido X, puesto Y al 50 % y añadido Z a la lista' for several actions."""
    return join_es([describe(h, a) for a in merge_actions(actions)])


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
    """'¿Qué luz has apagado?' -> answered from the history."""
    if not c.last:
        return False
    a = c.last[-1]
    dom, svc = a["service"].split(".", 1)
    ids = a["entity_id"]
    if dom in DOMAIN_NOUN and svc in PARTICIPLE and not a.get("data"):
        sg, pl = DOMAIN_NOUN[dom]
        q = (pl if len(ids) > 1 else sg) + f" has {PARTICIPLE[svc]}?"
        user = R.choice([q, q, "¿Qué acabas de hacer?", "¿Qué aparato era?"])
    else:
        user = R.choice(["¿Qué acabas de hacer?", "¿Qué has hecho exactamente?"])
    c.user(user)
    # the lamps an area stood for are known from "done" in the tool result
    parts = [describe(c.h, x, by_name=True) for x in merge_actions(c.last)]
    c.say(R.choice(["He ", "Hace un momento he "]) + join_es(parts) + ".")
    return True


def turn_summary(c):
    """'¿Qué has hecho hasta ahora?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    # "¿Qué radiadores has cambiado?" after the bathroom + "En la cocina
    # también": every action on that device type, not only the last one
    by_dom = {}
    for x in c.done:
        by_dom.setdefault(x["service"].split(".")[0], []).append(x)
    multi = [d for d, xs in by_dom.items() if len(xs) >= 2 and d in SUMMARY_ASK]
    if multi and R.random() < 0.6:
        d = R.choice(multi)
        c.user(R.choice(SUMMARY_ASK[d]))
        parts = [describe(c.h, x) for x in merge_actions(by_dom[d])]
    else:
        c.user(R.choice(["¿Qué has hecho hasta ahora?", "¿Qué has cambiado hasta ahora?",
                         "Resúmeme lo que has hecho"]))
        parts = [describe(c.h, x) for x in merge_actions(c.done)]
    c.say("He " + join_es(parts) + ".")
    return True


SUMMARY_ASK = {
    "climate": ["¿Qué calefacciones has cambiado?", "¿Qué termostatos has ajustado?",
                "¿Qué has cambiado en la calefacción?", "¿Qué radiadores has ajustado?"],
    "light": ["¿Qué luces has tocado?", "¿Qué lámparas has encendido o apagado?",
              "¿Qué has hecho con las luces?"],
    "cover": ["¿Qué persianas has movido?", "¿Qué estores has movido?",
              "¿Qué has hecho con las persianas?"],
}


def _clitic(h, ids):
    """'lo' / 'la' / 'los' / 'las' for the devices of the last action."""
    by_id = {e.eid: e for e in h.ents}
    ents = [by_id[i] for i in ids if i in by_id]
    if not ents:
        return "lo"
    if len(ents) == 1:
        return PRON[ents[0].gender]
    return PRON[group_g([plural_g(e.gender) for e in ents])]


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
    p = _clitic(c.h, inv[0]["entity_id"])
    c.user(R.choice(["Deshazlo", "Mejor no", "Vuelve a dejarlo como estaba",
                     "Anula eso"] +
                    ([f"Y apága{p} otra vez", f"Apága{p} otra vez"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    ([f"Y enciénde{p} otra vez"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "Vale, he " + describe_all(c.h, inv) + ".")
    return True


def turn_correct(c):
    """'No, quería decir en el dormitorio' -> undo + the right device."""
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
    en = en_room(o.area)
    if R.random() < 0.6:
        # "quería decir en el comedor": every device of that kind in the room
        c.user(R.choice([f"No, quería decir {en}", f"Me he equivocado, {en}",
                         f"No, {en}", f"Quería decir {en}"]))
        new_ids = [e.eid for e in c.h.of(dom, o.area, plain=dom == "cover")
                   if e.state != "unavailable"]
        by_room = len(new_ids) == len(c.h.of(dom, o.area, plain=dom == "cover"))
    else:
        c.user(R.choice([f"No, me refería {al(the(o))}", f"Quería decir {o.name}"]))
        new_ids = [o.eid]
        by_room = False
    undo = {"service": f"{dom}.{INVERSE[svc]}", "entity_id": cur_ids}
    if "area" in a:
        undo = {"service": undo["service"], "area": a["area"], "entity_id": cur_ids}
    new = {"service": a["service"], "entity_id": new_ids}
    if dom in S.AREA_DOMAINS and by_room:
        new = {"service": a["service"], "area": o.area, "entity_id": new_ids}
    acts = [undo, new]
    c.control(acts, f"Vale, he {describe(c.h, acts[0])} y {describe(c.h, acts[1])}.")
    return True


def turn_correct_recap(c):
    """Correction followed by "¿Qué has hecho hasta ahora?": the recap lists
    exactly the executed actions."""
    if not turn_correct(c):
        return False
    if R.random() < 0.7:
        return turn_summary(c)
    return turn_which(c)


def turn_also(c):
    """'Y en el baño también' -> same service on the same device type in
    another room."""
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
    en = en_room(r)
    ents = [e for e in c.h.of(dom, r, plain=dom == "cover") if e.state != "unavailable"]
    c.user(R.choice([f"Y {en} también", f"{cap(en)} también", f"Lo mismo {en}"]))
    act = {"service": a["service"],
           **({"area": r} if dom in S.AREA_DOMAINS and len(ents) == len(c.h.of(dom, r)) else {}),
           "entity_id": [e.eid for e in ents]}
    if a.get("data"):
        act["data"] = a["data"]
    c.control([act], R.choice(["Vale, he ", "Hecho, he "])
              + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["Un poco más de luz", "Más brillo", "Sube un poco más"]), \
            {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["Un poco menos de luz", "Menos brillo", "Baja un poco más"]), \
            {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = f"Al {p} %", {"brightness_pct": p}
    act = {"service": "light.turn_on",
           **({"area": c.last[0]["area"]} if "area" in c.last[0] else {}),
           "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "Vale, he " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'¿Ya está encendida?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    area = c.last[-1].get("area")
    ents = [e for e in c.h.ents if e.eid in ids]
    if not ents or ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    g = ents[0].gender if len(ents) == 1 else group_g([plural_g(e.gender) for e in ents])
    if len(ents) == 1 and is_plural_name(ents[0].name):
        g = plural_g(g)
    pairs = {"cover": [("abierto", "open"), ("cerrado", "closed"), ("bajado", "closed"),
                       ("subido", "open")],
             "lock": [("cerrado con llave", "locked"), ("abierto", "unlocked")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("encendido", "on"), ("apagado", "off")]))
    question = f"¿Ya {esta(g)} {agr(word_q, g)}?"
    keys = {_state(e)[0] for e in ents}
    states = ", ".join(f"{e.name} {_state(e, plural_g(e.gender) if is_plural_name(e.name) else None)[1]}"
                       for e in ents)
    if R.random() < 0.6 and len(keys) == 1 and keys <= {
            "on", "off", "open", "closed", "locked", "unlocked"}:
        c.user(question)
        key = keys.pop()
        yes = "Sí" if key == asked else "No"
        c.get_state(ents, f"{yes}, según Home Assistant: {states}.", area)
    else:
        c.user(R.choice(["¿Ha funcionado?", "¿Seguro?", "¿Puedes comprobarlo?"]))
        c.get_state(ents, f"Según Home Assistant: {states}.", area)
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


QKINDS = ["temp", "outdoor", "forecast", "climate", "onoff", "which_lights", "which_covers", "windows",
          "door", "car", "lock",
          "alarm", "media", "humidity", "power", "generic"]


def build_cases():
    """Every case gets the same share of the dataset, rare ones included:
    one case per command intent, per question kind, per error/unknown
    variant and per conversation turn type."""
    cases = [(f"command:{fn.__name__}", s_command, {"intent": fn}) for fn, _ in COMMANDS]
    cases += [(f"question:{k}", s_question, {"qkind": k}) for k in QKINDS]
    cases += [("compound", s_compound, {}), ("compound:question", s_compound_question, {}),
              ("relative_climate", s_relative_climate, {}),
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
    # "la tele del dormitorio" while the only TV is in the living room: one
    # mixed case plus the TV case (one case per kind doubled the refusals)
    cases += [("unknown:missing_type", s_missing_type, {"mtype": "other"}),
              ("unknown:missing_media_player", s_missing_type, {"mtype": "media_player"})]
    seen = set()
    for fn, _ in NEXT:
        if fn not in seen:
            seen.add(fn)
            cases.append((f"conversation:{fn.__name__}", s_conversation, {"conv_turn": fn}))
    # "Luz de la cocina" -> "No, quería decir en el comedor": undo every
    # kitchen light, switch every dining room light
    cases.append(("conversation:room_correct", s_conversation,
                  {"conv_turn": turn_correct, "intent": i_light_room_phrase}))
    cases.append(("conversation:correct_recap", s_conversation,
                  {"conv_turn": turn_correct_recap, "intent": i_light_room_phrase}))
    for hook in CASE_HOOKS:
        cases += hook()
    return cases


SHORT_OK = ["Hecho.", "Vale.", "Listo.", "Ya está."]
BARE = {"Hecho.", "Vale.", "Listo.", "Ya está.", "En pausa.", "Volumen subido.",
        "Volumen bajado.", "Siguiente canción.", "Detenido.", "Canción anterior."}
DESCRIBED = {"turn_on", "turn_off", "open_cover", "close_cover", "stop_cover", "lock", "unlock",
             "media_play", "media_pause", "media_stop", "media_next_track",
             "media_previous_track", "volume_up", "volume_down", "start", "pause",
             "return_to_base", "start_mowing", "dock", "open_valve", "close_valve", "press",
             "set_temperature", "set_cover_position", "set_percentage", "volume_set",
             # assist services, worded by DESCRIBE_HOOKS
             "add_item", "complete_item", "remove_item", "search_and_play", "seek_relative",
             "shuffle_set",
             "repeat_set", "volume_mute", "set_humidity", "set_valve_position"}


def apply_verbosity(h, msgs):
    """Answers after a successful control call follow the answer length:
    short -> a word or two, long/detailed -> never a bare 'Hecho.'."""
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
        m["content"] = "He " + describe_all(h, acts) + "."


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
        h.pick_sat_player(loc)
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
            # the answer is part of the key: an area query has the same
            # question and call in every house, but a different answer
            answer = s["messages"][-1].get("content") or ""
            key = last_user + str(s["meta"].get("gold")) + str(s["meta"].get("gold_last")) + answer
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
