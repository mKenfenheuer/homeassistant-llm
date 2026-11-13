"""Generate the French SFT dataset for Home Assistant tool calling.

Port of src/gen/de/core.py: the same house model, cases, tool calls and
metadata, with French sentences. Every sample uses a freshly randomised house
(rooms, device names, entity_ids, states), so the model learns to map names
from the device list to entity_ids instead of memorising one house. Prompt and
tool schema come from custom_components/ha_llm_agent/llm_schema.py.

French grammar helpers: the() (le/la/l'/les), de() (du/de la/de l'/des/d'),
a() (au/à la/aux), que() (que/qu'), agr() (agreement of participles:
allumé/allumée/allumés/allumées). Genders are "m", "f", "mp", "fp".

This module holds the house model and the device intents; src/gen/fr/assist.py
adds the Assist built-ins (music, timers, lists, weather, time, broadcasts)
through POPULATE_HOOKS, COMMANDS and CASE_HOOKS.

Usage: python src/gen_dataset.py --lang fr OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import re
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

LANG = "fr"
# answer length of the current sample (llm_schema.VERBOSITY), set in make_sample
VERBOSITY = "medium"
VERBOSITY_WEIGHTS = [("short", 2), ("medium", 4), ("long", 2), ("detailed", 2)]
R = random.Random()
# extensions (assist.py): fn(house) after populating, fn() -> extra cases
POPULATE_HOOKS: list = []
CASE_HOOKS: list = []
# assist.py: fn(h, action, who) -> description of an action or None
DESCRIBE_HOOKS: list = []
# set by the case loop in main(): forces one intent / question kind / turn type
FORCE: dict = {}


# --------------------------------------------------------------------------
# French grammar
# --------------------------------------------------------------------------
VOWELS = "aeiouàâäéèêëîïôöùûüœæAEIOUÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÆ"
H_MUET = ("humid", "halog", "huile", "heure", "horloge", "herbe", "hôte")
NO_ELISION = ("onze", "oui", "ouate")


def elides(word: str) -> bool:
    """True if le/la/de/que lose their vowel before `word` (l'entrée, d'Adele)."""
    w = word.lower()
    if w.startswith(NO_ELISION):
        return False
    return word[:1] in VOWELS or w.startswith(H_MUET)


def plural(g: str) -> bool:
    return g in ("mp", "fp")


def fem(g: str) -> bool:
    return g in ("f", "fp")


def the(g: str, word: str) -> str:
    """Definite article: le plafonnier, la lampe, l'applique, les spots."""
    if plural(g):
        return f"les {word}"
    if elides(word):
        return f"l'{word}"
    return f"{'la' if fem(g) else 'le'} {word}"


def de(x: str) -> str:
    """'de' + phrase: du salon, de la cuisine, de l'entrée, des volets, d'Adele."""
    if x.startswith("le "):
        return "du " + x[3:]
    if x.startswith("les "):
        return "des " + x[4:]
    if x.startswith(("la ", "l'")):
        return "de " + x
    if elides(x):
        return "d'" + x
    return "de " + x


def a(x: str) -> str:
    """'à' + phrase: au salon, à la cuisine, aux toilettes."""
    if x.startswith("le "):
        return "au " + x[3:]
    if x.startswith("les "):
        return "aux " + x[4:]
    return "à " + x


def que(x: str) -> str:
    return ("qu'" if elides(x) else "que ") + x


AGREEABLE = {"allumé", "éteint", "ouvert", "fermé", "verrouillé", "déverrouillé", "réglé",
             "baissé", "levé", "remonté", "connecté", "prêt", "activé", "désactivé",
             "déclenché", "arrêté", "mis", "coché", "supprimé", "seul", "déployé", "replié",
             "annulé", "raccourci", "lancé", "rayé"}


def agr_word(w: str, g: str) -> str:
    if fem(g) and not w.endswith("e"):
        w += "e"
    if plural(g) and not w.endswith(("s", "x")):
        w += "s"
    return w


def agr(phrase: str, g: str) -> str:
    """Agree the participle of a phrase: ('allumé', 'f') -> 'allumée',
    ('pas verrouillé', 'fp') -> 'pas verrouillées', ('mis en pause', 'f') ->
    'mise en pause'."""
    words = phrase.split(" ")
    for i, w in enumerate(words):
        core, tail = (w[:-1], w[-1]) if w.endswith(",") else (w, "")
        if core in AGREEABLE:
            words[i] = agr_word(core, g) + tail
            break
    return " ".join(words)


def est(g) -> str:
    return "sont" if (plural(g) if isinstance(g, str) else g) else "est"


def nest(g) -> str:
    return "ne sont" if (plural(g) if isinstance(g, str) else g) else "n'est"


def group_gender(ents) -> str:
    """Gender of a list of devices: plural, feminine only if all are."""
    if len(ents) == 1:
        return ents[0].gender
    return "fp" if all(fem(e.gender) for e in ents) else "mp"


def fr_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}".replace(".", ",")


def deg(x) -> str:
    """'21,5 degrés', '1 degré' (French singular below 2)."""
    return f"{fr_num(float(x))} degré" + ("s" if abs(float(x)) >= 2 else "")


def unit(x, sing, plur) -> str:
    return f"{fr_num(float(x))} {sing if abs(float(x)) < 2 else plur}"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def low(s: str) -> str:
    if len(s) > 1 and s[1].isupper():       # acronyms keep their case: VMC, LED
        return s
    return s[:1].lower() + s[1:]


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("œ", "oe").replace("Œ", "Oe"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, "dans" phrase, "de" phrase, slugs, gender
ROOMS = [
    ("Salon", "dans le salon", "du salon", ["salon", "living_room", "sejour"], "m"),
    ("Cuisine", "dans la cuisine", "de la cuisine", ["cuisine", "kitchen"], "f"),
    ("Chambre", "dans la chambre", "de la chambre", ["chambre", "bedroom"], "f"),
    ("Salle de bain", "dans la salle de bain", "de la salle de bain",
     ["salle_de_bain", "bathroom", "sdb"], "f"),
    ("Salle de bains", "dans la salle de bains", "de la salle de bains",
     ["salle_de_bains", "bathroom"], "f"),
    ("Chambre d'enfant", "dans la chambre d'enfant", "de la chambre d'enfant",
     ["chambre_enfant", "kids_room"], "f"),
    ("Bureau", "dans le bureau", "du bureau", ["bureau", "office"], "m"),
    ("Véranda", "dans la véranda", "de la véranda", ["veranda", "sunroom"], "f"),
    ("Couloir", "dans le couloir", "du couloir", ["couloir", "hallway"], "m"),
    ("Salle à manger", "dans la salle à manger", "de la salle à manger",
     ["salle_a_manger", "dining_room"], "f"),
    ("Chambre d'amis", "dans la chambre d'amis", "de la chambre d'amis",
     ["chambre_amis", "guest_room"], "f"),
    ("Sous-sol", "au sous-sol", "du sous-sol", ["sous_sol", "basement"], "m"),
    ("Garage", "dans le garage", "du garage", ["garage"], "m"),
    ("Jardin", "dans le jardin", "du jardin", ["jardin", "garden"], "m"),
    ("Terrasse", "sur la terrasse", "de la terrasse", ["terrasse", "terrace"], "f"),
    ("Balcon", "sur le balcon", "du balcon", ["balcon", "balcony"], "m"),
    ("Grenier", "au grenier", "du grenier", ["grenier", "attic"], "m"),
    ("Buanderie", "dans la buanderie", "de la buanderie", ["buanderie", "laundry"], "f"),
    ("Toilettes", "dans les toilettes", "des toilettes", ["toilettes", "wc"], "fp"),
    ("Dressing", "dans le dressing", "du dressing", ["dressing", "dressing_room"], "m"),
    ("Entrée", "dans l'entrée", "de l'entrée", ["entree", "entrance"], "f"),
    ("Escalier", "dans l'escalier", "de l'escalier", ["escalier", "stairs"], "m"),
    ("Extérieur", "à l'extérieur", "de l'extérieur", ["exterieur", "outdoor"], "m"),
    ("Débarras", "dans le débarras", "du débarras", ["debarras", "storage"], "m"),
    ("Chaufferie", "dans la chaufferie", "de la chaufferie", ["chaufferie", "boiler_room"], "f"),
    ("Cellier", "dans le cellier", "du cellier", ["cellier", "utility"], "m"),
    ("Allée", "dans l'allée", "de l'allée", ["allee", "driveway"], "f"),
    ("Cave", "à la cave", "de la cave", ["cave", "cellar"], "f"),
    ("Atelier", "dans l'atelier", "de l'atelier", ["atelier", "workshop"], "m"),
    ("Carport", "sous le carport", "du carport", ["carport"], "m"),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
BATH, BATH2 = "Salle de bain", "Salle de bains"
OUTDOOR = ("Jardin", "Terrasse", "Balcon", "Extérieur", "Allée", "Carport")
UTILITY = ("Débarras", "Chaufferie", "Cellier", "Buanderie", "Atelier", "Sous-sol",
           "Grenier", "Garage", "Cave")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garage",)]
NO_GROUP_COVERS = ("Store banne", "Porte de garage")
# rooms that often have a window and a balcony or patio door with covers
TWO_COVER_ROOMS = ("Salon", "Salle à manger", "Cuisine", "Chambre", "Véranda",
                   "Chambre d'enfant", "Bureau")


def prep(room):
    """'dans la cuisine'."""
    return ROOM_BY_NAME[room][1]


def droom(room):
    """'de la cuisine'."""
    return ROOM_BY_NAME[room][2]


def rgender(room):
    return ROOM_BY_NAME[room][4]


# device type: (label, gender, slugs, "de" name-able: "Lumière du salon")
TYPES = {
    "light": [
        ("Plafonnier", "m", ["plafonnier", "ceiling"], False),
        ("Lumière", "f", ["lumiere", "light"], True),
        ("Lampe", "f", ["lampe", "lamp"], True),
        ("Lampadaire", "m", ["lampadaire", "floor_lamp"], False),
        ("Liseuse", "f", ["liseuse", "reading_lamp"], False),
        ("Ruban LED", "m", ["ruban_led", "led_strip"], False),
        ("Spots", "mp", ["spots", "spotlights"], False),
        ("Applique", "f", ["applique", "wall_lamp"], False),
        ("Guirlande lumineuse", "f", ["guirlande", "fairy_lights"], False),
        ("Suspension", "f", ["suspension", "pendant"], False),
        ("Lampe de chevet", "f", ["lampe_chevet", "bedside"], False),
        ("Lumière miroir", "f", ["miroir", "mirror_light"], False),
        ("Éclairage", "m", ["eclairage", "lighting"], True),
    ],
    "switch": [
        ("Prise", "f", ["prise", "plug", "socket"], False),
        ("Machine à café", "f", ["machine_cafe", "coffee"], False),
        ("Chauffage d'appoint", "m", ["chauffage_appoint", "heater"], False),
        ("Déshumidificateur", "m", ["deshumidificateur", "dehumidifier"], False),
        ("Pompe du bassin", "f", ["pompe_bassin", "pond_pump"], False),
        ("Arrosage automatique", "m", ["arrosage_auto", "irrigation"], False),
        ("Sapin de Noël", "m", ["sapin", "xmas_tree"], False),
        ("Imprimante", "f", ["imprimante", "printer"], False),
        ("Bouilloire", "f", ["bouilloire", "kettle"], False),
        ("Écran", "m", ["ecran", "screen"], False),
    ],
    "cover": [
        ("Volet", "m", ["volet", "shutter"], True),
        ("Volets", "mp", ["volets", "shutters"], False),
        ("Volet roulant", "m", ["volet_roulant", "roller_shutter"], False),
        ("Store", "m", ["store", "blind"], False),
        ("Store banne", "m", ["store_banne", "awning"], False),
    ],
    "climate": [
        ("Chauffage", "m", ["chauffage", "heating", "thermostat"], True),
        ("Radiateur", "m", ["radiateur", "radiator"], False),
        ("Thermostat", "m", ["thermostat", "trv"], False),
        ("Climatisation", "f", ["clim", "aircon"], False),
        ("Plancher chauffant", "m", ["plancher_chauffant", "floor_heating"], False),
    ],
    "fan": [
        ("Ventilateur", "m", ["ventilateur", "fan"], False),
        ("Ventilateur de plafond", "m", ["ventilateur_plafond", "ceiling_fan"], False),
        ("VMC", "f", ["vmc", "ventilation"], False),
    ],
    "media_player": [
        ("Télé", "f", ["tv", "tele"], False),
        ("Enceinte", "f", ["speaker", "sonos", "enceinte"], False),
        ("Barre de son", "f", ["soundbar", "barre_de_son"], False),
        ("Radio", "f", ["radio"], False),
    ],
}
CLIMATE_ROOM_TYPES = 4      # the first four; the bathroom may get floor heating too

LOCKS = [("Porte d'entrée", "f", ["porte_entree", "front_door"]),
         ("Porte de la cave", "f", ["porte_cave", "basement_door"]),
         ("Porte du jardin", "f", ["porte_jardin", "garden_door"]),
         ("Portillon", "m", ["portillon", "garden_gate"]),
         ("Porte de derrière", "f", ["porte_arriere", "back_door"])]
VACUUMS = [("Robot aspirateur", "m", ["robot_aspirateur", "roborock", "robot"]),
           ("Aspirateur", "m", ["aspirateur", "vacuum"])]
SCENES = ["Soirée cinéma", "Bonne nuit", "Cuisiner", "Détente", "Fête", "Lecture",
          "Réveil", "Dîner", "Romantique", "Lumineux", "Concentration"]
SCRIPTS = ["Tout éteindre", "Mode vacances", "Routine du coucher", "Préparer le café",
           "Quitter la maison", "Arrivée"]

COLORS = {"rouge": "red", "vert": "green", "bleu": "blue", "jaune": "yellow",
          "orange": "orange", "violet": "purple", "rose": "pink", "blanc": "white",
          "turquoise": "turquoise", "mauve": "violet"}
COLOR_TEMPS = {"blanc chaud": 2700, "blanc froid": 6000, "blanc neutre": 4000}

# brand-named devices: user says the generic noun, the name is the product
BRAND_MEDIA = [("LG webOS Smart TV", "Télé", "f"), ("Samsung TV", "Télé", "f"),
               ("Sony Bravia", "Télé", "f"), ("Philips Ambilight", "Télé", "f"),
               ("Sonos One", "Enceinte", "f"), ("Echo Dot", "Enceinte", "f"),
               ("HomePod", "Enceinte", "f"), ("Bose Soundbar", "Barre de son", "f")]


# voice satellites expose a media_player that is no TV
def voice_media_name():
    hx = "".join(R.choice("0123456789abcdef") for _ in range(6))
    return R.choice([f"Home Assistant Voice {hx} Media Player", f"Home Assistant Voice {hx[:6]}",
                     f"ESPHome Voice {hx[:4]} Media Player", "Echo Show", "Nest Mini",
                     f"Atom Echo {hx[:6]} Media Player"])


BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Robot tondeuse", "Gardena Sileno", "Worx Landroid", "Tondeuse robot"]
# (name in the device list, the plant with its article)
PLANTS = [("Tomates", "les tomates"), ("Piment", "le piment"), ("Basilic", "le basilic"),
          ("Persil", "le persil"), ("Fraises", "les fraises"), ("Monstera", "le monstera"),
          ("Ficus", "le ficus"), ("Concombres", "les concombres"),
          ("Citronnier", "le citronnier"), ("Orchidée", "l'orchidée"), ("Olivier", "l'olivier")]
PLANT_ART = dict(PLANTS)
MOISTURE = "Humidité du sol"
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "PC Bureau", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Boost eau chaude", "Surpresseur", "Pompe aquarium"]
# appliances: a status sensor and a power switch with nearly the same name
# ("Sèche-linge" + "Sèche-linge Prise"): commands go to the switch, questions
# to the sensor
APPLIANCES = [("Sèche-linge", "m", "seche_linge"), ("Lave-linge", "m", "lave_linge"),
              ("Lave-vaisselle", "m", "lave_vaisselle"),
              ("Machine à laver", "f", "machine_a_laver"),
              ("Imprimante 3D", "f", "imprimante_3d"), ("Séchoir", "m", "dryer")]
# raw states as integrations report them (Home Connect, Miele, SmartThings, ...)
APP_RAW = {"running": ["running", "En marche", "En cours", "drying", "in_use", "run"],
           "finished": ["finished", "Programme terminé", "Terminé", "end", "program_ended"],
           "off": ["off", "Arrêt"], "idle": ["idle", "Prêt", "standby", "programmed",
                                             "waiting_to_start"],
           "paused": ["pause", "paused", "En pause"],
           "offline": ["not_connected", "disconnected", "Hors ligne"]}
# answer phrase (agreed with the appliance), word that must be in the answer
APP_ANSWER = {"running": ("est encore en marche", "en marche"),
              "finished": ("a terminé son programme", "terminé"),
              "off": ("est éteint", "éteint"),
              "idle": ("est prêt, mais ne tourne pas en ce moment", "ne tourne pas"),
              "paused": ("est en pause", "en pause"),
              "offline": ("n'est pas connecté en ce moment", "pas connecté")}
VACUUM_BUTTONS = [("Nettoyage complet", "le nettoyage complet"), ("Aspiration", "l'aspiration"),
                  ("Lavage serpillière", "le lavage de la serpillière")]
TOGGLES = ["Vacances", "Mode invités", "Mode fête", "Mode nuit", "Mode ménage"]

UNKNOWN_DEVICES = [("Sauna", "m"), ("Jacuzzi", "m"), ("Pompe de piscine", "f"),
                   ("Cheminée", "f"), ("Aquarium", "m"), ("Tondeuse", "f"),
                   ("Fontaine", "f"), ("Sèche-linge", "m"),
                   ("Lave-vaisselle", "m"), ("Four", "m"), ("Climatiseur", "m"),
                   ("Portail", "m"), ("Éclairage du portail", "m")]

OUTDOOR_LABEL = "Température extérieure"
FORECAST_LABEL = "Prévision"


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "Plafonnier"
    gender: str = "m"
    state: str = "off"
    attrs: dict = field(default_factory=dict)
    hidden: str = ""         # type label that the name no longer shows
    voice: bool = False      # media player of a voice satellite

    @property
    def plural(self) -> bool:
        return plural(self.gender)


CUR = {}      # the house of the sample being built (for fix_pl)


PLURAL_START = ("tous ", "toutes ", "spots", "volets", "les ")


def is_plural_name(name: str) -> bool:
    """'Toutes les lumières', 'Spots Salon', 'Volets maison' take a plural verb."""
    h = CUR.get("house")
    e = next((x for x in h.ents if x.name == name), None) if h else None
    if e is not None:
        return e.plural
    return name.lower().startswith(PLURAL_START)


def fix_pl(text: str, name: str) -> str:
    """'Spots Salon n'est pas joignable' -> 'Spots Salon ne sont pas
    joignables' when the named device is plural."""
    if not is_plural_name(name):
        return text
    for a_, b in ((f"{name} n'est cependant pas joignable", f"{name} ne sont cependant pas joignables"),
                  (f"{name} n'est toujours pas joignable", f"{name} ne sont toujours pas joignables"),
                  (f"{name} n'est pas joignable", f"{name} ne sont pas joignables"),
                  (f"{name} ne répond pas", f"{name} ne répondent pas")):
        text = text.replace(a_, b)
    return text


class House:
    def __init__(self, size: int):
        CUR["house"] = self
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
        rslugs = ROOM_BY_NAME[room][3] if room else ["maison", "home"]
        if domain in self.shared and R.random() < 0.9:
            tail = "".join(R.choices("0123456789abcdef", k=R.choice([2, 3])))
            base = self.shared[domain] + tail
        elif self.cryptic and R.random() < 0.6:
            vendor = R.choice(["shelly1", "shellyplus1pm", "hue_color_lamp",
                               "tz3000", "sonoff_basic", "zbmini", "tasmota",
                               "legrand_netatmo", "somfy_io", "tuya"])
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
        opts = []
        if room:
            opts += [f"{label} {room}", f"{label} {room}", f"{room} {label}"]
            if compound:
                opts += [f"{label} {droom(room)}"]       # "Lumière du salon"
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
        e = self.add("media_player", room, "Enceinte", "f",
                     [slugify(name), "voice_media_player"], name=name,
                     state=R.choice(["idle", "idle", "off"]),
                     attrs={"volume_level": R.choice([0.3, 0.5, 0.7])})
        e.voice = True
        return e

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
                # sample may call it "le plafonnier de la cuisine"; it takes
                # the gender of the room ("Cuisine est allumée")
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                room_lights[0].gender = rgender(room)
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Gauche", "Droite", "Centre"], ["Plafond", "Mur", "Coin"],
                                      ["Îlot", "Spots", "Comptoir"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        if f"{room} {side}" not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = f"{room} {side}"
                            lt.hidden, lt.label = lt.label, ""
                            lt.gender = rgender(room).replace("p", "")
                            self.used_names.add(lt.name)
            # many lamps carry a name without their room ("Ciel étoilé",
            # "Lumière buffet"); only the area column ties them to the room
            for lt in room_lights:
                if lt.name != room and R.random() < 0.3:
                    free = [n for n in FREE_LIGHT_NAMES if n[0] not in self.used_names]
                    if free:
                        self.used_names.discard(lt.name)
                        lt.name, lt.gender = R.choice(free)
                        lt.hidden, lt.label = lt.label or lt.hidden, ""
                        self.used_names.add(lt.name)
            if not outdoor and room not in ("Couloir", "Escalier", "Toilettes") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                if room in TWO_COVER_ROOMS and R.random() < 0.35:
                    # window and balcony door: "les volets de la salle à
                    # manger" means both
                    word = R.choice(["Volet", "Volet roulant", "Store", "Volet"])
                    door = R.choice(["Baie vitrée", "Porte-fenêtre", "Porte balcon", "Baie"])
                    for part, slug in (("Fenêtre", "fenetre"), (door, "baie")):
                        name = R.choice([f"{word} {room} {part}", f"{word} {low(part)} {room}"])
                        if name in self.used_names:
                            continue
                        p2 = R.choice([pos, pos, 0, 100])
                        self.add("cover", room, word, "m",
                                 [f"{ct[2][0]}_{slug}", f"{ct[2][1]}_{slug}"], name=name,
                                 state="closed" if p2 == 0 else "open",
                                 attrs={"current_position": p2})
                else:
                    self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                             state="closed" if pos == 0 else "open",
                             attrs={"current_position": pos})
            if room in ("Terrasse", "Balcon", "Extérieur") and R.random() < 0.5:
                self.add("cover", room, "Store banne", "m", ["store_banne", "awning"],
                         state=(st := R.choice(["open", "closed"])),
                         attrs={"current_position": 100 if st == "open" else 0})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:CLIMATE_ROOM_TYPES]
                              if room not in (BATH, BATH2) else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, "Température", "f",
                         ["temperature", "temperature"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "Humidité", "f",
                         ["humidite", "humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "Fenêtre", "f",
                         ["fenetre", "window"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Salon", "Chambre", "Cuisine", "Bureau",
                        "Chambre d'enfant") and R.random() < 0.4:
                mt = R.choice(TYPES["media_player"])
                brand = None
                if R.random() < 0.5:
                    brand = R.choice(BRAND_MEDIA)
                    mt = (brand[1], brand[2], [slugify(brand[0])])
                playing = R.random() < 0.4
                attrs = {"volume_level": R.choice([0.2, 0.3, 0.35, 0.5])}
                if playing:
                    attrs.update(media_title=R.choice(
                        ["Le 20 heures", "Bohemian Rhapsody", "Alors on danse",
                         "Koh-Lanta", "Les Minions", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "Stromae", ""]))
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if brand and brand[0] not in self.used_names else None,
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""})
            if not outdoor and R.random() < 0.12:
                self.add_voice_media(room)
            if room in ("Chambre", "Bureau", "Salon", "Grenier") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "Puissance", "f", ["puissance", "power"],
                         name=self._name(room, R.choice(["Prise Puissance", "Puissance"]), False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            self.add("alarm_control_panel", None, "Alarme", "f",
                     ["alarme", "alarm"], name="Alarme",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Couloir", "Entrée", None]) if lk[0] == "Porte d'entrée" else None
                room = room if room in self.rooms else None
                self.add("lock", room, lk[0], lk[1], lk[2], name=lk[0],
                         state=R.choice(["locked", "locked", "unlocked"]))
        if R.random() < 0.4:
            vt = R.choice(VACUUMS)
            vname = R.choice(BRAND_VACUUM) if R.random() < 0.6 else vt[0]
            self.add("vacuum", R.choice(self.rooms + [None, None]), vt[0], vt[1],
                     [slugify(vname)], name=vname,
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.25:
            mname = R.choice(MOWERS)
            room = "Jardin" if "Jardin" in self.rooms else None
            self.add("lawn_mower", room, "Robot tondeuse", "m",
                     [slugify(mname)], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = [("Arrosage", "m"), ("Vanne d'eau", "f")]
            if room:
                names += [(f"Arrosage {room}", "m"), (f"Goutte-à-goutte {room}", "m"),
                          (f"{room} Goutte-à-goutte", "m")]
                if room == "Jardin":
                    names.append(("Vanne jardin", "f"))
            vname, vg = R.choice(names)
            self.add("valve", room, "Arrosage", vg, ["arrosage", "irrigation"],
                     name=vname, state=R.choice(["open", "closed", "closed"]))
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
            room, names = None, ["Température extérieure", "Température dehors",
                                 "Station météo température"]
            orooms = [r for r in self.rooms if r in OUTDOOR]
            if orooms and R.random() < 0.4:
                # weather stations sit on the balcony or in the garden
                room = R.choice(orooms)
                names = [f"Température extérieure {room}", f"Température {room}"]
            self.add("sensor", room, OUTDOOR_LABEL, "f",
                     ["temperature_exterieure", "outdoor_temperature"],
                     name=R.choice(names), state=f"{out_t:.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        # heat pumps report their own outdoor sensor next to the climate entity
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            room = "Chaufferie" if "Chaufferie" in self.rooms and R.random() < 0.5 else None
            self.add("sensor", room, OUTDOOR_LABEL, "f",
                     ["pac_outdoor_temperature", "maison_outdoor_temperature"],
                     name=R.choice(["Température extérieure maison", "Température extérieure PAC",
                                    "PAC température extérieure"]),
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
                if label == FORECAST_LABEL:
                    val = out_t + R.uniform(-4, 6) if when == "aujourd'hui" else R.uniform(-6, 31)
                else:
                    val = out_t - R.uniform(0.5, 7)
                self.add("sensor", None, label, "f",
                         slugs, name=name,
                         state=f"{val:.1f}",
                         attrs={"device_class": "temperature", "unit_of_measurement": "°C",
                                "_when": when})
        if "Garage" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garage", "Porte de garage", "f",
                     ["porte_garage", "garage_door"], name="Porte de garage",
                     state=(st := R.choice(["open", "closed", "closed"])),
                     attrs={"current_position": 100 if st == "open" else 0})
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(n.startswith(app) for n in self.used_names):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Cuisine", "Buanderie", "Cellier", BATH, BATH2,
                    "Sous-sol", "Débarras", "Bureau")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "État", g, [slug, slug + "_etat"],
                                  name=R.choice([app, app, f"{app} État"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind})
                # the device is named after the appliance, so it takes its gender
                power = self.add("switch", room, "Prise", g,
                                 [slug + "_power", slug + "_prise"],
                                 name=R.choice([f"{app} Prise", f"{app} Alimentation",
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
            # name ("Arrosage jardin") then is the only hint where it waters
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Arrosage jardin", "Arrosage du jardin"] if room == "Jardin"
                         else [f"Arrosage {room}"])
            elif "Jardin" in valve_rooms:
                names = ["Lancer l'arrosage"]
            else:
                names = ["Lancer l'arrosage", "Arrosage jardin", "Arrosage du jardin"]
            self.add("button", room, "Arrosage", "m", ["arrosage_jardin",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # door contacts: patio doors in living rooms, the front door in the
        # hall; names often lose the hyphen or the accent
        if R.random() < 0.5:
            for room in [r for r in self.rooms if r in DOOR_ROOMS]:
                if R.random() < 0.55:
                    kind, spells, slug = R.choice(DOOR_KINDS)
                    spell = R.choice(spells)
                    name = R.choice([f"{spell} {room}", f"{room} {spell}"])
                    if name in self.used_names:
                        continue
                    self.add("binary_sensor", room, kind, "f", [slug, "door"],
                             name=name, state=R.choice(["off", "off", "on"]),
                             attrs={"device_class": "door"})
            if R.random() < 0.7:
                # a lock "Porte d'entrée" would make the question ambiguous
                base = "Porte palière" if R.random() < 0.5 or \
                    any(e.name == "Porte d'entrée" for e in self.ents) else "Porte d'entrée"
                name = R.choice([base, base, base.replace("é", "e").replace("è", "e")])
                if name in self.used_names:
                    name = f"{name} contact"
                hall = next((r for r in ("Couloir", "Entrée") if r in self.rooms), None)
                self.add("binary_sensor", hall, base, "f", ["porte_entree_state", "front_door_contact"],
                         name=name, state=R.choice(["off", "off", "on"]),
                         attrs={"device_class": "door"})
        # electric car: charge level and range next to each other
        if R.random() < 0.3:
            car, art, cslug = R.choice(CARS)
            self.add("sensor", None, "Niveau de batterie", "m",
                     ["battery_hv_state_of_charge", f"{cslug}_soc"],
                     name=f"{car} Niveau de batterie", state=str(R.randint(10, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
            self.add("sensor", None, "Autonomie", "f",
                     [f"{cslug}_range_ev_remaining_range", f"{cslug}_autonomie"],
                     name=f"{car} Autonomie", state=str(R.randint(40, 450)),
                     attrs={"device_class": "distance", "unit_of_measurement": "km"})
        # heating systems expose their modes as switches next to the climate
        # entity ("Chauffage maison mode absence")
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            mode, slug = R.choice(HEAT_MODES)
            base = R.choice(["Chauffage", "Chauffage maison", "Thermostat", "Maison"])
            room = "Chaufferie" if "Chaufferie" in self.rooms and R.random() < 0.5 else None
            if f"{base} {low(mode)}" not in self.used_names:
                self.add("switch", room, mode, "m", [f"chauffage_{slug}", f"maison_{slug}"],
                         name=f"{base} {low(mode)}", state=R.choice(["on", "off", "off"]))
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            name, g = R.choice([("Toutes les lumières", "fp"), ("Toutes les lampes", "fp"),
                                ("Éclairage maison", "m")])
            self.add("light", None, "__group__", g, ["toutes_les_lumieres", "all_lights"],
                     name=name, state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            name, g = R.choice([("Tous les volets", "mp"), ("Tous les stores", "mp"),
                                ("Volets maison", "mp")])
            self.add("cover", None, "__group__", g, ["tous_les_volets", "all_covers"],
                     name=name, state=R.choice(["open", "closed"]))
        # read-only distractors and odd names; fill up to the target size
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "Maison"
        if k == 0 and room:
            self.add("binary_sensor", room, "Présence", "f", ["presence"],
                     name=R.choice([f"Présence {room}", f"{room} Presence",
                                    f"Occupation {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "Mouvement", "m", ["motion"],
                     name=R.choice([f"{rname} Motion", f"Mouvement {rname}",
                                    f"Détecteur mouvement {rname}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant = R.choice(PLANTS)[0]
            self.add("sensor", room, MOISTURE, "f", ["plant_sensor", "moisture"],
                     name=f"{plant} {MOISTURE}", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "Batterie", "f", ["battery"],
                     name=R.choice(["Onduleur batterie", "Voiture niveau de charge",
                                    "Détecteur de fumée batterie", "Téléphone batterie",
                                    f"{rname} capteur batterie"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "Imprimante", "f", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, u = R.choice([("Vitesse du vent", "km/h"), ("Rafales de vent", "km/h"),
                              ("Précipitations", "mm/h"), ("Pression atmosphérique", "hPa"),
                              ("Niveau cuve d'eau", "L")])
            self.add("sensor", room, nm, "f", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": u})
        elif k == 6:
            self.add("sensor", room, "État", "m", ["status"],
                     name=R.choice(["Onduleur état", "Sèche-linge", "Lave-linge état",
                                    "Lave-vaisselle programme"]),
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
            self.add("sensor", room, "Température", "f", ["temperature"],
                     name=R.choice([f"{rname} Température", f"Température {rname}",
                                    f"{rname} Thermomètre"]),
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
    acc: str      # phrase used in commands ("la lampe du salon")
    nom: str      # phrase for confirmations (French: the same words)
    g: str        # gender of the phrase, for agreement ("allumée", "sont")
    ents: list
    area: str | None = None   # set for "la lumière du salon": control by area

    @property
    def plural(self) -> bool:
        return plural(self.g)


def rplace(room):
    """'du salon' or 'dans le salon' after a device noun."""
    return R.choice([droom(room), droom(room), prep(room)])


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    # "la lumière du couloir" / "la lampe du couloir" mean every light of the
    # room, so a single lamp only gets these words when it is the room's only light
    generic = e.domain == "light" and e.label in ("Lumière", "Lampe", "Éclairage") \
        and len(h.of("light", e.area)) > 1
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1 \
            and not generic:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 and e.name != e.label:
        opts += ["type_only"]           # "la télé" for "LG webOS Smart TV"
    choice = R.choice(opts)
    if choice == "type_only":
        t = the(e.gender, low(e.label))
        return Ref(t, t, e.gender, [e])
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        art = R.random() < 0.35 and e.label and e.label != "__group__" \
            and e.name.startswith(e.label) and not generic
        if art:
            return Ref(the(e.gender, low(name)), the(e.gender, low(e.name)), e.gender, [e])
        return Ref(name, e.name, e.gender, [e])
    t = f"{the(e.gender, low(e.label))} {rplace(e.area)}"
    return Ref(t, t, e.gender, [e])


# group nouns per domain: (phrase, gender)
GROUP = {
    "light": [("la lumière", "f"), ("les lumières", "fp"), ("les lampes", "fp"),
              ("l'éclairage", "m")],
    "cover": [("les volets", "mp"), ("les volets roulants", "mp"), ("les stores", "mp")],
    "climate": [("le chauffage", "m"), ("les radiateurs", "mp")],
    "media_player": [("la musique", "f")],
}
ALL_GROUP = {
    "light": [("toutes les lumières", "fp"), ("toutes les lampes", "fp"),
              ("la lumière partout", "f")],
    "cover": [("tous les volets", "mp"), ("tous les stores", "mp")],
    "climate": [("tous les radiateurs", "mp")],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    phrase, g = R.choice(GROUP[domain])
    area = room if domain in S.AREA_DOMAINS else None
    if location == room and R.random() < 0.7:
        return Ref(phrase, phrase, g, ents, area)    # "Allume la lumière"
    t = f"{phrase} {rplace(room)}"
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
    a_ = {"service": service}
    if area is not None:
        a_["area"] = area
    a_["entity_id"] = [e.eid for e in ents]
    if data:
        a_["data"] = data
    return a_


def wire(name, args):
    """What the model writes: area actions without their resolved entity_id."""
    if name == "control":
        return {"actions": [{k: v for k, v in x.items()
                             if not (k == "entity_id" and "area" in x)}
                            for x in args["actions"]]}
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
    if any("area" in x for x in actions):
        return S.control_result(None, [i for x in actions for i in x["entity_id"]])
    return S.control_result()


def voice(text: str) -> str:
    """Make user text look like speech-to-text output sometimes."""
    r = R.random()
    if r < 0.25:
        text = text.rstrip(" ?.!")
    elif r < 0.35:
        text = text.rstrip(" ?.!").lower()
    if R.random() < 0.05:
        text = R.choice(["Dis, ", "OK, ", "Hé, ", "S'il te plaît, "]) + text[0].lower() + text[1:]
    return text


def confirm(ref: Ref, adj: str, part: str) -> str:
    """adj: state participle ('allumé', agreed here), part: participle after
    'J'ai' ('allumé')."""
    w = agr(adj, ref.g)
    e_ = est(ref.g)
    return R.choice([
        f"{cap(ref.nom)} {e_} maintenant {w}.",
        f"{cap(ref.nom)} {e_} {w}.",
        f"J'ai {part} {ref.acc}.",
        f"C'est fait, {ref.nom} {e_} {w}.",
        f"D'accord, {ref.nom} {e_} maintenant {w}.",
        "C'est fait.",
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
            "Allume {r}", "Allume {r}, s'il te plaît", "Tu peux allumer {r} ?",
            "Peux-tu allumer {r} ?", "Allumer {r}", "Allume-moi {r}",
            "Est-ce que tu peux allumer {r} ?", "Tu veux bien allumer {r} ?",
            "J'ai besoin de lumière, allume {r}", "Allume {r} stp", "Allume {r}.",
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref)],
                confirm(ref, "allumé", "allumé"))
    user = R.choice([
        "Éteins {r}", "Éteins {r}, s'il te plaît", "Tu peux éteindre {r} ?",
        "Éteindre {r}", "Coupe {r}", "Éteins-moi {r}", "Peux-tu éteindre {r} ?",
        "Tu veux bien éteindre {r} ?", "Est-ce que tu peux éteindre {r} ?",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref)],
            confirm(ref, "éteint", "éteint"))


def i_light_room_phrase(h, loc):
    """'Allume la lumière dans la salle de bain' / 'Il fait trop sombre ici'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    p, d = prep(room), droom(room)
    ref = Ref(f"la lumière {d}", f"la lumière {d}", "f", ents, room)
    if R.random() < 0.5:
        user = R.choice([f"Allume la lumière {p}", f"Lumière {p}, s'il te plaît",
                         f"Il fait trop sombre {p}", f"{cap(p)}, allume la lumière",
                         f"Allume {p}", f"Lumière {d}"])
        return user, [action("light.turn_on", ref)], confirm(ref, "allumé", "allumé")
    user = R.choice([f"Éteins la lumière {p}", f"{cap(p)}, éteins la lumière",
                     f"Éteins {p}", f"Coupe la lumière {d}"])
    return user, [action("light.turn_off", ref)], confirm(ref, "éteint", "éteint")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        user = R.choice([
            "Règle {r} sur {p} %", "Mets {r} à {p} %",
            "{R} à {p} pour cent", "Luminosité {dr} à {p} %",
            "Mets {r} à {p} pour cent de luminosité", "Tamise {r} à {p} %",
        ]).format(r=ref.acc, R=cap(ref.acc), dr=de(ref.acc), p=p)
        conf = R.choice([f"{cap(ref.nom)} {est(ref.g)} maintenant à {p} %.",
                         f"J'ai réglé {ref.acc} sur {p} %.", "C'est fait."])
        return user, [action("light.turn_on", ref, {"brightness_pct": p})], conf
    up = r < 0.8
    user = R.choice(["Augmente la luminosité {dr}", "Monte la luminosité {dr}",
                     "Mets {r} plus fort"]
                    if up else
                    ["Baisse la luminosité {dr}", "Tamise {r}", "Mets {r} moins fort"]
                    ).format(r=ref.acc, dr=de(ref.acc))
    step = 20 if up else -20
    conf = (f"J'ai {'augmenté' if up else 'baissé'} la luminosité {de(ref.nom)}."
            if R.random() < 0.7 else "C'est fait.")
    return user, [action("light.turn_on", ref, {"brightness_step_pct": step})], conf


def i_light_color(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    if R.random() < 0.75:
        fr, en = R.choice(list(COLORS.items()))
        data = {"color_name": en}
    else:
        fr, k = R.choice(list(COLOR_TEMPS.items()))
        data = {"color_temp_kelvin": k}
    user = R.choice(["Mets {r} en {c}", "Passe {r} en {c}", "Change la couleur {dr} en {c}",
                     "{R} en {c}, s'il te plaît", "Tu peux mettre {r} en {c} ?",
                     "Je voudrais {r} en {c}"]).format(r=ref.acc, R=cap(ref.acc),
                                                       dr=de(ref.acc), c=fr)
    conf = R.choice([f"{cap(ref.nom)} {est(ref.g)} maintenant en {fr}.",
                     f"J'ai mis {ref.acc} en {fr}.", "C'est fait."])
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
        user = R.choice(["Allume {r}", "Mets en marche {r}", "Allumer {r}", "Active {r}",
                         "Tu peux allumer {r} ?", "Démarre {r}"])
        conf = confirm(ref, "allumé", "allumé")
    else:
        user = R.choice(["Éteins {r}", "Coupe {r}", "Éteindre {r}", "Arrête {r}",
                         "Tu peux couper {r} ?"])
        conf = confirm(ref, "éteint", "éteint")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    user = R.choice(["Mets {r} à {p} %", "{R} à {p} pour cent",
                     "Fais tourner {r} à {p} %"]).format(r=ref.acc, R=cap(ref.acc), p=p)
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {est(ref.g)} à {p} %.", "C'est fait."])


def i_awning(h, loc):
    """Store banne: déployer = open_cover, replier = close_cover."""
    ents = [e for e in h.of("cover") if e.label == "Store banne"]
    if not ents:
        return None
    e = R.choice(ents)
    n = "le store banne" if len(ents) == 1 and R.random() < 0.7 else e.name
    if R.random() < 0.5:
        user = R.choice([f"Déploie {n}", f"Sors {n}", f"Ouvre {n}",
                         f"Déployer {n}", "On a besoin d'ombre, sors le store"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["Le store banne se déploie.", "C'est fait, le store banne est déployé."])
    user = R.choice([f"Replie {n}", f"Rentre {n}", f"Ferme {n}",
                     f"Replier {n}", "Il pleut, rentre le store"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["Le store banne se replie.", "C'est fait, le store banne est replié."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["la tondeuse", "le robot tondeuse", e.name])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Lance {n}", f"Envoie {n} tondre", "Tonds la pelouse, s'il te plaît",
                         f"Démarre {n}"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{e.name} tond la pelouse."
    if r < 0.8:
        user = R.choice([f"Renvoie {n} à sa station", f"Fais rentrer {n}",
                         f"Ramène {n} à la base"])
        return user, [action("lawn_mower.dock", [e])], f"{e.name} retourne à sa station."
    user = R.choice([f"Mets {n} en pause", f"Arrête {n}"])
    return user, [action("lawn_mower.pause", [e])], f"{e.name} est en pause."


AREA_PLANTS = {"Balcon": "les plantes du balcon", "Terrasse": "les plantes de la terrasse",
               "Jardin": "le jardin"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == "Arrosage"]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Lance l'arrosage", "Arrose les plantes"]
    if e.area:
        p = prep(e.area)
        out += [f"Lance l'arrosage {p}", f"Arrose les plantes {p}",
                f"Arrose les plantes {droom(e.area)}"]
        if e.area in AREA_PLANTS:
            out.append(f"Arrose {AREA_PLANTS[e.area]}")
    elif "jardin" in e.name.lower():
        out += ["Arrose le jardin", "Lance l'arrosage du jardin",
                "Arrose le jardin, s'il te plaît"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name] if several else [e.name, "l'arrosage", "la vanne"])
    if R.random() < 0.5:
        user = R.choice([f"Ouvre {n}", f"Mets en marche {n}", f"Ouvrir {n}"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"{e.name} {est(e.gender)} {agr('ouvert', e.gender)}."
    user = R.choice([f"Ferme {n}", f"Coupe {n}", f"Fermer {n}",
                     "Arrête l'arrosage"])
    return user, [action("valve.close_valve", [e])], f"{e.name} {est(e.gender)} {agr('fermé', e.gender)}."


def mode_phrase(h, n):
    """'Vacances' is also called 'le mode vacances'; it stays an input_boolean.
    -> phrase with article, or the bare name when a script carries that name."""
    mode = n if n.startswith("Mode ") else f"Mode {n.lower()}"
    if mode != n and any(x.name == mode for x in h.ents):
        return n          # a script "Mode vacances" exists: only the plain name is clear
    return f"le {low(mode)}"


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    ph = mode_phrase(h, n)
    is_mode = ph.startswith("le ")
    if on:
        user = R.choice([f"Active {n}", f"Allume {n}", f"Active {ph}", f"Mets {ph}"]
                        + ([f"Passe en {ph[3:]}", f"Lance {ph}"] if is_mode else []))
        return user, [action("input_boolean.turn_on", [e])], f"{cap(ph)} est activé."
    user = R.choice([f"Désactive {n}", f"Éteins {n}", f"Désactive {ph}", f"Arrête {ph}"]
                    + ([f"Sors {de(ph)}", f"Quitte {ph}"] if is_mode else []))
    return user, [action("input_boolean.turn_off", [e])], f"{cap(ph)} est désactivé."


def i_appliance(h, loc):
    """'Éteins le sèche-linge' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = the(g, low(app))
    if R.random() < 0.5:
        user = R.choice([f"Éteins {n}", f"Coupe {n}", f"Éteindre {low(app)}",
                         f"Arrête {n}", f"Tu peux éteindre {n} ?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"{power.name} est {agr('éteint', g)}.", f"J'ai éteint {power.name}."])
    user = R.choice([f"Allume {n}", f"Mets en marche {n}", f"Allumer {low(app)}",
                     f"Démarre {n}"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"{power.name} est {agr('allumé', g)}.", f"J'ai allumé {power.name}."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        what = vac[e.label]
        user = R.choice([f"Lance {what}", f"Lance {what} du robot aspirateur",
                         f"Fais {what}", f"Démarre {e.name}"])
    elif e.label == "Arrosage":
        user = R.choice(irrigation_phrases(h, e) + [f"Appuie sur {e.name}"])
    else:
        user = R.choice([f"Appuie sur {e.name}", f"Déclenche {e.name}"])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} est {agr('déclenché', e.gender)}.", f"J'ai lancé {e.name}.", "C'est fait."])


# lamp names without a room (the names of the acceptance house and their
# translations are left out on purpose, so the acceptance test measures
# generalisation): (name, gender)
FREE_LIGHT_NAMES = [("Ciel étoilé", "m"), ("Lampe de sel", "f"), ("Lampe à lave", "f"),
                    ("Applique tableau", "f"), ("Vitrine", "f"), ("Hue Go", "f"),
                    ("Nanoleaf", "m"), ("Hue Play", "f"), ("Lampe nuage", "f"),
                    ("Lampe lune", "f"), ("Réglette", "f"), ("Cadre LED", "m"),
                    ("Boule lumineuse", "f"), ("Lampe d'angle", "f"), ("Lampe canapé", "f"),
                    ("Lumière buffet", "f"), ("Lumière bar", "f"), ("Lumière lit", "f"),
                    ("Lumière armoire", "f"), ("Lampe fenêtre", "f"), ("Lampe champignon", "f"),
                    ("Abat-jour", "m"), ("Enseigne néon", "f"), ("Govee Strip", "m"),
                    ("Lampe de table", "f"), ("Lumière commode", "f"), ("Lampe Pixar", "f"),
                    ("Lampe bureau", "f"), ("Lampe coin", "f"), ("Lanterne", "f"),
                    ("Lampe tripode", "f"), ("Lumière étagère", "f")]

DOOR_ROOMS = ("Salon", "Salle à manger", "Cuisine", "Chambre", "Véranda")
# door contacts: (label, spellings in names, slug)
DOOR_KINDS = [("Porte-fenêtre", ["Porte-fenêtre", "Porte fenêtre", "Porte-fenetre"], "porte_fenetre"),
              ("Baie vitrée", ["Baie vitrée", "Baie vitree", "Baie"], "baie_vitree")]
# (name, article, slug)
CARS = [("Zoe", "la", "zoe"), ("Mégane E-Tech", "la", "megane_e_tech"), ("e-208", "la", "e208"),
        ("Model 3", "la", "model_3"), ("Spring", "la", "spring"), ("ID.3", "l'", "id3"),
        ("Zoe", "la", "zoe_ze50")]

WEATHER_SENSORS = [  # (name, label, day, slugs)
    ("Température prévue aujourd'hui", FORECAST_LABEL, "aujourd'hui",
     ["temperature_prevue_aujourdhui", "forecast_temperature_today"]),
    ("Température prévue demain", FORECAST_LABEL, "demain",
     ["temperature_prevue_demain", "forecast_temperature_tomorrow"]),
    ("Température max aujourd'hui", FORECAST_LABEL, "aujourd'hui",
     ["temperature_max_aujourdhui", "forecast_high_today"]),
    ("Température max demain", FORECAST_LABEL, "demain",
     ["temperature_max_demain", "forecast_high_tomorrow"]),
    ("Température ressentie", "Valeur météo", None, ["temperature_ressentie", "feels_like_temperature"]),
    ("Point de rosée", "Valeur météo", None, ["point_de_rosee", "dew_point"]),
]

HEAT_MODES = [("Mode absence", "away_mode"), ("Mode éco", "eco_mode"),
              ("Mode hors-gel", "frost_protection")]


def car_article(car):
    art = next(x for c, x, _ in CARS if c == car)
    return f"l'{car}" if art == "l'" else f"{art} {car}"


def i_heating_mode(h, loc):
    """'Active le mode absence du chauffage' -> the mode switch, not the
    climate entity."""
    modes = [e for e in h.ents if e.domain == "switch"
             and e.label in [m for m, _ in HEAT_MODES]]
    if not modes or not h.of("climate"):
        return None
    e = R.choice(modes)
    m = low(e.label)
    on = R.random() < 0.6
    if on:
        user = R.choice([f"Active le {m} du chauffage", f"Mets le chauffage en {m}",
                         f"Chauffage en {m}", f"Active le {m}", f"Passe le chauffage en {m}",
                         f"Allume le {m}", f"Lance le {m}"])
    else:
        user = R.choice([f"Désactive le {m} du chauffage", f"Désactive le {m}",
                         f"Arrête le {m}", f"Sors le chauffage du {m}",
                         f"Coupe le {m} du chauffage"])
    word = "activé" if on else "désactivé"
    return (user, [action("switch.turn_on" if on else "switch.turn_off", [e])],
            R.choice([f"{e.name} est maintenant {word}.", f"J'ai {word} {e.name}."]))


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Arrose les plantes du balcon" -> balcony valve, not the garden button)."""
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
            [f"{e.name} {est(e.gender)} {agr('ouvert', e.gender)}.", f"J'ai ouvert {e.name}."])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} est {agr('déclenché', e.gender)}.", f"J'ai lancé {e.name}."])


def i_irrigation_pair(h, loc):
    """A plant valve on the balcony or patio next to an area-less garden
    button, as many real installs have it. "Arrose les plantes du balcon"
    opens the valve, "Arrose le jardin" presses the button."""
    rooms = [r for r in h.rooms if r in ("Balcon", "Terrasse")]
    if not rooms or irrigation(h):
        return None
    room = R.choice(rooms)
    vname, vg = R.choice([(f"Goutte-à-goutte {room}", "m"), (f"{room} Goutte-à-goutte", "m"),
                          (f"Arrosage {room}", "m"), ("Vanne d'eau", "f")])
    valve = h.add("valve", room, "Arrosage", vg, ["plantes", "arrosage"],
                  name=vname, state=R.choice(["open", "closed", "closed"]))
    button = h.add("button", None, "Arrosage", "m", ["arrosage_jardin"],
                   name=R.choice(["Arrosage jardin", "Arrosage du jardin"]),
                   state="unknown")
    if R.random() < 0.6:
        p = prep(room)
        user = R.choice([f"Arrose {AREA_PLANTS[room]}", f"Arrose les plantes {p}",
                         f"Lance l'arrosage {p}", f"Arrose les plantes {droom(room)}"])
        return user, [action("valve.open_valve", [valve])], R.choice(
            [f"{valve.name} {est(vg)} {agr('ouvert', vg)}.", f"J'ai ouvert {valve.name}."])
    user = R.choice(["Arrose le jardin", "Lance l'arrosage du jardin",
                     "Arrose le jardin, s'il te plaît", "Lance l'arrosage dans le jardin"])
    return user, [action("button.press", [button])], R.choice(
        [f"{button.name} est déclenché.", f"J'ai lancé {button.name}."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Ouvre {r}", "Remonte {r}", "Lève {r}", "Monte {r}",
                         "Ouvrir {r}", "Remonte-moi {r}", "Tu peux ouvrir {r} ?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "ouvert", "ouvert"))
    if r < 0.8:
        user = R.choice(["Ferme {r}", "Baisse {r}", "Descends {r}", "Fermer {r}",
                         "Baisse-moi {r}", "Tu peux fermer {r} ?", "Descendre {r}"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "fermé", "fermé"))
    if r < 0.88:
        user = R.choice(["Arrête {r}", "Stoppe {r}", "Stop, {r}", "Bloque {r}"])
        return (user.format(r=ref.acc), [action("cover.stop_cover", ref.ents)],
                R.choice(["C'est arrêté.", f"J'ai arrêté {ref.acc}.", "C'est fait."]))
    p, phrase = R.choice([(50, "à moitié"), (30, "à 30 %"), (70, "à 70 %"),
                          (20, "à 20 %"), (80, "à 80 pour cent"),
                          (50, "à 50 %"), (10, "à 10 %")])
    if phrase == "à moitié":
        user = R.choice(["Ferme {r} à moitié", "Baisse {r} à moitié", "{R} à moitié"]
                        ).format(r=ref.acc, R=cap(ref.acc))
    else:
        user = R.choice(["Mets {r} {ph}", "Ouvre {r} {ph}", "{R} {ph}"]
                        ).format(r=ref.acc, R=cap(ref.acc), ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} {est(ref.g)} maintenant à {p} %.", "C'est fait."])


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
            d = droom(room)
            user = R.choice([f"Mets le chauffage {d} à {{t}}",
                             f"Chauffage {d} à {{t}}",
                             f"Règle la température {d} sur {{t}}",
                             f"Je voudrais {{t}} {prep(room)}"]).format(t=deg(t))
            nom, g = f"le chauffage {d}", "m"
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Règle {r} sur {t}", "Mets {r} à {t}",
                             "{R} à {t}", "Monte {r} à {t}"]
                            ).format(r=ref.acc, R=cap(ref.acc), t=deg(t))
            nom, g = ref.nom, ref.g
        conf = R.choice([f"{cap(nom)} {est(g)} maintenant {agr('réglé', g)} sur {deg(t)}.",
                         f"J'ai réglé {nom} sur {deg(t)}.",
                         f"C'est fait, {deg(t)}."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Allume {r}", "Mets en marche {r}", "Allumer {r}"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "allumé", "allumé")
    user = R.choice(["Éteins {r}", "Coupe {r}", "Éteindre {r}"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "éteint", "éteint")


def pronoun_obj(g):
    """Object pronoun after an imperative: ferme-la, ferme-le, ferme-les."""
    return "les" if plural(g) else "la" if fem(g) else "le"


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc == e.name and R.random() < 0.7:
        t = the(e.gender, low(e.name))
        ref = Ref(t, t, e.gender, [e])
    if R.random() < 0.55:
        user = R.choice([f"Verrouille {ref.acc}", f"Ferme {ref.acc} à clé",
                         f"Tu peux verrouiller {ref.acc} ?",
                         f"{cap(ref.acc)} {est(ref.g)} {agr('fermé', ref.g)} ? "
                         f"Ferme-{pronoun_obj(ref.g)} à clé",
                         f"Verrouiller {ref.acc}"])
        return (user, [action("lock.lock", [e])], confirm(ref, "verrouillé", "verrouillé"))
    user = R.choice([f"Déverrouille {ref.acc}", f"Déverrouiller {ref.acc}",
                     f"Tu peux déverrouiller {ref.acc} ?", f"Ouvre la serrure {de(the(e.gender, low(e.name)))}"])
    return (user, [action("lock.unlock", [e])], confirm(ref, "déverrouillé", "déverrouillé"))


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
            "Active l'alarme", "Arme l'alarme",
            "Je pars, active l'alarme", "Mets l'alarme",
            "Alarme en mode absence"])
        conf = R.choice(["L'alarme est activée.",
                         "L'alarme est maintenant activée (mode absence)."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Alarme en mode présence", "Mets l'alarme en mode présence",
            "Active l'alarme en mode maison"])
        conf = "L'alarme est activée en mode présence."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Active l'alarme pour la nuit", "Mode nuit pour l'alarme",
            "Mets l'alarme en mode nuit"])
        conf = "L'alarme est activée en mode nuit."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Désactive l'alarme", "Désarme l'alarme", "Coupe l'alarme",
            "Je suis rentré, désactive l'alarme", "Arrête l'alarme"])
        conf = R.choice(["L'alarme est désactivée.",
                         "L'alarme est maintenant désarmée."])
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
        ("media_player.media_pause", ["Mets {r} en pause", "Pause sur {n}", "Mettre {r} en pause",
                                      "Mets la lecture en pause sur {n}"], None, "En pause."),
        ("media_player.media_play", ["Reprends la lecture sur {n}", "Relance la lecture sur {n}",
                                     "Lecture sur {n}", "Continue sur {n}"], None,
         "La lecture reprend."),
        ("media_player.media_next_track", ["Titre suivant sur {n}",
                                           "Passe à la chanson suivante sur {n}",
                                           "Chanson suivante sur {n}"], None, "Titre suivant."),
        ("media_player.volume_up", ["Monte le son {dr}", "Augmente le volume {dr}"], None,
         "Volume augmenté."),
        ("media_player.volume_down", ["Baisse le son {dr}", "Baisse le volume {dr}"], None,
         "Volume baissé."),
        ("media_player.turn_off", ["Éteins {r}", "Coupe {r}", "Éteindre {r}"], None,
         confirm(ref, "éteint", "éteint")),
        ("media_player.turn_on", ["Allume {r}", "Mets en marche {r}"], None,
         confirm(ref, "allumé", "allumé")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Mets le volume {dn} à {v} %",
                         "Volume {dn} à {v} pour cent",
                         "Règle le son {dr} sur {v} %"]).format(r=ref.acc, dr=de(ref.acc),
                                                               dn=de(e.name), v=v)
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"Le volume est maintenant à {v} %."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, dr=de(ref.acc), n=e.name), \
        [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = R.choice([the(e.gender, low(e.name)) if e.name == e.label else e.name,
                  "l'aspirateur", "le robot aspirateur", "le robot"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Lance {n}", f"Démarre {n}", f"Envoie {n} nettoyer",
                         "Passe l'aspirateur", f"Fais passer {n}"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{e.name} se met au travail.", "L'aspirateur démarre."])
    if r < 0.8:
        user = R.choice([f"Renvoie {n} à sa base", f"Renvoie {n} à la station de charge",
                         f"Fais rentrer {n}"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{e.name} retourne à sa base."
    user = R.choice([f"Mets {n} en pause", f"Arrête {n}", f"Stoppe {n}"])
    return user, [action("vacuum.pause", [e])], f"{e.name} est en pause."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Active la scène {e.name}", f"Scène {e.name}",
                         f"Active {e.name}", f"Lance la scène {e.name}",
                         f"Mets l'ambiance {e.name}"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"La scène {e.name} est activée.", f"Scène {e.name} activée.", "C'est fait."])
    if e.domain == "script":
        user = R.choice([f"Exécute {e.name}", f"Lance {e.name}", f"Démarre {e.name}",
                         f"Lance le script {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} est en cours.", f"J'ai lancé {e.name}.", "C'est fait."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    # a group entity ("Toutes les lumières") replaces the list of members
    if (grp := h.group(domain)) is not None:
        ents = [grp]
    phrase, g = R.choice(ALL_GROUP[domain])
    whole = S.ALL_AREAS if domain in S.AREA_DOMAINS and h.group(domain) is None else None
    ref = Ref(phrase, phrase, g, ents, whole)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Éteins {phrase}", f"Coupe {phrase}", f"Éteindre {phrase}",
                             "Éteins toutes les lumières de la maison"])
            return user, [action("light.turn_off", ref)], R.choice(
                [f"{cap(phrase)} {est(g)} {agr('éteint', g)}.",
                 "Toutes les lumières sont éteintes.", "C'est fait."])
        user = R.choice([f"Allume {phrase}", f"Allumer {phrase}", f"Allume-moi {phrase}"])
        return user, [action("light.turn_on", ref)], R.choice(
            [f"{cap(phrase)} {est(g)} {agr('allumé', g)}.",
             "Toutes les lumières sont allumées.", "C'est fait."])
    if R.random() < 0.6:
        user = R.choice([f"Ferme {phrase}", f"Baisse {phrase}", f"Descends {phrase}",
                         f"Fermer {phrase}"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{cap(phrase)} sont fermés.", "C'est fait, tout est fermé."])
    user = R.choice([f"Ouvre {phrase}", f"Remonte {phrase}", f"Lève {phrase}",
                     f"Ouvrir {phrase}"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{cap(phrase)} sont ouverts.", "C'est fait, tout est ouvert."])


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


# imperatives that may start the second half of "X et y"
VERBS = ("Allume", "Éteins", "Mets", "Règle", "Ouvre", "Ferme", "Baisse", "Monte", "Remonte",
         "Descends", "Lève", "Lance", "Démarre", "Active", "Désactive", "Passe", "Augmente",
         "Tamise", "Change", "Coupe", "Arrête", "Stoppe", "Verrouille", "Déverrouille",
         "Arme", "Désarme", "Reprends", "Relance", "Renvoie", "Envoie", "Fais", "Appuie",
         "Déclenche", "Arrose", "Exécute", "Déploie", "Replie", "Rentre", "Sors", "Joue",
         "Ajoute", "Note", "Coche", "Raye", "Supprime", "Enlève", "Retire", "Rajoute",
         "Inscris", "Bloque", "Ramène", "Quitte")


def join_commands(a_, b):
    """Two command intents -> one sentence 'X et y', or None."""
    if not a_ or not b or a_[1] == b[1]:
        return None
    ids_a = {i for x in a_[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a_[0].rstrip(" ?.!").replace(", s'il te plaît", "").replace(" stp", "")
    ub = b[0].rstrip(" ?.!").replace(", s'il te plaît", "").replace(" stp", "")
    # both halves must be plain imperatives ("Allume ... et ferme ...")
    if ua.split(" ", 1)[0] not in VERBS or ub.split(" ", 1)[0].split("-")[0] not in VERBS \
            or "?" in ua or "?" in ub or "," in ub:
        return None
    return f"{ua} et {low(ub)}"


def lower_start(text, h):
    """Lowercase the start of a sentence that goes behind 'et', unless it
    starts with a device name."""
    if any(text.startswith(e.name) for e in h.ents):
        return text
    return low(text)


def s_compound(h, loc):
    a_ = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a_, b)
    if not user:
        return None
    actions = a_[1] + b[1]
    plain = a_[2] != "C'est fait." and b[2] != "C'est fait." and \
        not b[2].startswith(("C'est fait", "D'accord"))
    conf = R.choice(["C'est fait.", "C'est fait pour les deux.",
                     f"{a_[2].rstrip('.')} et {lower_start(b[2], h)}" if plain else "C'est fait."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def _state_base(e: Ent) -> str:
    """State as a masculine word ('allumé'); agreed by _state_word()."""
    d, s = e.domain, e.state
    if s == "unavailable":
        return "pas joignable"
    if d in ("light", "switch", "fan", "input_boolean"):
        return "allumé" if s == "on" else "éteint"
    if d == "cover":
        return "ouvert" if s == "open" else "fermé"
    if d == "lock":
        return "verrouillé" if s == "locked" else "pas verrouillé"
    if d == "binary_sensor":
        return "ouvert" if s == "on" else "fermé"
    if d == "media_player":
        return {"playing": "en lecture", "paused": "en pause",
                "off": "éteint", "idle": "allumé, mais ne joue rien"}[s]
    return s


def _state_word(e: Ent) -> str:
    return agr(_state_base(e), e.gender)


def state_clause(e: Ent, g=None) -> str:
    """'est allumée', "n'est pas verrouillée", 'sont fermés'."""
    g = g or e.gender
    w = agr(_state_base(e), g)
    if w.startswith("pas "):
        return f"{nest(g)} {w}"
    return f"{est(g)} {w}"


def s_question(h, loc):
    """State question -> get_state -> grounded French answer."""
    kinds = []
    temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"
             and e.label not in (OUTDOOR_LABEL, FORECAST_LABEL)]
    if temps:
        kinds += ["temp"] * 2
    outs = [e for e in h.of("sensor") if e.label == OUTDOOR_LABEL]
    if outs:
        kinds += ["outdoor"] * 2
    fcs = [e for e in h.of("sensor") if e.label == FORECAST_LABEL]
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
        user = R.choice(["Quelle température fait-il dehors ?", "Il fait froid dehors ?",
                         "Quelle est la température extérieure ?", "Il fait combien dehors ?",
                         "Il fait chaud dehors ?"])
        vals = [fr_num(float(e.state)) for e in outs]
        if len(outs) == 1:
            ans = R.choice([f"Il fait {deg(outs[0].state)} dehors.",
                            f"La température extérieure est de {deg(outs[0].state)}."])
        else:
            ans = (f"{outs[0].name} indique {deg(outs[0].state)}, "
                   f"{outs[1].name} {deg(outs[1].state)}.")
        ids, must = outs, vals
    elif kind == "forecast":
        if h.of("weather"):
            return None       # the weather entity answers with condition and course
        when = R.choice(fcs).attrs["_when"]
        ids = [e for e in fcs if e.attrs["_when"] == when]
        vals = [fr_num(float(e.state)) for e in ids]
        user = R.choice([f"Quelle température fera-t-il {when} ?", f"Il va faire combien {when} ?",
                         f"Que disent les prévisions pour {when} ?"])
        if len(ids) == 1:
            ans = R.choice([f"D'après les prévisions, il fera {deg(ids[0].state)} {when}.",
                            f"{cap(when)}, les prévisions annoncent {deg(ids[0].state)}."])
        else:
            ans = (f"Pour {when}, {ids[0].name} indique {deg(ids[0].state)} "
                   f"et {ids[1].name} {deg(ids[1].state)}.")
        must = vals
    elif kind == "temp":
        e = R.choice(temps)
        val = fr_num(float(e.state))
        if e.area is None or e.area in OUTDOOR or e.label != "Température":
            # a sensor without area is not automatically outside, and a printer
            # bed or a freezer does not tell how warm the room is
            user = R.choice([f"Qu'indique {e.name} ?", f"Combien de degrés mesure {e.name} ?"])
            ans = f"{e.name} indique {deg(e.state)}."
        else:
            p = prep(e.area)
            user = R.choice([f"Quelle température fait-il {p} ?", f"Il fait combien {p} ?",
                             f"Il fait chaud {p} ?", f"Température {droom(e.area)} ?"])
            ans = R.choice([f"{cap(p)}, il fait {deg(e.state)}.",
                            f"La température {droom(e.area)} est de {deg(e.state)}."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        if e.area:
            user = R.choice([f"Quelle est l'humidité {droom(e.area)} ?",
                             f"C'est humide {prep(e.area)} ?",
                             f"Quel est le taux d'humidité {droom(e.area)} ?"])
            ans = f"L'humidité {droom(e.area)} est de {e.state} %."
        else:
            user = R.choice(["Quelle est l'humidité ?", "Quel est le taux d'humidité ?"])
            ans = f"L'humidité est de {e.state} %."
        ids, must = [e], [e.state]
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = PLANT_ART[e.name.replace(f" {MOISTURE}", "")]
            user = R.choice([f"Quelle est l'humidité du sol {de(plant)} ?",
                             f"Est-ce qu'il faut arroser {plant} ?",
                             f"La terre {de(plant)} est humide ?"])
            ans = f"L'humidité du sol {de(plant)} est de {e.state} %."
        else:
            user = R.choice([f"Que dit {e.name} ?", f"Combien indique {e.name} ?",
                             f"Où en est {e.name} ?"])
            ans = f"{e.name} indique {e.state} %."
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"Combien consomme {e.name} en ce moment ?",
                         f"Qu'indique {e.name} ?", f"Quelle est la valeur {de(e.name)} ?"])
        val = fr_num(float(e.state))
        ans = f"{e.name} indique actuellement {unit(e.state, 'watt', 'watts')}."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"Sur combien de degrés {est(ref.g)} {agr('réglé', ref.g)} {ref.nom} ?",
                             f"Quelle est la consigne {de(ref.nom)} ?"])
            val = fr_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {est(ref.g)} {agr('réglé', ref.g)} sur {deg(e.attrs['temperature'])}."
        else:
            if e.area:
                p = prep(e.area)
                user = R.choice([f"Quelle température fait-il {p} ?", f"Il fait combien {p} ?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None   # ambiguous: sensor and thermostat in same room
            else:
                user = f"Quelle température mesure {ref.nom} ?"
                p = ""
            if "current_temperature" not in e.attrs:
                return None   # thermostat without its own sensor
            val = fr_num(e.attrs["current_temperature"])
            cur = deg(e.attrs["current_temperature"])
            ans = (f"{cap(p)}, il fait {cur}." if p
                   else f"{cap(ref.nom)} mesure {cur}.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        n, g = ref.nom, ref.g
        if e.domain == "cover":
            user, asked = R.choice([(f"Est-ce {que(n)} {est(g)} {agr('ouvert', g)} ?", "ouvert"),
                                    (f"{cap(n)} {est(g)} {agr('baissé', g)} ?", "fermé"),
                                    (f"Est-ce {que(n)} {est(g)} {agr('fermé', g)} ?", "fermé"),
                                    (f"{cap(n)} {est(g)} {agr('fermé', g)} ?", "fermé")])
        else:
            user, asked = R.choice([(f"Est-ce {que(n)} {est(g)} {agr('allumé', g)} ?", "allumé"),
                                    (f"{cap(n)} {est(g)} encore {agr('allumé', g)} ?", "allumé"),
                                    (f"Est-ce {que(n)} {est(g)} {agr('éteint', g)} ?", "éteint"),
                                    (f"{cap(n)} {est(g)} {agr('éteint', g)} ?", "éteint")])
        base = _state_base(e)
        word = agr(base, g)
        ans = f"{cap(n)} {state_clause(e, g)}."
        if base in ("allumé", "éteint", "ouvert", "fermé") and R.random() < 0.8:
            ans = ("Oui, " if base == asked else "Non, ") + n + f" {state_clause(e, g)}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"allumé": "Oui, ", "éteint": "Non, "}[asked]
            ans = f"{yes}{n} {est(g)} {agr('allumé', g)}, à {pct} % de luminosité."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            p = prep(room)
            user = R.choice([f"Quelles lumières sont allumées {p} ?",
                             f"Il y a encore de la lumière {p} ?"])
        else:
            ents = h.of("light")
            user = R.choice(["Quelles lumières sont allumées ?",
                             "Il reste de la lumière allumée quelque part ?",
                             "Il y a encore des lampes allumées ?", "Quelles lampes sont allumées ?"])
        q_area = room or S.ALL_AREAS
        if any(e.state == "unavailable" for e in ents):
            return None
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "Aucune lumière n'est allumée en ce moment." if not room else \
                f"{cap(prep(room))}, aucune lumière n'est allumée en ce moment."
            must = ["aucune"]
        elif len(on) == 1:
            g = on[0].gender
            ans = f"{cap(agr('seul', g))} {on[0].name} {est(g)} {agr('allumé', g)}."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"Les lumières allumées sont : {', '.join(names[:-1])} et {names[-1]}."
            must = names
        ids = ents
    elif kind == "which_covers":
        room = None
        rooms = [r for r in h.rooms if len(h.of("cover", r)) >= 2]
        noun = R.choice(["volets", "volets", "stores"])
        if rooms and R.random() < 0.35:
            room = R.choice(rooms)
            p = prep(room)
            user, want_open = R.choice([
                (f"Quels {noun} sont ouverts {p} ?", True),
                (f"Il y a encore des {noun} ouverts {p} ?", True),
                (f"Quels {noun} sont levés {p} ?", True),
                (f"Quels {noun} sont fermés {p} ?", False)])
        else:
            user, want_open = R.choice([
                (f"Quels {noun} sont ouverts ?", True),
                (f"Il reste des {noun} ouverts ?", True),
                (f"Quels {noun} sont encore levés ?", True),
                (f"Quels {noun} sont fermés ?", False),
                (f"Quels {noun} sont baissés ?", False)])
        ents = h.of("cover", room or "*")
        q_area, q_dom = room or S.ALL_AREAS, "cover"
        if any(e.state == "unavailable" for e in ents):
            return None
        hits = [e for e in ents if (e.state == "open") == want_open]
        word = "ouvert" if want_open else "fermé"
        other = "fermé" if want_open else "ouvert"
        where = f" {prep(room)}" if room else ""
        if not hits:
            ans = R.choice([f"Tous les {noun}{where} sont {other}s.",
                            f"Aucun {noun[:-1]} n'est {word}{where}, tous sont {other}s."])
            must = [other + "s"]
        elif len(hits) == len(ents):
            ans = f"Tous les {noun}{where} sont {word}s."
            must = [word + "s"]
        elif len(hits) == 1:
            g = hits[0].gender
            ans = f"{cap(agr('seul', g))} {hits[0].name} {est(g)} {agr(word, g)}."
            must = [hits[0].name]
        else:
            names = [e.name for e in hits]
            ans = f"Les {noun} {word}s sont : {', '.join(names[:-1])} et {names[-1]}."
            must = names
        ids = ents
    elif kind == "windows":
        # only window sensors - presence and motion sensors are binary_sensors too
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") == "window"]
        if not ents:
            return None
        user, yes_if_open = R.choice([("Toutes les fenêtres sont fermées ?", False),
                                      ("Il y a encore une fenêtre ouverte ?", True),
                                      ("Quelles fenêtres sont ouvertes ?", None),
                                      ("Est-ce que les fenêtres sont fermées ?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "Non, toutes les fenêtres sont fermées.",
                   False: "Oui, toutes les fenêtres sont fermées.",
                   None: "Toutes les fenêtres sont fermées."}[yes_if_open]
            must = ["fermées"]
        else:
            names = [e.name for e in opn]
            ans = ("La seule fenêtre ouverte est : " + names[0] + "." if len(names) == 1 else
                   "Les fenêtres ouvertes sont : " + ", ".join(names[:-1]) + " et " + names[-1] + ".")
            if yes_if_open is not None:
                ans = ("Oui. " if yes_if_open else "Non. ") + ans
            must = names
        ids = ents
        q_area, q_dom, q_dc = S.ALL_AREAS, "binary_sensor", "window"
    elif kind == "door":
        # "La porte-fenêtre du salon est ouverte ?" -> that contact, not the front door
        doors = [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]
        e = R.choice(doors)
        same = [d for d in doors if d.label == e.label]
        if e.area and (len(same) > 1 or R.random() < 0.5):
            what = f"{the(e.gender, low(e.label))} {rplace(e.area)}"
        elif len(same) == 1:
            what = the(e.gender, low(e.label))
        else:
            return None
        if e.state == "unavailable":
            return None
        opn = e.state == "on"
        g = e.gender
        user, ask_open = R.choice([(f"Est-ce {que(what)} est {agr('ouvert', g)} ?", True),
                                   (f"{cap(what)} est {agr('fermé', g)} ?", False),
                                   (f"Est-ce {que(what)} est bien {agr('fermé', g)} ?", False),
                                   (f"{cap(what)} est restée {agr('ouvert', g)} ?", True)])
        word = agr("ouvert" if opn else "fermé", g)
        ans = f"{'Oui' if opn == ask_open else 'Non'}, {e.name} est {word}."
        ids, must = [e], [word]
    elif kind == "car":
        rng = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"])
        car = rng.name.rsplit(" ", 1)[0]
        art = car_article(car)
        soc = next((e for e in h.of("sensor") if e.name == f"{car} Niveau de batterie"), None)
        if soc is not None and R.random() < 0.45:
            e = soc
            user = R.choice([f"Quel est le niveau de charge {de(art)} ?",
                             f"La batterie {de(art)} est chargée à combien ?",
                             f"Est-ce {que(art)} est chargée ?"])
            ans = f"{cap(art)} est chargée à {e.state} %."
        else:
            e = rng
            user = R.choice([f"Quelle autonomie reste-t-il {a(art)} ?",
                             f"Combien de kilomètres peut encore faire {art} ?",
                             f"Il reste combien d'autonomie {a(art)} ?"])
            ans = f"{cap(art)} a encore {e.state} kilomètres d'autonomie."
        ids, must = [e], [e.state]
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = the(e.gender, low(e.name))
        g = e.gender
        user = R.choice([f"Est-ce {que(nom)} est {agr('verrouillé', g)} ?",
                         f"{cap(nom)} est {agr('fermé', g)} à clé ?",
                         f"J'ai bien verrouillé {nom} ?"])
        word = _state_word(e)
        ans = (f"Oui, {nom} est {agr('verrouillé', g)}." if e.state == "locked"
               else f"Non, {nom} n'est pas {agr('verrouillé', g)}.")
        ids, must = [e], [word]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        if e.state == "unavailable":
            return None   # unreachable devices: see _error_question
        user = R.choice(["Est-ce que l'alarme est activée ?", "L'alarme est allumée ?",
                         "Quel est l'état de l'alarme ?"])
        txt = {"disarmed": "L'alarme est désactivée.",
               "armed_away": "L'alarme est activée (mode absence).",
               "armed_home": "L'alarme est activée en mode présence.",
               "armed_night": "L'alarme est activée en mode nuit."}[e.state]
        ans, ids, must = txt, [e], ["désactivée" if e.state == "disarmed" else "activée"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"Qu'est-ce qui passe sur {e.name} ?", f"Que joue {e.name} ?"])
        t = e.attrs.get("media_title", "")
        a_ = e.attrs.get("media_artist")
        ans = f"En ce moment, {e.name} joue {t}" + (f" {de(a_)}." if a_ else ".")
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
    """Two questions in one sentence ("Quels volets sont ouverts et il fait
    combien dehors ?"): both are read and answered (v16 dropped the second
    part). Two entity_id lists go into one get_state call, a domain query and
    another question into two calls."""
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
    for pre in ("Dis, ", "OK, ", "Hé, ", "S'il te plaît, "):
        q2 = q2.removeprefix(pre)
    q2 = q2[0].lower() + q2[1:]
    msgs = m1[:-4] + [{"role": "user", "content": f"{q1} et {q2}"}]
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
    """'Le sèche-linge a fini ?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = the(g, low(app))
    user = R.choice([f"Est-ce {que(nom)} a fini ?", f"{cap(nom)} tourne encore ?",
                     f"Il en est où, {nom} ?", f"{cap(nom)} a terminé ?",
                     f"Quel est l'état {de(nom)} ?"])
    phrase, key = APP_ANSWER[kind]
    phrase = agr(phrase, g)
    ans = R.choice([f"{cap(nom)} {phrase}.", f"D'après Home Assistant, {nom} {phrase}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [agr(key, g)]}


def s_relative_climate(h, loc):
    """'J'ai froid dans la salle de bain' -> get_state -> set_temperature +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    p, d = prep(e.area), droom(e.area)
    up = R.random() < 0.6
    user = R.choice([f"Augmente un peu la température {d}", f"J'ai froid {p}",
                     f"Monte le chauffage {d} d'un degré"] if up else
                    [f"Baisse un peu la température {d}", f"Il fait trop chaud {p}",
                     f"Baisse le chauffage {d} d'un degré"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, ok_result()),
             {"role": "assistant",
              "content": f"J'ai réglé le chauffage {d} de {fr_num(e.attrs['temperature'])} "
                         f"à {deg(new)}."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


MISSING_TYPES = [
    # domain, phrase, "aucun ...", "pas de ...", verbs (on, off), services
    # (on, off), state words (on, off), label filter
    ("light", "la lumière", "aucune lumière", "pas de lumière", ("Allume", "Éteins"),
     ("light.turn_on", "light.turn_off"), ("allumé", "éteint"), None),
    ("light", "la lampe", "aucune lampe", "pas de lampe", ("Allume", "Éteins"),
     ("light.turn_on", "light.turn_off"), ("allumé", "éteint"), None),
    ("cover", "les volets", "aucun volet", "pas de volets", ("Ouvre", "Ferme"),
     ("cover.open_cover", "cover.close_cover"), ("ouvert", "fermé"), "plain"),
    ("climate", "le chauffage", "aucun chauffage", "pas de chauffage", ("Allume", "Éteins"),
     ("climate.turn_on", "climate.turn_off"), ("allumé", "éteint"), None),
    ("media_player", "la télé", "aucune télé", "pas de télé", ("Allume", "Éteins"),
     ("media_player.turn_on", "media_player.turn_off"), ("allumé", "éteint"), "Télé"),
    ("fan", "le ventilateur", "aucun ventilateur", "pas de ventilateur", ("Allume", "Éteins"),
     ("fan.turn_on", "fan.turn_off"), ("allumé", "éteint"), None),
]


def celui(g):
    return {"m": "celui", "f": "celle", "mp": "ceux", "fp": "celles"}[g]


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    mt = FORCE.get("mtype")
    dom, phrase, aucun, pas_de, verbs, svcs, words, filt = R.choice(
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
    p, d = prep(room), droom(room)
    # the room often holds a same-domain device of another kind (speaker, voice
    # satellite); it is no TV and must not stand in for the room in the answer
    if dom == "media_player" and R.random() < 0.6 and \
            not any(e.area == room for e in h.of(dom)):
        if R.random() < 0.6:
            h.add_voice_media(room)
        else:
            mt = R.choice([t for t in TYPES["media_player"] if t[0] != "Télé"])
            h.add("media_player", room, mt[0], mt[1], mt[2], state=R.choice(["idle", "off"]))
    turn_on = R.random() < 0.5
    verb = verbs[0] if turn_on else verbs[1]
    user = R.choice([f"{verb} {phrase} {d}", f"{verb} {phrase} {p}",
                     f"{cap(p)}, {low(verb)} {phrase}",
                     f"Tu peux {'allumer' if verb == 'Allume' else 'éteindre' if verb == 'Éteins' else 'ouvrir' if verb == 'Ouvre' else 'fermer'} {phrase} {d} ?"])
    others = [e for e in ents if e.area and e.area != room]
    # suggest the other device only if there are one or two candidates
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        op = prep(alt.area)
        where = "" if alt.area in alt.name else f", {op}"
        ans = R.choice([f"{cap(p)}, il n'y a {pas_de}. Il y en a {op} : {alt.name}.",
                        f"{cap(p)}, je ne trouve {aucun}. Tu veux dire {alt.name}{where} ?",
                        f"La liste des appareils ne contient {aucun} {p}. {cap(op)}, il y a {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"{cap(p)}, je ne trouve {aucun} que je puisse commander.",
                        f"La liste des appareils ne contient {aucun} {p}.",
                        f"Je n'ai trouvé {aucun} {p} que je puisse commander."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        # user accepts the device in the other room -> now it is switched
        op = prep(alt.area)
        c = celui(alt.gender)
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Alors {c} {droom(alt.area)}", f"Oui, c'est {c}-là", f"D'accord, alors {c}-là",
             f"Oui, {alt.name}", f"Alors {op}"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        word = agr(words[0] if turn_on else words[1], alt.gender)
        part = words[0] if turn_on else words[1]
        where = "" if alt.area in alt.name else f" {op}"
        msgs += [call, tool_msg(cid, ok_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"J'ai {part} {alt.name}{where}.",
                      f"C'est fait, {alt.name} {est(alt.gender)} maintenant {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        # user insists on the room, sometimes twice -> still no invented action
        pushes = ["Si, fais-le !", "Mais si, vas-y !", "Mais si, il y en a",
                  "Fais-le quand même", f"Si, {p} !", f"Non, je veux dire {p}",
                  "Ah bon, il n'y en a pas ?", "Si, il y en a un", "Vas-y, c'est tout"]
        refusals = [
            f"{cap(p)}, il n'y a {pas_de} dans ta liste d'appareils. Vérifie que l'appareil est bien exposé à Assist.",
            f"Je ne vois vraiment {aucun} {p} que je puisse commander.",
            f"Ma liste d'appareils ne contient {aucun} {p}, je ne peux donc rien y commander."]
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
        if dev.lower() in labels or (dev == "Tondeuse" and h.of("lawn_mower")):
            return None
        t = the(g, low(dev))
        user = R.choice([f"Allume {t}", f"Mets en marche {t}", f"Éteins {t}",
                         f"Est-ce {que(t)} est {agr('allumé', g)} ?", f"Lance {t}"])
        ans = R.choice([f"Je ne trouve pas d'appareil « {dev} » dans ta maison connectée.",
                        f"Je ne connais aucun appareil « {dev} ». Je ne peux malheureusement pas le commander.",
                        f"« {dev} » ne figure pas dans la liste des appareils."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != BATH2
                   and not (r_[0] == BATH and BATH2 in h.rooms)]
        room = R.choice(missing)
        p, d = prep(room), droom(room)
        what = R.choice([("la lumière", "aucune lumière", "Allume", "Éteins"),
                         ("les volets", "aucun volet", "Ouvre", "Ferme"),
                         ("le chauffage", "aucun chauffage", "Allume", "Éteins")])
        user = R.choice([f"{what[2]} {what[0]} {d}",
                         f"{what[3]} {what[0]} {d}",
                         f"{what[3]} {what[0]} {p}"])
        ans = R.choice([f"{cap(p)}, je ne trouve {what[1]} que je puisse commander.",
                        f"La liste des appareils ne contient {what[1]} {p}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


WEATHER_CHAT = "Quel temps fera-t-il demain ?"
CHAT = [
    (["Merci", "Merci beaucoup", "Super, merci", "Merci à toi"],
     ["De rien.", "Avec plaisir.", "Pas de souci."]),
    (["Bonjour", "Salut", "Coucou", "Bonsoir"],
     ["Bonjour, comment puis-je t'aider ?", "Salut ! Qu'est-ce que je peux faire pour toi ?"]),
    (["Qu'est-ce que tu sais faire ?", "Tu peux m'aider à quoi ?",
      "Qu'est-ce que tu peux commander ?"],
     ["Je peux commander les lumières, les volets, le chauffage et d'autres appareils de ta "
      "maison, et te dire dans quel état ils sont ou ce que mesurent les capteurs. Je peux "
      "aussi lancer de la musique, régler des minuteurs, gérer tes listes, te donner l'heure "
      "et la météo, et faire des annonces.",
      "Je commande tes appareils, je lance de la musique, je règle des minuteurs, j'ajoute "
      "des articles à ta liste de courses, je te donne l'heure et la météo et je fais des "
      "annonces dans la maison."]),
    (["Qui es-tu ?", "Tu es qui ?"], ["Je suis ton assistant vocal pour la maison connectée."]),
    (["Raconte-moi une blague"],
     ["Pourquoi la lampe n'a pas d'amis ? Parce qu'elle s'éteint trop vite.",
      "Que dit le volet à la fenêtre ? T'inquiète, je te couvre."]),
    ([WEATHER_CHAT],
     ["Je n'ai malheureusement pas de prévisions météo. Par contre, je peux te dire ce que "
      "mesurent tes capteurs en ce moment."]),
    (["Bonne nuit"], ["Bonne nuit. Je dois encore éteindre quelque chose ?", "Bonne nuit."]),
    # HassNevermind: nothing to do
    (["Laisse tomber", "Annule", "C'est bon", "Rien", "Peu importe", "Ah, rien",
      "Finalement non", "C'est réglé", "Oublie", "Non, rien",
      "J'ai rien dit", "Ça ira", "Non, laisse"],
     ["D'accord.", "Très bien.", "D'accord, on laisse tomber."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == WEATHER_CHAT and (
            h.of("weather") or any(e.label == FORECAST_LABEL for e in h.of("sensor"))):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


def _unreachable(name):
    return fix_pl(R.choice([f"Ça n'a malheureusement pas marché. {name} ne répond pas.",
                     f"Une erreur s'est produite : {name} n'est pas joignable.",
                     f"{name} n'est pas joignable pour le moment, je n'ai donc pas pu exécuter la commande."]), name)


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
            ["Réessaie", "Encore une fois, s'il te plaît", "Essaie encore", "Recommence"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": fix_pl(R.choice([
                     f"{name} n'est toujours pas joignable. Vérifie que l'appareil est alimenté et connecté.",
                     f"La deuxième tentative n'a pas marché non plus, {name} ne répond pas."]), name)}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        msgs.append({"role": "user", "content": voice(R.choice(
            ["C'est allumé maintenant ?", "Ah bon, ça ne marche pas ?", "Ça a marché ?",
             "Alors, c'est allumé ?"]
            if onoff else ["Ça a marché ?", "Alors, ça a fonctionné ?", "C'est vrai ?"]))})
        e = next(x for x in h.ents if x.eid == eid)
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": fix_pl(R.choice([
                     f"Non. {name} n'est pas joignable, je n'ai donc rien pu faire.",
                     f"D'après Home Assistant, {name} n'est pas joignable en ce moment. Rien n'a été commandé."]), name)}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def _down_txt(ents):
    g = group_gender(ents)
    return f"{_names_e(ents)} {nest(g)} pas {agr_word('joignable', 'mp' if plural(g) else 'm')} en ce moment."


def _names_e(ents):
    ns = [e.name for e in ents]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " et " + ns[-1]


def _error_room(h, loc):
    """Some lights of a room are unreachable, the others work. Questions and
    commands about the room report both groups by name."""
    rooms = [r for r in h.rooms if len([e for e in h.of("light", r)
                                         if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    # rooms with a light named like the room are the hard case ("Cuisine" is
    # down, "Cuisine Gauche" is not), so they come up more often
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
    p, d = prep(room), droom(room)
    ids = [e.eid for e in ents]
    nd = _names_e(down)
    down_txt = _down_txt(down)
    msgs = base_msgs(h, loc)
    if named_q or R.random() < 0.5:
        # question
        user, kind = R.choice([(f"Est-ce que la lumière est allumée {p} ?", "yn"),
                               (f"Il y a encore de la lumière {p} ?", "yn"),
                               (f"Quelles lumières sont allumées {p} ?", "which"),
                               (f"Les lampes {d} sont éteintes ?", "yn_off")])
        on = [e for e in up if e.state == "on"]
        off = [e for e in up if e.state != "on"]
        parts = []
        if on:
            g = group_gender(on)
            parts.append(f"{_names_e(on)} {est(g)} {agr('allumé', g)}.")
        if off and (not on or R.random() < 0.5):
            g = group_gender(off)
            parts.append(f"{_names_e(off)} {est(g)} {agr('éteint', g)}.")
        if not up:
            ans = f"{cap(down_txt)} Je ne peux donc pas dire si la lumière est allumée {p}."
        else:
            if kind == "yn":
                parts[0] = ("Oui, " if on else "Non, ") + parts[0]
            elif kind == "yn_off":
                # same wording as the "allumée ?" question, only the yes/no
                # flips; a separate phrase made the model mix up both questions
                parts[0] = ("Non, " if on else "Oui, ") + parts[0]
            ans = " ".join(parts) + " " + down_txt
        args = {"domain": "light", "area": room, "entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "error", "gold": [["get_state", args]]}
    # command for the whole room: the reachable lights switch
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    user = R.choice([f"Allume la lumière {d}", f"Allume la lumière {p}",
                     f"Lumière {d}, s'il te plaît"] if on_cmd else
                    [f"Éteins la lumière {d}", f"Éteins la lumière {p}",
                     f"Coupe la lumière {d}"])
    actions = [action(svc, ents, area=room)]
    done = [e.eid for e in up]
    word = "allumé" if on_cmd else "éteint"
    if up:
        did = _names_e(up)
        g = group_gender(up)
        ans = R.choice([
            f"J'ai {word} {did}. {down_txt}",
            f"Ça n'a fonctionné qu'en partie : {did} {est(g)} {agr(word, g)}, mais {down_txt[:-1]}.",
            f"{down_txt} J'ai {word} le reste : {did}.",
        ])
    else:
        ans = f"{down_txt} Je n'ai donc pas pu commander la lumière {d}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, S.control_result(
                 [f"Entity {', '.join(e.eid for e in down)} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _named_room_light(h, loc):
    """A light carries the room's name ("Cuisine") next to other lights there.
    "la lumière de la cuisine" means all lights of the room; only the bare
    name ("Allume Cuisine") means that single lamp."""
    rooms = [r for r in h.rooms
             if any(e.name == r for e in h.of("light", r))
             and len([e for e in h.of("light", r) if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = [e for e in h.of("light", room) if e.label != "__group__"]
    named = next(e for e in ents if e.name == room)
    p, d = prep(room), droom(room)
    msgs = base_msgs(h, loc)
    r = R.random()
    if r < 0.3:
        user, kind = R.choice([(f"Est-ce que la lumière est allumée {p} ?", "yn"),
                               (f"Il y a de la lumière {p} ?", "yn"),
                               (f"La lumière {d} est encore allumée ?", "yn"),
                               (f"Quelles lumières sont allumées {p} ?", "which")])
        on = [e for e in ents if e.state == "on"]
        off = [e for e in ents if e.state != "on"]
        if on:
            g = group_gender(on)
            ans = f"{_names_e(on)} {est(g)} {agr('allumé', g)}."
            if off and R.random() < 0.6:
                g2 = group_gender(off)
                ans += f" {_names_e(off)} {est(g2)} {agr('éteint', g2)}."
            if kind == "yn":
                ans = "Oui, " + ans
        elif kind == "yn":
            g2 = group_gender(off)
            ans = R.choice([f"Non, {_names_e(off)} {est(g2)} {agr('éteint', g2)}.",
                            f"Non, aucune lumière n'est allumée {p}."])
        else:
            ans = f"{cap(p)}, aucune lumière n'est allumée en ce moment."
        args = {"domain": "light", "area": room, "entity_ids": [e.eid for e in ents]}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        must = [e.name for e in on] if on else ["non" if kind == "yn" else "aucune"]
        return msgs, {"cat": "question", "gold": [["get_state", args]],
                      "answer_contains": must}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    word = "allumé" if on_cmd else "éteint"
    if r < 0.8:
        # the room: every light in it
        user = R.choice([f"Allume la lumière {d}", f"Allume la lumière {p}",
                         f"Lumière {d}, s'il te plaît", f"{cap(p)}, allume la lumière",
                         f"Tu peux allumer la lumière {d} ?"] if on_cmd else
                        [f"Éteins la lumière {d}", f"Éteins la lumière {p}",
                         f"Coupe la lumière {d}", f"{cap(p)}, éteins la lumière",
                         f"Tu peux éteindre la lumière {d} ?"])
        targets = Ref(f"la lumière {d}", f"la lumière {d}", "f", ents, room)
        conf = confirm(targets, word, word)
    else:
        # the bare name: only the lamp called like the room
        user = R.choice([f"Allume {room}", f"Allume la lampe {room}", f"Allumer {room}",
                         f"Allume {room}, s'il te plaît"] if on_cmd else
                        [f"Éteins {room}", f"Éteins la lampe {room}", f"Éteindre {room}",
                         f"Coupe {room}"])
        targets = [named]
        conf = confirm(Ref(room, room, named.gender, targets), word, word)
    actions = [action(svc, targets)]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, ok_result()), {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_no_areas(h, loc):
    """Home Assistant without any areas: a room cannot be resolved. The model
    says so and explains how to set up areas; device names and "toutes les
    lumières" keep working."""
    lights = h.of("light")
    rooms = [r for r in h.rooms if h.of("light", r) or h.of("cover", r, plain=True)]
    if len(lights) < 2 or not rooms:
        return None
    room = R.choice(rooms)
    p, d = prep(room), droom(room)
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
            user = R.choice([f"Allume la lumière {d}", f"Éteins la lumière {p}",
                             f"Est-ce que la lumière est allumée {p} ?",
                             f"Quelles lumières sont allumées {p} ?",
                             f"Allume la lumière {p}", f"{cap(p)}, éteins la lumière"])
            what = "lumières"
        else:
            user = R.choice([f"Baisse les volets {d}", f"Remonte les volets {d}",
                             f"Les volets {d} sont fermés ?", f"Ferme les volets {p}"])
            what = "volets"
        ans = R.choice([
            f"Aucune pièce n'est configurée dans ton Home Assistant, je ne sais donc pas "
            f"quels {what} se trouvent {p}. Crée les pièces dans les paramètres, sous "
            f"« Pièces, étiquettes et zones », puis attribue-leur les appareils.",
            f"Je ne sais pas ce qui se trouve {p}, car aucune pièce n'est définie dans "
            f"Home Assistant. Configure les pièces dans les paramètres, sous « Pièces, "
            f"étiquettes et zones », et attribue-leur les appareils."])
        if what == "lumières":
            ans = ans.replace("quels lumières", "quelles lumières")
        if named:
            ans += f" En attendant, tu peux aussi appeler un appareil par son nom, par exemple « {named[0].name} »."
        msgs += [{"role": "user", "content": voice(user)}, {"role": "assistant", "content": ans}]
        return msgs, {"cat": "unknown", "gold": [], "no_history": True}
    on = R.random() < 0.5
    svc = "light.turn_on" if on else "light.turn_off"
    word = "allumé" if on else "éteint"
    if r < 0.85 or h.group("light") is not None:
        # a device named directly still works
        e = R.choice(lights)
        user = R.choice([f"Allume {e.name}", f"Allumer {e.name}", f"Tu peux allumer {e.name} ?"] if on
                        else [f"Éteins {e.name}", f"Coupe {e.name}", f"Éteindre {e.name}"])
        actions = [action(svc, [e])]
        conf = R.choice([f"{e.name} {est(e.gender)} maintenant {agr(word, e.gender)}.",
                         f"J'ai {word} {e.name}."])
    else:
        # the whole house needs no areas
        user = R.choice(["Allume toutes les lumières", "Allume toutes les lampes"] if on
                        else ["Éteins toutes les lumières", "Éteins la lumière partout"])
        actions = [action(svc, lights, area=S.ALL_AREAS)]
        conf = f"Toutes les lumières sont {agr(word, 'fp')}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]],
                  "no_history": True}


def partial_answer(h, ok_actions, failed_name, conf=None):
    """Both outcomes, always: what worked and what did not."""
    return fix_pl(_partial_answer(h, ok_actions, failed_name, conf), failed_name)


def _partial_answer(h, ok_actions, failed_name, conf=None):
    if any(len(x["entity_id"]) > 3 for x in ok_actions):
        # long lists: the normal confirmation ("Toutes les lumières sont
        # allumées.") is clearer
        if conf and conf != "C'est fait.":
            return R.choice([f"{conf} {failed_name} n'est cependant pas joignable en ce moment.",
                             f"{failed_name} n'est pas joignable en ce moment. Pour le reste : {low(conf) if not conf.startswith(tuple(e.name for e in h.ents)) else conf}"])
        n = sum(len(x["entity_id"]) for x in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = PARTICIPLE.get(svc, "commandé {w}").format(w=f"{n} appareils", dw=f"de {n} appareils")
    else:
        did = describe_all(h, ok_actions)
    return R.choice([
        f"J'ai {did}. {failed_name} n'est cependant pas joignable en ce moment.",
        f"Ça n'a fonctionné qu'en partie : j'ai {did}, mais {failed_name} n'est pas joignable.",
        f"{failed_name} n'est pas joignable en ce moment. Pour le reste, c'est fait : j'ai {did}.",
    ])


def _error_partial(h, loc):
    """Two commands, one device is down: report both outcomes."""
    a_ = _single_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a_, b)
    if not user:
        return None
    eid = a_[1][0]["entity_id"][0]
    name = next(e.name for e in h.ents if e.eid == eid)
    actions = a_[1] + b[1]
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
    g = e.gender
    user = R.choice([f"Est-ce {que(e.name)} {est(g)} {agr('allumé', g)} ?",
                     f"Quel est l'état {de(e.name)} ?",
                     f"Que fait {e.name} ?", f"{e.name}, c'est allumé ?"])
    args = {"entity_ids": [e.eid]}
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": fix_pl(R.choice([
                 f"{e.name} n'est pas joignable en ce moment, je ne connais donc pas son état actuel.",
                 f"Je ne peux pas te le dire : {e.name} n'est pas joignable en ce moment."]), e.name)}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


def s_followup(h, loc):
    """Two-turn: command, then 'et rééteins-la' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    p1, p2 = prep(r1), prep(r2)
    d1, d2 = droom(r1), droom(r2)
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Allume la lumière {d1}",
                                                            f"Allume la lumière {p1}",
                                                            f"Lumière {d1}, s'il te plaît"]))}]
    a1 = [action("light.turn_on", e1, area=r1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": f"La lumière {d1} est allumée."}]
    k = R.random()
    if k < 0.15:
        # user doubts the result -> check the real state instead of guessing
        really = "Elle est vraiment allumée ?"
        u2 = R.choice(["Non, elle n'est pas allumée", really, "C'est vrai ?",
                       "Chez moi, il fait encore noir"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        q = {"domain": "light", "area": r1, "entity_ids": [e.eid for e in e1]}
        c2, cid2 = tool_call("get_state", q)
        ans = (f"D'après Home Assistant, la lumière {d1} est allumée." if really_on else
               f"D'après Home Assistant, la lumière {d1} est encore éteinte. Je la rallume ?")
        if u2 == really:
            ans = ("Oui. " if really_on else "Non. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", q]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["Et rééteins-la", "Éteins-la à nouveau", "Finalement, éteins-la"]), \
            [action("light.turn_off", e1, area=r1)], f"La lumière {d1} est de nouveau éteinte."
    elif k < 0.72:
        u2, a2, ans = R.choice([f"Et {p2} ?", f"{cap(p2)} aussi", f"Et {p2} aussi, s'il te plaît"]), \
            [action("light.turn_on", e2, area=r2)], f"La lumière {d2} est allumée aussi."
    else:
        kind = R.choice(["down", "up", "pct"])
        if kind == "down":
            u2 = R.choice(["Un peu moins fort", "Baisse un peu"])
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20}, area=r1)]
            ans = f"J'ai baissé la luminosité de la lumière {d1}."
        elif kind == "up":
            u2 = R.choice(["Un peu plus clair", "Monte un peu"])
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20}, area=r1)]
            ans = f"J'ai augmenté la luminosité de la lumière {d1}."
        else:
            u2 = R.choice(["À 30 %", "Mets-la à 30 pour cent"])
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30}, area=r1)]
            ans = f"La lumière {d1} est maintenant à 30 %."
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
# what "J'ai ..." says for a service: {w} the devices, {dw} "de" + devices
PARTICIPLE = {"turn_on": "allumé {w}", "turn_off": "éteint {w}",
              "open_cover": "ouvert {w}", "close_cover": "fermé {w}",
              "stop_cover": "arrêté {w}", "lock": "verrouillé {w}", "unlock": "déverrouillé {w}",
              "media_play": "relancé la lecture sur {w}", "media_pause": "mis {w} en pause",
              "media_stop": "arrêté la lecture sur {w}",
              "media_next_track": "passé au titre suivant sur {w}",
              "media_previous_track": "remis le titre précédent sur {w}",
              "volume_up": "monté le volume {dw}", "volume_down": "baissé le volume {dw}",
              "start": "lancé {w}", "pause": "mis {w} en pause",
              "return_to_base": "renvoyé {w} à sa base",
              "start_mowing": "lancé {w}", "dock": "renvoyé {w} à sa station",
              "open_valve": "ouvert {w}", "close_valve": "fermé {w}", "press": "déclenché {w}",
              "alarm_arm_home": "activé {w} en mode présence", "alarm_arm_away": "activé {w}",
              "alarm_arm_night": "activé {w} en mode nuit",
              "alarm_disarm": "désactivé {w}"}
# participles for "Quelle lumière as-tu allumée ?" (agreed with the noun)
QPART = {"turn_on": "allumé", "turn_off": "éteint", "open_cover": "ouvert",
         "close_cover": "fermé", "stop_cover": "arrêté", "lock": "verrouillé",
         "unlock": "déverrouillé", "media_pause": "mis en pause"}
DOMAIN_NOUN = {"light": ("Quelle lumière", "Quelles lampes", "f"),
               "cover": ("Quel volet", "Quels volets", "m"),
               "climate": ("Quel chauffage", "Quels radiateurs", "m"),
               "switch": ("Quel appareil", "Quels appareils", "m"),
               "media_player": ("Quel appareil", "Quels appareils", "m"),
               "fan": ("Quel ventilateur", "Quels ventilateurs", "m")}


def apply_actions(h, actions):
    """Change the simulated states, so later questions see the result."""
    by_id = {e.eid: e for e in h.ents}
    for x in actions:
        dom, svc = x["service"].split(".", 1)
        data = x.get("data") or {}
        for eid in x["entity_id"]:
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
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " et " + ns[-1]


def describe(h, x, by_name=False):
    """'allumé Plafonnier Couloir' for one action (after 'J'ai')."""
    dom, svc = x["service"].split(".", 1)
    data = x.get("data") or {}
    if x.get("area") == S.ALL_AREAS and not by_name:
        who = "toutes les lumières"
    elif x.get("area") and not by_name:
        who = f"la lumière {droom(x['area'])}"
    else:
        who = _names(h, x["entity_id"])
    for hook in DESCRIBE_HOOKS:
        out = hook(h, x, who)
        if out:
            return out
    if "brightness_pct" in data:
        return f"réglé {who} sur {data['brightness_pct']} %"
    if "brightness_step_pct" in data:
        return f"{'augmenté' if data['brightness_step_pct'] > 0 else 'baissé'} la luminosité {de(who)}"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"changé la couleur {de(who)}"
    if svc == "set_temperature":
        return f"réglé {who} sur {deg(data['temperature'])}"
    if svc == "set_cover_position":
        return f"mis {who} à {data['position']} %"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"réglé {who} sur {data.get('percentage')} %"
    if svc == "volume_set":
        return f"réglé le volume {de(who)} sur {round(data['volume_level'] * 100)} %"
    if dom == "scene":
        return f"activé la scène {who}"
    if dom == "script":
        return f"lancé {who}"
    return PARTICIPLE.get(svc, "commandé {w}").format(w=who, dw=de(who))


def merge_actions(actions):
    """Consecutive list items for the same list become one action, so the
    answer says 'ajouté du lait et du café à la liste de courses'."""
    out = []
    for x in actions:
        prev = out[-1] if out else None
        if prev and x["service"].startswith("todo.") and prev["service"] == x["service"] \
                and prev["entity_id"] == x["entity_id"] and "item" in (x.get("data") or {}):
            prev["data"]["items"] = prev["data"].get("items", [prev["data"]["item"]]) + [x["data"]["item"]]
            continue
        out.append({**x, "data": dict(x.get("data") or {})} if x.get("data") else dict(x))
    return out


def describe_all(h, actions):
    """'allumé X, réglé Y et ajouté Z' for several actions."""
    parts = [describe(h, x) for x in merge_actions(actions)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " et " + parts[-1]


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
        ok = [x for x in actions if not errors or
              all(i in (done or []) for i in x["entity_id"])]
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
    """'Quelle lumière as-tu éteinte ?' -> answered from the history."""
    if not c.last:
        return False
    x = c.last[-1]
    dom, svc = x["service"].split(".", 1)
    ids = x["entity_id"]
    if dom in DOMAIN_NOUN and svc in QPART and not x.get("data"):
        sg, pl, g = DOMAIN_NOUN[dom]
        many = len(ids) > 1
        noun = pl if many else sg
        gq = (g + "p") if many else g
        q = R.choice([f"{noun} as-tu {agr(QPART[svc], gq)} ?",
                      f"Tu as {QPART[svc]} {low(noun)} ?"])
        user = R.choice([q, q, "Qu'est-ce que tu viens de faire ?", "C'était quel appareil ?"])
    else:
        user = R.choice(["Qu'est-ce que tu viens de faire ?", "Qu'est-ce que tu as fait exactement ?"])
    c.user(user)
    # the lamps an area stood for are known from "done" in the tool result
    parts = [describe(c.h, y, by_name=True) for y in merge_actions(c.last)]
    c.say(R.choice(["J'ai ", "À l'instant, j'ai "]) + " et ".join(parts) + ".")
    return True


def turn_summary(c):
    """'Qu'est-ce que tu as fait ?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    # "Quels radiateurs as-tu réglés ?" after the bathroom + "Et dans la
    # cuisine aussi": every action on that device type, not only the last one
    by_dom = {}
    for x in c.done:
        by_dom.setdefault(x["service"].split(".")[0], []).append(x)
    multi = [d for d, xs in by_dom.items() if len(xs) >= 2 and d in SUMMARY_ASK]
    if multi and R.random() < 0.6:
        d = R.choice(multi)
        c.user(R.choice(SUMMARY_ASK[d]))
        parts = [describe(c.h, x) for x in merge_actions(by_dom[d])]
    else:
        c.user(R.choice(["Qu'est-ce que tu as fait jusqu'ici ?",
                         "Qu'est-ce que tu as commandé jusqu'à présent ?",
                         "Résume ce que tu as fait"]))
        parts = [describe(c.h, x) for x in merge_actions(c.done)]
    c.say("J'ai " + (", ".join(parts[:-1]) + " et " if len(parts) > 1 else "")
          + parts[-1] + ".")
    return True


SUMMARY_ASK = {
    "climate": ["Quels radiateurs as-tu réglés ?", "Quels thermostats as-tu modifiés ?",
                "Qu'est-ce que tu as changé au chauffage ?", "Quels chauffages as-tu réglés ?"],
    "light": ["Quelles lumières as-tu commandées ?", "Quelles lampes as-tu touchées ?",
              "Qu'est-ce que tu as fait avec les lumières ?"],
    "cover": ["Quels volets as-tu bougés ?", "Quels stores as-tu actionnés ?",
              "Qu'est-ce que tu as fait avec les volets ?"],
}


def turn_undo(c):
    if not c.last:
        return False
    inv = []
    for x in c.last:
        dom, svc = x["service"].split(".", 1)
        if svc not in INVERSE or x.get("data"):
            return False
        inv.append({"service": f"{dom}.{INVERSE[svc]}",
                    **({"area": x["area"]} if "area" in x else {}),
                    "entity_id": x["entity_id"]})
    c.user(R.choice(["Annule ça", "Finalement non", "Remets comme avant",
                     "Reviens en arrière"] +
                    (["Et rééteins", "Éteins à nouveau"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    (["Et rallume"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "D'accord, j'ai " + describe_all(c.h, inv) + ".")
    return True


def turn_correct(c):
    """'Non, je voulais dire dans la chambre' -> undo + the right device."""
    if not c.last or len(c.last) != 1:
        return False
    x = c.last[0]
    dom, svc = x["service"].split(".", 1)
    if svc not in ("turn_on", "turn_off", "open_cover", "close_cover") or x.get("data") \
            or dom not in ("light", "cover", "switch"):
        return False
    # undo everything the last action switched, not only its first entity
    cur_ids = list(x["entity_id"])
    cur_areas = {e.area for e in c.h.ents if e.eid in cur_ids}
    others = [e for e in c.h.of(dom, plain=dom == "cover")
              if e.area and e.area not in cur_areas and e.state != "unavailable"]
    if not others:
        return False
    o = R.choice(others)
    p = prep(o.area)
    if R.random() < 0.6:
        # "je voulais dire dans la salle à manger": every device of that kind
        # in the room
        c.user(R.choice([f"Non, je voulais dire {p}", f"Pas là, {p}",
                         f"Non, {p}", f"Je parlais {droom(o.area)}"]))
        new_ids = [e.eid for e in c.h.of(dom, o.area, plain=dom == "cover")
                   if e.state != "unavailable"]
        by_room = len(new_ids) == len(c.h.of(dom, o.area, plain=dom == "cover"))
    else:
        c.user(R.choice([f"Pas ça, {o.name}", f"Je voulais dire {o.name}"]))
        new_ids = [o.eid]
        by_room = False
    undo = {"service": f"{dom}.{INVERSE[svc]}", "entity_id": cur_ids}
    if "area" in x:
        undo = {"service": undo["service"], "area": x["area"], "entity_id": cur_ids}
    new = {"service": x["service"], "entity_id": new_ids}
    if dom in S.AREA_DOMAINS and by_room:
        new = {"service": x["service"], "area": o.area, "entity_id": new_ids}
    acts = [undo, new]
    c.control(acts, f"D'accord, j'ai {describe(c.h, acts[0])} et {describe(c.h, acts[1])}.")
    return True


def turn_correct_recap(c):
    """Correction followed by "Qu'est-ce que tu as fait ?": the recap lists
    exactly the executed actions."""
    if not turn_correct(c):
        return False
    if R.random() < 0.7:
        return turn_summary(c)
    return turn_which(c)


def turn_also(c):
    """'Et dans la salle de bain aussi' -> same service on the same device
    type in another room."""
    if not c.last or len(c.last) != 1:
        return False
    x = c.last[0]
    dom = x["service"].split(".")[0]
    if dom not in ("light", "cover", "climate"):
        return False
    first = next(e for e in c.h.ents if e.eid == x["entity_id"][0])
    rooms = [r for r in c.h.rooms if r != first.area and
             [e for e in c.h.of(dom, r, plain=dom == "cover") if e.state != "unavailable"]]
    if not rooms:
        return False
    r = R.choice(rooms)
    p = prep(r)
    ents = [e for e in c.h.of(dom, r, plain=dom == "cover") if e.state != "unavailable"]
    c.user(R.choice([f"Et {p} aussi", f"{cap(p)} aussi", f"Pareil {p}"]))
    act = {"service": x["service"],
           **({"area": r} if dom in S.AREA_DOMAINS and len(ents) == len(c.h.of(dom, r)) else {}),
           "entity_id": [e.eid for e in ents]}
    if x.get("data"):
        act["data"] = x["data"]
    c.control([act], R.choice(["D'accord, j'ai ", "C'est fait, j'ai "])
              + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["Un peu plus clair", "Encore plus lumineux", "Plus de lumière"]), \
            {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["Un peu plus sombre", "Encore moins de lumière",
                               "Baisse un peu la lumière"]), {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = R.choice([f"À {p} %", f"Mets à {p} pour cent"]), {"brightness_pct": p}
    act = {"service": "light.turn_on",
           **({"area": c.last[0]["area"]} if "area" in c.last[0] else {}),
           "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "D'accord, j'ai " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'Elle est allumée maintenant ?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    area = c.last[-1].get("area")
    ents = [e for e in c.h.ents if e.eid in ids]
    if not ents or ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    g = group_gender(ents)
    pron = {"m": "Il", "f": "Elle", "mp": "Ils", "fp": "Elles"}[g]
    pairs = {"cover": [("ouvert", "ouvert"), ("fermé", "fermé"), ("baissé", "fermé"),
                       ("remonté", "ouvert")],
             "lock": [("verrouillé", "verrouillé"), ("ouvert", "pas verrouillé")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("allumé", "allumé"), ("éteint", "éteint")]))
    question = R.choice([f"{pron} {est(g)} {agr(word_q, g)} maintenant ?",
                         f"C'est {word_q} maintenant ?"])
    words = {_state_base(e) for e in ents}
    states = ", ".join(f"{e.name} {state_clause(e)}" for e in ents)
    if R.random() < 0.6 and len(words) == 1 and words <= {
            "allumé", "éteint", "ouvert", "fermé", "verrouillé", "pas verrouillé"}:
        c.user(question)
        word = words.pop()
        yes = "Oui" if word == asked else "Non"
        c.get_state(ents, f"{yes}, d'après Home Assistant : {states}.", area)
    else:
        c.user(R.choice(["Ça a marché ?", "C'est vraiment fait ?", "Tu peux vérifier ?"]))
        c.get_state(ents, f"D'après Home Assistant : {states}.", area)
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
    # "la télé de la chambre" while the only TV is in the living room: one
    # mixed case plus the TV case (one case per kind made the model refuse)
    cases += [("unknown:missing_type", s_missing_type, {"mtype": "other"}),
              ("unknown:missing_media_player", s_missing_type, {"mtype": "media_player"})]
    seen = set()
    for fn, _ in NEXT:
        if fn not in seen:
            seen.add(fn)
            cases.append((f"conversation:{fn.__name__}", s_conversation, {"conv_turn": fn}))
    # "Allume la lumière de la cuisine" -> "Non, je voulais dire dans la salle
    # à manger": undo every kitchen light, switch every dining room light
    cases.append(("conversation:room_correct", s_conversation,
                  {"conv_turn": turn_correct, "intent": i_light_room_phrase}))
    cases.append(("conversation:correct_recap", s_conversation,
                  {"conv_turn": turn_correct_recap, "intent": i_light_room_phrase}))
    for hook in CASE_HOOKS:
        cases += hook()
    return cases


SHORT_OK = ["C'est fait.", "D'accord.", "Fait.", "C'est bon."]
BARE = {"C'est fait.", "D'accord.", "Fait.", "C'est bon.", "En pause.", "Volume augmenté.",
        "Volume baissé.", "Titre suivant.", "C'est arrêté.", "Titre précédent."}
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
    short -> a word or two, long/detailed -> never a bare 'C'est fait.'."""
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
        if not all(x["service"].split(".", 1)[1] in DESCRIBED for x in acts):
            continue
        for x in acts:
            x.setdefault("entity_id", [])
        m["content"] = "J'ai " + describe_all(h, acts) + "."


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
