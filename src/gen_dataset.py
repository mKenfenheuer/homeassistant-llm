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
    ("Außenbereich", "im Außenbereich", "Außen", ["aussen", "outdoor"]),
    ("Abstellraum", "im Abstellraum", "Abstellraum", ["abstellraum", "storage"]),
    ("Heizungsraum", "im Heizungsraum", "Heizungsraum", ["heizungsraum", "boiler_room"]),
    ("Waschraum", "im Waschraum", "Waschraum", ["waschraum", "utility"]),
    ("Einfahrt", "in der Einfahrt", "Einfahrts", ["einfahrt", "driveway"]),
    ("Hauswirtschaftsraum", "im Hauswirtschaftsraum", "HWR", ["hwr", "utility_room"]),
    ("Werkstatt", "in der Werkstatt", "Werkstatt", ["werkstatt", "workshop"]),
    ("Carport", "im Carport", "Carport", ["carport"]),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
OUTDOOR = ("Garten", "Terrasse", "Balkon", "Außenbereich", "Einfahrt", "Carport")
UTILITY = ("Abstellraum", "Heizungsraum", "Waschraum", "Hauswirtschaftsraum",
           "Werkstatt", "Keller", "Dachboden", "Garage", "Waschküche")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garage",)]
NO_GROUP_COVERS = ("Markise", "Garagentor")

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

# brand-named devices: user says the generic noun, the name is the product
BRAND_MEDIA = [("LG webOS Smart TV", "Fernseher", "m"), ("Samsung TV", "Fernseher", "m"),
               ("Sony Bravia", "Fernseher", "m"), ("Philips Ambilight", "Fernseher", "m"),
               ("Sonos One", "Lautsprecher", "m"), ("Echo Dot", "Lautsprecher", "m"),
               ("HomePod", "Lautsprecher", "m"), ("Bose Soundbar", "Soundbar", "f")]
BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Mählanie", "Rasenroboter", "Gardena Sileno", "Worx Landroid"]
PLANTS = ["Tomaten", "Chili", "Basilikum", "Petersilie", "Erdbeeren", "Monstera",
          "Ficus", "Gurken", "Zitronenbaum"]
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "Max-PC", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Warmwasser Boost", "Hauswasserwerk", "Aquarium Pumpe"]
# appliances: a status sensor and a power switch with nearly the same name
# ("Trockner" + "Trockner Power"): commands go to the switch, questions to the
# sensor
APPLIANCES = [("Trockner", "m", "trockner"), ("Waschmaschine", "f", "waschmaschine"),
              ("Spülmaschine", "f", "spuelmaschine"),
              ("Geschirrspüler", "m", "geschirrspueler"),
              ("3D-Drucker", "m", "drucker_3d"), ("Wäschetrockner", "m", "dryer")]
# raw states as integrations report them (Home Connect, Miele, SmartThings, ...)
APP_RAW = {"running": ["running", "In Betrieb", "Läuft", "drying", "in_use", "run"],
           "finished": ["finished", "Programm beendet", "Ende", "end", "program_ended"],
           "off": ["off", "Aus"], "idle": ["idle", "Bereit", "standby", "programmed",
                                          "waiting_to_start"],
           "paused": ["pause", "paused", "Pausiert"],
           "offline": ["not_connected", "disconnected", "Offline"]}
APP_ANSWER = {"running": ("läuft noch", "läuft"), "finished": ("ist fertig", "fertig"),
              "off": ("ist aus", "aus"), "idle": ("ist bereit, läuft aber gerade nicht",
                                                  "läuft aber gerade nicht"),
              "paused": ("ist pausiert", "pausiert"),
              "offline": ("ist gerade nicht verbunden", "nicht verbunden")}
VACUUM_BUTTONS = [("Vollreinigung", "die Vollreinigung"), ("Absaugen", "das Absaugen"),
                  ("Mopp waschen", "das Mopp-Waschen")]
TOGGLES = ["Urlaub", "Gästemodus", "Partymodus", "Schlafmodus", "Putzmodus"]

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
    hidden: str = ""         # type label that the name no longer shows

    @property
    def plural(self) -> bool:
        return self.gender == "pl"


class House:
    def __init__(self, size: int):
        self.ents: list[Ent] = []
        self.used_ids: set[str] = set()
        self.used_names: set[str] = set()
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
        names = R.sample([r[0] for r in ROOMS if r[0] != "Badezimmer"], n_rooms)
        if "Bad" in names and R.random() < 0.4:
            names[names.index("Bad")] = "Badezimmer"
        self.rooms = names
        self.appliances: list[tuple[str, str, Ent, Ent]] = []
        self._populate(size)

    # ---- id / name helpers ----
    def _slug(self, room: str | None, type_slugs: list[str], domain: str) -> str:
        rslugs = ROOM_BY_NAME[room][3] if room else ["haus", "home"]
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
                # sample may call it "die Deckenlampe in der Küche"
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Links", "Rechts", "Mitte"], ["Decke", "Wand", "Ecke"],
                                      ["Insel", "Spots", "Theke"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        if f"{room} {side}" not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = f"{room} {side}"
                            lt.hidden, lt.label = lt.label, ""
                            self.used_names.add(lt.name)
            if not outdoor and room not in ("Flur", "Treppenhaus", "Gäste-WC") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                         state="closed" if pos == 0 else "open",
                         attrs={"current_position": pos})
            if room in ("Terrasse", "Balkon", "Außenbereich") and R.random() < 0.5:
                self.add("cover", room, "Markise", "f", ["markise", "awning"],
                         state=R.choice(["open", "closed"]),
                         attrs={"current_position": R.choice([0, 100])})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
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
                brand = None
                if R.random() < 0.5:
                    brand = R.choice(BRAND_MEDIA)
                    mt = (brand[1], brand[2], [brand[0].lower().replace(" ", "_")
                                               .replace("-", "_")])
                playing = R.random() < 0.4
                attrs = {"volume_level": R.choice([0.2, 0.3, 0.35, 0.5])}
                if playing:
                    attrs.update(media_title=R.choice(
                        ["Tagesschau", "Bohemian Rhapsody", "Atemlos",
                         "Tatort", "Die Maus", "Morning Jazz"]),
                        media_artist=R.choice(["", "Queen", "Helene Fischer", ""]))
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if brand and brand[0] not in self.used_names else None,
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
        if R.random() < 0.4:
            vt = R.choice(VACUUMS)
            vname = R.choice(BRAND_VACUUM) if R.random() < 0.6 else vt[0]
            self.add("vacuum", R.choice(self.rooms + [None, None]), vt[0], vt[1],
                     [vname.lower().replace(" ", "_")], name=vname,
                     state=R.choice(["docked", "docked", "cleaning", "paused"]))
        if R.random() < 0.25:
            mname = R.choice(MOWERS)
            room = "Garten" if "Garten" in self.rooms else None
            self.add("lawn_mower", room, "Mähroboter", "m",
                     [mname.lower().replace(" ", "_").replace("ä", "ae")], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = ["Bewässerung", "Wasserventil"]
            if room:
                names += [f"Bewässerung {room}", f"{room} Pflanzen", f"Pflanzen {room}"]
                if room == "Garten":
                    names.append("Gartenventil")
            self.add("valve", room, "Bewässerung", "f", ["bewaesserung", "irrigation"],
                     name=R.choice(names), state=R.choice(["open", "closed", "closed"]))
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
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(n.startswith(app) for n in self.used_names):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Küche", "Waschraum", "Hauswirtschaftsraum", "Bad", "Badezimmer",
                    "Keller", "Abstellraum", "Büro")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "Status", "m", [slug, slug + "_status"],
                                  name=R.choice([app, app, f"{app} Status"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind})
                power = self.add("switch", room, "Steckdose", "f",
                                 [slug + "_power", slug + "_steckdose"],
                                 name=R.choice([f"{app} Power", f"{app} Steckdose",
                                                f"{app} Strom"]),
                                 state="off" if kind == "off" and R.random() < 0.5 else "on")
                self.appliances.append((app, g, sensor, power))
        vac = [e for e in self.ents if e.domain == "vacuum"]
        if vac and R.random() < 0.45:
            v = vac[0]
            for label, _ in R.sample(VACUUM_BUTTONS, R.choice([1, 2])):
                self.add("button", v.area, label, "f",
                         [v.eid.split(".")[1] + "_" + label.lower().replace(" ", "_")],
                         name=f"{v.name} {label}", state="unknown")
        valve_rooms = {e.area for e in self.ents if e.domain == "valve"}
        if R.random() < (0.5 if valve_rooms else 0.15):
            free = [r for r in self.rooms if r in OUTDOOR and r not in valve_rooms]
            # many installs leave the irrigation button without an area; its
            # name ("Bewässerung Garten") then is the only hint where it waters
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Bewässerung Garten", "Gartenbewässerung"] if room == "Garten"
                         else [f"Bewässerung {room}"])
            elif "Garten" in valve_rooms:
                names = ["Bewässerung starten"]
            else:
                names = ["Bewässerung starten", "Bewässerung Garten", "Gartenbewässerung"]
            self.add("button", room, "Bewässerung", "f", ["bewaesserung_garten",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            e = self.add("light", None, "__group__", "pl", ["alle_lichter", "all_lights"],
                         name=R.choice(["Alle Lichter", "Alle Lampen", "Licht Haus"]),
                         state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            self.add("cover", None, "__group__", "pl", ["alle_rolladen", "all_covers"],
                     name=R.choice(["Alle Rolladen", "Alle Rollläden", "Rollos Haus"]),
                     state=R.choice(["open", "closed"]))
        # read-only distractors and odd names; fill up to the target size
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "Haus"
        if k == 0 and room:
            self.add("binary_sensor", room, "Anwesenheit", "f", ["presence"],
                     name=R.choice([f"Anwesenheit {room}", f"{room} Presence",
                                    f"Präsenz {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "Bewegung", "f", ["motion"],
                     name=R.choice([f"{rname} Motion", f"Bewegung {rname}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant = R.choice(PLANTS)
            self.add("sensor", room, "Bodenfeuchte", "f", ["plant_sensor", "moisture"],
                     name=f"{plant} Bodenfeuchte", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "Akku", "m", ["battery"],
                     name=R.choice(["USV Akku", "Auto Ladestand", "Rauchmelder Batterie",
                                    "Handy Akku", f"{rname} Sensor Batterie"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "Drucker", "m", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, unit = R.choice([("Windgeschwindigkeit", "km/h"), ("Windböen", "km/h"),
                                 ("Regenmenge", "mm/h"), ("Luftdruck", "hPa"),
                                 ("Wassertank Füllmenge", "L")])
            self.add("sensor", room, nm, "f", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": unit})
        elif k == 6:
            self.add("sensor", room, "Status", "m", ["status"],
                     name=R.choice(["USV Status", "Trockner", "Waschmaschine Status",
                                    "Spülmaschine Programm"]),
                     state=R.choice(["idle", "OL", "running", "off"]))
        elif k == 7:
            nm = R.choice(TOGGLES)
            if nm not in self.used_names:
                self.add("input_boolean", None, nm, "m", [nm.lower()], name=nm,
                         state=R.choice(["on", "off"]))
        elif k == 8:
            nm = R.choice(JUNK_SWITCHES)
            if nm not in self.used_names:
                self.add("switch", room, nm, "m", [nm.lower().replace(" ", "_")
                         .replace("-", "_")], name=nm, state=R.choice(["on", "off"]))
        else:
            self.add("sensor", room, "Temperatur", "f", ["temperature"],
                     name=R.choice([f"{rname} Temperatur", f"Temperatur {rname}",
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
            [(e.eid, e.name, e.area) for e in self.ents], location)

    def state_of(self, ents):
        # unreachable entities report no attribute values in Home Assistant
        return S.dumps([S.format_state(e.eid, e.name, e.state,
                                       {} if e.state == "unavailable" else e.attrs)
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
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 and e.name != e.label:
        opts += ["type_only"]           # "den Fernseher" for "LG webOS Smart TV"
    choice = R.choice(opts)
    if choice == "type_only":
        return Ref(f"{ART_ACC[e.gender]} {e.label}", f"{ART_NOM[e.gender]} {e.label}",
                   e.plural, [e])
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


def i_awning(h, loc):
    """Markise: ausfahren = open_cover, einfahren = close_cover."""
    ents = [e for e in h.of("cover") if e.label == "Markise"]
    if not ents:
        return None
    e = R.choice(ents)
    n = "die Markise" if len(ents) == 1 and R.random() < 0.7 else e.name
    if R.random() < 0.5:
        user = R.choice([f"Fahr {n} aus", f"{cap(n)} ausfahren", f"Mach {n} auf",
                         f"{cap(n)} raus", "Wir brauchen Schatten, fahr die Markise aus"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["Die Markise fährt aus.", "Erledigt, die Markise ist ausgefahren."])
    user = R.choice([f"Fahr {n} ein", f"{cap(n)} einfahren", f"Mach {n} zu",
                     f"{cap(n)} rein", "Es regnet, fahr die Markise ein"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["Die Markise fährt ein.", "Erledigt, die Markise ist eingefahren."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["den Mähroboter", "den Rasenmäher", e.name])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Starte {n}", f"Lass {n} mähen", "Rasen mähen bitte",
                         f"Schick {n} los"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{e.name} mäht jetzt."
    if r < 0.8:
        user = R.choice([f"Schick {n} in die Ladestation", f"{cap(n)} nach Hause",
                         f"Hol {n} rein"])
        return user, [action("lawn_mower.dock", [e])], f"{e.name} fährt zur Ladestation."
    user = R.choice([f"Pausiere {n}", f"Halte {n} an"])
    return user, [action("lawn_mower.pause", [e])], f"{e.name} ist pausiert."


AREA_PLANTS = {"Balkon": "die Balkonpflanzen", "Terrasse": "die Terrassenpflanzen",
          "Garten": "den Garten"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == "Bewässerung"]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Starte die Bewässerung", "Bewässere die Pflanzen"]
    if e.area:
        prep = ROOM_BY_NAME[e.area][1]
        out += [f"Starte die Bewässerung {prep}", f"Bewässere die Pflanzen {prep}",
                f"Gieß die Pflanzen {prep}"]
        if e.area in AREA_PLANTS:
            out.append(f"Bewässere {AREA_PLANTS[e.area]}")
    elif "garten" in e.name.lower():
        out += ["Bewässere den Garten", "Starte die Gartenbewässerung",
                "Gieß den Garten"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name] if several else [e.name, "die Bewässerung", "das Ventil"])
    if R.random() < 0.5:
        user = R.choice([f"Öffne {n}", f"Schalte {n} an", f"Mach {n} auf"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"{e.name} ist geöffnet."
    user = R.choice([f"Schließe {n}", f"Schalte {n} aus", f"Mach {n} zu",
                     "Stopp die Bewässerung"])
    return user, [action("valve.close_valve", [e])], f"{e.name} ist geschlossen."


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    if on:
        user = R.choice([f"Aktiviere {n}", f"{n} an", f"Schalte {n} ein",
                         f"Aktiviere den {n}smodus" if not n.endswith("modus") else f"Aktiviere den {n}"])
        return user, [action("input_boolean.turn_on", [e])], f"{n} ist aktiviert."
    user = R.choice([f"Deaktiviere {n}", f"{n} aus", f"Schalte {n} aus"])
    return user, [action("input_boolean.turn_off", [e])], f"{n} ist deaktiviert."


def i_appliance(h, loc):
    """'Schalte den Trockner aus' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = f"{ART_ACC[g]} {app}"
    if R.random() < 0.5:
        user = R.choice([f"Schalte {n} aus", f"Mach {n} aus", f"{app} aus",
                         f"Schalte {n} ab", f"Kannst du {n} ausschalten?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"{power.name} ist aus.", f"Ich habe {power.name} ausgeschaltet."])
    user = R.choice([f"Schalte {n} ein", f"Mach {n} an", f"{app} an",
                     f"Schalte {n} an"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"{power.name} ist an.", f"Ich habe {power.name} eingeschaltet."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        what = vac[e.label]
        user = R.choice([f"Starte {what}", f"Starte {what} vom Saugroboter",
                         f"Mach {what}", f"{e.name} starten"])
    elif e.label == "Bewässerung":
        user = R.choice(irrigation_phrases(h, e) + [f"Drück {e.name}"])
    else:
        user = R.choice([f"Drück {e.name}", f"Löse {e.name} aus"])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} ist ausgelöst.", f"Ich habe {e.name} gestartet.", "Erledigt."])


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Bewässere die Balkonpflanzen" -> balcony valve, not the garden button)."""
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
            [f"{e.name} ist geöffnet.", f"Ich habe {e.name} geöffnet."])
    return user, [action("button.press", [e])], R.choice(
        [f"{e.name} ist ausgelöst.", f"Ich habe {e.name} gestartet."])


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
    n = R.choice([f"{ART_ACC[e.gender]} {e.name}" if e.name == e.label else e.name,
                  "den Staubsauger", "den Saugroboter", "den Staubsauger"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Starte {n}", f"Lass {n} saugen", f"{cap(n)} starten",
                         "Staubsaugen bitte", f"Schick {n} los"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{e.name} legt los.", "Der Sauger startet."])
    if r < 0.8:
        user = R.choice([f"Schick {n} zur Ladestation", f"{cap(n)} zurück zur Basis",
                         f"Schick {n} nach Hause"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{e.name} fährt zurück zur Ladestation."
    user = R.choice([f"Pausiere {n}", f"{cap(n)} anhalten", f"Halte {n} an"])
    return user, [action("vacuum.pause", [e])], f"{e.name} ist pausiert."


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
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    # a group entity ("Alle Lichter") replaces the list of members
    if (grp := h.group(domain)) is not None:
        ents = [grp]
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
    (i_climate, 7), (i_lock, 3), (i_alarm, 3), (i_media, 5), (i_vacuum, 3),
    (i_scene, 4), (i_all, 6), (i_awning, 2), (i_mower, 2), (i_valve, 1),
    (i_toggle, 2), (i_appliance, 3), (i_button, 2), (i_irrigation, 12),
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


def join_commands(a, b):
    """Two command intents -> one sentence 'X und y', or None."""
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    if ub.startswith(("Kannst", "Würdest", "Ich ", "Bitte", "Es ist", "Hey", "Wir ")) or \
            ua.startswith(("Kannst", "Würdest", "Es ist")):
        return None
    # lowercase the leading verb of the second command ("... und mach ...")
    first = ub.split(" ", 1)
    if first[0] in ("Mach", "Schalte", "Schalt", "Stell", "Öffne", "Schließe", "Fahr",
                    "Dimme", "Starte", "Aktiviere", "Pausiere", "Spiel", "Setze",
                    "Setz", "Lass", "Färbe", "Schließ", "Sperre", "Verriegle",
                    "Entriegle", "Halte", "Stopp", "Stoppe", "Dreh", "Zieh", "Führe",
                    "Schick", "Deaktiviere", "Drück", "Löse", "Hol", "Bewässere"):
        ub = first[0].lower() + (" " + first[1] if len(first) > 1 else "")
    return f"{ua} und {ub}"


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
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
    if s == "unavailable":
        return "nicht erreichbar"
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
    if [e for e in h.of("sensor") if e.attrs.get("device_class") in ("moisture", "battery")]:
        kinds += ["generic"] * 2
    kind = R.choice(kinds)
    must = []
    if kind == "temp":
        temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"]
        if not temps:
            return None
        e = R.choice(temps)
        val = de_num(float(e.state))
        if e.label == "Außentemperatur":
            user = R.choice(["Wie warm ist es draußen?", "Wie kalt ist es draußen?",
                             "Wie ist die Außentemperatur?", "Wie viel Grad hat es draußen?"])
            ans = R.choice([f"Draußen sind es {val} Grad.", f"Die Außentemperatur liegt bei {val} Grad."])
        elif e.area is None or e.area in OUTDOOR or e.label != "Temperatur":
            # a sensor without area is not automatically outside, and a printer
            # bed or a freezer does not tell how warm the room is
            user = R.choice([f"Was zeigt {e.name}?", f"Wie viel Grad misst {e.name}?"])
            ans = f"{e.name} misst {val} Grad."
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
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = e.name.replace(" Bodenfeuchte", "")
            user = R.choice([f"Wie feucht ist die Erde bei den {plant}?",
                             f"Müssen die {plant} gegossen werden?",
                             f"Wie ist die Bodenfeuchte bei {plant}?"])
            ans = f"Die Bodenfeuchte bei {plant} liegt bei {e.state} Prozent."
        else:
            user = R.choice([f"Wie voll ist {e.name}?", f"Wie steht es um {e.name}?",
                             f"Was zeigt {e.name}?"])
            ans = f"{e.name} steht bei {e.state} Prozent."
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
            if "current_temperature" not in e.attrs:
                return None   # thermostat without its own sensor
            val = de_num(e.attrs["current_temperature"])
            ans = (f"{cap(prep)} sind es {val} Grad." if prep
                   else f"{cap(ref.nom)} misst {val} Grad.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        if e.domain == "cover":
            user, asked = R.choice([(f"Ist {ref.nom} offen?", "offen"),
                                    (f"Sind {ref.nom} unten?", "geschlossen")
                                    if ref.plural else
                                    (f"Ist {ref.nom} geschlossen?", "geschlossen"),
                                    (f"{sein(ref.plural).capitalize()} {ref.nom} zu?",
                                     "geschlossen")])
        else:
            user, asked = R.choice([(f"Ist {ref.nom} an?", "an"),
                                    (f"Ist {ref.nom} noch an?", "an"),
                                    (f"Ist {ref.nom} ausgeschaltet?", "aus"),
                                    (f"Ist {ref.nom} aus?", "aus")]) \
                if not ref.plural else R.choice([(f"Sind {ref.nom} an?", "an"),
                                                 (f"Sind {ref.nom} aus?", "aus")])
        word = _state_word(e)
        ans = f"{cap(ref.nom)} {sein(ref.plural)} {word}."
        if word in ("an", "aus", "offen", "geschlossen") and R.random() < 0.8:
            ans = ("Ja, " if word == asked else "Nein, ") + ref.nom + \
                f" {sein(ref.plural)} {word}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"an": "Ja, ", "aus": "Nein, "}[asked]
            ans = f"{yes}{ref.nom} {sein(ref.plural)} an, mit {pct} Prozent Helligkeit."
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
        if any(e.state == "unavailable" for e in ents):
            return None
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
        # only window sensors - presence and motion sensors are binary_sensors too
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") in ("window", "door")]
        if not ents:
            return None
        user, yes_if_open = R.choice([("Sind alle Fenster zu?", False),
                                      ("Ist noch ein Fenster offen?", True),
                                      ("Welche Fenster sind offen?", None),
                                      ("Sind die Fenster geschlossen?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "Nein, alle Fenster sind geschlossen.",
                   False: "Ja, alle Fenster sind geschlossen.",
                   None: "Alle Fenster sind geschlossen."}[yes_if_open]
            must = ["geschlossen"]
        else:
            names = [e.name for e in opn]
            ans = ("Offen ist: " + names[0] + "." if len(names) == 1 else
                   "Offen sind: " + ", ".join(names[:-1]) + " und " + names[-1] + ".")
            if yes_if_open is not None:
                ans = ("Ja. " if yes_if_open else "Nein. ") + ans
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


def s_appliance_question(h, loc):
    """'Ist der Trockner fertig?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = f"{ART_NOM[g]} {app}"
    user = R.choice([f"Ist {nom} fertig?", f"Läuft {nom} noch?", f"Was macht {nom}?",
                     f"Ist {nom} schon durch?", f"Wie ist der Status von {ART_DAT[g]} {app}?"])
    phrase, key = APP_ANSWER[kind]
    ans = R.choice([f"{cap(nom)} {phrase}.", f"Laut Home Assistant {phrase.split()[0]} "
                    f"{nom} {' '.join(phrase.split()[1:])}.".replace("  ", " ")])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [key]}


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


MISSING_TYPES = [
    # domain, phrase, "kein ...", rel. pronoun, verb on, verb off, "ein ...",
    # services (on, off), state words (on, off), label filter
    ("light", "das Licht", "kein Licht", "das", "an", "aus", "Licht",
     ("light.turn_on", "light.turn_off"), ("an", "aus"), None),
    ("light", "die Lampe", "keine Lampe", "die", "an", "aus", "Eine Lampe",
     ("light.turn_on", "light.turn_off"), ("an", "aus"), None),
    ("cover", "die Rollläden", "keine Rollläden", "die", "hoch", "runter", "Rollläden",
     ("cover.open_cover", "cover.close_cover"), ("offen", "geschlossen"), "plain"),
    ("climate", "die Heizung", "keine Heizung", "die", "an", "aus", "Eine Heizung",
     ("climate.turn_on", "climate.turn_off"), ("an", "aus"), None),
    ("media_player", "den Fernseher", "keinen Fernseher", "den", "an", "aus",
     "Einen Fernseher", ("media_player.turn_on", "media_player.turn_off"), ("an", "aus"),
     "Fernseher"),
    ("fan", "den Ventilator", "keinen Ventilator", "den", "an", "aus", "Einen Ventilator",
     ("fan.turn_on", "fan.turn_off"), ("an", "aus"), None),
]


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    dom, phrase, kein, rel, on, off, ein, svcs, words, filt = R.choice(MISSING_TYPES)

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
    turn_on = R.random() < 0.5
    verb = on if turn_on else off
    user = R.choice([f"Mach {phrase} {prep} {verb}",
                     f"Schalte {phrase} {prep} {'ein' if turn_on else 'aus'}",
                     f"{cap(phrase.split()[1])} {prep} {verb}", f"Mach {prep} {phrase} {verb}"])
    others = [e for e in ents if e.area and e.area != room]
    # suggest the other device only if there are one or two candidates
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        ans = R.choice([f"{cap(prep)} gibt es {kein}. {ein} gibt es {oprep}: {alt.name}.",
                        f"{cap(prep)} finde ich {kein}. Meinst du {alt.name}{where}?",
                        f"{cap(prep)} habe ich {kein} in der Geräteliste. {oprep[0].upper() + oprep[1:]} gibt es {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"{cap(prep)} finde ich {kein}, {rel} ich steuern kann.",
                        f"{cap(prep)} gibt es {kein} in der Geräteliste.",
                        f"Ich habe {prep} {kein} gefunden, {rel} ich schalten kann."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        # user accepts the device in the other room -> now it is switched
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if alt.area in alt.name else f" {oprep}"
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Dann {ART_ACC[alt.gender]} {oprep}", "Ja, den meine ich", "Okay, dann den",
             f"Ja, {alt.name}", f"Dann eben {oprep}"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        word = words[0] if turn_on else words[1]
        msgs += [call, tool_msg(cid, S.control_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"{alt.name}{where} ist jetzt {word}.",
                      f"Erledigt, {alt.name}{where} ist {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        # user insists on the room -> still no invented action
        msgs += [{"role": "user", "content": voice(R.choice(
                     ["Doch, mach es an!", "Ja, schalte es an!", "Nee, das gibt es doch",
                      "Mach es trotzdem", f"Doch, {prep}!", f"Nein, ich meine {prep}"]))},
                 {"role": "assistant", "content": R.choice([
                     f"{cap(prep)} gibt es in deiner Geräteliste {kein}. Prüfe bitte, ob das Gerät für Assist freigegeben ist.",
                     f"Ich sehe {prep} wirklich {kein}, {rel} ich steuern kann."])}]
        cat = "unknown"
    else:
        cat = "unknown"
    meta = {"cat": cat, "gold": gold}
    if cat == "followup":
        meta["gold_last"] = gold
    return msgs, meta


def s_unknown(h, loc):
    """User asks for a device or room that does not exist -> no tool call."""
    if R.random() < 0.45:
        return s_missing_type(h, loc)
    r = R.random()
    if r < 0.5:
        dev, g = R.choice(UNKNOWN_DEVICES)
        labels = " ".join(f"{e.name} {e.label} {e.hidden}" for e in h.ents).lower()
        if dev.lower() in labels or (dev == "Rasenmäher" and h.of("lawn_mower")):
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


def _unreachable(name):
    return R.choice([f"Das hat leider nicht funktioniert. {name} reagiert gerade nicht.",
                     f"Leider ist ein Fehler aufgetreten: {name} ist nicht erreichbar.",
                     f"{name} ist gerade nicht erreichbar, deshalb konnte ich den Befehl nicht ausführen."])


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
            ["Versuch es nochmal", "Nochmal bitte", "Probier es noch einmal",
             "Mach es nochmal"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": R.choice([
                     f"{name} ist immer noch nicht erreichbar. Prüfe bitte, ob das Gerät Strom und Verbindung hat.",
                     f"Auch der zweite Versuch hat nicht funktioniert, {name} reagiert nicht."])}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        msgs.append({"role": "user", "content": voice(R.choice(
            ["Ist es jetzt an?", "Nee, ist es nicht?", "Hat es geklappt?", "Und, ist es an?"]
            if onoff else ["Hat es geklappt?", "Und, hat es funktioniert?", "Stimmt das?"]))})
        e = next(x for x in h.ents if x.eid == eid)
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": R.choice([
                     f"Nein. {name} ist nicht erreichbar, deshalb konnte ich es nicht schalten.",
                     f"Laut Home Assistant ist {name} gerade nicht erreichbar. Geschaltet wurde nichts."])}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def partial_answer(h, ok_actions, failed_name, conf=None):
    """Both outcomes, always: what worked and what did not."""
    if any(len(a["entity_id"]) > 3 for a in ok_actions):
        # long lists: the normal confirmation ("Alle Lichter sind an.") is clearer
        if conf and conf != "Erledigt.":
            return R.choice([f"{conf} {failed_name} ist aber gerade nicht erreichbar.",
                             f"{failed_name} ist gerade nicht erreichbar. Sonst gilt: {conf}"])
        n = sum(len(a["entity_id"]) for a in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = f"{n} Geräte {PARTICIPLE.get(svc, 'geschaltet')}"
    else:
        did = " und ".join(describe(h, a) for a in ok_actions)
    return R.choice([
        f"Ich habe {did}. {failed_name} ist aber gerade nicht erreichbar.",
        f"Das hat nur teilweise geklappt: Ich habe {did}, aber {failed_name} ist gerade nicht erreichbar.",
        f"{failed_name} ist gerade nicht erreichbar. Den Rest habe ich erledigt: {did}.",
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
    user = R.choice([f"Ist {e.name} an?", f"Wie ist der Status von {e.name}?",
                     f"Was macht {e.name}?", f"Ist {e.name} eingeschaltet?"])
    args = {"entity_ids": [e.eid]}
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": R.choice([
                 f"{e.name} ist gerade nicht erreichbar, deshalb kenne ich den aktuellen Zustand nicht.",
                 f"Das kann ich nicht sagen: {e.name} ist gerade nicht erreichbar."])}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


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
    if k < 0.15:
        # user doubts the result -> check the real state instead of guessing
        u2 = R.choice(["Nee, ist es nicht", "Ist es wirklich an?", "Stimmt das?",
                       "Bei mir ist es noch dunkel"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        c2, cid2 = tool_call("get_state", {"entity_ids": [e.eid for e in e1]})
        ans = (f"Laut Home Assistant ist das Licht {p1} an." if really_on else
               f"Laut Home Assistant ist das Licht {p1} noch aus. Soll ich es erneut einschalten?")
        if u2 == "Ist es wirklich an?":
            ans = ("Ja. " if really_on else "Nein. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", {"entity_ids": [e.eid for e in e1]}]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["Und wieder aus", "Mach es wieder aus", "Doch wieder aus"]), \
            [action("light.turn_off", e1)], f"Das Licht {p1} ist wieder aus."
    elif k < 0.72:
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


# --------------------------------------------------------------------------
# multi-turn conversations: later turns refer to earlier ones
# --------------------------------------------------------------------------
INVERSE = {"turn_on": "turn_off", "turn_off": "turn_on", "open_cover": "close_cover",
           "close_cover": "open_cover", "lock": "unlock", "unlock": "lock",
           "media_play": "media_pause", "media_pause": "media_play",
           "open_valve": "close_valve", "close_valve": "open_valve",
           "start": "return_to_base", "start_mowing": "dock"}
PARTICIPLE = {"turn_on": "eingeschaltet", "turn_off": "ausgeschaltet",
              "open_cover": "geöffnet", "close_cover": "geschlossen",
              "stop_cover": "angehalten", "lock": "abgeschlossen", "unlock": "aufgeschlossen",
              "media_play": "fortgesetzt", "media_pause": "pausiert", "media_stop": "gestoppt",
              "media_next_track": "zum nächsten Titel geschaltet",
              "media_previous_track": "zum vorherigen Titel geschaltet",
              "volume_up": "lauter gestellt", "volume_down": "leiser gestellt",
              "start": "gestartet", "pause": "pausiert", "return_to_base": "zur Ladestation geschickt",
              "start_mowing": "gestartet", "dock": "zur Ladestation geschickt",
              "open_valve": "geöffnet", "close_valve": "geschlossen", "press": "ausgelöst",
              "alarm_arm_home": "scharf geschaltet", "alarm_arm_away": "scharf geschaltet",
              "alarm_arm_night": "im Nachtmodus scharf geschaltet",
              "alarm_disarm": "unscharf geschaltet"}
DOMAIN_NOUN = {"light": ("Welches Licht", "Welche Lampen"),
               "cover": ("Welchen Rollladen", "Welche Rollläden"),
               "climate": ("Welche Heizung", "Welche Heizungen"),
               "switch": ("Welches Gerät", "Welche Geräte"),
               "media_player": ("Welchen Lautsprecher", "Welche Geräte"),
               "fan": ("Welchen Ventilator", "Welche Ventilatoren")}


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
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " und " + ns[-1]


def describe(h, a):
    """'Flur Deckenlicht eingeschaltet' for one action."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    who = _names(h, a["entity_id"])
    if "brightness_pct" in data:
        return f"{who} auf {data['brightness_pct']} Prozent gestellt"
    if "brightness_step_pct" in data:
        return f"{who} {'heller' if data['brightness_step_pct'] > 0 else 'dunkler'} gemacht"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"bei {who} die Farbe geändert"
    if svc == "set_temperature":
        return f"{who} auf {de_num(data['temperature'])} Grad gestellt"
    if svc == "set_cover_position":
        return f"{who} auf {data['position']} Prozent gefahren"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"{who} auf {data.get('percentage')} Prozent gestellt"
    if svc == "volume_set":
        return f"bei {who} die Lautstärke auf {round(data['volume_level'] * 100)} Prozent gestellt"
    if dom == "scene":
        return f"die Szene {who} aktiviert"
    if dom == "script":
        return f"{who} gestartet"
    return f"{who} {PARTICIPLE.get(svc, 'geschaltet')}"


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
        self.msgs += [call, tool_msg(cid, S.control_result(errors, done)),
                      {"role": "assistant", "content": answer}]
        ok = [a for a in actions if not errors or
              all(i in (done or []) for i in a["entity_id"])]
        apply_actions(self.h, ok)
        self.done += ok
        self.last = ok or None
        self.gold_last = [["control", {"actions": actions}]]

    def get_state(self, ents, answer):
        args = {"entity_ids": [e.eid for e in ents]}
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
    """'Welches Licht hast du ausgeschaltet?' -> answered from the history."""
    if not c.last:
        return False
    a = c.last[-1]
    dom, svc = a["service"].split(".", 1)
    ids = a["entity_id"]
    if dom in DOMAIN_NOUN and svc in PARTICIPLE and not a.get("data"):
        sg, pl = DOMAIN_NOUN[dom]
        q = (pl if len(ids) > 1 else sg) + f" hast du {PARTICIPLE[svc]}?"
        user = R.choice([q, q, "Was hast du gerade gemacht?", "Welches Gerät war das?"])
    else:
        user = R.choice(["Was hast du gerade gemacht?", "Was genau hast du gemacht?"])
    c.user(user)
    parts = [describe(c.h, x) for x in c.last]
    c.say(R.choice(["Ich habe ", "Gerade habe ich "]) + " und ".join(parts) + ".")
    return True


def turn_summary(c):
    """'Was hast du alles gemacht?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    c.user(R.choice(["Was hast du alles gemacht?", "Was hast du bisher geschaltet?",
                     "Fass mal zusammen, was du gemacht hast"]))
    parts = [describe(c.h, x) for x in c.done]
    c.say("Ich habe " + ", ".join(parts[:-1]) + " und " + parts[-1] + ".")
    return True


def turn_undo(c):
    if not c.last:
        return False
    inv = []
    for a in c.last:
        dom, svc = a["service"].split(".", 1)
        if svc not in INVERSE or a.get("data"):
            return False
        inv.append({"service": f"{dom}.{INVERSE[svc]}", "entity_id": a["entity_id"]})
    c.user(R.choice(["Mach das rückgängig", "Doch nicht", "Und wieder zurück",
                     "Nimm das zurück"] +
                    (["Und wieder aus", "Mach es wieder aus"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    (["Und wieder an"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "Okay, ich habe " + " und ".join(describe(c.h, x) for x in inv) + ".")
    return True


def turn_correct(c):
    """'Nein, ich meinte das im Schlafzimmer' -> undo + the right device."""
    if not c.last or len(c.last) != 1:
        return False
    a = c.last[0]
    dom, svc = a["service"].split(".", 1)
    if svc not in ("turn_on", "turn_off", "open_cover", "close_cover") or a.get("data") \
            or dom not in ("light", "cover", "switch"):
        return False
    cur = next(e for e in c.h.ents if e.eid == a["entity_id"][0])
    others = [e for e in c.h.of(dom, plain=dom == "cover")
              if e.area and e.area != cur.area and e.state != "unavailable"]
    if not others:
        return False
    o = R.choice(others)
    prep = ROOM_BY_NAME[o.area][1]
    c.user(R.choice([f"Nein, ich meinte {prep}", f"Falsch, {prep}",
                     f"Nicht das, sondern {o.name}", f"Ich meinte {o.name}"]))
    acts = [{"service": f"{dom}.{INVERSE[svc]}", "entity_id": [cur.eid]},
            {"service": a["service"], "entity_id": [o.eid]}]
    c.control(acts, f"Okay, ich habe {describe(c.h, acts[0])} und {describe(c.h, acts[1])}.")
    return True


def turn_also(c):
    """'Im Bad auch' -> same service on the same device type in another room."""
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
    c.user(R.choice([f"Und {prep} auch", f"{cap(prep)} auch", f"Das gleiche {prep}"]))
    act = {"service": a["service"], "entity_id": [e.eid for e in ents]}
    if a.get("data"):
        act["data"] = a["data"]
    c.control([act], R.choice(["Okay, ich habe ", "Erledigt, ich habe "])
              + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["Etwas heller", "Noch heller", "Mach es heller"]), {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["Etwas dunkler", "Noch dunkler", "Mach es dunkler"]), {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = f"Auf {p} Prozent", {"brightness_pct": p}
    act = {"service": "light.turn_on", "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "Okay, ich habe " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'Ist es jetzt an?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    ents = [e for e in c.h.ents if e.eid in ids]
    if ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    pron = "sie" if len(ents) > 1 or ents[0].plural else \
        {"m": "er", "f": "sie", "n": "es"}[ents[0].gender]
    verb = "Sind" if pron == "sie" and (len(ents) > 1 or ents[0].plural) else "Ist"
    pairs = {"cover": [("offen", "offen"), ("zu", "geschlossen"), ("unten", "geschlossen"),
                       ("oben", "offen")],
             "lock": [("abgeschlossen", "abgeschlossen"), ("offen", "nicht abgeschlossen")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("an", "an"), ("aus", "aus")]))
    question = f"{verb} {pron} jetzt {word_q}?"
    words = {_state_word(e) for e in ents}
    states = ", ".join(f"{e.name} ist {_state_word(e)}" for e in ents)
    if R.random() < 0.6 and len(words) == 1 and words <= {
            "an", "aus", "offen", "geschlossen", "abgeschlossen", "nicht abgeschlossen"}:
        c.user(question)
        word = words.pop()
        yes = "Ja" if word == asked else "Nein"
        c.get_state(ents, f"{yes}, laut Home Assistant: {states}.")
    else:
        c.user(R.choice(["Hat das geklappt?", "Stimmt das wirklich?", "Kannst du das prüfen?"]))
        c.get_state(ents, f"Laut Home Assistant: {states}.")
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
    for _ in range(n * 6):
        fn = R.choices([p[0] for p in pool], [p[1] for p in pool])[0]
        if fn(c):
            turns += 1
            pool = NEXT
            if turns >= n:
                break
    if turns < 3:
        return None
    return c.msgs, {"cat": "conversation", "gold": [], "gold_last": c.gold_last}


CATEGORIES = [
    (s_command, 46), (s_compound, 8), (s_question, 24), (s_relative_climate, 2),
    (s_unknown, 9), (s_chat, 3), (s_error, 5), (s_followup, 8),
    (s_missing_type, 6), (s_appliance_question, 3), (s_conversation, 12),
]


def history_prefix(h, loc):
    """An earlier exchange in the same conversation (small talk or a command)."""
    if R.random() < 0.4:
        users, answers = R.choice(CHAT)
        return [{"role": "user", "content": voice(R.choice(users))},
                {"role": "assistant", "content": R.choice(answers)}]
    out = s_command(h, loc)
    return out[0][1:] if out else []


def make_sample():
    size = R.choice([R.randint(8, 18), R.randint(18, 35), R.randint(25, 45),
                     R.randint(35, 60), R.randint(60, 110)])
    h = House(size)
    loc = R.choice(h.rooms) if R.random() < 0.25 else None
    for _ in range(20):
        fn = R.choices([c[0] for c in CATEGORIES], [c[1] for c in CATEGORIES])[0]
        out = fn(h, loc)
        if out:
            msgs, meta = out
            if meta["cat"] not in ("followup", "conversation") and R.random() < 0.2:
                msgs = msgs[:1] + history_prefix(h, loc) + msgs[1:]
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
        last_user = [m["content"] for m in s["messages"] if m["role"] == "user"][-1]
        key = last_user + str(s["meta"].get("gold"))
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
