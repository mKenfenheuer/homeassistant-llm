"""Generate the English SFT dataset for Home Assistant tool calling.

Port of src/gen/de/core.py: same cases, categories, tool calls and gold
metadata; only the language differs. Every sample uses a freshly randomised
house (rooms, device names, entity_ids, states), so the model learns to map
names from the device list to entity_ids instead of memorising one house.
Prompt and tool schema come from custom_components/ha_llm_agent/llm_schema.py.

Each JSONL row: {"messages": [...], "meta": {"cat": ..., "gold": [...],
"answer_contains": [...]}}. "meta" is ignored by training and used by
src/eval_model.py.

This module holds the house model and the device intents; src/gen/en/assist.py
adds the Assist built-ins (music, timers, lists, weather, time, broadcasts)
through POPULATE_HOOKS, COMMANDS and CASE_HOOKS.

Usage: python src/gen_dataset.py --lang en OUT.jsonl N [SEED]
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

LANG = "en"
# answer length of the current sample (llm_schema.VERBOSITY), set in make_sample
VERBOSITY = "medium"
VERBOSITY_WEIGHTS = [("short", 2), ("medium", 4), ("long", 2), ("detailed", 2)]
R = random.Random()
# extensions (assist.py): fn(house) after populating, fn() -> extra cases
POPULATE_HOOKS: list = []
CASE_HOOKS: list = []
# set by the case loop in main(): forces one intent / question kind / turn type
FORCE: dict = {}

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
# area name, preposition phrase, (unused compound prefix), slugs
ROOMS = [
    ("Living Room", "in the living room", "Living Room", ["living_room", "lounge", "lr"]),
    ("Kitchen", "in the kitchen", "Kitchen", ["kitchen", "kit"]),
    ("Bedroom", "in the bedroom", "Bedroom", ["bedroom", "master_bedroom", "br"]),
    ("Bathroom", "in the bathroom", "Bathroom", ["bathroom", "bath"]),
    ("Main Bathroom", "in the main bathroom", "Main Bathroom", ["main_bathroom", "bathroom"]),
    ("Kids Room", "in the kids' room", "Kids Room", ["kids_room", "nursery"]),
    ("Office", "in the office", "Office", ["office", "home_office"]),
    ("Study", "in the study", "Study", ["study", "den"]),
    ("Hallway", "in the hallway", "Hallway", ["hallway", "hall", "corridor"]),
    ("Dining Room", "in the dining room", "Dining Room", ["dining_room", "dining"]),
    ("Guest Room", "in the guest room", "Guest Room", ["guest_room", "spare_room"]),
    ("Basement", "in the basement", "Basement", ["basement", "cellar"]),
    ("Garage", "in the garage", "Garage", ["garage"]),
    ("Garden", "in the garden", "Garden", ["garden", "backyard"]),
    ("Patio", "on the patio", "Patio", ["patio", "terrace"]),
    ("Balcony", "on the balcony", "Balcony", ["balcony"]),
    ("Attic", "in the attic", "Attic", ["attic", "loft"]),
    ("Laundry Room", "in the laundry room", "Laundry Room", ["laundry_room", "laundry"]),
    ("Guest Toilet", "in the guest toilet", "Guest Toilet", ["guest_toilet", "wc"]),
    ("Dressing Room", "in the dressing room", "Dressing Room", ["dressing_room", "closet"]),
    ("Entrance", "in the entrance", "Entrance", ["entrance", "entryway"]),
    ("Stairs", "on the stairs", "Stairs", ["stairs", "staircase"]),
    ("Outdoor", "outside", "Outdoor", ["outdoor", "outside"]),
    ("Storage Room", "in the storage room", "Storage Room", ["storage_room", "storage"]),
    ("Boiler Room", "in the boiler room", "Boiler Room", ["boiler_room", "plant_room"]),
    ("Mudroom", "in the mudroom", "Mudroom", ["mudroom", "boot_room"]),
    ("Driveway", "on the driveway", "Driveway", ["driveway", "drive"]),
    ("Utility Room", "in the utility room", "Utility Room", ["utility_room", "utility"]),
    ("Workshop", "in the workshop", "Workshop", ["workshop", "shed"]),
    ("Carport", "in the carport", "Carport", ["carport"]),
]
ROOM_BY_NAME = {r[0]: r for r in ROOMS}
OUTDOOR = ("Garden", "Patio", "Balcony", "Outdoor", "Driveway", "Carport")
UTILITY = ("Storage Room", "Boiler Room", "Mudroom", "Utility Room",
           "Workshop", "Basement", "Attic", "Garage", "Laundry Room")
INDOOR = [r[0] for r in ROOMS if r[0] not in OUTDOOR + ("Garage",)]
NO_GROUP_COVERS = ("Awning", "Garage Door")
# rooms that often have a window and a balcony or patio door with covers
TWO_COVER_ROOMS = ("Living Room", "Dining Room", "Kitchen", "Bedroom", "Study",
                   "Kids Room", "Office")

# device type: (label, number s/pl, slugs, compound-able)
TYPES = {
    "light": [
        ("Ceiling Light", "s", ["ceiling_light", "ceiling"], False),
        ("Light", "s", ["light", "lights"], True),
        ("Ceiling Lamp", "s", ["ceiling_lamp", "ceiling"], False),
        ("Lamp", "s", ["lamp", "lamp_1"], True),
        ("Floor Lamp", "s", ["floor_lamp", "standing_lamp"], False),
        ("Reading Lamp", "s", ["reading_lamp", "reading_light"], False),
        ("LED Strip", "s", ["led_strip", "lightstrip"], False),
        ("Spotlights", "pl", ["spotlights", "spots"], False),
        ("Wall Light", "s", ["wall_light", "sconce"], False),
        ("Fairy Lights", "pl", ["fairy_lights", "string_lights"], False),
        ("Pendant Light", "s", ["pendant_light", "pendant"], False),
        ("Bedside Lamp", "s", ["bedside_lamp", "nightstand"], False),
        ("Mirror Light", "s", ["mirror_light", "mirror"], False),
    ],
    "switch": [
        ("Plug", "s", ["plug", "smart_plug", "socket"], False),
        ("Coffee Machine", "s", ["coffee_machine", "coffee"], False),
        ("Space Heater", "s", ["space_heater", "heater"], False),
        ("Dehumidifier", "s", ["dehumidifier", "dehumidifier_plug"], False),
        ("Pond Pump", "s", ["pond_pump", "pump"], False),
        ("Irrigation", "s", ["irrigation", "sprinkler"], False),
        ("Christmas Tree", "s", ["christmas_tree", "xmas_tree"], False),
        ("Printer", "s", ["printer", "printer_plug"], False),
        ("Kettle", "s", ["kettle", "kettle_plug"], False),
        ("Monitor", "s", ["monitor", "screen"], False),
    ],
    "cover": [
        ("Shutter", "s", ["shutter", "roller_shutter"], True),
        ("Shutters", "pl", ["shutters", "roller_shutters"], False),
        ("Blinds", "pl", ["blinds", "venetian_blinds"], False),
        ("Roller Blind", "s", ["roller_blind", "blind"], False),
        ("Awning", "s", ["awning", "sun_awning"], False),
    ],
    "climate": [
        ("Heating", "s", ["heating", "thermostat", "radiator"], True),
        ("Thermostat", "s", ["thermostat", "trv"], False),
        ("Air Conditioner", "s", ["air_conditioner", "ac"], False),
        ("Underfloor Heating", "s", ["underfloor_heating", "floor_heating"], False),
    ],
    "fan": [
        ("Fan", "s", ["fan", "desk_fan"], False),
        ("Ceiling Fan", "s", ["ceiling_fan", "fan"], False),
        ("Ventilation", "s", ["ventilation", "extractor"], False),
    ],
    "media_player": [
        ("TV", "s", ["tv", "television"], False),
        ("Speaker", "s", ["speaker", "sonos"], False),
        ("Soundbar", "s", ["soundbar"], False),
        ("Radio", "s", ["radio"], False),
    ],
}

LOCKS = [("Front Door", "s", ["front_door", "front_door_lock"]),
         ("Basement Door", "s", ["basement_door", "cellar_door"]),
         ("Patio Door", "s", ["patio_door", "patio_door_lock"]),
         ("Garden Gate", "s", ["garden_gate", "gate"]),
         ("Back Door", "s", ["back_door", "back_door_lock"])]
VACUUMS = [("Robot Vacuum", "s", ["robot_vacuum", "roborock", "robot"]),
           ("Vacuum", "s", ["vacuum", "vacuum_cleaner"])]
SCENES = ["Movie Night", "Good Night", "Cooking", "Relax", "Party", "Reading",
          "Wake Up", "Dinner", "Romantic", "Bright", "Focus"]
SCRIPTS = ["All Off", "Vacation Mode", "Bedtime Routine", "Make Coffee",
           "Leaving Home", "Arriving Home"]
BUTTONS = [("Garage Door", "s", ["garage_door_button", "garage_opener"]),
           ("Doorbell", "s", ["doorbell", "doorbell_button"])]

COLORS = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
          "orange": "orange", "purple": "purple", "pink": "pink", "white": "white",
          "turquoise": "turquoise", "violet": "violet"}
COLOR_TEMPS = {"warm white": 2700, "cool white": 6000, "neutral white": 4000}

# brand-named devices: user says the generic noun, the name is the product
BRAND_MEDIA = [("LG webOS Smart TV", "TV", "s"), ("Samsung TV", "TV", "s"),
               ("Sony Bravia", "TV", "s"), ("Philips Ambilight", "TV", "s"),
               ("Sonos One", "Speaker", "s"), ("Echo Dot", "Speaker", "s"),
               ("HomePod", "Speaker", "s"), ("Bose Soundbar", "Soundbar", "s")]


# voice satellites expose a media_player that is no TV
def voice_media_name():
    hx = "".join(R.choice("0123456789abcdef") for _ in range(6))
    return R.choice([f"Home Assistant Voice {hx} Media Player", f"Home Assistant Voice {hx[:6]}",
                     f"ESPHome Voice {hx[:4]} Media Player", "Echo Show", "Nest Mini",
                     f"Atom Echo {hx[:6]} Media Player"])


BRAND_VACUUM = ["S8 MaxV Ultra", "Roborock S7", "Dreame L10s", "Roomba i7",
                "Deebot X1", "Valetudo"]
MOWERS = ["Automower", "Robot Mower", "Lawn Robot", "Gardena Sileno", "Worx Landroid"]
PLANTS = ["Tomatoes", "Chili", "Basil", "Parsley", "Strawberries", "Monstera",
          "Ficus", "Cucumbers", "Lemon Tree"]
PLURAL_PLANTS = {"Tomatoes", "Strawberries", "Cucumbers"}
JUNK_SWITCHES = ["Gosund Sp1 03 Relay", "ACE 2 Drying", "Gaming PC", "Sonoff Basic 4",
                 "Shelly Plug S 2", "Enable guest Wi-Fi", "Tasmota Relay 1",
                 "Hot Water Boost", "Well Pump", "Aquarium Pump"]
# appliances: a status sensor and a power switch with nearly the same name
# ("Dryer" + "Dryer Power"): commands go to the switch, questions to the sensor
APPLIANCES = [("Dryer", "s", "dryer"), ("Washing Machine", "s", "washing_machine"),
              ("Dishwasher", "s", "dishwasher"),
              ("Washer", "s", "washer"),
              ("3D Printer", "s", "printer_3d"), ("Tumble Dryer", "s", "tumble_dryer")]
# raw states as integrations report them (Home Connect, Miele, SmartThings, ...)
APP_RAW = {"running": ["running", "Running", "In use", "drying", "in_use", "run"],
           "finished": ["finished", "Program finished", "End", "end", "program_ended"],
           "off": ["off", "Off"], "idle": ["idle", "Ready", "standby", "programmed",
                                         "waiting_to_start"],
           "paused": ["pause", "paused", "Paused"],
           "offline": ["not_connected", "disconnected", "Offline"]}
APP_ANSWER = {"running": ("is still running", "running"), "finished": ("is finished", "finished"),
              "off": ("is off", "off"), "idle": ("is ready, but it isn't running right now",
                                                 "isn't running"),
              "paused": ("is paused", "paused"),
              "offline": ("isn't connected right now", "connected")}
# (button label, how the user asks for it)
VACUUM_BUTTONS = [("Full Clean", ["Start a full clean", "Start a full clean with the robot vacuum",
                                  "Run a full clean", "Do a full clean"]),
                  ("Empty Bin", ["Empty the vacuum's dust bin", "Empty the dust bin",
                                 "Have the robot vacuum empty its bin"]),
                  ("Wash Mop", ["Wash the mop", "Start mop washing",
                                "Have the robot vacuum wash its mop"])]
TOGGLES = ["Vacation", "Guest Mode", "Party Mode", "Sleep Mode", "Cleaning Mode"]

UNKNOWN_DEVICES = [("Sauna", "s"), ("Hot Tub", "s"), ("Pool Pump", "s"),
                   ("Fireplace", "s"), ("Aquarium", "s"), ("Lawn Mower", "s"),
                   ("Fountain", "s"), ("Tumble Dryer", "s"),
                   ("Dishwasher", "s"), ("Oven", "s"), ("AC Unit", "s"),
                   ("Garage Door", "s"), ("Gate Lights", "pl")]


def en_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}"


de_num = en_num   # name used by shared helper code


def cap(s: str) -> str:
    if s[:1] in "\"“":
        return s[:1] + s[1:2].upper() + s[2:]
    return s[:1].upper() + s[1:]


def low(label: str) -> str:
    """'LED Strip' -> 'LED strip': lowercase, acronyms and model names stay."""
    return " ".join(w if (len(w) > 1 and w.isupper()) or w[:1].isdigit() or w in PROPER
                    else w.lower() for w in label.split(" "))


PROPER = {"Christmas", "Hue", "Sonos", "Echo", "HomePod", "Nest", "Govee", "Nanoleaf"}


def be(pl) -> str:
    return "are" if pl else "is"


PLURAL_ENDINGS = ("lights", "spotlights", "spots", "shutters", "blinds", "sprinklers",
                  "lamps", "covers")


def pl_name(name: str) -> bool:
    """'Office Fairy Lights', 'All Lights' take a plural verb."""
    return name.lower().endswith(PLURAL_ENDINGS)


def is_n(name: str) -> str:
    return "are" if pl_name(name) else "is"


def isnt(name: str) -> str:
    return "aren't" if pl_name(name) else "isn't"


def a_an(word: str) -> str:
    """'an oven timer', 'an 8-minute timer', 'a one-hour timer'."""
    w = word.lower()
    if w.startswith(("one", "uni", "eu")):
        return "a"
    if w[:1] in "aeiou" or w.startswith(("8", "11", "18")):
        return "an"
    return "a"


def join_en(items, word="and"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


# --------------------------------------------------------------------------
# house model
# --------------------------------------------------------------------------
@dataclass
class Ent:
    eid: str
    name: str
    area: str | None
    domain: str
    label: str = ""          # device type label, e.g. "Ceiling Light"
    gender: str = "s"        # "s" singular, "pl" plural noun
    state: str = "off"
    attrs: dict = field(default_factory=dict)
    hidden: str = ""         # type label that the name no longer shows
    voice: bool = False      # media player of a voice satellite

    @property
    def plural(self) -> bool:
        return self.gender == "pl"


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
        self.english_ids = R.random() < 0.4      # here: the alternative slugs
        # naming style of the install: "Living Room Lamp" or "Living room lamp"
        self.sentence_case = R.random() < 0.35
        n_rooms = max(2, min(len(ROOMS) - 4, size // 4 + R.randint(-1, 2)))
        names = R.sample([r[0] for r in ROOMS if r[0] != "Main Bathroom"], n_rooms)
        if "Bathroom" in names and R.random() < 0.4:
            names[names.index("Bathroom")] = "Main Bathroom"
        self.rooms = names
        self.appliances: list[tuple[str, str, Ent, Ent]] = []
        self._populate(size)
        for hook in POPULATE_HOOKS:
            hook(self)

    # ---- id / name helpers ----
    def _slug(self, room: str | None, type_slugs: list[str], domain: str) -> str:
        rslugs = ROOM_BY_NAME[room][3] if room else ["home", "house"]
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

    def styled(self, room: str, rest: str) -> str:
        """'Living Room' + 'Ceiling Light' in the naming style of the house."""
        if self.sentence_case:
            return cap(low(f"{room} {rest}"))
        return f"{room} {rest}"

    def _name(self, room: str | None, label: str, compound: bool) -> str:
        name = self.styled(room, label) if room else label
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
        e = self.add("media_player", room, "Speaker", "s",
                     [name.lower().replace(" ", "_"), "voice_media_player"], name=name,
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
                # sample may call it "the ceiling lamp in the kitchen"
                room_lights[0].name = room
                room_lights[0].hidden, room_lights[0].label = room_lights[0].label, ""
                room_lights[0].gender = "s"
                self.used_names.add(room)
                if len(room_lights) > 1 and R.random() < 0.6:
                    sides = R.choice([["Left", "Right", "Center"], ["Ceiling", "Wall", "Corner"],
                                      ["Island", "Spots", "Counter"], ["1", "2", "3"]])
                    for lt, side in zip(room_lights[1:], sides):
                        nm = self.styled(room, side)
                        if nm not in self.used_names:
                            self.used_names.discard(lt.name)
                            lt.name = nm
                            lt.hidden, lt.label = lt.label, ""
                            lt.gender = "s"
                            self.used_names.add(lt.name)
            # many lamps carry a name without their room ("Starry Sky",
            # "Sideboard Light"); only the area column ties them to the room
            for lt in room_lights:
                if lt.name != room and R.random() < 0.3:
                    free = [n for n in FREE_LIGHT_NAMES if n not in self.used_names]
                    if free:
                        self.used_names.discard(lt.name)
                        lt.name = R.choice(free)
                        lt.hidden, lt.label = lt.label or lt.hidden, ""
                        lt.gender = "s"
                        self.used_names.add(lt.name)
            if not outdoor and room not in ("Hallway", "Stairs", "Guest Toilet") \
                    and room not in UTILITY and R.random() < 0.55:
                ct = R.choice(TYPES["cover"][:4])
                pos = R.choice([0, 0, 100, 100, 50, 30])
                if room in TWO_COVER_ROOMS and R.random() < 0.35:
                    # window and balcony door: "the shutters in the dining
                    # room" means both
                    word = R.choice(["Shutter", "Roller Shutter", "Blind", "Roller Blind"])
                    door = R.choice(["Balcony Door", "Patio Door", "Door", "French Door"])
                    for part, slug in (("Window", "window"), (door, "door")):
                        name = self.styled(room, f"{part} {word}")
                        if name in self.used_names:
                            continue
                        p2 = R.choice([pos, pos, 0, 100])
                        self.add("cover", room, word, "s",
                                 [f"{ct[2][0]}_{slug}", f"{ct[2][1]}_{slug}"], name=name,
                                 state="closed" if p2 == 0 else "open",
                                 attrs={"current_position": p2})
                else:
                    self.add("cover", room, ct[0], ct[1], ct[2], ct[3],
                             state="closed" if pos == 0 else "open",
                             attrs={"current_position": pos})
            if room in ("Patio", "Balcony", "Outdoor") and R.random() < 0.5:
                self.add("cover", room, "Awning", "s", ["awning", "sun_awning"],
                         state=(st := R.choice(["open", "closed"])),
                         attrs={"current_position": 100 if st == "open" else 0})
            if not outdoor and room not in UTILITY and R.random() < 0.5:
                ct = R.choice(TYPES["climate"][:3] if room != "Bathroom" else TYPES["climate"])
                target = R.choice([18, 19, 20, 20.5, 21, 21.5, 22, 23])
                cur = round(target + R.uniform(-2, 1.5), 1)
                self.add("climate", room, ct[0], ct[1], ct[2], ct[3],
                         state=R.choice(["heat", "heat", "off", "auto"]),
                         attrs={"current_temperature": cur, "temperature": target,
                                "hvac_action": R.choice(["heating", "idle"])})
            if not outdoor and R.random() < 0.45:
                t = round(R.uniform(17, 24.5), 1)
                self.add("sensor", room, "Temperature", "s",
                         ["temperature", "temp"],
                         state=str(t), attrs={"device_class": "temperature",
                                              "unit_of_measurement": "°C"})
            if not outdoor and R.random() < 0.3:
                self.add("sensor", room, "Humidity", "s",
                         ["humidity", "relative_humidity"],
                         state=str(R.randint(35, 70)),
                         attrs={"device_class": "humidity",
                                "unit_of_measurement": "%"})
            if not outdoor and R.random() < 0.35:
                self.add("binary_sensor", room, "Window", "s",
                         ["window", "window_contact"],
                         state=R.choice(["off", "off", "off", "on"]),
                         attrs={"device_class": "window"})
            if R.random() < 0.25:
                st = R.choice(TYPES["switch"])
                self.add("switch", room, st[0], st[1], st[2],
                         state=R.choice(["on", "off"]))
            if room in ("Living Room", "Bedroom", "Kitchen", "Office",
                        "Kids Room") and R.random() < 0.4:
                mt = R.choice(TYPES["media_player"])
                brand = None
                if R.random() < 0.5:
                    brand = R.choice(BRAND_MEDIA)
                    mt = (brand[1], brand[2], [brand[0].lower().replace(" ", "_")
                                               .replace("-", "_")])
                playing = R.random() < 0.4
                attrs = {"volume_level": R.choice([0.2, 0.3, 0.35, 0.5])}
                if playing:
                    title, artist = R.choice(
                        [("BBC News", ""), ("Bohemian Rhapsody", "Queen"),
                         ("Shape of You", "Ed Sheeran"), ("The Crown", ""), ("Bluey", ""),
                         ("Morning Jazz", "")])
                    attrs.update(media_title=title, media_artist=artist)
                self.add("media_player", room, mt[0], mt[1], mt[2],
                         name=brand[0] if brand and brand[0] not in self.used_names else None,
                         state="playing" if playing else R.choice(["off", "paused", "idle"]),
                         attrs={k: v for k, v in attrs.items() if v != ""})
            if not outdoor and R.random() < 0.12:
                self.add_voice_media(room)
            if room in ("Bedroom", "Office", "Living Room", "Attic") \
                    and R.random() < 0.2:
                ft = R.choice(TYPES["fan"])
                on = R.random() < 0.4
                self.add("fan", room, ft[0], ft[1], ft[2],
                         state="on" if on else "off",
                         attrs={"percentage": R.choice([33, 66, 100])} if on else {})
            if R.random() < 0.12:
                self.add("sensor", room, "Power", "s", ["power", "power_consumption"],
                         name=self._name(room, R.choice(["Plug Power", "Power"]), False),
                         state=str(R.choice([0, 3.2, 45, 120, 860, 2100])),
                         attrs={"device_class": "power", "unit_of_measurement": "W"})
        # house-wide devices
        if R.random() < 0.5:
            area = R.choice(self.rooms + [None])
            self.add("alarm_control_panel", None, "Alarm System", "s",
                     ["alarm_system", "alarm"], name="Alarm System",
                     state=R.choice(["disarmed", "disarmed", "armed_away",
                                     "armed_home", "armed_night"]))
        if R.random() < 0.55:
            for lk in R.sample(LOCKS, R.choice([1, 1, 2])):
                room = R.choice(["Hallway", "Entrance", None]) if lk[0] == "Front Door" else None
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
            room = "Garden" if "Garden" in self.rooms else None
            self.add("lawn_mower", room, "Robot Mower", "s",
                     [mname.lower().replace(" ", "_")], name=mname,
                     state=R.choice(["docked", "mowing", "paused"]))
        if R.random() < 0.2:
            room = R.choice([r for r in self.rooms if r in OUTDOOR] or [None])
            names = ["Irrigation", "Water Valve"]
            if room:
                names += [f"{room} Irrigation", f"{room} Plant Valve", f"{room} Plant Watering"]
                if room == "Garden":
                    names.append("Garden Valve")
            self.add("valve", room, "Irrigation", "s", ["irrigation", "water_valve"],
                     name=R.choice(names), state=R.choice(["open", "closed", "closed"]))
        if R.random() < 0.6:
            for sc in R.sample(SCENES, R.randint(1, 4)):
                self.add("scene", None, sc, "s", [sc.lower().replace(" ", "_")],
                         name=sc, state="2026-10-30T19:12:00+00:00")
        if R.random() < 0.35:
            for sc in R.sample(SCRIPTS, R.randint(1, 2)):
                self.add("script", None, sc, "s", [sc.lower().replace(" ", "_")
                         .replace("-", "_")], name=sc, state="off")
        out_t = round(R.uniform(-6, 31), 1)
        has_out = R.random() < 0.5
        if has_out:
            room, names = None, ["Outdoor Temperature", "Outside Temperature",
                                 "Weather Station Temperature"]
            orooms = [r for r in self.rooms if r in OUTDOOR]
            if orooms and R.random() < 0.4:
                # weather stations sit on the balcony or in the garden
                room = R.choice(orooms)
                names = [f"{room} Outdoor Temperature", f"{room} Temperature"]
            self.add("sensor", room, "Outdoor Temperature", "s",
                     ["outdoor_temperature", "outside_temperature"],
                     name=R.choice(names), state=f"{out_t:.1f}",
                     attrs={"device_class": "temperature",
                            "unit_of_measurement": "°C"})
        # heat pumps report their own outdoor sensor next to the climate entity
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            room = "Boiler Room" if "Boiler Room" in self.rooms and R.random() < 0.5 else None
            self.add("sensor", room, "Outdoor Temperature", "s",
                     ["heat_pump_outdoor_temperature", "home_outdoor_temperature"],
                     name=R.choice(["House Outdoor Temperature", "Heat Pump Outdoor Temperature",
                                    "Heating Outdoor Temperature"]),
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
                if label == "Forecast":
                    val = out_t + R.uniform(-4, 6) if when == "today" else R.uniform(-6, 31)
                else:
                    val = out_t - R.uniform(0.5, 7)
                self.add("sensor", None, label, "s",
                         slugs, name=name,
                         state=f"{val:.1f}",
                         attrs={"device_class": "temperature", "unit_of_measurement": "°C",
                                "_when": when})
        if "Garage" in self.rooms and R.random() < 0.6:
            self.add("cover", "Garage", "Garage Door", "s",
                     ["garage_door", "garage"], name="Garage Door",
                     state=(st := R.choice(["open", "closed", "closed"])),
                     attrs={"current_position": 100 if st == "open" else 0})
        if R.random() < 0.35:
            for app, g, slug in R.sample(APPLIANCES, R.choice([1, 1, 2])):
                if any(n.startswith(app) for n in self.used_names):
                    continue
                rooms = [r for r in self.rooms if r in (
                    "Kitchen", "Mudroom", "Utility Room", "Bathroom", "Main Bathroom",
                    "Basement", "Storage Room", "Office")]
                room = R.choice(rooms + [None]) if rooms else None
                kind = R.choice(list(APP_RAW))
                sensor = self.add("sensor", room, "Status", "s", [slug, slug + "_status"],
                                  name=R.choice([app, app, f"{app} Status"]),
                                  state=R.choice(APP_RAW[kind]), attrs={"_kind": kind})
                power = self.add("switch", room, "Plug", "s",
                                 [slug + "_power", slug + "_plug"],
                                 name=R.choice([f"{app} Power", f"{app} Plug",
                                                f"{app} Socket"]),
                                 state="off" if kind == "off" and R.random() < 0.5 else "on")
                self.appliances.append((app, g, sensor, power))
        vac = [e for e in self.ents if e.domain == "vacuum"]
        if vac and R.random() < 0.45:
            v = vac[0]
            for label, _ in R.sample(VACUUM_BUTTONS, R.choice([1, 2])):
                self.add("button", v.area, label, "s",
                         [v.eid.split(".")[1] + "_" + label.lower().replace(" ", "_")],
                         name=f"{v.name} {label}", state="unknown")
        valve_rooms = {e.area for e in self.ents if e.domain == "valve"}
        if R.random() < (0.5 if valve_rooms else 0.15):
            free = [r for r in self.rooms if r in OUTDOOR and r not in valve_rooms]
            # many installs leave the irrigation button without an area; its
            # name ("Garden Irrigation") then is the only hint where it waters
            room = R.choice(free) if free and R.random() < 0.65 else None
            if room:
                names = (["Garden Irrigation", "Garden Sprinklers"] if room == "Garden"
                         else [f"{room} Irrigation"])
            elif "Garden" in valve_rooms:
                names = ["Start Irrigation"]
            else:
                names = ["Start Irrigation", "Garden Irrigation", "Garden Sprinklers"]
            self.add("button", room, "Irrigation", "s", ["garden_irrigation",
                     "irrigation_start"], name=R.choice(names), state="unknown")
        # door contacts: balcony/patio doors in living rooms, the apartment
        # door in the hall; names often add "Contact" or "Sensor"
        if R.random() < 0.5:
            suffix = R.choice(["", " Contact", " Sensor"])
            for room in [r for r in self.rooms if r in DOOR_ROOMS]:
                if R.random() < 0.55:
                    kind, slug = R.choice([("Balcony", "balcony_door"), ("Patio", "patio_door")])
                    name = self.styled(room, f"{kind} Door{suffix}")
                    if name in self.used_names:
                        continue
                    self.add("binary_sensor", room, f"{kind} Door", "s", [slug, "door"],
                             name=name, state=R.choice(["off", "off", "on"]),
                             attrs={"device_class": "door"})
            if R.random() < 0.7:
                base = R.choice(["Apartment Door", "Entry Door", "Apartment Door"])
                name = base + suffix
                if name in self.used_names:
                    name = f"{name} Contact"
                hall = next((r for r in ("Hallway", "Entrance") if r in self.rooms), None)
                self.add("binary_sensor", hall, base, "s", ["apartment_door_state",
                                                             "entry_door_contact"],
                         name=name, state=R.choice(["off", "off", "on"]),
                         attrs={"device_class": "door"})
        # electric car: charge level and range next to each other
        if R.random() < 0.3:
            car, art, cslug = R.choice(CARS)
            self.add("sensor", None, "Battery", "s", ["battery_hv_state_of_charge", f"{cslug}_soc"],
                     name=f"{car} Battery", state=str(R.randint(10, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
            self.add("sensor", None, "Range", "s",
                     [f"{cslug}_range_ev_remaining_range", f"{cslug}_range"],
                     name=f"{car} Range", state=str(R.randint(40, 450)),
                     attrs={"device_class": "distance", "unit_of_measurement": "km"})
        # heating systems expose their modes as switches next to the climate
        # entity ("Home Heating Away Mode")
        if any(e.domain == "climate" for e in self.ents) and R.random() < 0.3:
            mode, slug = R.choice(HEAT_MODES)
            base = R.choice(["Heating", "Home Heating", "Thermostat", "Home"])
            room = "Boiler Room" if "Boiler Room" in self.rooms and R.random() < 0.5 else None
            if f"{base} {mode}" not in self.used_names:
                self.add("switch", room, mode, "s", [f"heating_{slug}", f"home_{slug}"],
                         name=f"{base} {mode}", state=R.choice(["on", "off", "off"]))
        # group entities as they exist in many installs
        if len([e for e in self.ents if e.domain == "light"]) >= 3 and R.random() < 0.45:
            e = self.add("light", None, "__group__", "pl", ["all_lights", "house_lights"],
                         name=R.choice(["All Lights", "All Lamps", "House Lights"]),
                         state=R.choice(["on", "off"]))
        if len([e for e in self.ents if e.domain == "cover"]) >= 3 and R.random() < 0.45:
            self.add("cover", None, "__group__", "pl", ["all_shutters", "all_covers"],
                     name=R.choice(["All Shutters", "All Blinds", "House Shutters"]),
                     state=R.choice(["open", "closed"]))
        # read-only distractors and odd names; fill up to the target size
        while len(self.ents) < size:
            self._distractor()

    def _distractor(self):
        room = R.choice(self.rooms + [None])
        k = R.randrange(10)
        rname = room or "House"
        if k == 0 and room:
            self.add("binary_sensor", room, "Presence", "s", ["presence"],
                     name=R.choice([f"{room} Presence", f"{room} Occupancy",
                                    f"Presence {room}"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "occupancy"})
        elif k == 1:
            self.add("binary_sensor", room, "Motion", "s", ["motion"],
                     name=R.choice([f"{rname} Motion", f"{rname} Motion Sensor"]),
                     state=R.choice(["on", "off"]), attrs={"device_class": "motion"})
        elif k == 2:
            plant = R.choice(PLANTS)
            self.add("sensor", room, "Soil Moisture", "s", ["plant_sensor", "moisture"],
                     name=f"{plant} Soil Moisture", state=str(R.randint(8, 70)),
                     attrs={"device_class": "moisture", "unit_of_measurement": "%"})
        elif k == 3:
            self.add("sensor", room, "Battery", "s", ["battery"],
                     name=R.choice(["UPS Battery", "Car Battery Level", "Smoke Detector Battery",
                                    "Phone Battery", f"{rname} Sensor Battery"]),
                     state=str(R.randint(5, 100)),
                     attrs={"device_class": "battery", "unit_of_measurement": "%"})
        elif k == 4:
            pr = R.choice(["Anycubic Kobra S1 Max", "Prusa MK4", "Bambu X1C"])
            part = R.choice(["Nozzle temperature", "Bed temperature", "Chamber temperature"])
            self.add("sensor", room, "Printer", "s", ["printer"], name=f"{pr} {part}",
                     state=str(R.randint(22, 220)),
                     attrs={"device_class": "temperature", "unit_of_measurement": "°C"})
        elif k == 5:
            nm, unit = R.choice([("Wind Speed", "km/h"), ("Wind Gusts", "km/h"),
                                 ("Rainfall", "mm/h"), ("Air Pressure", "hPa"),
                                 ("Water Tank Level", "L")])
            self.add("sensor", room, nm, "s", ["weather_station"], name=nm,
                     state=str(R.randint(0, 40)), attrs={"unit_of_measurement": unit})
        elif k == 6:
            self.add("sensor", room, "Status", "s", ["status"],
                     name=R.choice(["UPS Status", "Dryer", "Washing Machine Status",
                                    "Dishwasher Program"]),
                     state=R.choice(["idle", "OL", "running", "off"]))
        elif k == 7:
            nm = R.choice(TOGGLES)
            if nm not in self.used_names:
                self.add("input_boolean", None, nm, "s", [nm.lower().replace(" ", "_")],
                         name=nm, state=R.choice(["on", "off"]))
        elif k == 8:
            nm = R.choice(JUNK_SWITCHES)
            if nm not in self.used_names:
                self.add("switch", room, nm, "s", [nm.lower().replace(" ", "_")
                         .replace("-", "_")], name=nm, state=R.choice(["on", "off"]))
        else:
            self.add("sensor", room, "Temperature", "s", ["temperature"],
                     name=R.choice([f"{rname} Temperature", f"{rname} Temp",
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


def in_name(room: str | None, name: str) -> bool:
    return bool(room) and room.lower() in name.lower()


def article_ok(e: Ent) -> bool:
    """'the Kitchen Ceiling Light' reads well; 'the Kitchen' or 'the Kitchen
    Left' (lights named after the room) does not."""
    if e.name in ROOM_BY_NAME:
        return False
    if e.area and e.name.lower().startswith(e.area.lower()) and not e.label:
        return False
    return True


# --------------------------------------------------------------------------
# references: how the user names a device or a group of devices
# --------------------------------------------------------------------------
@dataclass
class Ref:
    acc: str      # phrase used in commands ("the light in the bathroom")
    nom: str      # phrase for confirmations
    plural: bool
    ents: list
    area: str | None = None   # set for "the light in the bathroom": control by area


def ref_single(h: House, e: Ent, allow_room_ref=True) -> Ref:
    same_type_in_room = [x for x in h.ents if x.domain == e.domain
                         and x.area == e.area and x.label == e.label]
    same_type = [x for x in h.ents if x.domain == e.domain and x.label == e.label]
    opts = ["name", "name"]
    # "the light in the hallway" / "the lamp in the hallway" mean every light
    # of the room, so a single lamp only gets these words when it is the
    # room's only light
    generic = e.domain == "light" and e.label in ("Light", "Lamp") \
        and len(h.of("light", e.area)) > 1
    if allow_room_ref and e.label and e.area and len(same_type_in_room) == 1 \
            and not generic:
        opts += ["type_room", "type_room"]
    if e.label and e.label != "__group__" and len(same_type) == 1 and e.name != e.label:
        opts += ["type_only"]           # "the TV" for "LG webOS Smart TV"
    choice = R.choice(opts)
    if choice == "type_only":
        return Ref(f"the {low(e.label)}", f"the {low(e.label)}", e.plural, [e])
    if choice == "name":
        name = e.name if R.random() < 0.8 else e.name.lower()
        if article_ok(e) and R.random() < 0.5:
            return Ref(f"the {name}", f"the {e.name}", e.plural, [e])
        return Ref(name, e.name, e.plural, [e])
    prep = ROOM_BY_NAME[e.area][1]
    return Ref(f"the {low(e.label)} {prep}", f"the {low(e.label)} {prep}", e.plural, [e])


# group nouns per domain: (phrase, phrase, plural)
GROUP = {
    "light": [("the light", "the light", False), ("the lights", "the lights", True),
              ("the lamps", "the lamps", True), ("the lighting", "the lighting", False)],
    "cover": [("the shutters", "the shutters", True), ("the blinds", "the blinds", True),
              ("the roller blinds", "the roller blinds", True)],
    "climate": [("the heating", "the heating", False), ("the radiators", "the radiators", True)],
    "media_player": [("the music", "the music", False)],
}
ALL_GROUP = {
    "light": [("all the lights", "All the lights"), ("all lights", "All lights"),
              ("every light", "Every light in the house")],
    "cover": [("all the shutters", "All the shutters"), ("all the blinds", "All the blinds")],
    "climate": [("all the radiators", "All the radiators")],
}


def ref_room_group(domain, room, ents, location=None) -> Ref:
    acc, nom, pl = R.choice(GROUP[domain])
    area = room if domain in S.AREA_DOMAINS else None
    if location == room and R.random() < 0.7:
        return Ref(acc, nom, pl, ents, area)    # "Turn on the light"
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


# first words that lose their capital behind "Hey, " or "Please " (names and
# "I" keep it)
LOWER_OK = set("""Turn Switch Set Open Close Lower Raise Dim Start Activate Deactivate Pause
Play Put Make Run Lock Unlock Stop Move Send Press Trigger Water Extend Retract Roll Shut
Resume Skip Arm Disarm Enable Disable What What's Which Is Are How How's Can Could Would
Do Does Did Tell Add Read Remind Wake Cancel Delete Remove Check Cross Give Let Lights
Light The Mute Unmute Repeat Shuffle Announce Announcement: Broadcast: Mow Brighten Bring
Have Next Timer Where Will Never Forget Nothing Doesn't Oh, Actually, Not Just No No,
Nah, It It's Thanks Thank Hello Hi Good Great, Who Why Undo Again, Try Louder Quieter
Off All Both Yes, Okay, Then Same Wrong Sorry, And A Even To Got Alarm We We're Sound
Previous One Music Night Scene Script Heating Shutters Temperature Hold End Clear Get
Please Write Stick Mark Shorten Extend Subtract Take Unpause Continue Lights Leave
Weather? Date? Time? Weather Date Time Lamps Shutters Blinds Heating Every Shower
There Isn't Any Change Give Wash Empty Mop Full In On Outside Louder, Quieter,""".split())


IMPERATIVE = {"Turn", "Switch", "Set", "Open", "Close", "Lower", "Raise", "Dim", "Start",
              "Activate", "Deactivate", "Pause", "Play", "Put", "Make", "Run", "Lock",
              "Unlock", "Stop", "Move", "Send", "Press", "Water", "Extend", "Retract",
              "Roll", "Shut", "Resume", "Skip", "Arm", "Disarm", "Tell", "Add", "Read",
              "Remind", "Wake", "Cancel", "Delete", "Remove", "Check", "Cross", "Give",
              "Mute", "Unmute", "Repeat", "Shuffle", "Announce", "Brighten", "Mark", "Take",
              "Extend", "Shorten", "Subtract", "Unpause", "Write", "Stick", "Trigger",
              "Empty", "Wash", "Mow", "Bring", "Have", "Let", "Undo", "Change", "Clear",
              "End", "Hold", "Try"}


def voice(text: str) -> str:
    """Make user text look like speech-to-text output sometimes."""
    # "Lights on on the patio" -> "Lights on the patio on"
    text = re.sub(r"\b([Ll]ights?|[Ll]amps?) on (on the \w+)", r"\1 \2 on", text)
    common = text[:1].islower()
    text = cap(text)
    r = R.random()
    if r < 0.25:
        text = text.rstrip("?.!")
    elif r < 0.35:
        text = text.rstrip("?.!").lower()
    if R.random() < 0.05:
        first = text.split(" ", 1)[0]
        pre = R.choice(["Hey, ", "Okay, ", "So, ", "Please "])
        if pre == "Please " and first not in IMPERATIVE:
            pre = "Hey, "
        if common or first in LOWER_OK:
            text = text[0].lower() + text[1:]
        text = pre + text
    return text


def confirm(ref: Ref, adj: str, part: str) -> str:
    """adj: state word ('on'), part: past participle ('turned on')."""
    return R.choice([
        f"{cap(ref.nom)} {be(ref.plural)} now {adj}.",
        f"{cap(ref.nom)} {be(ref.plural)} {adj}.",
        f"I've {part} {ref.nom}.",
        f"Done, {ref.nom} {be(ref.plural)} {adj}.",
        f"Okay, {ref.nom} {be(ref.plural)} now {adj}.",
        "Done.",
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
            "Turn on {r}", "Switch on {r}", "Turn {r} on", "{r} on",
            "Please turn on {r}", "Can you turn on {r}?",
            "Turn on {r}, please", "Switch {r} on", "Could you switch on {r}?",
            "I need some light, turn on {r}", "Turn {r} on, please.",
        ])
        return (user.format(r=ref.acc), [action("light.turn_on", ref)],
                confirm(ref, "on", "turned on"))
    user = R.choice([
        "Turn off {r}", "Switch off {r}", "Turn {r} off", "{r} off",
        "Please turn off {r}", "Can you turn off {r}?", "Turn off {r}, please.",
        "Switch {r} off", "Could you switch off {r}?",
    ])
    return (user.format(r=ref.acc), [action("light.turn_off", ref)],
            confirm(ref, "off", "turned off"))


def i_light_room_phrase(h, loc):
    """'Lights on in the bathroom' / 'It's too dark in the bathroom'."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if not rooms:
        return None
    room = R.choice(rooms)
    ents = h.of("light", room)
    prep = ROOM_BY_NAME[room][1]
    ref = Ref(f"the light {prep}", f"the light {prep}", False, ents, room)
    if R.random() < 0.5:
        user = R.choice([f"Lights on {prep}", f"Turn on the light {prep}",
                         f"It's too dark {prep}", f"{cap(prep)}, lights on please",
                         f"Light on {prep}"])
        return user, [action("light.turn_on", ref)], confirm(ref, "on", "turned on")
    user = R.choice([f"Lights off {prep}", f"Turn off the light {prep}",
                     f"Light off {prep}", f"Switch off the lights {prep}"])
    return user, [action("light.turn_off", ref)], confirm(ref, "off", "turned off")


def i_light_brightness(h, loc):
    ref = pick_target(h, "light", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.6:
        p = R.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 5])
        user = R.choice([
            "Dim {r} to {p} percent", "Set {r} to {p}%",
            "{r} to {p} percent", "Put {r} at {p} percent",
            "Set {r} to {p} percent brightness",
        ]).format(r=ref.acc, p=p)
        conf = R.choice([f"{cap(ref.nom)} {be(ref.plural)} now at {p} percent.",
                         f"I've set {ref.nom} to {p} percent.", "Done."])
        return user, [action("light.turn_on", ref, {"brightness_pct": p})], conf
    up = r < 0.8
    user = R.choice(["Make {r} brighter", "{r} brighter", "Brighten {r} a bit, please"]
                    if up else
                    ["Make {r} dimmer", "Dim {r}", "{r} a bit darker"]).format(r=ref.acc)
    step = 20 if up else -20
    conf = (f"{cap(ref.nom)} {be(ref.plural)} "
            f"{'brighter' if up else 'dimmer'} now.") if R.random() < 0.7 else "Done."
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
    user = R.choice(["Make {r} {c}", "Set {r} to {c}", "Turn {r} {c}",
                     "{r} in {c}, please", "Can you make {r} {c}?",
                     "I'd like {r} in {c}"]).format(r=ref.acc, c=word)
    conf = R.choice([f"{cap(ref.nom)} {be(ref.plural)} {word} now.",
                     f"I've set {ref.nom} to {word}.", "Done."])
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
        user = R.choice(["Turn on {r}", "Switch on {r}", "{r} on", "Turn {r} on",
                         "Can you switch on {r}?", "Start {r}"])
        conf = confirm(ref, "on", "turned on")
    else:
        user = R.choice(["Turn off {r}", "Switch {r} off", "{r} off", "Shut off {r}",
                         "Can you switch off {r}?"])
        conf = confirm(ref, "off", "turned off")
    return user.format(r=ref.acc), [action(svc, [e])], conf


def i_fan_pct(h, loc):
    ents = h.of("fan")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e)
    p = R.choice([25, 33, 50, 66, 75, 100])
    user = R.choice(["Set {r} to {p} percent", "{r} to {p}%",
                     "Run {r} at {p} percent"]).format(r=ref.acc, p=p)
    return user, [action("fan.set_percentage", [e], {"percentage": p})], \
        R.choice([f"{cap(ref.nom)} {be(ref.plural)} at {p} percent.", "Done."])


def the_name(e: Ent) -> str:
    """'the Patio Awning' for type-named devices, the bare name otherwise."""
    return f"the {e.name}" if e.label and e.name.lower().endswith(e.label.lower()) else e.name


def i_awning(h, loc):
    """Awning: extend = open_cover, retract = close_cover."""
    ents = [e for e in h.of("cover") if e.label == "Awning"]
    if not ents:
        return None
    e = R.choice(ents)
    n = "the awning" if len(ents) == 1 and R.random() < 0.7 else the_name(e)
    if R.random() < 0.5:
        user = R.choice([f"Extend {n}", f"Roll out {n}", f"Open {n}",
                         f"{cap(n)} out", "We need some shade, extend the awning"])
        return user, [action("cover.open_cover", [e])], R.choice(
            ["The awning is extending.", "Done, the awning is out."])
    user = R.choice([f"Retract {n}", f"Roll in {n}", f"Close {n}",
                     f"{cap(n)} in", "It's raining, retract the awning"])
    return user, [action("cover.close_cover", [e])], R.choice(
        ["The awning is retracting.", "Done, the awning is retracted."])


def i_mower(h, loc):
    ents = h.of("lawn_mower")
    if not ents:
        return None
    e = ents[0]
    n = R.choice(["the robot mower", "the lawn mower", e.name, f"the {e.name}"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Start {n}", f"Have {n} mow the lawn", "Mow the lawn, please",
                         f"Send {n} out"])
        return user, [action("lawn_mower.start_mowing", [e])], f"{e.name} is mowing now."
    if r < 0.8:
        user = R.choice([f"Send {n} back to its dock", f"Send {n} home",
                         f"Bring {n} in"])
        return user, [action("lawn_mower.dock", [e])], f"{e.name} is heading back to its dock."
    user = R.choice([f"Pause {n}", f"Stop {n} for now"])
    return user, [action("lawn_mower.pause", [e])], f"{e.name} is paused."


AREA_PLANTS = {"Balcony": "the balcony plants", "Patio": "the patio plants",
               "Garden": "the garden"}


def irrigation(h):
    return h.of("valve") + [e for e in h.of("button") if e.label == "Irrigation"]


def irrigation_phrases(h, e):
    """How the user may name this irrigation device. With several devices only
    phrases that point to one of them (name or area) are valid."""
    several = len(irrigation(h)) > 1
    out = [] if several else ["Start the irrigation", "Water the plants"]
    if e.area:
        prep = ROOM_BY_NAME[e.area][1]
        out += [f"Start the irrigation {prep}", f"Water the plants {prep}",
                f"Give the plants {prep} some water"]
        if e.area in AREA_PLANTS:
            out.append(f"Water {AREA_PLANTS[e.area]}")
    elif "garden" in e.name.lower():
        out += ["Water the garden", "Start the garden irrigation",
                "Give the garden some water"]
    return out


def i_valve(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    several = len(irrigation(h)) > 1
    n = R.choice([e.name, f"the {e.name}"] if several else
                 [e.name, f"the {e.name}", "the irrigation", "the valve"])
    if R.random() < 0.5:
        user = R.choice([f"Open {n}", f"Turn on {n}", f"Open up {n}"]
                        + irrigation_phrases(h, e) * 2)
        return user, [action("valve.open_valve", [e])], f"{e.name} is open."
    user = R.choice([f"Close {n}", f"Turn off {n}", f"Shut {n}",
                     "Stop the irrigation"])
    return user, [action("valve.close_valve", [e])], f"{e.name} is closed."


def i_toggle(h, loc):
    ents = h.of("input_boolean")
    if not ents:
        return None
    e = R.choice(ents)
    on = R.random() < 0.5
    n = e.name
    # "Vacation" is also called "Vacation Mode"; it stays an input_boolean
    mode = n if n.endswith("Mode") else f"{n} Mode"
    if mode != n and any(x.name == mode for x in h.ents):
        mode = n          # a script "Vacation Mode" exists: only the plain name is clear
    m = low(mode)
    if on:
        user = R.choice([f"Activate {n}", f"{n} on", f"Turn on {n}",
                         f"Activate {m}", f"{cap(m)} on", f"Turn on {m}",
                         f"Switch on {m}"])
        return user, [action("input_boolean.turn_on", [e])], f"{mode} is on."
    user = R.choice([f"Deactivate {n}", f"{n} off", f"Turn off {n}",
                     f"Deactivate {m}", f"{cap(m)} off", f"End {m}"])
    return user, [action("input_boolean.turn_off", [e])], f"{mode} is off."


def i_appliance(h, loc):
    """'Turn off the dryer' -> the power switch, not the status sensor."""
    if not h.appliances:
        return None
    app, g, _sensor, power = R.choice(h.appliances)
    n = f"the {low(app)}"
    if R.random() < 0.5:
        user = R.choice([f"Turn off {n}", f"Switch off {n}", f"{app} off",
                         f"Shut off {n}", f"Can you turn off {n}?"])
        return user, [action("switch.turn_off", [power])], R.choice(
            [f"{power.name} is off.", f"I've turned off {power.name}."])
    user = R.choice([f"Turn on {n}", f"Switch on {n}", f"{app} on",
                     f"Turn {n} on"])
    return user, [action("switch.turn_on", [power])], R.choice(
        [f"{power.name} is on.", f"I've turned on {power.name}."])


def i_button(h, loc):
    ents = h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    vac = dict(VACUUM_BUTTONS)
    if e.label in vac:
        user = R.choice(vac[e.label] + [f"Press {e.name}"])
    elif e.label == "Irrigation":
        user = R.choice(irrigation_phrases(h, e) + [f"Press {e.name}"])
    else:
        user = R.choice([f"Press {e.name}", f"Trigger {e.name}"])
    return user, [action("button.press", [e])], R.choice(
        [f"I've pressed {e.name}.", f"I've started {e.name}.", "Done."])


# lamp names without a room (the names of the acceptance house and their
# translations are left out on purpose, so the acceptance test measures
# generalisation)
FREE_LIGHT_NAMES = ["Starry Sky", "Salt Lamp", "Lava Lamp", "Picture Light", "Display Cabinet",
                    "Hue Go", "Nanoleaf", "Hue Play", "Cloud Lamp", "Moon Lamp",
                    "Light Bar", "LED Picture Frame", "Paper Lantern", "Globe Light",
                    "Arc Lamp", "Corner Lamp", "Sofa Lamp", "Sideboard Light", "Bar Light",
                    "Bed Light", "Cabinet Light", "Window Light", "Desk Lamp",
                    "Hanging Lamp", "Light Arch", "Neon Sign", "Govee Strip", "Table Lamp",
                    "Dresser Light", "Vanity Light", "Reading Light", "Accent Lamp"]

DOOR_ROOMS = ("Living Room", "Dining Room", "Kitchen", "Bedroom", "Study")
# (name, article, slug)
CARS = [("Enyaq", "the", "enyaq_iv80"), ("ID.3", "the", "id3"), ("Model 3", "the", "model_3"),
        ("Zoe", "the", "zoe"), ("Enyaq", "the", "enyaq"), ("e-Golf", "the", "egolf"),
        ("EQA", "the", "eqa")]

WEATHER_SENSORS = [  # (name, label, day, slugs)
    ("Forecast Temperature Today", "Forecast", "today",
     ["forecast_temperature_today", "forecast_temp_today"]),
    ("Forecast Temperature Tomorrow", "Forecast", "tomorrow",
     ["forecast_temperature_tomorrow", "forecast_temp_tomorrow"]),
    ("High Temperature Today", "Forecast", "today",
     ["high_temperature_today", "forecast_high_today"]),
    ("High Temperature Tomorrow", "Forecast", "tomorrow",
     ["high_temperature_tomorrow", "forecast_high_tomorrow"]),
    ("Feels Like Temperature", "Weather Value", None, ["feels_like_temperature", "apparent_temperature"]),
    ("Dew Point", "Weather Value", None, ["dew_point", "dewpoint"]),
]

HEAT_MODES = [("Away Mode", "away_mode"), ("Eco Mode", "eco_mode"),
              ("Frost Protection", "frost_protection")]


def i_heating_mode(h, loc):
    """'Turn on the heating's away mode' -> the mode switch, not the climate
    entity."""
    modes = [e for e in h.ents if e.domain == "switch"
             and e.label in [m for m, _ in HEAT_MODES]]
    if not modes or not h.of("climate"):
        return None
    e = R.choice(modes)
    m = low(e.label)
    on = R.random() < 0.6
    if on:
        user = R.choice([f"Turn on the heating's {m}", f"Put the heating in {m}",
                         f"Heating to {m}", f"{cap(m)} on", f"Switch on {m}",
                         f"Set the heating to {m}", f"Turn on {m}"])
    else:
        user = R.choice([f"Turn off the heating's {m}", f"{cap(m)} off",
                         f"Switch off {m}", f"End the heating's {m}",
                         f"Take the heating out of {m}"])
    adj, part = ("on", "turned on") if on else ("off", "turned off")
    return (user, [action("switch.turn_on" if on else "switch.turn_off", [e])],
            R.choice([f"{e.name} is now {adj}.", f"I've {part} {e.name}."]))


def i_irrigation(h, loc):
    """Several irrigation devices: the phrase must pick the right one
    ("Water the balcony plants" -> balcony valve, not the garden button)."""
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
            [f"{e.name} is open.", f"I've opened {e.name}."])
    return user, [action("button.press", [e])], R.choice(
        [f"I've pressed {e.name}.", f"I've started {e.name}."])


def i_irrigation_pair(h, loc):
    """A plant valve on the balcony or patio next to an area-less garden
    button, as many real installs have it. "Water the balcony plants" opens
    the valve, "Water the garden" presses the button."""
    rooms = [r for r in h.rooms if r in ("Balcony", "Patio")]
    if not rooms or irrigation(h):
        return None
    room = R.choice(rooms)
    valve = h.add("valve", room, "Irrigation", "s", ["plants", "irrigation"],
                  name=R.choice([f"{room} Plant Valve", f"{room} Plant Watering",
                                 f"{room} Irrigation", "Water Valve"]),
                  state=R.choice(["open", "closed", "closed"]))
    button = h.add("button", None, "Irrigation", "s", ["garden_irrigation"],
                   name=R.choice(["Garden Irrigation", "Garden Sprinklers"]),
                   state="unknown")
    if R.random() < 0.6:
        prep = ROOM_BY_NAME[room][1]
        user = R.choice([f"Water {AREA_PLANTS[room]}", f"Give the plants {prep} some water",
                         f"Start the irrigation {prep}", f"Water the plants {prep}"])
        return user, [action("valve.open_valve", [valve])], R.choice(
            [f"{valve.name} is open.", f"I've opened {valve.name}."])
    user = R.choice(["Water the garden", "Start the garden irrigation",
                     "Give the garden some water", "Start the irrigation in the garden"])
    return user, [action("button.press", [button])], R.choice(
        [f"I've pressed {button.name}.", f"I've started {button.name}."])


def i_cover(h, loc):
    ref = pick_target(h, "cover", loc)
    if not ref:
        return None
    r = R.random()
    if r < 0.4:
        user = R.choice(["Open {r}", "Raise {r}", "Roll up {r}", "{r} up",
                         "Open up {r}", "Put {r} up", "Can you open {r}?"])
        return (user.format(r=ref.acc), [action("cover.open_cover", ref.ents)],
                confirm(ref, "open", "opened"))
    if r < 0.8:
        user = R.choice(["Close {r}", "Lower {r}", "Roll down {r}", "{r} down",
                         "Shut {r}", "Let {r} down", "Can you close {r}?"])
        return (user.format(r=ref.acc), [action("cover.close_cover", ref.ents)],
                confirm(ref, "closed", "closed"))
    if r < 0.88:
        user = R.choice(["Stop {r}", "Hold {r}", "Stop {r} now", "Stop moving {r}"])
        return (user.format(r=ref.acc), [action("cover.stop_cover", ref.ents)],
                R.choice(["Stopped.", f"I've stopped {ref.nom}.", "Done."]))
    p, phrase = R.choice([(50, "halfway"), (30, "to 30 percent"), (70, "to 70 percent"),
                          (20, "to 20 percent"), (80, "to 80 percent"),
                          (50, "to 50 percent"), (10, "to 10 percent")])
    if phrase == "halfway":
        user = R.choice(["Close {r} halfway", "Lower {r} halfway", "{r} halfway"]).format(r=ref.acc)
    else:
        user = R.choice(["Move {r} {ph}", "Set {r} {ph}", "{r} {ph}"]).format(r=ref.acc, ph=phrase)
    return user, [action("cover.set_cover_position", ref.ents, {"position": p})], \
        R.choice([f"{cap(ref.nom)} {be(ref.plural)} now at {p} percent.", "Done."])


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
            user = R.choice([f"Set the heating {prep} to {{t}} degrees",
                             f"Heating {prep} to {{t}} degrees",
                             f"Make it {{t}} degrees {prep}",
                             f"I'd like it to be {{t}} degrees {prep}"]).format(t=en_num(t))
            nom = f"the heating {prep}"
            pl = False
        else:
            e = R.choice(ents)
            ref = ref_single(h, e)
            tgt = [e]
            user = R.choice(["Set {r} to {t} degrees", "{r} to {t} degrees",
                             "Put {r} on {t} degrees", "Turn {r} to {t} degrees"]
                            ).format(r=ref.acc, t=en_num(t))
            nom, pl = ref.nom, ref.plural
        conf = R.choice([f"{cap(nom)} {be(pl)} now set to {en_num(t)} degrees.",
                         f"I've set {nom} to {en_num(t)} degrees.",
                         f"Done, {en_num(t)} degrees."])
        return user, [action("climate.set_temperature", tgt, {"temperature": t})], conf
    e = R.choice(ents)
    ref = ref_single(h, e)
    on = r < 0.85
    if on:
        user = R.choice(["Turn on {r}", "Switch on {r}", "{r} on"]).format(r=ref.acc)
        return user, [action("climate.turn_on", [e])], confirm(ref, "on", "turned on")
    user = R.choice(["Turn off {r}", "Switch off {r}", "{r} off"]).format(r=ref.acc)
    return user, [action("climate.turn_off", [e])], confirm(ref, "off", "turned off")


def i_lock(h, loc):
    ents = h.of("lock")
    if not ents:
        return None
    e = R.choice(ents)
    ref = ref_single(h, e, allow_room_ref=False)
    if ref.acc.lower() == e.name.lower() and R.random() < 0.7:
        ref = Ref(f"the {low(e.name)}", f"the {low(e.name)}", False, [e])
    if R.random() < 0.55:
        user = R.choice(["Lock {r}", "Lock up {r}", "Please lock {r}",
                         "Is {r} locked? Please lock it", "Lock {r}, please"])
        return (user.format(r=ref.acc), [action("lock.lock", [e])],
                confirm(ref, "locked", "locked"))
    user = R.choice(["Unlock {r}", "Please unlock {r}", "Open the lock on {r}",
                     "Unlock {r}, please", "Can you unlock {r}?"])
    return (user.format(r=ref.acc, n=e.name), [action("lock.unlock", [e])],
            confirm(ref, "unlocked", "unlocked"))


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
            "Arm the alarm", "Turn on the alarm system",
            "I'm leaving now, arm the alarm", "Alarm on",
            "Set the alarm to away"])
        conf = R.choice(["The alarm is armed.",
                         "The alarm is now armed in away mode."])
    elif r < 0.45:
        svc, user = "alarm_control_panel.alarm_arm_home", R.choice([
            "Set the alarm to home", "Arm the alarm, I'm staying home",
            "Arm the alarm in home mode"])
        conf = "The alarm is armed in home mode."
    elif r < 0.6:
        svc, user = "alarm_control_panel.alarm_arm_night", R.choice([
            "Arm the alarm for the night", "Night mode for the alarm",
            "Arm the alarm in night mode"])
        conf = "The alarm is armed in night mode."
    else:
        svc, user = "alarm_control_panel.alarm_disarm", R.choice([
            "Turn off the alarm system", "Disarm the alarm", "Switch off the alarm",
            "I'm home, disarm the alarm", "Alarm off"])
        conf = R.choice(["The alarm is disarmed.",
                         "The alarm is now off."])
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
        ("media_player.media_pause", ["Pause {r}", "Pause {n}", "Pause {r}, please",
                                      "Pause playback on {n}"], None,
         "Paused."),
        ("media_player.media_play", ["Resume playback on {n}", "Continue playing on {n}",
                                     "Unpause {n}", "Continue on {n}"], None,
         "Playback is resuming."),
        ("media_player.media_next_track", ["Next track on {n}", "Skip to the next song on {n}",
                                           "Next song on {n}"], None, "Next track."),
        ("media_player.volume_up", ["Turn {r} up", "Turn up {r}", "{r} louder"], None,
         "Volume up."),
        ("media_player.volume_down", ["Turn {r} down", "Turn down {r}", "{r} quieter"], None,
         "Volume down."),
        ("media_player.turn_off", ["Turn off {r}", "Switch off {r}", "{r} off"], None,
         confirm(ref, "off", "turned off")),
        ("media_player.turn_on", ["Turn on {r}", "Switch on {r}"], None,
         confirm(ref, "on", "turned on")),
    ]
    if r < 0.2:
        v = R.choice([10, 20, 25, 30, 40, 50, 60, 80])
        user = R.choice(["Set the volume of {n} to {v} percent",
                         "{n} volume {v} percent",
                         "Set {r} to {v} percent volume"]).format(r=ref.acc, n=e.name, v=v)
        return user, [action("media_player.volume_set", [e], {"volume_level": v / 100})], \
            f"The volume is now at {v} percent."
    svc, users, data, conf = R.choice(opts)
    return R.choice(users).format(r=ref.acc, n=e.name), [action(svc, [e], data)], conf


def i_vacuum(h, loc):
    ents = h.of("vacuum")
    if not ents:
        return None
    e = ents[0]
    n = R.choice([f"the {low(e.name)}" if e.name == e.label else R.choice([e.name, f"the {e.name}"]),
                  "the vacuum", "the robot vacuum", "the vacuum"])
    r = R.random()
    if r < 0.5:
        user = R.choice([f"Start {n}", f"Let {n} clean", f"{cap(n)}, start",
                         "Start vacuuming, please", f"Send {n} out"])
        return user, [action("vacuum.start", [e])], R.choice(
            [f"{e.name} is starting.", "The vacuum is starting."])
    if r < 0.8:
        user = R.choice([f"Send {n} back to the dock", f"{cap(n)} back to base",
                         f"Send {n} home"])
        return user, [action("vacuum.return_to_base", [e])], \
            f"{e.name} is heading back to the dock."
    user = R.choice([f"Pause {n}", f"{cap(n)}, pause", f"Stop {n} for a moment"])
    return user, [action("vacuum.pause", [e])], f"{e.name} is paused."


def i_scene(h, loc):
    ents = h.of("scene") + h.of("script") + h.of("button")
    if not ents:
        return None
    e = R.choice(ents)
    if e.domain == "scene":
        user = R.choice([f"Activate the {e.name} scene", f"Scene {e.name}",
                         f"Activate {e.name}", f"Start the {e.name} scene",
                         f"Turn on the {e.name} scene"])
        return user, [action("scene.turn_on", [e])], R.choice(
            [f"The {e.name} scene is active.", f"Scene {e.name} activated.", "Done."])
    if e.domain == "script":
        user = R.choice([f"Run {e.name}", f"Start {e.name}", f"Run the {e.name} script",
                         f"Script {e.name}"])
        return user, [action("script.turn_on", [e])], R.choice(
            [f"{e.name} is running.", f"I've started {e.name}.", "Done."])
    return None


def i_all(h, loc):
    domain = R.choice(["light", "light", "cover"])
    ents = h.of(domain, plain=True)
    if len(ents) < 2:
        return None
    # a group entity ("All Lights") replaces the list of members
    if (grp := h.group(domain)) is not None:
        ents = [grp]
    acc, nom = R.choice(ALL_GROUP[domain])
    whole = S.ALL_AREAS if domain in S.AREA_DOMAINS and h.group(domain) is None else None
    ref = Ref(acc, nom, not acc.startswith("every"), ents, whole)
    if domain == "light":
        if R.random() < 0.6:
            user = R.choice([f"Turn off {acc}", f"Switch off {acc}", f"{cap(acc)} off",
                             "Lights off everywhere"])
            return user, [action("light.turn_off", ref)], R.choice(
                [f"{nom} {be(ref.plural)} off.", "All the lights are off.", "Done."])
        user = R.choice([f"Turn on {acc}", f"Switch on {acc}", f"{cap(acc)} on"])
        return user, [action("light.turn_on", ref)], R.choice(
            [f"{nom} {be(ref.plural)} on.", "All the lights are on.", "Done."])
    if R.random() < 0.6:
        user = R.choice([f"Close {acc}", f"Lower {acc}", f"{cap(acc)} down",
                         f"Shut {acc}"])
        return user, [action("cover.close_cover", ents)], R.choice(
            [f"{nom} are closed.", "Done, everything is closed."])
    user = R.choice([f"Open {acc}", f"Raise {acc}", f"{cap(acc)} up", f"Open up {acc}"])
    return user, [action("cover.open_cover", ents)], R.choice(
        [f"{nom} are open.", "Done, everything is open."])


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


# leading verbs that are lowercased behind "... and"
JOIN_VERBS = ("Turn", "Switch", "Set", "Open", "Close", "Lower", "Raise", "Dim", "Start",
              "Activate", "Deactivate", "Pause", "Play", "Put", "Make", "Run", "Lock",
              "Unlock", "Stop", "Move", "Send", "Press", "Trigger", "Water", "Bring",
              "Extend", "Retract", "Roll", "Shut", "Resume", "Skip", "Arm", "Disarm",
              "Give", "Have", "Let", "Brighten", "Hold", "Continue", "Unpause", "Mute",
              "Unmute", "Add", "Write", "Check", "Cross", "Remove", "Delete", "Take",
              "Repeat", "Shuffle", "Mow", "End", "Empty", "Wash", "Do", "Stick",
              "Tune", "Go", "Mark", "Listen")


def join_commands(a, b):
    """Two command intents -> one sentence 'X and y', or None."""
    if not a or not b or a[1] == b[1]:
        return None
    ids_a = {i for x in a[1] for i in x["entity_id"]}
    ids_b = {i for x in b[1] for i in x["entity_id"]}
    if ids_a & ids_b:
        return None
    ua = a[0].rstrip("?.!")
    ub = b[0].rstrip("?.!")
    if ub.startswith(("Can ", "Could ", "Would ", "I ", "I'", "Please", "It's", "Hey",
                      "We", "Is ", "Are ")) or \
            ua.startswith(("Can ", "Could ", "Would ", "It's")):
        return None
    # lowercase the leading verb of the second command ("... and close ...")
    first = ub.split(" ", 1)
    if first[0] in JOIN_VERBS or first[0] in LOWER_OK:
        ub = first[0].lower() + (" " + first[1] if len(first) > 1 else "")
    return f"{ua} and {ub}"


def tail_sentence(s: str) -> str:
    """Second confirmation behind 'and': 'The light is on.' -> 'the light is on.'"""
    for pre in ("Okay, ", "Done, "):
        if s.startswith(pre):
            s = s[len(pre):]
    if s.startswith("The "):
        return "the " + s[4:]
    return s


def s_compound(h, loc):
    a = pick_command(h, loc)
    b = pick_command(h, loc)
    user = join_commands(a, b)
    if not user:
        return None
    actions = a[1] + b[1]
    conf = R.choice(["Done.", "Both done.",
                     f"{a[2].rstrip('.')} and {tail_sentence(b[2])}"
                     if a[2] != "Done." and b[2] != "Done." else "Done."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "compound", "gold": [["control", {"actions": actions}]]}


def _state_word(e: Ent) -> str:
    d, s = e.domain, e.state
    if s == "unavailable":
        return "unreachable"
    if d in ("light", "switch", "fan", "input_boolean"):
        return "on" if s == "on" else "off"
    if d == "cover":
        return "open" if s == "open" else "closed"
    if d == "lock":
        return "locked" if s == "locked" else "unlocked"
    if d == "binary_sensor":
        return "open" if s == "on" else "closed"
    if d == "media_player":
        return {"playing": "playing", "paused": "paused",
                "off": "off", "idle": "on, but not playing anything"}[s]
    return s


def s_question(h, loc):
    """State question -> get_state -> grounded English answer."""
    kinds = []
    temps = [e for e in h.of("sensor") if e.attrs.get("device_class") == "temperature"
             and e.label not in ("Outdoor Temperature", "Forecast")]
    if temps:
        kinds += ["temp"] * 2
    outs = [e for e in h.of("sensor") if e.label == "Outdoor Temperature"]
    if outs:
        kinds += ["outdoor"] * 2
    fcs = [e for e in h.of("sensor") if e.label == "Forecast"]
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
        user = R.choice(["How warm is it outside?", "How cold is it outside?",
                         "What's the outside temperature?", "What's the temperature outside?",
                         "How many degrees is it outside?"])
        vals = [en_num(float(e.state)) for e in outs]
        if len(outs) == 1:
            ans = R.choice([f"It's {vals[0]} degrees outside.",
                            f"The outside temperature is {vals[0]} degrees."])
        else:
            ans = (f"{outs[0].name} reads {vals[0]} degrees and "
                   f"{outs[1].name} reads {vals[1]} degrees.")
        ids, must = outs, vals
    elif kind == "forecast":
        if h.of("weather"):
            return None       # the weather entity answers with condition and course
        when = R.choice(fcs).attrs["_when"]
        ids = [e for e in fcs if e.attrs["_when"] == when]
        vals = [en_num(float(e.state)) for e in ids]
        user = R.choice([f"How warm will it get {when}?",
                         f"How many degrees is it supposed to be {when}?",
                         f"What's the forecast for {when}?"])
        if len(ids) == 1:
            ans = R.choice([f"According to the forecast, it'll be {vals[0]} degrees {when}.",
                            f"{cap(when)} it's supposed to reach {vals[0]} degrees, "
                            f"according to the forecast."])
        else:
            ans = (f"For {when}, {ids[0].name} says {vals[0]} degrees and "
                   f"{ids[1].name} says {vals[1]} degrees.")
        must = vals
    elif kind == "temp":
        e = R.choice(temps)
        val = en_num(float(e.state))
        if e.area is None or e.area in OUTDOOR or e.label != "Temperature":
            # a sensor without area is not automatically outside, and a printer
            # bed or a freezer does not tell how warm the room is
            user = R.choice([f"What does {e.name} show?", f"What temperature does {e.name} read?"])
            ans = f"{e.name} reads {val} degrees."
        else:
            prep = ROOM_BY_NAME[e.area][1]
            user = R.choice([f"How warm is it {prep}?", f"What's the temperature {prep}?",
                             f"How many degrees is it {prep}?", f"Temperature {prep}?"])
            ans = R.choice([f"It's {val} degrees {prep}.",
                            f"The temperature {prep} is {val} degrees."])
        ids, must = [e], [val]
    elif kind == "humidity":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "humidity"])
        prep = ROOM_BY_NAME[e.area][1] if e.area else ""
        user = R.choice([f"What's the humidity {prep}?",
                         f"How humid is it {prep}?"]).replace(" ?", "?")
        ans = f"The humidity {prep} is {e.state} percent.".replace("  ", " ")
        ids, must = [e], [e.state]
    elif kind == "generic":
        e = R.choice([e for e in h.of("sensor")
                      if e.attrs.get("device_class") in ("moisture", "battery")])
        if e.attrs["device_class"] == "moisture":
            plant = e.name.replace(" Soil Moisture", "")
            p = low(plant)
            pl = plant in PLURAL_PLANTS
            user = R.choice([f"How moist is the soil for the {p}?",
                             f"{'Do' if pl else 'Does'} the {p} need watering?",
                             f"What's the soil moisture for the {p}?"])
            ans = f"The soil moisture for the {p} is {e.state} percent."
        else:
            user = R.choice([f"What level is {e.name} at?", f"How's {e.name} doing?",
                             f"What does {e.name} show?"])
            ans = f"{e.name} is at {e.state} percent."
        ids, must = [e], [e.state]
    elif kind == "power":
        e = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "power"])
        user = R.choice([f"What's {e.name} showing right now?",
                         f"What does {e.name} read?", f"How many watts does {e.name} show?"])
        val = en_num(float(e.state))
        ans = f"{e.name} shows {val} watts right now."
        ids, must = [e], [val]
    elif kind == "climate":
        e = R.choice(h.of("climate"))
        ref = ref_single(h, e)
        if R.random() < 0.5:
            user = R.choice([f"What temperature is {ref.nom} set to?",
                             f"What's the target temperature of {ref.nom}?"])
            val = en_num(e.attrs["temperature"])
            ans = f"{cap(ref.nom)} {be(ref.plural)} set to {val} degrees."
        else:
            if e.area:
                prep = ROOM_BY_NAME[e.area][1]
                user = R.choice([f"How warm is it {prep}?", f"How many degrees is it {prep}?"])
                if [s for s in h.of("sensor", e.area)
                        if s.attrs.get("device_class") == "temperature"]:
                    return None   # ambiguous: sensor and thermostat in same room
            else:
                user = f"What temperature does {ref.nom} measure?"
                prep = ""
            if "current_temperature" not in e.attrs:
                return None   # thermostat without its own sensor
            val = en_num(e.attrs["current_temperature"])
            ans = (f"It's {val} degrees {prep}." if prep
                   else f"{cap(ref.nom)} measures {val} degrees.")
        ids, must = [e], [val]
    elif kind == "onoff":
        cands = h.of("light") + h.of("switch") + h.of("cover")
        if not cands:
            return None
        e = R.choice(cands)
        ref = ref_single(h, e)
        if e.domain == "cover":
            user, asked = R.choice([(f"{cap(be(ref.plural))} {ref.nom} open?", "open"),
                                    (f"Are {ref.nom} down?", "closed")
                                    if ref.plural else
                                    (f"Is {ref.nom} closed?", "closed"),
                                    (f"{cap(be(ref.plural))} {ref.nom} shut?", "closed")])
        else:
            user, asked = R.choice([(f"Is {ref.nom} on?", "on"),
                                    (f"Is {ref.nom} still on?", "on"),
                                    (f"Is {ref.nom} switched off?", "off"),
                                    (f"Is {ref.nom} off?", "off")]) \
                if not ref.plural else R.choice([(f"Are {ref.nom} on?", "on"),
                                                 (f"Are {ref.nom} off?", "off")])
        word = _state_word(e)
        ans = f"{cap(ref.nom)} {be(ref.plural)} {word}."
        if word in ("on", "off", "open", "closed") and R.random() < 0.8:
            ans = ("Yes, " if word == asked else "No, ") + ref.nom + \
                f" {be(ref.plural)} {word}."
        if e.domain == "light" and e.state == "on" and "brightness" in e.attrs and R.random() < 0.5:
            pct = round(e.attrs["brightness"] / 255 * 100)
            yes = {"on": "Yes, ", "off": "No, "}[asked]
            ans = f"{yes}{ref.nom} {be(ref.plural)} on, at {pct} percent brightness."
        ids, must = [e], [word]
    elif kind == "which_lights":
        room = None
        if R.random() < 0.4:
            rooms = [r for r in h.rooms if len(h.of("light", r)) >= 1]
            room = R.choice(rooms)
            ents = h.of("light", room)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Which lights are on {prep}?", f"Is there still a light on {prep}?"])
        else:
            ents = h.of("light")
            user = R.choice(["Which lights are on?", "Is there a light on anywhere?",
                             "Are any lamps still on?", "Which lamps are switched on?"])
        q_area = room or S.ALL_AREAS
        if any(e.state == "unavailable" for e in ents):
            return None
        on = [e for e in ents if e.state == "on"]
        if not on:
            ans = "No lights are on right now." if not room else \
                f"No lights are on {ROOM_BY_NAME[room][1]} right now."
            must = ["no"]
        elif len(on) == 1:
            ans = f"Only {on[0].name} is on."
            must = [on[0].name]
        else:
            names = [e.name for e in on]
            ans = f"Currently on: {', '.join(names[:-1])} and {names[-1]}."
            must = names
        ids = ents
    elif kind == "which_covers":
        room = None
        rooms = [r for r in h.rooms if len(h.of("cover", r)) >= 2]
        if rooms and R.random() < 0.35:
            room = R.choice(rooms)
            prep = ROOM_BY_NAME[room][1]
            user = R.choice([f"Which shutters are open {prep}?",
                             f"Are any shutters still open {prep}?",
                             f"Which blinds are up {prep}?"])
        else:
            user = R.choice(["Which shutters are open?", "Are any shutters still open?",
                             "Which blinds are up?", "Which shutters are still open?",
                             "Which shutters are closed?", "Which blinds are down?"])
        ents = h.of("cover", room or "*")
        q_area, q_dom = room or S.ALL_AREAS, "cover"
        if any(e.state == "unavailable" for e in ents):
            return None
        want_open = not any(w in user for w in ("closed", "down"))
        hits = [e for e in ents if (e.state == "open") == want_open]
        word = "open" if want_open else "closed"
        where = f" {ROOM_BY_NAME[room][1]}" if room else ""
        if not hits:
            other = "closed" if want_open else "open"
            ans = R.choice([f"All the shutters{where} are {other}.",
                            f"None of the shutters{where} are {word}, they're all {other}."])
            must = [other]
        elif len(hits) == len(ents):
            ans = f"All the shutters{where} are {word}."
            must = [word]
        elif len(hits) == 1:
            ans = f"Only {hits[0].name} is {word}."
            must = [hits[0].name]
        else:
            names = [e.name for e in hits]
            ans = f"Currently {word}: {', '.join(names[:-1])} and {names[-1]}."
            must = names
        ids = ents
    elif kind == "windows":
        # only window sensors - presence and motion sensors are binary_sensors too
        ents = [e for e in h.of("binary_sensor")
                if e.attrs.get("device_class") == "window"]
        if not ents:
            return None
        user, yes_if_open = R.choice([("Are all the windows closed?", False),
                                      ("Is any window still open?", True),
                                      ("Which windows are open?", None),
                                      ("Are the windows shut?", False),
                                      ("Are any windows open?", True),
                                      ("Which windows are still open?", None),
                                      ("Did I close all the windows?", False)])
        opn = [e for e in ents if e.state == "on"]
        if not opn:
            ans = {True: "No, all the windows are closed.",
                   False: "Yes, all the windows are closed.",
                   None: "All the windows are closed."}[yes_if_open]
            must = ["closed"]
        else:
            names = [e.name for e in opn]
            ans = ("Currently open: " + names[0] + "." if len(names) == 1 else
                   "Currently open: " + ", ".join(names[:-1]) + " and " + names[-1] + ".")
            if yes_if_open is not None:
                ans = ("Yes. " if yes_if_open else "No. ") + ans
            must = names
        ids = ents
        q_area, q_dom, q_dc = S.ALL_AREAS, "binary_sensor", "window"
    elif kind == "door":
        # "Is the balcony door in the living room open?" -> that contact, not the front door
        doors = [e for e in h.of("binary_sensor") if e.attrs.get("device_class") == "door"]
        e = R.choice(doors)
        same = [d for d in doors if d.label == e.label]
        if e.area and (len(same) > 1 or R.random() < 0.5):
            what = f"the {low(e.label)} {ROOM_BY_NAME[e.area][1]}"
        elif len(same) == 1:
            what = f"the {low(e.label)}"
        else:
            return None
        if e.state == "unavailable":
            return None
        opn = e.state == "on"
        user, ask_open = R.choice([(f"Is {what} open?", True), (f"Is {what} shut?", False),
                                   (f"Is {what} closed?", False),
                                   (f"Is {what} still open?", True)])
        word = "open" if opn else "closed"
        ans = f"{'Yes' if opn == ask_open else 'No'}, {e.name} is {word}."
        ids, must = [e], [word]
    elif kind == "car":
        rng = R.choice([e for e in h.of("sensor") if e.attrs.get("device_class") == "distance"])
        car = rng.name.rsplit(" ", 1)[0]
        art = next(a for c, a, _ in CARS if c == car)
        soc = next((e for e in h.of("sensor") if e.name == f"{car} Battery"), None)
        if soc is not None and R.random() < 0.45:
            e = soc
            user = R.choice([f"What's the charge level of {art} {car}?",
                             f"How charged is {art} {car}?",
                             f"Is {art} {car} charged yet?"])
            ans = f"{cap(art)} {car} is charged to {e.state} percent."
        else:
            e = rng
            user = R.choice([f"How far can {art} {car} still go?",
                             f"How much range does {art} {car} have left?",
                             f"What's the range of {art} {car}?"])
            ans = f"{cap(art)} {car} has {e.state} km of range left."
        ids, must = [e], [e.state]
    elif kind == "lock":
        e = R.choice(h.of("lock"))
        nom = f"the {low(e.name)}"
        user = R.choice([f"Is {nom} locked?", f"Is {nom} locked up?",
                         f"Did I lock {nom}?"])
        word = _state_word(e)
        ans = (f"Yes, {nom} is locked." if e.state == "locked"
               else f"No, {nom} is unlocked.")
        ids, must = [e], [word]
    elif kind == "alarm":
        e = h.of("alarm_control_panel")[0]
        if e.state == "unavailable":
            return None   # unreachable devices: see _error_question
        user = R.choice(["Is the alarm armed?", "Is the alarm on?",
                         "What's the status of the alarm?"])
        txt = {"disarmed": "The alarm is disarmed.",
               "armed_away": "The alarm is armed in away mode.",
               "armed_home": "The alarm is armed in home mode.",
               "armed_night": "The alarm is armed in night mode."}[e.state]
        ans, ids, must = txt, [e], ["disarmed" if e.state == "disarmed" else "armed"]
    elif kind == "media":
        e = R.choice([e for e in h.of("media_player") if e.state == "playing"])
        user = R.choice([f"What's playing on {e.name}?", f"What is {e.name} playing?"])
        t = e.attrs.get("media_title", "")
        a = e.attrs.get("media_artist")
        ans = f"{e.name} is playing {t}" + (f" by {a}." if a else ".")
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
    """Two questions in one sentence ("Which shutters are open and how warm
    is it outside?"): both are read and answered (v16 dropped the second
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
    for pre in ("Hey, ", "Okay, ", "So, ", "Please "):
        q2 = q2.removeprefix(pre)
    if q2.split(" ", 1)[0] in LOWER_OK:
        q2 = q2[0].lower() + q2[1:]
    msgs = m1[:-4] + [{"role": "user", "content": f"{q1} and {q2}"}]
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
    """'Is the dryer done?' -> the status sensor, not the power switch."""
    if not h.appliances:
        return None
    app, g, sensor, _power = R.choice(h.appliances)
    kind = sensor.attrs["_kind"]
    nom = f"the {low(app)}"
    user = R.choice([f"Is {nom} done?", f"Is {nom} still running?", f"What's {nom} doing?",
                     f"Is {nom} finished yet?", f"What's the status of {nom}?"])
    phrase, key = APP_ANSWER[kind]
    ans = R.choice([f"{cap(nom)} {phrase}.", f"According to Home Assistant, {nom} {phrase}."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    args = {"entity_ids": [sensor.eid]}
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, h.state_of([sensor])),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "question", "gold": [["get_state", args]],
                  "answer_contains": [key]}


def s_relative_climate(h, loc):
    """'Make it warmer in the bathroom' -> get_state -> set_temperature target +/- 1."""
    ents = [e for e in h.of("climate") if e.area]
    if not ents:
        return None
    e = R.choice(ents)
    if len([x for x in h.of("climate", e.area)]) > 1:
        return None
    prep = ROOM_BY_NAME[e.area][1]
    up = R.random() < 0.6
    user = R.choice([f"Make it a bit warmer {prep}", f"I'm cold {prep}",
                     f"Turn the heating up a degree {prep}"] if up else
                    [f"Make it a bit cooler {prep}", f"It's too warm {prep}",
                     f"Turn the heating down a degree {prep}"])
    new = e.attrs["temperature"] + (1 if up else -1)
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    c1, id1 = tool_call("get_state", {"entity_ids": [e.eid]})
    acts = [action("climate.set_temperature", [e], {"temperature": new})]
    c2, id2 = tool_call("control", {"actions": acts})
    msgs += [c1, tool_msg(id1, h.state_of([e])), c2, tool_msg(id2, ok_result()),
             {"role": "assistant",
              "content": f"I've turned the heating {prep} {'up' if up else 'down'} from "
                         f"{en_num(e.attrs['temperature'])} to {en_num(new)} degrees."}]
    return msgs, {"cat": "relative", "gold": [["get_state", {"entity_ids": [e.eid]}],
                                              ["control", {"actions": acts}]]}


MISSING_TYPES = [
    # domain, phrase, noun, "There's no ...", "no ...", "a ..." / "any ...",
    # "There's a ...", verbs (on, off), short words (on, off), services,
    # state words (on, off), label filter
    ("light", "the light", "light", "There's no light", "no light", "a light",
     "There's a light", ("turn on", "turn off"), ("on", "off"),
     ("light.turn_on", "light.turn_off"), ("on", "off"), None),
    ("light", "the lamp", "lamp", "There's no lamp", "no lamp", "a lamp",
     "There's a lamp", ("turn on", "turn off"), ("on", "off"),
     ("light.turn_on", "light.turn_off"), ("on", "off"), None),
    ("cover", "the shutters", "shutters", "There are no shutters", "no shutters",
     "any shutters", "There are shutters", ("open", "close"), ("up", "down"),
     ("cover.open_cover", "cover.close_cover"), ("open", "closed"), "plain"),
    ("climate", "the heating", "heating", "There's no heating", "no heating",
     "any heating", "There's heating", ("turn on", "turn off"), ("on", "off"),
     ("climate.turn_on", "climate.turn_off"), ("on", "off"), None),
    ("media_player", "the TV", "TV", "There's no TV", "no TV", "a TV",
     "There's a TV", ("turn on", "turn off"), ("on", "off"),
     ("media_player.turn_on", "media_player.turn_off"), ("on", "off"), "TV"),
    ("fan", "the fan", "fan", "There's no fan", "no fan", "a fan",
     "There's a fan", ("turn on", "turn off"), ("on", "off"),
     ("fan.turn_on", "fan.turn_off"), ("on", "off"), None),
]


def s_missing_type(h, loc):
    """Room exists, but has no device of the requested kind -> no tool call.
    If the kind exists in another room, name it instead of switching it."""
    mt = FORCE.get("mtype")
    dom, phrase, noun, none_s, none, any_, there, verbs, shorts, svcs, words, filt = R.choice(
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
    verb = verbs[0] if turn_on else verbs[1]
    short = shorts[0] if turn_on else shorts[1]
    user = R.choice([f"{cap(verb)} {phrase} {prep}",
                     f"Can you {verb} {phrase} {prep}?",
                     f"{cap(noun)} {prep} {short}", f"{cap(verb)} {phrase} {prep}, please"])
    others = [e for e in ents if e.area and e.area != room]
    # suggest the other device only if there are one or two candidates
    alt = R.choice(others) if 0 < len(others) <= 2 else None
    if alt is not None and R.random() < 0.8:
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if in_name(alt.area, alt.name) else f" {oprep}"
        ans = R.choice([f"{none_s} {prep}. {there} {oprep}: {alt.name}.",
                        f"I can't find {any_} {prep}. Do you mean {alt.name}{where}?",
                        f"I don't have {any_} {prep} in the device list. "
                        f"{cap(oprep)}, there's {alt.name}."])
    else:
        alt = None
        ans = R.choice([f"I can't find {any_} {prep} that I can control.",
                        f"The device list has {none} {prep}.",
                        f"I didn't find {any_} {prep} that I can switch."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    gold: list = []
    r = R.random()
    if alt is not None and r < 0.35:
        # user accepts the device in the other room -> now it is switched
        oprep = ROOM_BY_NAME[alt.area][1]
        where = "" if in_name(alt.area, alt.name) else f" {oprep}"
        msgs.append({"role": "user", "content": voice(R.choice(
            [f"Then the one {oprep}", "Yes, that one", "Okay, that one then",
             f"Yes, {alt.name}", f"{cap(oprep)} then"]))})
        act = [action(svcs[0] if turn_on else svcs[1], [alt])]
        call, cid = tool_call("control", {"actions": act})
        word = words[0] if turn_on else words[1]
        msgs += [call, tool_msg(cid, ok_result()),
                 {"role": "assistant", "content": R.choice(
                     [f"{alt.name}{where} is now {word}.",
                      f"Done, {alt.name}{where} is {word}."])}]
        gold = [["control", {"actions": act}]]
        cat = "followup"
    elif r < 0.7:
        # user insists on the room, sometimes twice -> still no invented action
        pushes = ["Yes, turn it on!", "Just switch it on!", "No, there is one",
                  "Do it anyway", f"Yes, {prep}!", f"No, I mean {prep}",
                  "Isn't there?", "There is one, though", "Just do it"]
        refusals = [
            f"Your device list has {none} {prep}. Please check whether the device is exposed to Assist.",
            f"I really don't see {any_} {prep} that I can control.",
            f"{none_s} {prep} in my device list, so I can't switch anything there."]
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
        if dev.lower() in labels or (dev == "Lawn Mower" and h.of("lawn_mower")):
            return None
        d = low(dev)
        pl = g == "pl"
        user = R.choice([f"Turn on the {d}", f"Switch on the {d}", f"Turn off the {d}",
                         f"{cap(be(pl))} the {d} on?", f"Start the {d}"])
        ans = R.choice([f"I can't find a device called “{d}” in your smart home.",
                        f"I don't know any device called “{d}”. Sorry, I can't control that.",
                        f"{'There are no' if pl else 'There is no'} {d} in the device list."])
    else:
        missing = [r_[0] for r_ in ROOMS if r_[0] not in h.rooms and r_[0] != "Main Bathroom"
                   and not (r_[0] == "Bathroom" and "Main Bathroom" in h.rooms)]
        room = R.choice(missing)
        prep = ROOM_BY_NAME[room][1]
        what = R.choice([("the light", "a light", "Turn on", "Turn off", "Light", "off"),
                         ("the shutters", "any shutters", "Open", "Close", "Shutters", "down"),
                         ("the heating", "any heating", "Turn on", "Turn off", "Heating", "off")])
        user = R.choice([f"{what[2]} {what[0]} {prep}",
                         f"{what[3]} {what[0]} {prep}",
                         f"{what[4]} {prep} {what[5]}"])
        ans = R.choice([f"I can't find {what[1]} {prep} that I can control.",
                        f"I didn't find {what[1]} {prep} in the device list."])
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)},
                                {"role": "assistant", "content": ans}]
    return msgs, {"cat": "unknown", "gold": []}


WEATHER_CHAT = "What's the weather going to be like tomorrow?"
CHAT = [
    (["Thanks", "Thank you", "Thanks a lot", "Great, thanks"],
     ["You're welcome.", "Sure thing.", "No problem."]),
    (["Hello", "Hi", "Good morning", "Good evening"],
     ["Hello, how can I help?", "Hi. What can I do for you?"]),
    (["What can you do?", "What can you help me with?", "What can you control?"],
     ["I can control the lights, shutters, heating and other devices in your home "
      "and tell you what state they're in or what the sensors are measuring. "
      "I can also play music, set timers, keep your lists, tell you the time "
      "and the weather, and make announcements.",
      "I control your devices, play music, set timers, add things to your "
      "shopping list, tell you the time and the weather, and make announcements in the house."]),
    (["Who are you?"], ["I'm the voice assistant for your smart home."]),
    (["Tell me a joke"],
     ["Why did the light bulb fail its exam? It wasn't very bright.",
      "What did the blind say to the window? Don't worry, I've got you covered."]),
    ([WEATHER_CHAT],
     ["Sorry, I don't have a weather forecast. But I can tell you what your "
      "sensors are measuring right now."]),
    (["Good night"], ["Good night. Should I turn anything off?", "Good night."]),
    # HassNevermind: nothing to do
    (["Never mind", "Cancel", "Forget it", "Nothing", "Doesn't matter", "Oh, nothing",
      "Actually, no", "Not needed anymore", "Just forget it", "No, nothing",
      "I didn't say anything", "It's fine", "Nah, leave it", "Don't bother",
      "Skip it", "Leave it"],
     ["Okay.", "All right.", "Okay, never mind."]),
]


def s_chat(h, loc):
    users, answers = R.choice(CHAT)
    if users[0] == WEATHER_CHAT and (
            h.of("weather") or any(e.label == "Forecast" for e in h.of("sensor"))):
        return None
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(R.choice(users))},
                                {"role": "assistant", "content": R.choice(answers)}]
    return msgs, {"cat": "chat", "gold": []}


def _unreachable(name):
    return R.choice([f"Sorry, that didn't work. {name} {isnt(name)} responding right now.",
                     f"Sorry, there was an error: {name} {isnt(name)} reachable.",
                     f"{name} {isnt(name)} reachable right now, so I couldn't carry out the command."])


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
            ["Try again", "Again, please", "Give it another try",
             "Do it again"]))})
        call, cid = tool_call("control", {"actions": actions})
        msgs += [call, tool_msg(cid, S.control_result([err])),
                 {"role": "assistant", "content": R.choice([
                     f"{name} still {isnt(name)} reachable. Please check that the device has power and a connection.",
                     f"The second attempt didn't work either, {name} {isnt(name)} responding."])}]
        meta.update(cat="followup", gold_last=[["control", {"actions": actions}]])
    else:
        onoff = actions[0]["service"].endswith(("turn_on", "open_cover", "open_valve"))
        msgs.append({"role": "user", "content": voice(R.choice(
            ["Is it on now?", "It isn't, is it?", "Did it work?", "So, is it on?"]
            if onoff else ["Did it work?", "So, did that work?", "Is that right?"]))})
        e = next(x for x in h.ents if x.eid == eid)
        args = {"entity_ids": [eid]}
        call, cid = tool_call("get_state", args)
        msgs += [call, tool_msg(cid, S.dumps([S.format_state(eid, e.name, "unavailable", {})])),
                 {"role": "assistant", "content": R.choice([
                     f"No. {name} {isnt(name)} reachable, so I couldn't switch {'them' if pl_name(name) else 'it'}.",
                     f"According to Home Assistant, {name} {isnt(name)} reachable right now. Nothing was switched."])}]
        meta.update(cat="followup", gold_last=[["get_state", args]])
    return msgs, meta


def _error_room(h, loc):
    """Some lights of a room are unreachable, the others work. Questions and
    commands about the room report both groups by name."""
    rooms = [r for r in h.rooms if len([e for e in h.of("light", r)
                                         if e.label != "__group__"]) >= 2]
    if not rooms:
        return None
    # rooms with a light named like the room are the hard case ("Kitchen" is
    # down, "Kitchen Left" is not), so they come up more often
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
    dpl = len(down) > 1 or pl_name(down[0].name)
    down_txt = f"{nd} {'aren' if dpl else 'isn'}'t reachable right now."
    msgs = base_msgs(h, loc)
    if named_q or R.random() < 0.5:
        # question
        user, kind = R.choice([(f"Is the light on {prep}?", "yn"),
                               (f"Is there still a light on {prep}?", "yn"),
                               (f"Which lights are on {prep}?", "which"),
                               (f"Are the lamps {prep} off?", "yn_off")])
        on = [e for e in up if e.state == "on"]
        off = [e for e in up if e.state != "on"]
        parts = []
        if on:
            parts.append(f"{_names(h, [e.eid for e in on])} {be(len(on) > 1 or pl_name(on[0].name))} on.")
        if off and (not on or R.random() < 0.5):
            parts.append(f"{_names(h, [e.eid for e in off])} {be(len(off) > 1 or pl_name(off[0].name))} off.")
        if not up:
            ans = f"{cap(down_txt)} So I can't tell whether any light is on there."
        else:
            if kind == "yn":
                parts[0] = ("Yes, " if on else "No, ") + parts[0]
            elif kind == "yn_off":
                # same wording as the "on?" question, only the yes/no flips;
                # a separate phrase made the model mix up both questions
                parts[0] = ("No, " if on else "Yes, ") + parts[0]
            ans = " ".join(parts) + " " + down_txt
        args = {"domain": "light", "area": room, "entity_ids": ids}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "error", "gold": [["get_state", args]]}
    # command for the whole room: the reachable lights switch
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    user = R.choice([f"Turn on the light {prep}", f"Lights on {prep}",
                     f"Switch on the light {prep}"] if on_cmd else
                    [f"Turn off the light {prep}", f"Lights off {prep}",
                     f"Switch off the light {prep}"])
    actions = [action(svc, ents, area=room)]
    done = [e.eid for e in up]
    word = "turned on" if on_cmd else "turned off"
    if up:
        did = _names(h, done)
        ans = R.choice([
            f"I've {word} {did}. {down_txt}",
            f"That only partly worked: {did} {be(len(up) > 1 or pl_name(up[0].name))} {'on' if on_cmd else 'off'}, "
            f"but {nd} {'aren' if dpl else 'isn'}'t reachable right now.",
            f"{down_txt} I've {word} the rest: {did}.",
        ])
    else:
        ans = f"{down_txt} So I couldn't switch the light {prep}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, S.control_result(
                 [f"Entity {', '.join(e.eid for e in down)} is unavailable"], done)),
             {"role": "assistant", "content": ans}]
    return msgs, {"cat": "error", "gold": [["control", {"actions": actions}]]}


def _named_room_light(h, loc):
    """A light carries the room's name ("Kitchen") next to other lights there.
    "the light in the kitchen" means all lights of the room; only the bare
    name ("Turn on Kitchen") means that single lamp."""
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
        user, kind = R.choice([(f"Is the light on {prep}?", "yn"),
                               (f"Is there a light on {prep}?", "yn"),
                               (f"Is any light still on {prep}?", "yn"),
                               (f"Which lights are on {prep}?", "which")])
        on = [e for e in ents if e.state == "on"]
        off = [e for e in ents if e.state != "on"]
        if on:
            ans = f"{_names(h, [e.eid for e in on])} {be(len(on) > 1 or pl_name(on[0].name))} on."
            if off and R.random() < 0.6:
                ans += f" {_names(h, [e.eid for e in off])} {be(len(off) > 1 or pl_name(off[0].name))} off."
            if kind == "yn":
                ans = "Yes, " + ans
        elif kind == "yn":
            ans = R.choice([f"No, {_names(h, [e.eid for e in off])} {be(len(off) > 1 or pl_name(off[0].name))} off.",
                            f"No, no light is on {prep}."])
        else:
            ans = f"No lights are on {prep} right now."
        args = {"domain": "light", "area": room, "entity_ids": [e.eid for e in ents]}
        call, cid = tool_call("get_state", args)
        msgs += [{"role": "user", "content": voice(user)}, call,
                 tool_msg(cid, h.state_of(ents)), {"role": "assistant", "content": ans}]
        must = [e.name for e in on] if on else ["no"]
        return msgs, {"cat": "question", "gold": [["get_state", args]],
                      "answer_contains": must}
    on_cmd = R.random() < 0.5
    svc = "light.turn_on" if on_cmd else "light.turn_off"
    adj, part = ("on", "turned on") if on_cmd else ("off", "turned off")
    if r < 0.8:
        # the room: every light in it
        user = R.choice([f"Turn on the light {prep}", f"Switch on the light {prep}",
                         f"Lights on {prep}", f"Turn the lights on {prep}",
                         f"Can you turn on the light {prep}?"] if on_cmd else
                        [f"Turn off the light {prep}", f"Switch off the light {prep}",
                         f"Lights off {prep}", f"Turn the lights off {prep}",
                         f"Can you turn off the light {prep}?"])
        targets = Ref(f"the light {prep}", f"the light {prep}", False, ents, room)
        conf = confirm(targets, adj, part)
    else:
        # the bare name: only the lamp called like the room
        user = R.choice([f"Turn on {room}", f"Switch on {room}", f"{room} on",
                         f"Turn on the {room} lamp"] if on_cmd else
                        [f"Turn off {room}", f"Switch off {room}", f"{room} off",
                         f"Turn off the {room} lamp"])
        targets = [named]
        conf = confirm(Ref(room, room, False, targets), adj, part)
    actions = [action(svc, targets)]
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call,
             tool_msg(cid, ok_result()), {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]]}


def s_no_areas(h, loc):
    """Home Assistant without any areas: a room cannot be resolved. The model
    says so and explains how to set up areas; device names and "all the
    lights" keep working."""
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
            user = R.choice([f"Turn on the light {prep}", f"Lights off {prep}",
                             f"Is the light on {prep}?", f"Which lights are on {prep}?",
                             f"Switch on the light {prep}", f"Turn off the lights {prep}"])
            what = "lights"
        else:
            user = R.choice([f"Lower the shutters {prep}", f"Shutters up {prep}",
                             f"Are the shutters {prep} closed?", f"Close the shutters {prep}"])
            what = "shutters"
        ans = R.choice([
            f"There are no areas set up in your Home Assistant, so I don't know which "
            f"{what} are {prep}. Please create the rooms under Settings → Areas, labels "
            f"& zones and assign the devices to them.",
            f"I can't tell what's {prep}, because no areas are set up in Home Assistant. "
            f"Please set up the rooms under Settings → Areas, labels & zones and assign "
            f"the devices."])
        if named:
            ans += f" Until then, you can also call a device by its name, for example “{named[0].name}”."
        msgs += [{"role": "user", "content": voice(user)}, {"role": "assistant", "content": ans}]
        return msgs, {"cat": "unknown", "gold": [], "no_history": True}
    on = R.random() < 0.5
    svc = "light.turn_on" if on else "light.turn_off"
    adj, part = ("on", "turned on") if on else ("off", "turned off")
    if r < 0.85 or h.group("light") is not None:
        # a device named directly still works
        e = R.choice(lights)
        user = R.choice([f"Turn on {e.name}", f"Switch on {e.name}", f"{e.name} on"] if on
                        else [f"Turn off {e.name}", f"Switch off {e.name}", f"{e.name} off"])
        actions = [action(svc, [e])]
        conf = R.choice([f"{e.name} is now {adj}.", f"I've {part} {e.name}."])
    else:
        # the whole house needs no areas
        user = R.choice(["Turn on all the lights", "Switch on all lamps"] if on
                        else ["Turn off all the lights", "Lights off everywhere"])
        actions = [action(svc, lights, area=S.ALL_AREAS)]
        conf = f"All the lights are {adj}."
    call, cid = tool_call("control", {"actions": actions})
    msgs += [{"role": "user", "content": voice(user)}, call, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": conf}]
    return msgs, {"cat": "command", "gold": [["control", {"actions": actions}]],
                  "no_history": True}


def partial_answer(h, ok_actions, failed_name, conf=None):
    """Both outcomes, always: what worked and what did not."""
    if any(len(a["entity_id"]) > 3 for a in ok_actions):
        # long lists: the normal confirmation ("All the lights are on.") is clearer
        if conf and conf != "Done.":
            return R.choice([f"{conf} But {failed_name} {isnt(failed_name)} reachable right now.",
                             f"{failed_name} {isnt(failed_name)} reachable right now. Otherwise: {conf}"])
        n = sum(len(a["entity_id"]) for a in ok_actions)
        svc = ok_actions[0]["service"].split(".", 1)[1]
        did = PARTICIPLE.get(svc, "switched {w}").format(w=f"{n} devices")
    else:
        did = describe_all(h, ok_actions)
    return R.choice([
        f"I've {did}. But {failed_name} {isnt(failed_name)} reachable right now.",
        f"That only partly worked: I've {did}, but {failed_name} {isnt(failed_name)} reachable right now.",
        f"{failed_name} {isnt(failed_name)} reachable right now. I took care of the rest: I've {did}.",
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
    v = cap(is_n(e.name))
    word = {"cover": "open", "lock": "locked"}.get(e.domain, "on")
    user = R.choice([f"{v} {e.name} {word}?", f"What's the status of {e.name}?",
                     f"What's {e.name} doing?",
                     f"{v} {e.name} switched on?" if word == "on" else f"{v} {e.name} {word} right now?"])
    args = {"entity_ids": [e.eid]}
    msgs = base_msgs(h, loc) + [{"role": "user", "content": voice(user)}]
    call, cid = tool_call("get_state", args)
    msgs += [call, tool_msg(cid, S.dumps([S.format_state(e.eid, e.name, "unavailable", {})])),
             {"role": "assistant", "content": R.choice([
                 f"{e.name} {isnt(e.name)} reachable right now, so I don't know "
                 f"{'their' if pl_name(e.name) else 'its'} current state.",
                 f"I can't tell: {e.name} {isnt(e.name)} reachable right now."])}]
    return msgs, {"cat": "error", "gold": [["get_state", args]]}


def s_followup(h, loc):
    """Two-turn: command, then 'and off again' / other room / brighter."""
    rooms = [r for r in h.rooms if h.of("light", r)]
    if len(rooms) < 2:
        return None
    r1, r2 = R.sample(rooms, 2)
    e1, e2 = h.of("light", r1), h.of("light", r2)
    p1, p2 = ROOM_BY_NAME[r1][1], ROOM_BY_NAME[r2][1]
    msgs = base_msgs(h, loc) + [{"role": "user",
                                 "content": voice(R.choice([f"Turn on the light {p1}",
                                                            f"Lights on {p1}"]))}]
    a1 = [action("light.turn_on", e1, area=r1)]
    c, cid = tool_call("control", {"actions": a1})
    msgs += [c, tool_msg(cid, ok_result()),
             {"role": "assistant", "content": f"The light {p1} is on."}]
    k = R.random()
    if k < 0.15:
        # user doubts the result -> check the real state instead of guessing
        u2 = R.choice(["No, it isn't", "Is it really on?", "Are you sure?",
                       "It's still dark here"])
        really_on = R.random() < 0.5
        for e in e1:
            e.state = "on" if really_on else "off"
        msgs.append({"role": "user", "content": voice(u2)})
        q = {"domain": "light", "area": r1, "entity_ids": [e.eid for e in e1]}
        c2, cid2 = tool_call("get_state", q)
        ans = (f"According to Home Assistant, the light {p1} is on." if really_on else
               f"According to Home Assistant, the light {p1} is still off. Should I turn it on again?")
        if u2 == "Is it really on?":
            ans = ("Yes. " if really_on else "No. ") + ans
        msgs += [c2, tool_msg(cid2, h.state_of(e1)), {"role": "assistant", "content": ans}]
        return msgs, {"cat": "followup",
                      "gold_last": [["get_state", q]],
                      "gold": [["control", {"actions": a1}]]}
    if k < 0.4:
        u2, a2, ans = R.choice(["And off again", "Turn it off again", "Actually, off again"]), \
            [action("light.turn_off", e1, area=r1)], f"The light {p1} is off again."
    elif k < 0.72:
        u2, a2, ans = R.choice([f"And {p2}?", f"{cap(p2)} too", f"And {p2} too, please"]), \
            [action("light.turn_on", e2, area=r2)], f"The light {p2} is on too."
    else:
        u2, a2, ans = R.choice(["A bit dimmer", "A little brighter", "To 30 percent"]), \
            None, None
        if "dimmer" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": -20}, area=r1)]
            ans = f"The light {p1} is dimmer now."
        elif "brighter" in u2:
            a2 = [action("light.turn_on", e1, {"brightness_step_pct": 20}, area=r1)]
            ans = f"The light {p1} is brighter now."
        else:
            a2 = [action("light.turn_on", e1, {"brightness_pct": 30}, area=r1)]
            ans = f"The light {p1} is now at 30 percent."
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
# what was done, {w} = the devices
PARTICIPLE = {"turn_on": "turned on {w}", "turn_off": "turned off {w}",
              "open_cover": "opened {w}", "close_cover": "closed {w}",
              "stop_cover": "stopped {w}", "lock": "locked {w}", "unlock": "unlocked {w}",
              "media_play": "resumed {w}", "media_pause": "paused {w}",
              "media_stop": "stopped {w}",
              "media_next_track": "skipped to the next track on {w}",
              "media_previous_track": "switched {w} to the previous track",
              "volume_up": "turned up {w}", "volume_down": "turned down {w}",
              "start": "started {w}", "pause": "paused {w}",
              "return_to_base": "sent {w} back to the dock",
              "start_mowing": "started {w}", "dock": "sent {w} back to the dock",
              "open_valve": "opened {w}", "close_valve": "closed {w}", "press": "pressed {w}",
              "alarm_arm_home": "armed {w} in home mode", "alarm_arm_away": "armed {w}",
              "alarm_arm_night": "armed {w} in night mode",
              "alarm_disarm": "disarmed {w}"}
# the same as a question verb: "Which light did you turn off?"
QVERB = {"turn_on": "turn on", "turn_off": "turn off", "open_cover": "open",
         "close_cover": "close", "stop_cover": "stop", "lock": "lock", "unlock": "unlock",
         "media_play": "resume", "media_pause": "pause", "media_stop": "stop",
         "media_next_track": "skip", "media_previous_track": "skip back on",
         "volume_up": "turn up", "volume_down": "turn down", "start": "start",
         "pause": "pause", "return_to_base": "send back", "start_mowing": "start",
         "dock": "send back", "open_valve": "open", "close_valve": "close", "press": "press",
         "alarm_arm_home": "arm", "alarm_arm_away": "arm", "alarm_arm_night": "arm",
         "alarm_disarm": "disarm"}
DOMAIN_NOUN = {"light": ("Which light", "Which lights"),
               "cover": ("Which shutter", "Which shutters"),
               "climate": ("Which thermostat", "Which thermostats"),
               "switch": ("Which device", "Which devices"),
               "media_player": ("Which speaker", "Which devices"),
               "fan": ("Which fan", "Which fans")}


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
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def describe(h, a, by_name=False):
    """'turned on Hallway Ceiling Light' for one action."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if a.get("area") == S.ALL_AREAS and not by_name:
        who = "all the lights"
    elif a.get("area") and not by_name:
        who = f"the light {ROOM_BY_NAME[a['area']][1]}"
    else:
        who = _names(h, a["entity_id"])
    if "brightness_pct" in data:
        return f"set {who} to {data['brightness_pct']} percent"
    if "brightness_step_pct" in data:
        return f"made {who} {'brighter' if data['brightness_step_pct'] > 0 else 'dimmer'}"
    if "color_name" in data or "color_temp_kelvin" in data:
        return f"changed the color of {who}"
    if svc == "set_temperature":
        return f"set {who} to {en_num(data['temperature'])} degrees"
    if svc == "set_cover_position":
        return f"moved {who} to {data['position']} percent"
    if svc in ("set_percentage",) or "percentage" in data:
        return f"set {who} to {data.get('percentage')} percent"
    if svc == "volume_set":
        return f"set the volume of {who} to {round(data['volume_level'] * 100)} percent"
    if dom == "todo" and "item" in data:
        lst = LIST_PHRASES.get(who, who)
        item = join_en([f"“{i}”" if " " in i and i.split(" ", 1)[0] in TASK_VERBS else i
                        for i in data.get("items") or [data["item"]]])
        if svc == "add_item":
            return f"added {item} to {lst}"
        if svc == "complete_item":
            return f"checked off {item} on {lst}"
        return f"removed {item} from {lst}"
    if svc == "search_and_play":
        return f"started {data.get('query', 'music')} on {who}"
    if svc == "seek_relative":
        n = int(data.get("seconds") or 0)
        word = "a minute" if abs(n) == 60 else f"{abs(n)} seconds"
        return f"{'skipped ahead' if n >= 0 else 'went back'} {word} on {who}"
    if svc == "shuffle_set":
        return f"turned shuffle {'on' if data.get('shuffle') else 'off'} on {who}"
    if svc == "repeat_set":
        return f"set repeat to {data.get('repeat')} on {who}"
    if svc == "volume_mute":
        return f"{'muted' if data.get('is_volume_muted') else 'unmuted'} {who}"
    if svc == "set_humidity":
        return f"set {who} to {data.get('humidity')} percent humidity"
    if svc == "set_valve_position":
        return f"opened {who} to {data.get('position')} percent"
    if dom == "scene":
        return f"activated the scene {who}"
    if dom == "script":
        return f"started {who}"
    return PARTICIPLE.get(svc, "switched {w}").format(w=who)


def merge_actions(actions):
    """Consecutive list items for the same list become one action, so the
    answer says 'added milk and eggs to the shopping list'."""
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
    """'turned on X, set Y and added Z' for several actions."""
    return join_en([describe(h, a) for a in merge_actions(actions)])


# filled by assist.py: list name -> "the shopping list"; first words of tasks
LIST_PHRASES: dict = {}
TASK_VERBS: set = set()


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
    """'Which light did you turn off?' -> answered from the history."""
    if not c.last:
        return False
    a = c.last[-1]
    dom, svc = a["service"].split(".", 1)
    ids = a["entity_id"]
    if dom in DOMAIN_NOUN and svc in PARTICIPLE and not a.get("data"):
        sg, pl = DOMAIN_NOUN[dom]
        q = (pl if len(ids) > 1 else sg) + f" did you {QVERB[svc]}?"
        user = R.choice([q, q, "What did you just do?", "Which device was that?"])
    else:
        user = R.choice(["What did you just do?", "What exactly did you do?"])
    c.user(user)
    # the lamps an area stood for are known from "done" in the tool result
    parts = [describe(c.h, x, by_name=True) for x in merge_actions(c.last)]
    c.say(R.choice(["I've ", "I just "]) + " and ".join(parts) + ".")
    return True


def turn_summary(c):
    """'What have you done so far?' over the whole conversation."""
    if len(c.done) < 2:
        return False
    # "Which thermostats did you change?" after the bathroom + "And in the
    # kitchen too": every action on that device type, not only the last one
    by_dom = {}
    for x in c.done:
        by_dom.setdefault(x["service"].split(".")[0], []).append(x)
    multi = [d for d, xs in by_dom.items() if len(xs) >= 2 and d in SUMMARY_ASK]
    if multi and R.random() < 0.6:
        d = R.choice(multi)
        c.user(R.choice(SUMMARY_ASK[d]))
        parts = [describe(c.h, x) for x in merge_actions(by_dom[d])]
    else:
        c.user(R.choice(["What have you done so far?", "What did you switch so far?",
                         "Give me a quick summary of what you did"]))
        parts = [describe(c.h, x) for x in merge_actions(c.done)]
    c.say("I've " + join_en(parts) + ".")
    return True


SUMMARY_ASK = {
    "climate": ["Which thermostats did you change?", "Which radiators did you adjust?",
                "What did you change on the heating?", "Which thermostats did you set?"],
    "light": ["Which lights did you switch?", "Which lamps did you switch?",
              "What did you do with the lights?"],
    "cover": ["Which shutters did you move?", "Which blinds did you move?",
              "What did you do with the shutters?"],
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
    c.user(R.choice(["Undo that", "Actually, no", "Change it back",
                     "Take that back"] +
                    (["And off again", "Turn it off again"]
                     if inv[0]["service"].endswith("turn_off") else []) +
                    (["And on again"] if inv[0]["service"].endswith("turn_on") else [])))
    c.control(inv, "Okay, I've " + describe_all(c.h, inv) + ".")
    return True


def turn_correct(c):
    """'No, I meant in the bedroom' -> undo + the right device."""
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
        # "I meant in the dining room": every device of that kind in the room
        c.user(R.choice([f"No, I meant {prep}", f"Wrong room, {prep}",
                         f"No, {prep}", f"Sorry, I meant {prep}"]))
        new_ids = [e.eid for e in c.h.of(dom, o.area, plain=dom == "cover")
                   if e.state != "unavailable"]
        by_room = len(new_ids) == len(c.h.of(dom, o.area, plain=dom == "cover"))
    else:
        c.user(R.choice([f"Not that one, {o.name}", f"I meant {o.name}"]))
        new_ids = [o.eid]
        by_room = False
    undo = {"service": f"{dom}.{INVERSE[svc]}", "entity_id": cur_ids}
    if "area" in a:
        undo = {"service": undo["service"], "area": a["area"], "entity_id": cur_ids}
    new = {"service": a["service"], "entity_id": new_ids}
    if dom in S.AREA_DOMAINS and by_room:
        new = {"service": a["service"], "area": o.area, "entity_id": new_ids}
    acts = [undo, new]
    c.control(acts, f"Okay, I've {describe(c.h, acts[0])} and {describe(c.h, acts[1])}.")
    return True


def turn_correct_recap(c):
    """Correction followed by "What have you done?": the recap lists
    exactly the executed actions."""
    if not turn_correct(c):
        return False
    if R.random() < 0.7:
        return turn_summary(c)
    return turn_which(c)


def turn_also(c):
    """'In the bathroom too' -> same service on the same device type in another room."""
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
    c.user(R.choice([f"And {prep} too", f"{cap(prep)} too", f"Same thing {prep}"]))
    act = {"service": a["service"],
           **({"area": r} if dom in S.AREA_DOMAINS and len(ents) == len(c.h.of(dom, r)) else {}),
           "entity_id": [e.eid for e in ents]}
    if a.get("data"):
        act["data"] = a["data"]
    c.control([act], R.choice(["Okay, I've ", "Done, I've "])
              + describe(c.h, act) + ".")
    return True


def turn_brightness(c):
    if not c.last or len(c.last) != 1 or c.last[0]["service"] != "light.turn_on":
        return False
    ids = c.last[0]["entity_id"]
    k = R.random()
    if k < 0.4:
        user, data = R.choice(["A bit brighter", "Even brighter", "Make it brighter"]), \
            {"brightness_step_pct": 20}
    elif k < 0.8:
        user, data = R.choice(["A bit dimmer", "Even dimmer", "Make it darker"]), \
            {"brightness_step_pct": -20}
    else:
        p = R.choice([10, 30, 50, 70])
        user, data = R.choice([f"To {p} percent", f"Set it to {p} percent"]), {"brightness_pct": p}
    act = {"service": "light.turn_on",
           **({"area": c.last[0]["area"]} if "area" in c.last[0] else {}),
           "entity_id": ids, "data": data}
    c.user(user)
    c.control([act], "Okay, I've " + describe(c.h, act) + ".")
    return True


def turn_check(c):
    """'Is it on now?' -> get_state on the last devices."""
    if not c.last:
        return False
    ids = c.last[-1]["entity_id"]
    area = c.last[-1].get("area")
    ents = [e for e in c.h.ents if e.eid in ids]
    if not ents or ents[0].domain not in ("light", "switch", "cover", "lock", "fan", "media_player"):
        return False
    pl = len(ents) > 1 or ents[0].plural or pl_name(ents[0].name)
    pron, verb = ("they", "Are") if pl else ("it", "Is")
    pairs = {"cover": [("open", "open"), ("closed", "closed"), ("down", "closed"),
                       ("up", "open")],
             "lock": [("locked", "locked"), ("unlocked", "unlocked")]}
    word_q, asked = R.choice(pairs.get(ents[0].domain, [("on", "on"), ("off", "off")]))
    question = f"{verb} {pron} {word_q} now?"
    words = {_state_word(e) for e in ents}
    states = join_en([f"{e.name} {'are' if e.plural else is_n(e.name)} {_state_word(e)}"
                      for e in ents])
    if R.random() < 0.6 and len(words) == 1 and words <= {
            "on", "off", "open", "closed", "locked", "unlocked"}:
        c.user(question)
        word = words.pop()
        yes = "Yes" if word == asked else "No"
        c.get_state(ents, f"{yes}, according to Home Assistant, {states}.", area)
    else:
        c.user(R.choice(["Did that work?", "Are you sure?", "Can you check that?"]))
        c.get_state(ents, f"According to Home Assistant, {states}.", area)
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
    # one mixed missing-type case plus the TV case (see the German generator)
    cases += [("unknown:missing_type", s_missing_type, {"mtype": "other"}),
              ("unknown:missing_media_player", s_missing_type, {"mtype": "media_player"})]
    seen = set()
    for fn, _ in NEXT:
        if fn not in seen:
            seen.add(fn)
            cases.append((f"conversation:{fn.__name__}", s_conversation, {"conv_turn": fn}))
    # "Lights on in the kitchen" -> "No, I meant in the dining room": undo every
    # kitchen light, switch every dining room light
    cases.append(("conversation:room_correct", s_conversation,
                  {"conv_turn": turn_correct, "intent": i_light_room_phrase}))
    cases.append(("conversation:correct_recap", s_conversation,
                  {"conv_turn": turn_correct_recap, "intent": i_light_room_phrase}))
    for hook in CASE_HOOKS:
        cases += hook()
    return cases


SHORT_OK = ["Done.", "Okay.", "All set.", "Got it."]
BARE = {"Done.", "Okay.", "All set.", "Got it.", "Paused.", "Volume up.",
        "Volume down.", "Next track.", "Stopped.", "Previous track."}
DESCRIBED = {"turn_on", "turn_off", "open_cover", "close_cover", "stop_cover", "lock", "unlock",
             "media_play", "media_pause", "media_stop", "media_next_track",
             "media_previous_track", "volume_up", "volume_down", "start", "pause",
             "return_to_base", "start_mowing", "dock", "open_valve", "close_valve", "press",
             "set_temperature", "set_cover_position", "set_percentage", "volume_set",
             # assist services, worded by describe()
             "add_item", "complete_item", "remove_item", "search_and_play", "seek_relative",
             "shuffle_set",
             "repeat_set", "volume_mute", "set_humidity", "set_valve_position"}


def apply_verbosity(h, msgs):
    """Answers after a successful control call follow the answer length:
    short -> a word or two, long/detailed -> never a bare 'Done.'."""
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
        m["content"] = "I've " + describe_all(h, acts) + "."


# "1 degrees" -> "1 degree"
ONE_UNIT = re.compile(r"(?<![\d.,])(-?1) (degree|watt|minute|hour|second)s\b")


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
            for m in msgs[1:]:
                if m["role"] in ("user", "assistant") and m.get("content"):
                    m["content"] = ONE_UNIT.sub(r"\1 \2", m["content"])
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
        low_n = min(count[c[0]] for c in active)
        case = R.choice([c for c in active if count[c[0]] == low_n])
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
