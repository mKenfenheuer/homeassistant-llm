"""Dutch samples for the Home Assistant Assist built-ins.

Port of src/gen/de/assist.py. Music (search and play, the media library of a
player, shuffle, repeat, mute), timers on the voice device (start, named, delayed commands, status, cancel,
pause, resume, add, subtract), to-do lists, weather with forecast, time and
date, broadcasts, humidifiers and valve positions.

Registers itself in core: entities through POPULATE_HOOKS, device commands
through COMMANDS (they get a case each and take part in compound commands and
conversations), conversation turns through NEXT, the rest through CASE_HOOKS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import core as C
from .core import R, S, action, base_msgs, cap, low, nl_num, tool_call, tool_msg, voice

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


def join_nl(items, word="en"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


# verbs after which a time phrase fits ("Doe over 10 minuten het licht uit");
# the same templates qualify as in German
VERBS = ("Doe", "Zet", "Schakel", "Open", "Sluit", "Start", "Pauzeer", "Activeer", "Dim",
         "Speel", "Laat", "Maak", "Hervat")


def lower_first(text):
    """An imperative moves behind another sentence part; names keep their capital."""
    return low(text) if text.split(" ", 1)[0] in C.IMPERATIVES else text


def sample(msgs, cat, gold, **meta):
    out = {"cat": cat, "gold": gold}
    out.update(meta)
    return msgs, out


def call_turn(name, args, result, answer):
    """[tool call, tool result, answer] messages."""
    call, cid = tool_call(name, args)
    return [call, tool_msg(cid, result), {"role": "assistant", "content": answer}]


def user(text):
    return {"role": "user", "content": voice(text)}


def say(text):
    return {"role": "assistant", "content": text}


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
ARTISTS = ["Queen", "André Hazes", "Coldplay", "Marco Borsato", "BLØF", "Golden Earring",
           "Doe Maar", "Ed Sheeran", "Adele", "Taylor Swift", "Racoon", "Frans Bauer",
           "Davina Michelle", "ABBA", "The Beatles", "Suzan & Freek", "Nirvana", "Beyoncé",
           "Guus Meeuwis", "Tiësto", "Billie Eilish", "Linkin Park", "Typhoon", "Froukje",
           "Kensington", "Mozart", "Ludovico Einaudi", "Within Temptation", "De Dijk",
           "Pink Floyd", "Daft Punk", "Dua Lipa", "Kinderen voor Kinderen",
           "Armin van Buuren", "Duncan Laurence"]
TRACKS = [("Bohemian Rhapsody", "Queen"), ("Zij gelooft in mij", "André Hazes"),
          ("Yellow", "Coldplay"), ("Dromen zijn bedrog", "Marco Borsato"),
          ("Aan de kust", "BLØF"), ("Radar Love", "Golden Earring"),
          ("Smoorverliefd", "Doe Maar"), ("Shape of You", "Ed Sheeran"),
          ("Hello", "Adele"), ("Arcade", "Duncan Laurence"), ("Dancing Queen", "ABBA"),
          ("Let It Be", "The Beatles"), ("Het is een nacht", "Guus Meeuwis"),
          ("Bad Guy", "Billie Eilish"), ("In the End", "Linkin Park"),
          ("Laat me", "Ramses Shaffy"), ("Kom van dat dak af", "Peter Koelewijn"),
          ("Mag ik dan bij jou", "Claudia de Breij"), ("Ik neem je mee", "Gers Pardoel"),
          ("Levitating", "Dua Lipa"), ("Get Lucky", "Daft Punk"),
          ("Wish You Were Here", "Pink Floyd")]
ALBUMS = [("A Night at the Opera", "Queen"), ("Parachutes", "Coldplay"),
          ("Moontan", "Golden Earring"), ("Abbey Road", "The Beatles"),
          ("Divide", "Ed Sheeran"), ("21", "Adele"), ("Umoja", "BLØF"),
          ("Nevermind", "Nirvana"), ("Back to Black", "Amy Winehouse"), ("Random Access Memories", "Daft Punk"),
          ("The Dark Side of the Moon", "Pink Floyd"), ("Folklore", "Taylor Swift")]
PLAYLISTS = ["Chill", "Koken", "Feest", "Workout", "Ontspanning", "Zomerhits",
             "Favorieten", "Slapen", "Rock Classics", "Kinderliedjes",
             "Kerstmuziek", "Focus", "Discover Weekly", "Roadtrip", "Jaren 80"]
STATIONS = ["NPO Radio 1", "NPO Radio 2", "NPO 3FM", "NPO Klassiek", "Qmusic", "Radio 538",
            "Sky Radio", "Radio 10", "100% NL", "Radio Veronica", "SLAM!", "Studio Brussel",
            "Joe", "Sublime", "BNR Nieuwsradio", "Omroep Brabant"]
GENRES = ["jazz", "klassieke muziek", "rock", "ontspannende muziek", "kindermuziek",
          "kerstmuziek", "lo-fi", "Nederlandstalige muziek", "hiphop", "muziek om bij te slapen",
          "muziek om bij te koken", "hits uit de jaren 80", "popmuziek", "smartlappen"]

SPEAKER_NAMES = ["Sonos {room}", "{room} Speaker", "Echo {room}", "HomePod {room}",
                 "Speaker {room}", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "{room} Luidspreker"]
MUSIC_LABELS = ("Speaker", "Radio", "Soundbar")

GROCERIES = ["melk", "eieren", "boter", "brood", "kaas", "appels", "bananen", "tomaten",
             "koffie", "pasta", "rijst", "yoghurt", "bloem", "suiker", "wc-papier",
             "afwasmiddel", "wasmiddel", "kattenvoer", "havermelk", "uien",
             "aardappelen", "bruiswater", "sinaasappelsap", "vuilniszakken", "havermout",
             "komkommer", "paprika", "gehakt", "slagroom",
             "hagelslag", "pindakaas", "stroopwafels"]
LISTS = {  # name: (phrase, items, slugs)
    "Boodschappenlijst": ("de boodschappenlijst", GROCERIES,
                          ["shopping_list", "boodschappenlijst"]),
    "Taken": ("de takenlijst",
              ["vuilnis buitenzetten", "belastingaangifte doen", "auto wassen",
               "gras maaien", "oma bellen", "ramen lappen", "zolder opruimen",
               "pakketje ophalen", "afspraak bij de huisarts maken", "planten water geven"],
              ["taken", "tasks"]),
    "Bouwmarkt": ("de bouwmarktlijst",
                  ["schroeven", "pluggen", "muurverf", "kit", "tape",
                   "ledlampen", "potgrond", "tie-wraps", "schuurpapier"],
                  ["bouwmarkt", "hardware_store"]),
    "Drogisterij": ("de drogisterijlijst",
                    ["tandpasta", "shampoo", "douchegel", "zonnebrand", "deodorant",
                     "wattenschijfjes", "zakdoekjes", "handcrème"],
                    ["drogisterij", "drugstore"]),
    "Paklijst": ("de paklijst",
                 ["oplader", "zonnebril", "paspoort", "zwembroek", "tandenborstel",
                  "regenjas", "koptelefoon", "verloopstekker"],
                 ["paklijst", "packing_list"]),
}
SHOPPING, TASKS = "Boodschappenlijst", "Taken"

CITIES = ["Utrecht", "Groningen", "Zwolle", "Leiden", "Haarlem", "Nijmegen", "Breda",
          "Amersfoort", "Gent", "Maastricht"]
WEATHER_NAMES = ["Forecast Home", "Thuis", "Weer", "OpenWeatherMap", "Buienradar",
                 "Met.no", "Weersverwachting"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "Morgen ..." + fragment (verb second: "Morgen wordt het zonnig")
FORECAST_FRAG = {
    "sunny": "wordt het zonnig", "clear-night": "blijft het helder", "cloudy": "wordt het bewolkt",
    "partlycloudy": "wordt het half bewolkt", "rainy": "gaat het regenen",
    "pouring": "gaat het hard regenen", "snowy": "gaat het sneeuwen",
    "snowy-rainy": "valt er natte sneeuw", "fog": "wordt het mistig", "hail": "gaat het hagelen",
    "lightning": "komt er onweer", "lightning-rainy": "komt er onweer met regen",
    "windy": "wordt het winderig", "windy-variant": "wordt het winderig en bewolkt",
    "exceptional": "wordt er noodweer verwacht",
}
NOW_SENT = {
    "sunny": "Het is zonnig", "clear-night": "Het is helder", "cloudy": "Het is bewolkt",
    "partlycloudy": "Het is half bewolkt", "rainy": "Het regent",
    "pouring": "Het regent hard", "snowy": "Het sneeuwt", "snowy-rainy": "Er valt natte sneeuw",
    "fog": "Het is mistig", "hail": "Het hagelt", "lightning": "Het onweert",
    "lightning-rainy": "Het onweert en regent", "windy": "Het is winderig",
    "windy-variant": "Het is winderig en bewolkt", "exceptional": "Er is noodweer",
}

# (name, "voor de ...", compound noun)
TIMER_NAMES = [("Pizza", "voor de pizza", "pizzatimer"), ("Pasta", "voor de pasta", "pastatimer"),
               ("Eieren", "voor de eieren", "eierwekker"), ("Thee", "voor de thee", "theetimer"),
               ("Taart", "voor de taart", "taarttimer"), ("Was", "voor de was", "wastimer"),
               ("Brood", "voor het brood", "broodtimer"), ("Rijst", "voor de rijst", "rijsttimer"),
               ("Aardappels", "voor de aardappels", "aardappeltimer"),
               ("Oven", "voor de oven", "oventimer"), ("Pauze", "voor de pauze", "pauzetimer"),
               ("Parkeren", "voor het parkeren", "parkeertimer"),
               ("Huiswerk", "voor het huiswerk", "huiswerktimer"),
               ("Biefstuk", "voor de biefstuk", "biefstuktimer"),
               ("Lasagne", "voor de lasagne", "lasagnetimer"), ("Friet", "voor de friet", "friettimer"),
               ("Wafels", "voor de wafels", "wafeltimer"),
               ("Groente", "voor de groente", "groentetimer")]
TIMER_FOR = {n: f for n, f, _ in TIMER_NAMES}
TIMER_COMP = {n: c for n, _, c in TIMER_NAMES}
NUM_WORDS = {2: "twee", 3: "drie", 4: "vier", 5: "vijf", 6: "zes", 7: "zeven",
             8: "acht", 10: "tien", 12: "twaalf", 15: "vijftien", 20: "twintig",
             30: "dertig"}

# (reported speech "dat ...", message)
BROADCASTS = [("het eten klaar is", "Het eten is klaar"),
              ("we over vijf minuten vertrekken", "We vertrekken over vijf minuten"),
              ("het tijd is om te gaan slapen", "Het is tijd om te gaan slapen"),
              ("de pizza er is", "De pizza is er"),
              ("er iemand aan de deur is", "Er is iemand aan de deur"),
              ("iedereen aan tafel moet komen", "Iedereen aan tafel"),
              ("de film zo begint", "De film begint zo"),
              ("ik zo thuiskom", "Ik kom zo thuis"),
              ("de container nog aan de straat moet", "De container moet nog aan de straat"),
              ("we bezoek krijgen", "We krijgen bezoek"),
              ("het ontbijt klaar is", "Het ontbijt is klaar"),
              ("de taart klaar is", "De taart is klaar"),
              ("iedereen naar beneden moet komen", "Iedereen naar beneden komen"),
              ("we zo gaan wandelen", "We gaan zo wandelen"),
              ("het nu tijd is voor huiswerk", "Het is nu tijd voor huiswerk")]

# items that take a plural verb ("bananen staan")
PLURAL_ITEMS = {"eieren", "appels", "bananen", "tomaten", "uien", "aardappelen",
                "vuilniszakken", "stroopwafels", "schroeven", "pluggen", "ledlampen",
                "tie-wraps", "wattenschijfjes", "zakdoekjes"}


def staat(item):
    return "staan" if item in PLURAL_ITEMS else "staat"


def is_item(item):
    return "zijn" if item in PLURAL_ITEMS else "is"


def with_target(sentence, where):
    """Put the player before a closing verb or particle:
    'Ik wil Queen horen' -> 'Ik wil Queen op Sonos horen'."""
    for end in (" horen", " luisteren", " aan"):
        if where and sentence.endswith(end):
            return sentence[:-len(end)] + where + end
    return sentence + where


def rand_dt():
    base = datetime(2026, 1, 1) + timedelta(days=R.randint(0, 730))
    return base.replace(hour=R.randint(0, 23), minute=R.randint(0, 59))


# --------------------------------------------------------------------------
# house extensions
# --------------------------------------------------------------------------
def music_players(h):
    return [e for e in h.of("media_player")
            if e.label in MUSIC_LABELS and e.state != "unavailable"]


def add_speaker(h, room=None):
    room = room or R.choice([r for r in h.rooms if r not in C.OUTDOOR] or h.rooms)
    for _ in range(5):
        name = R.choice(SPEAKER_NAMES).format(room=room)
        if name not in h.used_names:
            break
    else:
        return None
    playing = R.random() < 0.3
    attrs = {"volume_level": R.choice([0.2, 0.3, 0.4, 0.5, 0.6])}
    if playing:
        title, artist = R.choice(TRACKS)
        attrs.update(media_title=title, media_artist=artist)
    return h.add("media_player", room, "Speaker", "de", [C.ascii_slug(name), "speaker"],
                 name=name, state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs)


def add_list(h, name, items=None):
    phrase, pool, slugs = LISTS[name]
    if items is None:
        items = R.sample(pool, R.choice([0, 1, 2, 3, 3, 4, 5, 6]))
    e = h.add("todo", None, name, "de", slugs, name=name, state=str(len(items)),
              attrs={"items": list(items)})
    eid = f"todo.{R.choice(slugs)}"
    if eid not in h.used_ids:
        h.used_ids.discard(e.eid)
        e.eid = eid
        h.used_ids.add(eid)
    return e


SNOW = ("snowy", "snowy-rainy", "hail")


def fitting_conditions(temp):
    """No snow in summer, no thunderstorm in deep frost."""
    conds = [c for c in S.WEATHER_CONDITIONS if c not in ("clear-night", "exceptional")]
    if temp > 4:
        conds = [c for c in conds if c not in ("snowy", "snowy-rainy")]
    if temp > 20:
        conds = [c for c in conds if c != "hail"]
    if temp < 3:
        conds = [c for c in conds if c not in ("lightning", "lightning-rainy")]
    if temp < -1:
        conds = [c for c in conds if c not in ("rainy", "pouring")]
    return conds


def forecast_entries(today, cond_now, t_now):
    out = []
    for i in range(S.FORECAST_DAYS):
        hi = round(t_now + R.uniform(-1, 5)) if i == 0 else round(t_now + R.uniform(-7, 7))
        if i == 0 and (cond_now in RAINY or R.random() < 0.6) and cond_now != "clear-night":
            cond = cond_now
        else:
            cond = R.choice(fitting_conditions(hi))
        f = {"datetime": (today + timedelta(days=i)).isoformat(), "condition": cond,
             "temperature": hi}
        if R.random() < 0.85:
            f["templow"] = hi - R.randint(3, 11)
        if R.random() < 0.75:
            f["precipitation_probability"] = R.choice(
                [70, 80, 90, 100, 60] if cond in RAINY + SNOW else [0, 0, 5, 10, 20, 30])
        out.append(f)
    return out


STAY = {
    "sunny": "blijft het zonnig", "clear-night": "blijft het helder", "cloudy": "blijft het bewolkt",
    "partlycloudy": "blijft het half bewolkt", "rainy": "blijft het regenen",
    "pouring": "blijft het hard regenen", "snowy": "blijft het sneeuwen",
    "snowy-rainy": "blijft er natte sneeuw vallen", "fog": "blijft het mistig",
    "hail": "blijft het hagelen", "lightning": "blijft het onweren",
    "lightning-rainy": "blijft het onweren en regenen", "windy": "blijft het winderig",
    "windy-variant": "blijft het winderig en bewolkt", "exceptional": "houdt het noodweer aan",
}


def hourly_entries(now, cond, temp):
    """Hourly forecast until midnight; the condition changes at most once,
    the temperature falls towards the night."""
    out = []
    change_at = R.randint(now.hour + 1, 24) if R.random() < 0.55 else 99
    later = R.choice(fitting_conditions(temp))
    t = temp
    for hr in range(now.hour + 1, 24):
        c = cond if hr < change_at else later
        if c == "sunny" and hr >= 20:
            c = "clear-night"
        if c == "clear-night" and hr < 19:
            c = "sunny"
        t += R.uniform(-0.3, 0.6) if hr < 15 else R.uniform(-1.4, 0.2)
        f = {"datetime": now.replace(hour=hr, minute=0).isoformat(), "condition": c,
             "temperature": round(t)}
        if R.random() < 0.8:
            f["precipitation_probability"] = R.choice([60, 70, 80, 90]) if c in RAINY + SNOW \
                else R.choice([0, 0, 5, 10, 20])
        out.append(f)
    return out


def add_weather(h):
    name = R.choice(WEATHER_NAMES + CITIES)
    if name in h.used_names:
        return None
    now = rand_dt()
    temp = round(R.uniform(-8, 33), 1)
    # no rain below zero: snow instead
    cond = R.choice(fitting_conditions(temp) + ["clear-night", "sunny", "cloudy",
                                                "partlycloudy"]
                    + (["rainy"] if temp > -1 else ["snowy"]))
    if cond == "clear-night" and 7 <= now.hour <= 18:
        cond = "sunny"
    entries = forecast_entries(now.date(), cond, temp)
    hourly = hourly_entries(now, cond, temp)
    later = S.format_hourly(hourly, now, LANG)
    raw_by_time = {datetime.fromisoformat(f["datetime"]).strftime("%H:00"): f["condition"]
                   for f in hourly}
    attrs = {"temperature": temp, "humidity": R.randint(30, 98),
             "later_today": later or None,
             "_later_raw": [raw_by_time[x["time"]] for x in later],
             "wind_speed": round(R.uniform(0, 45), 1), "wind_speed_unit": "km/h",
             "forecast": S.format_forecast(entries, now.date(), LANG),
             "_entries": entries}
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Buiten", "Tuin")])
    slug = "forecast_home" if name in ("Forecast Home", "Thuis") else C.ascii_slug(name)
    e = h.add("weather", room, "Weer", "het", [slug], name=name, state=cond, attrs=attrs)
    eid = f"weather.{slug}"
    if eid not in h.used_ids:
        h.used_ids.discard(e.eid)
        e.eid = eid
        h.used_ids.add(eid)
    return e


def populate(h):
    if R.random() < 0.4:
        for _ in range(R.choice([1, 1, 2])):
            add_speaker(h)
    if R.random() < 0.5:
        add_weather(h)
    if R.random() < 0.55:
        add_list(h, SHOPPING)
        for name in R.sample([n for n in LISTS if n != SHOPPING], R.choice([0, 0, 1, 2])):
            add_list(h, name)
    elif R.random() < 0.15:
        add_list(h, R.choice([n for n in LISTS if n != SHOPPING]))
    if R.random() < 0.12:
        rooms = [r for r in h.rooms if r in ("Slaapkamer", "Kinderkamer", "Woonkamer", "Kantoor")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "Luchtbevochtiger", "de",
                  ["luchtbevochtiger", "humidifier"],
                  state="on" if on else "off",
                  attrs={"humidity": R.choice([40, 45, 50, 55]),
                         "current_humidity": R.randint(30, 60)})


C.POPULATE_HOOKS.append(populate)


# --------------------------------------------------------------------------
# music
# --------------------------------------------------------------------------
def music_request():
    """-> (sentence without target, data, description for the answer)."""
    k = R.choice(["artist", "artist", "track", "track", "album", "playlist", "radio",
                  "radio", "genre", "plain", "library"])
    if k == "library":
        # no title: the integration plays the favorites of the library
        return R.choice(["Speel muziek", "Speel iets", "Zet wat muziek op",
                         "Speel maar wat muziek", "Speel iets uit mijn bibliotheek",
                         "Ik wil muziek luisteren", "Zet muziek aan", "Speel iets leuks"]), {}, \
            "muziek"
    if k == "artist":
        a = R.choice(ARTISTS)
        u = R.choice([f"Speel muziek van {a}", f"Speel iets van {a}", f"Ik wil {a} horen",
                      f"Speel nummers van {a}", f"Draai iets van {a}",
                      f"Ik wil naar muziek van {a} luisteren"])
        return u, {"query": a, "media_type": "artist"}, f"muziek van {a}"
    if k == "track":
        t, a = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Speel {t} van {a}", f"Speel het nummer {t} van {a}",
                          f"Ik wil {t} van {a} horen", f"Draai {t} van {a}"])
            return u, {"query": t, "media_type": "track", "artist": a}, f"{t} van {a}"
        u = R.choice([f"Speel het nummer {t}", f"Speel het liedje {t}", f"Draai het nummer {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al, a = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Speel het album {al} van {a}", f"Draai het album {al} van {a}"]),
                    {"query": al, "media_type": "album", "artist": a}, f"het album {al}")
        return f"Speel het album {al}", {"query": al, "media_type": "album"}, f"het album {al}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Speel mijn playlist {p}", f"Speel de playlist {p}",
                      f"Start de playlist {p}", f"Speel de {p}-playlist"])
        return u, {"query": p, "media_type": "playlist"}, f"de playlist {p}"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Speel de zender {s}", f"Speel radio {s}", f"Zet {s} aan",
                      f"Zet de radiozender {s} aan", f"Ik wil naar {s} luisteren"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        g = R.choice(GENRES)
        return f"Speel {g}", {"query": g}, g
    # "Speel Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a for a in ARTISTS])
    return R.choice([f"Speel {t}", f"Speel eens {t}"]), {"query": t}, t


def music_target(h, loc):
    """-> (player, target words) or None when the player is not clear."""
    ps = music_players(h)
    if not ps:
        return None
    sat = h.sat_player if h.sat_player in ps and loc and h.sat_loc == loc else None
    if sat is not None and (F("sat") or R.random() < 0.6):
        # no player named: the speaker of the voice device comes first
        return sat, ""
    opts = ["name"]
    single_rooms = [r for r in C.room_order({e.area for e in ps if e.area})
                    if len([e for e in ps if e.area == r]) == 1]
    if single_rooms:
        opts.append("room")
    in_loc = [e for e in ps if loc and e.area == loc]
    if sat is None and len(in_loc) == 1:
        opts += ["loc", "loc"]
    if len(ps) == 1:
        opts += ["only", "only"]
    k = R.choice(opts)
    if k == "name":
        e = R.choice(ps)
        return e, R.choice([f" op {e.name}", f" op {e.name}", f" via {e.name}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.ROOM_BY_NAME[room][1]
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Ik speel {desc} af op {e.name}.", f"Oké, {desc} op {e.name}.",
                     f"{cap(desc)} is nu te horen op {e.name}.", f"Ik speel {desc} af."])


def i_music_play(h, loc):
    tgt = music_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    u, data, desc = music_request()
    return with_target(u, where), [action("media_player.search_and_play", [e], data)], \
        play_answer(desc, e)


def i_media_extra(h, loc):
    """Shuffle, repeat, mute, previous track, stop."""
    tgt = music_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    opts = [
        ("media_player.shuffle_set", {"shuffle": True},
         ["Zet shuffle aan{w}", "Shuffle aan{w}", "Willekeurige volgorde aan{w}",
          "Speel de nummers in willekeurige volgorde{w}"],
         ["Shuffle staat aan.", "De nummers worden in willekeurige volgorde afgespeeld."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Shuffle uit{w}", "Zet shuffle uit{w}", "Willekeurige volgorde uit{w}"],
         ["Shuffle staat uit."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Herhaal dit nummer{w}", "Speel dit nummer steeds opnieuw{w}",
          "Herhaal het huidige nummer{w}"], ["Het nummer wordt herhaald."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Herhaal alles{w}", "Herhaal de playlist{w}", "Alle nummers herhalen{w}"],
         ["Alle nummers worden herhaald."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Herhalen uit{w}", "Zet herhalen uit{w}", "Stop met herhalen{w}"],
         ["Herhalen staat uit."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Geluid uit{w}", "Dempen{w}", "Zet het geluid uit{w}", "Mute{w}"],
         [f"{e.name} is gedempt.", "Het geluid is uit."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Geluid weer aan{w}", "Dempen uit{w}", "Zet het geluid weer aan{w}"],
         ["Het geluid is weer aan.", f"{e.name} is niet meer gedempt."]),
        ("media_player.media_previous_track", None,
         ["Vorig nummer{w}", "Terug naar het vorige nummer{w}", "Speel het vorige nummer{w}",
          "Een nummer terug{w}"], ["Ik ben naar het vorige nummer gegaan.", "Vorig nummer."]),
        ("media_player.media_stop", None,
         ["Stop de muziek{w}", "Stop het afspelen{w}", "Muziek stoppen{w}", "Stop{w}"],
         ["Het afspelen is gestopt.", "Gestopt."]),
        ("media_player.media_play", None,
         ["Play{w}", "Speel verder{w}", "Hervat de muziek{w}", "Ga verder{w}"],
         ["Het afspelen gaat verder.", "We gaan verder."]),
        ("media_player.media_next_track", None,
         ["Volgende{w}", "Volgend nummer{w}", "Volgende nummer{w}"], ["Volgend nummer."]),
        ("media_player.media_previous_track", None,
         ["Terug{w}", "Vorige nummer{w}"], ["Vorig nummer."]),
        ("media_player.volume_up", None, ["Harder{w}", "Zet harder{w}"], ["Harder gezet."]),
        ("media_player.volume_down", None, ["Zachter{w}", "Zet zachter{w}"], ["Zachter gezet."]),
    ]
    sk = R.choice([10, 15, 30, 30, 60])
    sk_word = "een minuut" if sk == 60 else f"{sk} seconden"
    opts += [
        ("media_player.seek_relative", {"seconds": sk},
         [f"Spoel {sk_word} vooruit{{w}}",
          "Vooruitspoelen{w}" if sk == 30 else f"Spring {sk_word} vooruit{{w}}",
          f"Ga {sk_word} verder{{w}}"],
         [f"Ik heb {sk_word} vooruitgespoeld.", "Vooruitgespoeld."]),
        ("media_player.seek_relative", {"seconds": -sk},
         [f"Spoel {sk_word} terug{{w}}",
          "Terugspoelen{w}" if sk == 30 else f"Spring {sk_word} terug{{w}}",
          f"Ga {sk_word} terug{{w}}"],
         [f"Ik heb {sk_word} teruggespoeld.", "Teruggespoeld."]),
    ]
    svc, data, users, confs = R.choice(opts)
    return (R.choice(users).format(w=where), [action(svc, [e], data)], R.choice(confs))


def s_music_ask(h, loc):
    """Several speakers, none named and none in the room of the voice device:
    read all players first (some may be unreachable), play on the only
    reachable speaker, otherwise ask which one; optionally the user answers
    (or gives up). v16 picked an unreachable voice satellite."""
    while len(music_players(h)) < 2:
        if not add_speaker(h):
            return None
    if loc and (any(e.area == loc for e in music_players(h)) or h.sat_loc == loc):
        loc = None
    # make some speakers unreachable: none, all but one, or a few
    ps = music_players(h)
    mode = R.choice(["all", "all", "one", "few", "none"])
    if mode == "none":
        for e in ps:
            e.state = "unavailable"
            e.attrs.pop("media_title", None)
            e.attrs.pop("media_artist", None)
    elif mode == "one":
        keep = R.choice(ps)
        for e in ps:
            if e is not keep:
                e.state = "unavailable"
                e.attrs.pop("media_title", None)
                e.attrs.pop("media_artist", None)
    elif mode == "few" and len(ps) > 2:
        for e in R.sample(ps, R.randint(1, len(ps) - 2)):
            e.state = "unavailable"
            e.attrs.pop("media_title", None)
            e.attrs.pop("media_artist", None)
    ps = music_players(h)              # reachable ones
    u, data, desc = music_request()
    args = {"domain": "media_player", "area": S.ALL_AREAS}
    ents = sorted(h.of("media_player"), key=lambda e: S.sort_key((e.eid, e.name, e.area)))
    call, cid = tool_call("get_state", args)
    msgs = base_msgs(h, loc) + [user(u), call, tool_msg(cid, h.state_of(ents))]
    gold = [["get_state", args]]
    gone = [e.name for e in ents if e.state == "unavailable"]
    if not ps:
        msgs += [say(R.choice(["Er is op dit moment helaas geen speaker bereikbaar.",
                               "Geen van je speakers is nu bereikbaar, dus ik kan geen "
                               "muziek afspelen.",
                               f"{join_nl(gone[:4])} {'is' if len(gone) == 1 else 'zijn'} "
                               "op dit moment niet bereikbaar, ik kan helaas niets afspelen."]))]
        return sample(msgs, "command", gold, no_history=True)
    if len(ps) == 1:
        e = ps[0]
        acts = [action("media_player.search_and_play", [e], data)]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Ik speel {desc} af op {e.name}, dat is nu de enige "
                                    f"beschikbare speaker.",
                                    play_answer(desc, e)]))
        return sample(msgs, "command", gold, no_history=True)
    opts = ps[:4]
    # numbers in the request (NPO Radio 2, 3FM) are not in the tool result
    what = "de muziek" if any(ch.isdigit() for ch in desc) else desc
    ask = R.choice([f"Op welke speaker moet ik {what} afspelen: "
                    f"{join_nl([e.name for e in opts], 'of')}?",
                    f"Waar moet ik {what} afspelen? Je kunt kiezen uit "
                    f"{join_nl([e.name for e in opts], 'en')}."])
    gone = [e.name for e in ents if e.state == "unavailable"]
    if gone:
        ask = (f"{join_nl(gone)} {'is' if len(gone) == 1 else 'zijn'} op dit moment niet "
               f"bereikbaar. ") + ask
    msgs += [say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", gold, no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Laat maar", "Toch niet", "Maakt niet uit"])),
                 say("Oké, dan niet.")]
        return sample(msgs, "conversation", gold, gold_last=[])
    e = R.choice(opts)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"Op {e.name}", f"Op {e.name} graag"]
    if e.area and len(same_room) == 1:
        replies += [cap(C.ROOM_BY_NAME[e.area][1]), e.area]
    acts = [action("media_player.search_and_play", [e], data)]
    msgs += [user(R.choice(replies))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), play_answer(desc, e))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]])


def s_music_sat(h, loc):
    """'Speel muziek' on a voice device with its own speaker: play there,
    even when the room or the house has other speakers."""
    rooms = [r for r in h.rooms if r not in C.OUTDOOR]
    if not rooms:
        return None
    loc = R.choice(rooms)
    own = [e for e in h.of("media_player", loc) if e.voice and e.state != "unavailable"]
    sat = R.choice(own) if own else h.add_voice_media(loc)
    if sat is None:
        return None
    if sat.state == "unavailable":
        sat.state = "idle"
    h.set_sat_player(sat, loc)
    if R.random() < 0.5:
        add_speaker(h, loc)
    if R.random() < 0.5:
        add_speaker(h)
    u, data, desc = music_request()
    acts = [action("media_player.search_and_play", [sat], data)]
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), play_answer(desc, sat))
    return sample(msgs, "command", [["control", {"actions": acts}]])


UNCLEAR_MEDIA = [
    ("media_player.media_next_track", None, ["Volgend nummer", "Volgende nummer graag",
                                             "Spring naar het volgende nummer",
                                             "Sla dit nummer over", "Volgende"],
     "Volgend nummer op {n}."),
    ("media_player.media_previous_track", None, ["Vorig nummer", "Een nummer terug",
                                                 "Vorige nummer", "Terug"],
     "Vorig nummer op {n}."),
    ("media_player.media_pause", None, ["Pauze", "Pauzeer de muziek", "Zet de muziek even stil"],
     "De muziek op {n} is gepauzeerd."),
    ("media_player.media_stop", None, ["Stop de muziek", "Muziek uit", "Stop het afspelen", "Stop"],
     "Het afspelen op {n} is gestopt."),
    ("media_player.volume_up", None, ["Zet harder", "Harder graag", "Zet de muziek harder",
                                      "Harder"],
     "{n} staat nu harder."),
    ("media_player.volume_down", None, ["Zet zachter", "Zachter graag", "Zet de muziek zachter",
                                        "Zachter"],
     "{n} staat nu zachter."),
    ("media_player.seek_relative", {"seconds": 30}, ["Vooruitspoelen", "Spoel vooruit",
                                                     "Spoel 30 seconden vooruit"],
     "Op {n} 30 seconden vooruitgespoeld."),
    ("media_player.seek_relative", {"seconds": -30}, ["Terugspoelen", "Spoel terug",
                                                      "Spoel 30 seconden terug"],
     "Op {n} 30 seconden teruggespoeld."),
    # play goes on with the paused title of the player
    ("media_player.media_play", None, ["Play", "Speel verder", "Ga verder", "Hervat",
                                       "Ga verder met de muziek"],
     "Het afspelen op {n} gaat verder."),
]


def s_media_unclear(h, loc):
    """'Volgend nummer' with several speakers, none named and none in the room
    of the voice device: read all players, act on the one that is playing
    ('Play': the one that is paused), ask when none or several are (v16
    picked an unreachable player)."""
    while len(music_players(h)) < 2:
        if not add_speaker(h):
            return None
    ps = music_players(h)
    if loc and (any(e.area == loc for e in ps) or h.sat_loc == loc):
        loc = None
    mode = R.choice(["one", "one", "one", "none", "several"])
    active = {"one": R.sample(ps, 1), "none": [], "several": R.sample(ps, 2)}[mode]
    svc, data, users, conf = R.choice(UNCLEAR_MEDIA)
    resume = svc == "media_player.media_play"
    for e in h.of("media_player"):
        if e in active:
            title, artist = R.choice(TRACKS)
            e.state = "paused" if resume else "playing"
            e.attrs.update(media_title=title, media_artist=artist)
        elif e.state in ("playing", "paused"):
            e.state = R.choice(["idle", "off"]) if resume else R.choice(["idle", "paused", "off"])
            e.attrs.pop("media_title", None)
            e.attrs.pop("media_artist", None)
    args = {"domain": "media_player", "area": S.ALL_AREAS}
    ents = sorted(h.of("media_player"), key=lambda e: S.sort_key((e.eid, e.name, e.area)))
    call, cid = tool_call("get_state", args)
    msgs = base_msgs(h, loc) + [user(R.choice(users)), call, tool_msg(cid, h.state_of(ents))]
    gold = [["get_state", args]]
    if mode == "one":
        e = active[0]
        acts = [action(svc, [e], data)]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), conf.format(n=e.name))
        return sample(msgs, "command", gold)
    names = [e.name for e in (active or ps[:4])]
    if mode == "none" and resume:
        ask = R.choice([f"Er staat nu nergens iets op pauze. Wat moet ik op welke speaker "
                        f"afspelen: {join_nl(names, 'of')}?",
                        "Op geen enkele speaker staat nu iets op pauze. Wat wil je horen?"])
    elif mode == "none":
        ask = R.choice([f"Er speelt nu nergens muziek. Welke speaker bedoel je: "
                        f"{join_nl(names, 'of')}?",
                        "Op dit moment speelt geen enkele speaker iets. Op welke moet ik dat doen?"])
    elif resume:
        ask = f"Er staat muziek op pauze op {join_nl(names)}. Waar moet ik verder spelen?"
    else:
        ask = f"Er speelt muziek op {join_nl(names)}. Welke bedoel je?"
    msgs += [say(ask)]
    return sample(msgs, "command", gold)


def s_music_error(h, loc):
    """Search found nothing, or the player cannot search."""
    out = None
    for _ in range(6):
        out = i_music_play(h, loc)
        if not out and not add_speaker(h):
            return None
        if out and (out[1][0].get("data") or {}).get("query"):
            break
    else:
        return None
    if not out:
        return None
    u, acts, _ = out
    e = next(x for x in h.ents if x.eid == acts[0]["entity_id"][0])
    q = acts[0]["data"]["query"]
    if R.random() < 0.75:
        err = f"media_player.search_and_play failed for {e.eid}: No results found for {q}"
        ans = R.choice([f"Ik heb {q} helaas niet gevonden.",
                        f"Helaas heb ik niets gevonden voor {q}.",
                        f"Voor {q} heb ik helaas niets gevonden. Zal ik iets anders afspelen?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Op {e.name} kan ik helaas geen muziek zoeken.",
                        f"{e.name} ondersteunt het zoeken naar muziek helaas niet."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


# --------------------------------------------------------------------------
# media library of a player (music_library, media_player.browse_media)
# --------------------------------------------------------------------------
# mood -> (request words, playlist names that fit)
LIB_MOODS = {
    "relax": (["om te ontspannen", "om tot rust te komen", "om te chillen",
               "voor een rustige avond"],
              ["Chillout Lounge", "Puur ontspannen", "Relax & Unwind", "Peaceful Piano",
               "Chill Out Music", "Totally Stress Free", "Rustige avonden"]),
    "sport": (["om te sporten", "voor de training", "voor de sportschool", "om mee te hardlopen"],
              ["Gym Hits", "Workout Mix", "Running Beats", "Power Workout", "Training 2026",
               "Beast Mode"]),
    "sleep": (["om in slaap te vallen", "om bij te slapen", "voor de nacht"],
              ["Deep Sleep", "Slaapmuziek", "Sleep Sounds", "Inslapen", "Welterusten"]),
    "party": (["voor het feest", "om te feesten", "om op te dansen"],
              ["Party Hits", "Feestknallers", "Dance Party", "Club Hits 2026", "Dansen!"]),
    "xmas": (["met kerstliedjes", "voor de kerst", "met kerstmuziek"],
             ["Kerstliedjes", "Christmas Classics", "Kerst 2025", "Xmas Hits"]),
    "cook": (["om bij te koken", "bij het avondeten", "bij het eten"],
             ["Koken met muziek", "Dinner Jazz", "Cooking Vibes", "Dinner for Two"]),
    "focus": (["om bij te werken", "om me te concentreren", "om bij te studeren",
               "om bij te programmeren"],
              ["Deep Focus", "Beats to think to", "Lofi Coders", "Concentratie",
               "Programming Beats", "Studeermuziek"]),
    "rock": (["met rock", "met rockmuziek", "met metal"],
             ["Rock Classics", "Rock Anthems", "Metal Love", "Hard Rock Hits"]),
    "kids": (["voor de kinderen", "met kinderliedjes"],
             ["Kinderliedjes", "Kinderdisco", "Liedjes voor kinderen"]),
    "eighties": (["met hits uit de jaren tachtig", "met muziek uit de jaren tachtig"],
                 ["Jaren 80 Hits", "Best of the 80s", "Jaren 80 Feest"]),
    "car": (["voor in de auto", "voor onderweg"],
            ["Roadtrip", "Autorit", "Driving Hits"]),
}
FAV_NAMES = ["Favoriete nummers", "Liked Songs", "Favorieten", "Mijn favorieten", "Leuke nummers",
             "Favs", "Liked Songs Max", "Liked Songs Anna", "Mijn favoriete nummers",
             "Lievelingsmuziek"]
FAV_USERS = ["Speel mijn favoriete nummers", "Speel mijn lievelingsnummers", "Speel mijn favorieten",
             "Speel mijn lievelingsmuziek", "Ik wil mijn favoriete nummers horen",
             "Speel de nummers die ik leuk vind", "Speel mijn favoriete playlist"]
NEUTRAL_PLAYLISTS = ["Discover Weekly", "Release Radar", "Mijn playlist nr. 3",
                     "Mijn playlist nr. 7", "Verjaardag Alex", "Vakantie 2024", "Zomer 2025",
                     "Bruiloft", "Mix van de week", "Nieuw ontdekt", "Shazam-nummers",
                     "Teen Wolf Soundtrack", "Electro Swing", "Deep House 2026",
                     "Psytrance Supernova", "Goa Mix", "Recently added tracks",
                     "Dobby + Maximilian", "Fietsen", "Boomerfeest", "Welcome Back 20s"]
LIB_KINDS = {  # media_type -> (plural, user question words, pool)
    "playlist": ("playlists", ["Welke playlists heb ik", "Wat voor playlists heb ik",
                               "Welke playlists zijn er", "Welke playlists heb ik allemaal",
                               "Wat heb ik aan playlists"], None),
    "album": ("albums", ["Welke albums heb ik", "Wat voor albums heb ik",
                         "Welke albums staan er in mijn bibliotheek"], None),
    "artist": ("artiesten", ["Welke artiesten heb ik", "Van welke artiesten heb ik muziek",
                             "Welke artiesten staan er in mijn bibliotheek"], None),
    "radio": ("radiozenders", ["Welke radiozenders heb ik", "Welke zenders heb ik opgeslagen",
                               "Wat voor radiozenders zijn er"], None),
    "podcast": ("podcasts", ["Welke podcasts heb ik", "Op welke podcasts ben ik geabonneerd"],
                None),
}
PODCASTS = ["Zware Jongens", "Echt Gebeurd", "Dit is de Dag", "NRC Vandaag",
            "De Volkskrant Elke Dag", "Man Man Man", "Damn, Honey", "Betrouwbare Bronnen",
            "De Universiteit van Nederland", "Kalm aan"]


def lib_pool(kind, dash=False):
    """dash: albums and tracks as 'Artist - Title' (Music Assistant style)."""
    if kind == "playlist":
        return NEUTRAL_PLAYLISTS + [n for _, ns in LIB_MOODS.values() for n in ns] + FAV_NAMES
    if kind == "album":
        return [f"{a} - {al}" if dash else al for al, a in ALBUMS]
    if kind == "track":
        return [f"{a} - {t}" if dash else t for t, a in TRACKS]
    if kind == "artist":
        return list(ARTISTS)
    if kind == "radio":
        return list(STATIONS)
    return list(PODCASTS)


def make_library(kind, must=(), avoid=(), dash=False):
    """Names in the library of one kind, sorted like a browse result."""
    pool = [n for n in lib_pool(kind, dash) if n not in avoid and n not in must]
    n = R.choice([R.randint(1, 4), R.randint(4, 9), R.randint(8, 20), R.randint(15, 40),
                  R.randint(40, 80)])
    names = list(must) + R.sample(pool, min(n, len(pool)))
    return sorted(dict.fromkeys(names), key=str.casefold)


def lib_answer(kind, names, total, who):
    plural = LIB_KINDS[kind][0]
    if not names:
        return R.choice([f"Op {who} heb ik geen {plural} gevonden.",
                         f"In de bibliotheek van {who} staan geen {plural}."])
    k = {"short": 3, "medium": 5, "long": 6, "detailed": 8}[C.VERBOSITY]
    if total <= k:
        if total == 1:
            return f"Je hebt er maar één: {names[0]}."
        return f"Je hebt {total} {plural}: {join_nl(names)}."
    shown = R.sample(names, k) if R.random() < 0.3 else names[:k]
    return f"Je hebt {total} {plural}, bijvoorbeeld {join_nl(shown)}."


def lib_target(h, loc):
    tgt = music_target(h, loc)
    if not tgt:
        if not add_speaker(h):
            return None
        tgt = music_target(h, loc)
    return tgt


def lib_where(sentence, where):
    """'Welke playlists heb ik' + ' op Sonos Keuken' + '?'."""
    return sentence + (where.replace(" via ", " op ") if where else "") + "?"


def s_library_list(h, loc):
    """'Welke playlists heb ik?' -> music_library -> names; optionally
    'Speel <name>' afterwards plays one of them."""
    tgt = lib_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    kind = R.choice(["playlist", "playlist", "playlist", "album", "album", "artist", "radio",
                     "podcast"])
    plural, questions, _ = LIB_KINDS[kind]
    u = lib_where(R.choice(questions), where)
    args = {"entity_id": e.eid, "media_type": kind}
    if kind == "album" and R.random() < 0.5:
        # 'Welke albums van Queen heb ik?': albums are listed as 'Queen - Innuendo'
        a = R.choice(sorted({a for _, a in ALBUMS}))
        mine = [f"{a} - {al}" for al, aa in ALBUMS if aa == a]
        names = make_library("album", must=mine if R.random() < 0.85 else [], dash=True,
                             avoid=[] if R.random() < 0.85 else mine)
        hits = [n for n in names if a.casefold() in n.casefold()]
        args["search"] = a
        u = lib_where(R.choice([f"Welke albums van {a} heb ik",
                                f"Wat heb ik aan albums van {a}",
                                f"Welke albums van {a} staan er in mijn bibliotheek"]), where)
        if hits:
            titles = [n.split(" - ", 1)[1] for n in hits]
            ans = (f"Van {a} heb je het album {titles[0]}." if len(titles) == 1 else
                   f"Van {a} heb je {len(titles)} albums: {join_nl(titles)}.")
        else:
            ans = f"Van {a} heb ik geen albums in je bibliotheek gevonden."
        msgs = base_msgs(h, loc) + [user(u)]
        msgs += call_turn("music_library", args, S.library_result(hits), ans)
        return sample(msgs, "question", [["music_library", args]], no_history=True)
    msgs = base_msgs(h, loc) + [user(u)]
    gold = [["music_library", args]]
    if R.random() < 0.08:
        res = S.library_result(error=f"media library is not supported by {e.eid}")
        msgs += call_turn("music_library", args, res,
                          R.choice([f"{e.name} ondersteunt het doorzoeken van de bibliotheek "
                                    f"helaas niet.",
                                    f"De bibliotheek van {e.name} kan ik helaas niet lezen."]))
        return sample(msgs, "question", gold, no_history=True)
    names = make_library(kind, dash=R.random() < 0.5) if R.random() > 0.06 else []
    res = S.library_result(names)
    total = len(names)
    msgs += call_turn("music_library", args, res, lib_answer(kind, names, total, e.name))
    if not names or R.random() < 0.6:
        return sample(msgs, "question", gold, no_history=True)
    # play one of them in the next turn
    pick = R.choice(names[:S.LIBRARY_ITEMS])
    data = {"query": pick, "media_type": kind}
    acts = [action("media_player.search_and_play", [e], data)]
    msgs += [user(R.choice([f"Speel {pick}", f"Speel dan {pick}", f"Speel {pick} maar",
                            f"{pick} graag"]))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                      R.choice([f"Ik speel {pick} af op {e.name}.", f"{pick} is nu te horen."]))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]],
                  no_history=True)


def s_library_search(h, loc):
    """'Heb ik een playlist met de naam Party?' -> music_library with search."""
    tgt = lib_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    kind = R.choice(["playlist", "playlist", "album", "artist", "radio"])
    pool = lib_pool(kind)
    word = R.choice(pool)
    found = R.random() < 0.65
    names = make_library(kind, must=[word] if found else [], avoid=[] if found else [word])
    hits = [n for n in names if word.casefold() in n.casefold()]
    q = {"playlist": [f"Heb ik een playlist met de naam {word}", f"Is er een playlist {word}",
                      f"Staat de playlist {word} in mijn bibliotheek"],
         "album": [f"Heb ik het album {word}", f"Staat het album {word} in mijn bibliotheek"],
         "artist": [f"Heb ik muziek van {word}", f"Staat {word} in mijn bibliotheek"],
         "radio": [f"Heb ik de zender {word} opgeslagen", f"Heb ik de zender {word}"]}[kind]
    args = {"entity_id": e.eid, "media_type": kind, "search": word}
    res = S.library_result(hits)
    if hits:
        ans = R.choice([f"Ja, {hits[0]} staat in je bibliotheek." if len(hits) == 1 else
                        f"Ja, ik heb {len(hits)} treffers: {join_nl(hits[:5])}.",
                        f"Ja, {join_nl(hits[:5])}."])
    else:
        ans = R.choice([f"Nee, {word} heb ik niet gevonden in je bibliotheek.",
                        f"Nee, {word} staat niet in de bibliotheek van {e.name}."])
    msgs = base_msgs(h, loc) + [user(lib_where(R.choice(q), where))]
    msgs += call_turn("music_library", args, res, ans)
    return sample(msgs, "question", [["music_library", args]], no_history=True)


def s_library_play(h, loc, favorites=False):
    """'Speel een playlist om te ontspannen' / 'Speel mijn favoriete nummers':
    read the playlists, play the one that fits; none fits -> say so and
    name a few."""
    tgt = lib_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    if favorites:
        u = R.choice(FAV_USERS)
        fit_pool, avoid = FAV_NAMES, FAV_NAMES
        what = "je favoriete nummers"
    else:
        mood = R.choice(list(LIB_MOODS))
        words, fit_pool = LIB_MOODS[mood]
        w = R.choice(words)
        u = R.choice([f"Speel een playlist {w}", f"Speel iets {w}", f"Speel muziek {w}",
                      f"Zoek een playlist {w} voor me", f"Speel een van mijn playlists {w}"])
        avoid = fit_pool
        what = f"muziek {w}"
    fits = R.random() < 0.8
    must = [R.choice(fit_pool)] if fits else []
    names = make_library("playlist", must=must, avoid=[n for n in avoid if n not in must])
    if not names:
        return None
    args = {"entity_id": e.eid, "media_type": "playlist"}
    msgs = base_msgs(h, loc) + [user(with_target(u, where))]
    call, cid = tool_call("music_library", args)
    msgs += [call, tool_msg(cid, S.library_result(names))]
    gold = [["music_library", args]]
    if fits:
        pick = must[0]
        acts = [action("media_player.search_and_play", [e],
                       {"query": pick, "media_type": "playlist"})]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Ik speel de playlist {pick} af op {e.name}.",
                                    f"Oké, {pick} is nu te horen op {e.name}.",
                                    f"Ik heb {pick} gestart."]))
        return sample(msgs, "command", gold, no_history=True)
    some = R.sample(names, min(3, len(names)))
    ans = R.choice([f"Een playlist met {what} heb ik niet gevonden. Je hebt bijvoorbeeld "
                    f"{join_nl(some)}.",
                    f"Daar past geen van je playlists bij. Zal ik er een van afspelen: "
                    f"{join_nl(some, 'of')}?"])
    msgs += [say(ans)]
    return sample(msgs, "command", gold, no_history=True)


# --------------------------------------------------------------------------
# to-do lists
# --------------------------------------------------------------------------
def pick_list(h, with_items=False):
    lists = [e for e in h.of("todo") if not with_items or e.attrs.get("items")]
    if not lists:
        return None
    shop = [e for e in lists if e.name == SHOPPING]
    return shop[0] if shop and R.random() < 0.65 else R.choice(lists)


def i_todo_add(h, loc):
    lst = pick_list(h)
    if not lst:
        return None
    li, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return None
    items = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    it = join_nl(items)
    users = [f"Zet {it} op {li}", f"Schrijf {it} op {li}", f"Voeg {it} toe aan {li}",
             f"{it} op {li}", f"Zet {it} even op {li}", f"Noteer {it} op {li}"]
    if lst.name == SHOPPING:
        users += [f"We hebben {it} nodig", f"Ik moet {it} kopen", f"Zet {it} op de lijst"
                  if len(h.of("todo")) == 1 else f"Zet {it} op de boodschappenlijst"]
    if lst.name == TASKS:
        users += [f"Voeg {it} toe aan mijn taken", f"Zet {it} op mijn takenlijst"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    pl = len(items) > 1 or items[0] in PLURAL_ITEMS
    conf = R.choice([f"Ik heb {it} op {li} gezet.",
                     f"{cap(it)} {'staan' if pl else 'staat'} nu op {li}.",
                     f"Oké, {it} {'staan' if pl else 'staat'} op {li}."])
    return R.choice(users), acts, conf


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    li, _, _ = LISTS[lst.name]
    item = R.choice(lst.attrs["items"])
    if R.random() < 0.6:
        it_pron = "ze" if item in PLURAL_ITEMS else "het"
        users = [f"Vink {item} af op {li}", f"Streep {item} door op {li}",
                 f"{item} afvinken op {li}"]
        if lst.name == SHOPPING:
            users += [f"Ik heb {item} gekocht, vink {it_pron} maar af"]
        if lst.name == TASKS:
            users += [f"{item} is gedaan, vink het af op {li}"]
        svc = "todo.complete_item"
        conf = R.choice([f"Ik heb {item} afgevinkt.",
                         f"{cap(item)} {is_item(item)} afgevinkt op {li}."])
    else:
        users = [f"Verwijder {item} van {li}", f"Haal {item} van {li} af",
                 f"Haal {item} van {li}"]
        svc = "todo.remove_item"
        conf = R.choice([f"{cap(item)} {is_item(item)} van {li} verwijderd.",
                         f"Ik heb {item} van {li} gehaald."])
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], conf


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOPPING)
    li, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [i for i in pool if i not in items] or items)
        u = f"{cap(staat(item))} {item} op {li}?"
        yes = item in items
        ans = f"Ja, {item} {staat(item)} op {li}." if yes else \
            f"Nee, {item} {staat(item)} niet op {li}."
        must = [item]
    else:
        users = [f"Wat staat er op {li}?", f"Lees {li} voor", f"Wat staat er allemaal op {li}?"]
        if lst.name == SHOPPING:
            users += ["Wat moet ik nog kopen?", "Wat hebben we nog nodig?"]
        if lst.name == TASKS:
            users += ["Wat staat er vandaag op de planning?", "Wat moet ik nog doen?"]
        u = R.choice(users)
        if not items:
            ans = R.choice([f"{cap(li)} is leeg.", f"Er staat nu niets op {li}."])
            must = ["leeg" if "leeg" in ans else "niets"]
        elif len(items) == 1:
            ans = f"Op {li} staat alleen {items[0]}."
            must = items
        else:
            ans = f"Op {li} staan {join_nl(items)}."
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOPPING)
    li, pool, _ = LISTS[lst.name]
    missing = [i for i in pool if i not in lst.attrs["items"]]
    item = R.choice(missing)
    u, svc = R.choice([(f"Vink {item} af op {li}", "todo.complete_item"),
                       (f"Streep {item} door op {li}", "todo.complete_item"),
                       (f"Verwijder {item} van {li}", "todo.remove_item")])
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"{cap(item)} {staat(item)} helaas niet op {li}.",
                    f"Ik heb {item} niet gevonden op {li}."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    li, pool, _ = LISTS[name]
    item = R.choice(pool)
    u = R.choice([f"Zet {item} op {li}", f"Wat staat er op {li}?", f"Schrijf {item} op {li}"])
    others = [LISTS[e.name][0] for e in h.of("todo")]
    ans = f"Ik heb {li} niet gevonden."
    if others:
        ans += f" Er is alleen {join_nl(others)}."
    else:
        ans = R.choice([ans, f"Ik heb helaas geen lijst met de naam {name} gevonden."])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def s_weather(h, loc):
    ws = [e for e in h.of("weather") if e.state != "unavailable"]
    if not ws:
        if not add_weather(h):
            return None
        ws = h.of("weather")
    e = R.choice(ws)
    fc = e.attrs["forecast"]
    cond_word = S.lang_texts(LANG)["conditions"][e.state]
    temp = nl_num(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == "Buitentemperatuur" for x in h.of("sensor")):
            return None
        u = R.choice(["Hoe warm is het buiten?", "Hoe koud is het buiten?",
                      "Hoeveel graden is het buiten?", "Wat is de buitentemperatuur?"])
        ans, must = f"Buiten is het {temp} graden.", [temp]
    elif kind == "now":
        u = R.choice(["Hoe is het weer?", "Hoe is het weer nu?",
                      "Wat voor weer is het buiten?", "Wat doet het weer?",
                      "Hoe is het buiten?", "Hoe is het weer vandaag?",
                      "Weer?", "Wat zegt het weerbericht?", "Vertel eens iets over het weer"])
        ans, must = weather_report(e, temp, cond_word), [temp]
    else:
        idx = 1
        if kind == "day":
            idx = R.choice([0, 1, 1, 2, 3])
        if len(fc) <= idx:
            return None
        f = fc[idx]
        raw = e.attrs["_entries"][idx]
        day = f["day"]
        day_cap = cap(day)
        frag = FORECAST_FRAG[raw["condition"]]
        hi, lo = f["temperature"], f.get("templow")
        if kind == "rain":
            u = R.choice([f"Gaat het {day} regenen?", f"Regent het {day}?",
                          f"Heb ik {day} een paraplu nodig?"])
            p = f.get("precipitation_probability")
            if raw["condition"] in RAINY:
                ans = f"Ja, {day} {frag}" + (f", de kans op regen is {p} procent."
                                             if p is not None else ".")
                must = ["Ja"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"Nee, maar {day} {frag}" + (f", de kans op neerslag is {p} procent."
                                                   if p is not None else ".")
                must = ["Nee"]
            else:
                ans = f"Nee, {day} {frag}" + (f", de kans op regen is {p} procent."
                                              if p is not None else ".")
                must = ["Nee"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"Hoe warm wordt het {day}?", f"Hoeveel graden wordt het {day}?"])
                ans, must = f"{day_cap} wordt het maximaal {hi} graden.", [str(hi)]
            else:
                u = R.choice([f"Hoe koud wordt het {day}?", f"Hoe koud wordt het {day} 's nachts?"])
                ans, must = f"{day_cap} koelt het af tot {lo} graden.", [str(lo)]
        else:
            u = R.choice([f"Hoe wordt het weer {day}?", f"Wat zegt het weerbericht voor {day}?",
                          f"Wat voor weer wordt het {day}?"])
            ans = f"{day_cap} {frag}"
            ans += f", met {lo} tot {hi} graden." if lo is not None else f", tot {hi} graden."
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = now_sentence(e.state, temp) + " " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Vandaag wordt het tussen de {lo} en {hi} graden."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" De kans op regen is {p} procent."
            if C.VERBOSITY == "short":
                ans = f"{day_cap} {frag}, tot {hi} graden."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def op_dit_moment(sentence):
    """'Het regent hard' -> 'Op dit moment regent het hard'."""
    w = sentence.split(" ")
    if w[0] not in ("Het", "Er"):
        return sentence
    return " ".join(["Op dit moment", w[1], w[0].lower()] + w[2:])


def now_sentence(cond, temp, moment=False):
    """'Het is zonnig en 12 graden.' / 'Het regent en het is 8 graden.'"""
    s = NOW_SENT[cond]
    if moment:
        s = op_dit_moment(s)
    if s.startswith(("Het is ", "Op dit moment is het ")):
        return f"{s} en {temp} graden."
    if moment:
        return f"{s} bij {temp} graden."
    return f"{s} en het is {temp} graden."


def rest_of_day(e):
    """'Vanaf 18 uur gaat het regenen bij 12 graden en om 21 uur is het nog 8 graden.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    end = f"om {int(last['time'].split(':')[0])} uur is het nog {last['temperature']} graden"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        return cap(R.choice(["tot de avond", "de rest van de dag", "ook later"])) + \
            f" {STAY[e.state]}, {end}."
    f = later[change]
    out = (f"Vanaf {int(f['time'].split(':')[0])} uur {FORECAST_FRAG[raw[change]]} "
           f"bij {f['temperature']} graden")
    if change < len(later) - 1:
        out += f" en {end}"
    return out + "."


def weather_report(e, temp, cond_word):
    """'Hoe is het weer?' by answer length: short - now; medium - now and
    the rest of the day; long - plus today's range and rain chance;
    detailed - plus humidity, wind and tomorrow."""
    a, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    first = R.choice([now_sentence(e.state, temp),
                      f"Buiten is het nu {temp} graden, {cond_word}.",
                      now_sentence(e.state, temp, moment=True)])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {temp} graden.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Vandaag wordt het tussen de {t['templow']} en {t['temperature']} graden.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"De luchtvochtigheid is {a['humidity']} procent en de wind waait "
                     f"met {nl_num(a['wind_speed'])} {a['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([f"Vandaag wordt het tussen de {t['templow']} en {t['temperature']} graden.",
                                   f"Vandaag wordt het maximaal {t['temperature']} graden, "
                                   f"de minimumtemperatuur is {t['templow']} graden."]))
        else:
            parts.append(f"Vandaag wordt het maximaal {t['temperature']} graden.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"De kans op regen is {p} procent." if p else
                         "Vandaag wordt er geen regen verwacht.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Morgen {FORECAST_FRAG[raw['condition']]}, tot {fc[1]['temperature']} graden.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# time and date
# --------------------------------------------------------------------------
def s_time(h, loc):
    now = rand_dt()
    res = S.time_result(now, LANG)
    t = S.lang_texts(LANG)
    wd = t["weekdays"][now.weekday()]
    date = t["date_format"].format(day=now.day, month=t["months"][now.month - 1], year=now.year)
    kind = F("tkind") or R.choice(["time", "date", "weekday"])
    if kind == "time":
        u = R.choice(["Hoe laat is het?", "Hoe laat is het nu?", "Hoe laat hebben we het?",
                      "Zeg me hoe laat het is", "Wat is de tijd?", "Tijd?",
                      "Weet je hoe laat het is?"])
        hm = now.strftime("%H:%M")
        ans, must = R.choice([f"Het is {hm}.", f"Het is nu {hm} uur."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm}."
        elif C.VERBOSITY == "detailed":
            ans = f"Het is {hm} uur op {wd} {date}."
    elif kind == "date":
        u = R.choice(["Welke datum is het vandaag?", "De hoeveelste is het vandaag?",
                      "Welke datum hebben we?", "Wat is de datum van vandaag?",
                      "De hoeveelste hebben we vandaag?", "Zeg me de datum",
                      "Welke datum is het?", "Datum?", "Wat is het vandaag voor datum?"])
        ans, must = f"Vandaag is het {wd} {date}.", [date.split(" ")[0] + " " + date.split(" ")[1]]
    else:
        u = R.choice(["Welke dag is het vandaag?", "Welke dag van de week is het?",
                      "Wat voor dag is het vandaag?", "Welke weekdag is het vandaag?",
                      "Welke dag hebben we vandaag?", "Is het vandaag al vrijdag?",
                      "Wat is het vandaag voor dag?", "Zeg me welke dag het vandaag is",
                      "Welke weekdag is het?", "Is het vandaag maandag?",
                      "Wat is het vandaag?", "Is het al weekend?"])
        ans, must = R.choice([f"Vandaag is het {wd}.", f"Vandaag is het {wd} {date}."]), [wd]
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("get_time", {}, res, ans)
    return sample(msgs, "question", [["get_time", {}]], answer_contains=must)


# --------------------------------------------------------------------------
# timers
# --------------------------------------------------------------------------
def round_time(hours, minutes, seconds):
    """Home Assistant's rounding of the time left (intent/timers.py)."""
    if hours > 0:
        if minutes > 45:
            return hours + 1, 0, 0
        return hours, 0 if minutes < 15 else 30, 0
    if minutes > 0:
        if seconds > 45:
            return 0, minutes + 1, 0
        return 0, minutes, 0 if seconds < 15 else 30
    if seconds > 50:
        return 0, 1, 0
    return 0, 0, seconds if seconds < 10 else seconds - seconds % 10


def numw(n, word_ok=True):
    return NUM_WORDS[n] if word_ok and n in NUM_WORDS and R.random() < 0.3 else str(n)


def duration():
    """-> (hours, minutes, seconds, spoken)."""
    k = R.random()
    if k < 0.5:
        m = R.choice([1, 2, 3, 4, 5, 5, 6, 7, 8, 10, 10, 12, 15, 20, 25, 30, 40, 45, 50, 60, 90])
        return 0, m, 0, "een minuut" if m == 1 else f"{numw(m)} minuten"
    if k < 0.65:
        return R.choice([(0, 30, 0, "een half uur"), (0, 15, 0, "een kwartier"),
                         (0, 45, 0, "drie kwartier"), (1, 0, 0, "een uur"),
                         (1, 30, 0, "anderhalf uur"), (1, 30, 0, "een uur en een half"),
                         (2, 0, 0, "twee uur"), (2, 0, 0, "2 uur"),
                         (3, 0, 0, "drie uur")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        return 0, 0, s, f"{numw(s)} seconden"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = "een minuut" if m == 1 else f"{numw(m)} minuten"
        return 0, m, s, f"{mw} en {s} seconden"
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = "een uur" if hh == 1 else f"{numw(hh)} uur"
    return hh, m, 0, f"{hw} en {m} minuten"


def dur_args(hh, m, s):
    return {k: v for k, v in (("hours", hh), ("minutes", m), ("seconds", s)) if v}


def running_timers(n, named=True, unique=True):
    """n simulated timers: [(name, h, m, s, active)] with the time left."""
    names = R.sample(TIMER_NAMES, n)
    out = []
    for i in range(n):
        left = R.randint(5, 5400)
        m, s = divmod(left, 60)
        hh, m = divmod(m, 60)
        nm = names[i][0] if named else ""
        out.append((nm, *round_time(hh, m, s), R.random() > 0.15))
    return out


def status_entries(timers):
    return [S.timer_status_entry(nm, hh, m, s, act, LANG) for nm, hh, m, s, act in timers]


def timer_label(name):
    """'timer voor de pizza' / 'timer'."""
    return f"timer {TIMER_FOR[name]}" if name else "timer"


def s_timer_start(h, loc):
    hh, m, s, spoken = duration()
    named = F("named", R.random() < 0.35)
    name, fuer, comp = R.choice(TIMER_NAMES) if named else ("", "", "")
    if named:
        u = R.choice([f"Zet een timer {fuer} op {spoken}", f"Zet een {comp} op {spoken}",
                      f"{cap(comp)} {spoken}", f"Timer {fuer}, {spoken}",
                      f"Zet een timer met de naam {name} op {spoken}",
                      f"Timer van {spoken} {fuer}", f"Start een timer {fuer} van {spoken}"])
    else:
        u = R.choice([f"Zet een timer op {spoken}", f"Timer {spoken}",
                      f"Zet een timer van {spoken}", f"Start een timer van {spoken}",
                      f"Wek me over {spoken}", f"Timer van {spoken}",
                      f"Kun je een timer zetten op {spoken}?",
                      f"Geef me over {spoken} een seintje", f"Timer op {spoken}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Op dit apparaat kan ik helaas geen timer zetten. Dat kan wel "
                        "via een spraakapparaat zoals Home Assistant Voice.",
                        "Timers werken helaas alleen op een spraakapparaat, niet hier.",
                        "Hier kan ik helaas geen timer zetten, alleen op een spraakapparaat."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        ans = R.choice([f"De {timer_label(name)} loopt: {dur}.",
                        f"De {timer_label(name)} staat op {dur}.",
                        f"Oké, {dur} {fuer}."])
    else:
        ans = R.choice([f"De timer loopt: {dur}.", f"Ik heb een timer gezet op {dur}.",
                        f"Timer van {dur} gestart.", f"Oké, {dur}. De timer loopt."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Doe over 10 minuten het licht uit' -> timer with the command; the
    command runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = cap(out[0].rstrip(".!?"))
    hh, m, s, spoken = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2:
        return None       # "Kun je ..." does not take a time phrase
    sp = spoken
    # Dutch puts the time right after the verb ("Doe over 10 minuten het
    # licht uit") or at the end
    forms = [f"{cmd} over {sp}", f"Zet een timer op {spoken} en {low(words[0])} dan {words[1]}",
             f"{words[0]} over {sp} {words[1]}", f"{words[0]} over {sp} {words[1]}",
             f"Over {sp}: {lower_first(cmd)}"]
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"Oké, over {dur} voer ik dat uit.", f"Komt goed, dat doe ik over {dur}.",
                    f"Over {dur} regel ik dat."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"Er loopt nu geen {timer_label(asked_name)}.", ["geen"]
        return R.choice(["Er loopt nu geen timer.", "Er loopt op dit moment geen timer."]), ["geen"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        if e.get("paused"):
            return f"De {lab} is gepauzeerd, er staat nog {e['remaining']} op.", [e["remaining"]]
        return R.choice([f"De {lab} loopt nog {e['remaining']}.",
                         f"Nog {e['remaining']}."]), [e["remaining"]]
    parts = [f"{e['name'] or 'Een timer'}: nog {e['remaining']}"
             + (", gepauzeerd" if e.get("paused") else "") for e in entries]
    return f"Er lopen {len(entries)} timers. " + ". ".join(parts) + ".", \
        [e["remaining"] for e in entries]


def s_timer_status(h, loc):
    n = R.choice([0, 1, 1, 1, 2, 2, 3])
    timers = running_timers(n, named=n > 1 or R.random() < 0.4)
    named = [t for t in timers if t[0]]
    args = {"action": "status"}
    asked = None
    if named and R.random() < 0.35:
        asked = R.choice(named)[0] if R.random() < 0.8 else R.choice(TIMER_NAMES)[0]
        fuer = TIMER_FOR[asked]
        u = R.choice([f"Hoe lang loopt de {TIMER_COMP[asked]} nog?", f"Hoe lang nog {fuer}?",
                      f"Hoeveel tijd heeft de timer {fuer} nog?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["Hoe lang loopt de timer nog?", "Hoeveel tijd staat er nog op de timer?",
                      "Hoe lang nog?", "Welke timers lopen er?", "Loopt er nog een timer?",
                      "Hoe lang duurt de timer nog?", "Timer status"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


TIMER_OPS = {
    "cancel": (["Annuleer de {t}", "Stop de {t}", "{T} annuleren", "Verwijder de {t}",
                "Zet de {t} uit", "Haal de {t} weg"],
               ["De {t} is geannuleerd.", "Ik heb de {t} verwijderd.", "Oké, {t} geannuleerd."]),
    "pause": (["Pauzeer de {t}", "Zet de {t} even stil", "{T} pauzeren", "Stop de {t} even"],
              ["De {t} is gepauzeerd.", "Ik heb de {t} stilgezet."]),
    "resume": (["Hervat de {t}", "Laat de {t} verder lopen", "{T} verder",
                "Start de {t} weer"],
               ["De {t} loopt weer.", "Oké, de {t} loopt verder."]),
    "add": (["Doe er {d} bij op de {t}", "Verleng de {t} met {d}", "Nog {d} op de {t}",
             "Zet er {d} bij op de {t}"],
            ["Ik heb {D} bij de {t} opgeteld.", "De {t} loopt nu {D} langer."]),
    "subtract": (["Haal {d} van de {t} af", "Verkort de {t} met {d}", "Trek {d} af van de {t}"],
                 ["De {t} is nu {D} korter.", "Ik heb {D} van de {t} afgehaald."]),
}


def s_timer_op(h, loc):
    op = F("op") or R.choice(list(TIMER_OPS))
    users, answers = TIMER_OPS[op]
    hh, m, s, spoken = duration() if op in ("add", "subtract") else (0, 0, 0, "")
    if op in ("add", "subtract") and hh > 1:
        return None
    r = R.random()
    if r < 0.6:      # one timer, matches
        timers = running_timers(1, named=R.random() < 0.4)
        nm = timers[0][0] if timers[0][0] and R.random() < 0.6 else ""
        result = S.timer_result()
    elif r < 0.8:    # nothing running
        timers, nm = [], ""
        result = S.timer_result("no matching timer", [])
    else:            # name does not match the running timers
        timers = running_timers(R.choice([1, 2]))
        nm = R.choice([n for n, _, _ in TIMER_NAMES if n not in [t[0] for t in timers]])
        result = S.timer_result("no matching timer", status_entries(timers))
    t = timer_label(nm)
    u = R.choice(users).format(t=t, T=cap(t), d=spoken)
    args = {"action": op}
    if op in ("add", "subtract"):
        args.update(dur_args(hh, m, s))
    if nm:
        args["name"] = nm
    if r < 0.6:
        ans = R.choice(answers).format(t=t, D=S.fmt_duration(hh, m, s, LANG))
    elif r < 0.8:
        ans = R.choice(["Er loopt nu geen timer.", "Er loopt helaas geen timer."])
    else:
        others = ["de " + timer_label(x[0]) for x in timers]
        ans = f"Er is geen {t}. Alleen {join_nl(others)} {'lopen' if len(others) > 1 else 'loopt'}."
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, result, ans)
    return sample(msgs, "timer", [["timer", args]])


def s_timer_ambiguous(h, loc):
    """Several named timers, the user does not say which: ask, then act."""
    op = R.choice(["cancel", "cancel", "pause", "status_skip"])
    if op == "status_skip":
        op = "cancel"
    timers = running_timers(R.choice([2, 2, 3]))
    entries = status_entries(timers)
    users, answers = TIMER_OPS[op]
    u = R.choice(users).format(t="timer", T="Timer", d="")
    args = {"action": op}
    names = [t[0] for t in timers]
    ask = R.choice([f"Welke timer bedoel je: {join_nl(names, 'of')}?",
                    f"Er lopen meerdere timers: {join_nl(names)}. Welke bedoel je?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["Allemaal", "Alle timers", "Allebei" if len(names) == 2 else "Alle"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          f"Ik heb de {len(names)} timers geannuleerd.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    fuer = TIMER_FOR[pick]
    msgs += [user(R.choice([pick, f"De {TIMER_COMP[pick]}", f"Die {fuer}", f"De timer {fuer}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      R.choice(answers).format(t=timer_label(pick), D=""))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Verwijder alle timers", "Annuleer alle timers", "Stop alle timers",
                  "Alle timers annuleren", "Zet alle timers uit", "Haal alle timers weg",
                  "Alle timers verwijderen", "Schakel alle timers uit", "Stop met alle timers",
                  "Kun je alle timers annuleren?", "Alle timers stoppen graag",
                  "Wis alle timers", "Alle timers uit"])
    args = {"action": "cancel_all"}
    ans = {0: "Er liep geen timer.", 1: "Ik heb de timer geannuleerd."}.get(
        n, f"Ik heb de {n} timers geannuleerd.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh}:{mm:02d}" if mm else f"{hh} uur"
    u = R.choice([f"Maak me wakker om {t}", f"Zet een wekker op {t}",
                  f"Zet de wekker voor morgenochtend om {t}", f"Wekker {t}"])
    ans = R.choice(["Een wekker voor een bepaalde tijd kan ik helaas niet zetten. "
                    "Ik kan wel een timer voor je zetten.",
                    "Een wekker op een vaste tijd lukt helaas niet, alleen een timer. "
                    "Zal ik een timer zetten?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Doe een omroepbericht: {main}", f"Zeg tegen iedereen dat {sub}",
                  f"Zeg in het hele huis dat {sub}", f"Omroepen: {main}",
                  f"Roep door het huis: {main}", f"Laat iedereen weten dat {sub}",
                  f"Meld overal dat {sub}", f"Kondig aan dat {sub}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Helaas is er geen ander spraakapparaat voor een omroepbericht.",
                        "Omroepen lukt helaas niet, er zijn geen andere spraakapparaten."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["Ik heb het omgeroepen.", "Het omroepbericht is verstuurd.",
                        "Gedaan, ik heb het in het hele huis omgeroepen."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("broadcast", args, res, ans)
    return sample(msgs, "broadcast", [["broadcast", args]])


def s_nevermind(h, loc):
    users, answers = C.CHAT[-1]
    msgs = base_msgs(h, loc) + [user(R.choice(users)), say(R.choice(answers))]
    return sample(msgs, "chat", [])


# --------------------------------------------------------------------------
# humidifier and valve position
# --------------------------------------------------------------------------
def i_humidifier(h, loc):
    ents = h.of("humidifier")
    if not ents:
        return None
    e = R.choice(ents)
    ref = C.ref_single(h, e)
    r = R.random()
    if r < 0.55:
        v = R.choice([35, 40, 45, 50, 55, 60])
        users = [f"Zet {ref.acc} op {v} procent", f"{cap(ref.acc)} op {v} procent",
                 f"Zet de luchtvochtigheid van {ref.acc} op {v} procent"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Zet de luchtvochtigheid {C.ROOM_BY_NAME[e.area][1]} op {v} procent")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} staat op {v} procent."
    on = r < 0.8
    users = [f"Zet {ref.acc} {'aan' if on else 'uit'}", f"Doe {ref.acc} {'aan' if on else 'uit'}"]
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, "aan" if on else "uit", "aangezet" if on else "uitgezet")


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "half"), (25, "op 25 procent"), (30, "op 30 procent"),
                        (75, "op 75 procent"), (50, "op 50 procent")])
    u = R.choice([f"Open {e.name} {word}", f"Zet {e.name} {word} open" if word == "half"
                  else f"Zet {e.name} {word}"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"{e.name} staat {p} procent open."


def music_desc(data):
    """search_and_play data -> 'muziek van Queen', 'het album 21', ..."""
    q, kind, artist = data.get("query", ""), data.get("media_type"), data.get("artist")
    if not q:
        return "muziek"
    if kind == "artist":
        return f"muziek van {q}"
    if kind == "track" and artist:
        return f"{q} van {artist}"
    if kind == "album":
        return f"het album {q}"
    if kind == "playlist":
        return f"de playlist {q}"
    return q


def describe_assist(h, a, who):
    """Wording for core.describe(): 'melk op de boodschappenlijst gezet'."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if dom == "todo":
        lst = next((e for e in h.ents if e.eid in a["entity_id"]), None)
        li = LISTS[lst.name][0] if lst is not None and lst.name in LISTS else who
        item = join_nl(data["items"]) if data.get("items") else data.get("item", "")
        return {"add_item": f"{item} op {li} gezet",
                "complete_item": f"{item} op {li} afgevinkt",
                "remove_item": f"{item} van {li} verwijderd"}.get(svc)
    if svc == "search_and_play":
        return f"{music_desc(data)} op {who} aangezet"
    if svc == "seek_relative":
        n = int(data.get("seconds") or 0)
        word = "een minuut" if abs(n) == 60 else f"{abs(n)} seconden"
        return f"op {who} {word} {'vooruitgespoeld' if n >= 0 else 'teruggespoeld'}"
    if svc == "shuffle_set":
        return f"shuffle op {who} {'aangezet' if data.get('shuffle') else 'uitgezet'}"
    if svc == "repeat_set":
        return {"one": f"herhalen van het nummer op {who} aangezet",
                "all": f"herhalen van alle nummers op {who} aangezet"}.get(
                    data.get("repeat"), f"herhalen op {who} uitgezet")
    if svc == "volume_mute":
        return f"{who} gedempt" if data.get("is_volume_muted") else \
            f"het geluid van {who} weer aangezet"
    if svc == "set_humidity":
        return f"{who} op {data.get('humidity')} procent luchtvochtigheid gezet"
    if svc == "set_valve_position":
        return f"{who} {data.get('position')} procent opengezet"
    return None


C.DESCRIBE_HOOKS.append(describe_assist)

C.COMMANDS += [(i_music_play, 6), (i_media_extra, 4), (i_todo_add, 4), (i_todo_complete, 2),
               (i_humidifier, 1), (i_valve_position, 1)]


# --------------------------------------------------------------------------
# conversation turns (each brings its own first step, so it can be forced)
# --------------------------------------------------------------------------
def conv_tool(c, name, args, result, answer):
    c.msgs += call_turn(name, args, result, answer)
    c.gold_last = [[name, args]]


def turn_music_follow(c):
    """'Speel Queen in de keuken' -> 'Harder' / 'Volgend nummer': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Harder", "Zet harder", "Iets harder graag"], "Harder gezet."),
        ("media_player.volume_down", ["Zachter", "Zet zachter", "Iets zachter"], "Zachter gezet."),
        ("media_player.media_next_track", ["Volgend nummer", "Volgende", "Overslaan"],
         "Volgend nummer."),
        ("media_player.media_pause", ["Pauze", "Even pauze", "Zet even op pauze"], "Gepauzeerd."),
        ("media_player.turn_off", ["Zet de muziek uit", "Uit"], "De muziek is uit."),
        ("media_player.shuffle_set", ["En graag door elkaar"], "Shuffle staat aan."),
        ("media_player.media_previous_track", ["Terug", "Vorige nummer", "Nog een keer het vorige"],
         "Vorig nummer."),
        ("media_player.seek_relative", ["Vooruitspoelen", "Spoel een stukje vooruit"],
         "Vooruitgespoeld."),
        ("media_player.seek_relative-", ["Terugspoelen", "Spoel een stukje terug"],
         "Teruggespoeld."),
        ("media_player.media_stop", ["Stop", "Stop de muziek"], "Gestopt."),
    ])
    data = {"shuffle": True} if svc.endswith("shuffle_set") else None
    if svc.startswith("media_player.seek_relative"):
        data = {"seconds": -30 if svc.endswith("-") else 30}
        svc = "media_player.seek_relative"
    c.user(R.choice(users))
    c.control([action(svc, [e], data)], ans)
    if svc == "media_player.media_pause" and R.random() < 0.6:
        c.user(R.choice(["Play", "Speel verder", "Hervat", "Ga verder"]))
        c.control([action("media_player.media_play", [e])], "Het afspelen gaat verder.")
    return True


def turn_timer_follow(c):
    """Start a timer, then ask how long it still runs or cancel it."""
    hh, m, s, spoken = duration()
    if not m and not hh:
        return False
    named = R.random() < 0.4
    name, fuer, _ = R.choice(TIMER_NAMES) if named else ("", "", "")
    c.user(f"Zet een timer {fuer + ' ' if named else ''}op {spoken}")
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"De {timer_label(name)} loopt: {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["Hoe lang nog?", "Hoe lang loopt hij nog?", "En hoe lang nog?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Annuleer hem maar weer", "Toch niet, verwijder de timer",
                         "Stop hem maar weer"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"De {timer_label(name)} is geannuleerd.")
    return True


def turn_list_follow(c):
    """'Zet melk op de boodschappenlijst' -> 'en eieren'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    li, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item = R.choice(free)
    c.user(R.choice([f"En {item}", f"{cap(item)} ook nog", f"En nog {item}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    c.control(a2, R.choice([f"{cap(item)} {staat(item)} nu ook op {li}.",
                            f"Ik heb {item} toegevoegd."]))
    return True


C.NEXT += [(turn_music_follow, 2), (turn_timer_follow, 2), (turn_list_follow, 1)]


def cases():
    return [
        ("music:ask", s_music_ask, {}), ("music:ask_answer", s_music_ask, {"answer": True}),
        ("music:error", s_music_error, {}),
        ("music:unclear_player", s_media_unclear, {}),
        ("music:satellite", s_music_sat, {}),
        ("music:library_list", s_library_list, {}), ("music:library_search", s_library_search, {}),
        ("music:library_play", s_library_play, {}),
        ("music:library_favorites", lambda h, loc: s_library_play(h, loc, True), {}),
        ("todo:question", s_todo_question, {}), ("todo:missing_item", s_todo_missing, {}),
        ("todo:no_list", s_todo_no_list, {}),
        ("weather:now", s_weather, {"wkind": "now"}), ("weather:rain", s_weather, {"wkind": "rain"}),
        ("weather:day", s_weather, {"wkind": "day"}), ("weather:temp", s_weather, {"wkind": "temp"}),
        ("weather:outdoor", s_weather, {"wkind": "outdoor"}),
        ("time:time", s_time, {"tkind": "time"}), ("time:date", s_time, {"tkind": "date"}),
        ("time:weekday", s_time, {"tkind": "weekday"}),
        ("timer:start", s_timer_start, {"named": False}),
        ("timer:start_named", s_timer_start, {"named": True}),
        ("timer:unsupported", s_timer_start, {"unsupported": True}),
        ("timer:delayed", s_timer_delayed, {}), ("timer:status", s_timer_status, {}),
        ("timer:cancel", s_timer_op, {"op": "cancel"}), ("timer:pause", s_timer_op, {"op": "pause"}),
        ("timer:resume", s_timer_op, {"op": "resume"}), ("timer:add", s_timer_op, {"op": "add"}),
        ("timer:subtract", s_timer_op, {"op": "subtract"}),
        ("timer:ambiguous", s_timer_ambiguous, {}), ("timer:cancel_all", s_timer_cancel_all, {}),
        ("timer:alarm_clock", s_alarm_clock, {}),
        ("broadcast", s_broadcast, {}), ("broadcast:error", s_broadcast, {"error": True}),
        ("chat:nevermind", s_nevermind, {}),
    ]


C.CASE_HOOKS.append(cases)
