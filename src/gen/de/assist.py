"""German samples for the Home Assistant Assist built-ins.

Music (search and play, shuffle, repeat, mute), timers on the voice device
(start, named, delayed commands, status, cancel, pause, resume, add,
subtract), to-do lists, weather with forecast, time and date, broadcasts,
humidifiers and valve positions.

Registers itself in core: entities through POPULATE_HOOKS, device commands
through COMMANDS (they get a case each and take part in compound commands and
conversations), conversation turns through NEXT, the rest through CASE_HOOKS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import core as C
from .core import R, S, action, base_msgs, cap, de_num, tool_call, tool_msg, voice

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


def join_de(items, word="und"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


VERBS = ("Mach", "Schalte", "Schalt", "Öffne", "Schließe", "Stell", "Fahr", "Starte",
         "Pausiere", "Aktiviere", "Dimme", "Spiel", "Setz", "Lass")


def lower_first(text):
    """Imperatives move behind 'In 10 Minuten'; nouns keep their capital."""
    return text[:1].lower() + text[1:] if text.split(" ", 1)[0] in VERBS else text


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
ARTISTS = ["Queen", "Helene Fischer", "Coldplay", "Die Ärzte", "Die Toten Hosen",
           "AnnenMayKantereit", "Herbert Grönemeyer", "Ed Sheeran", "Adele",
           "Taylor Swift", "Rammstein", "Peter Fox", "Nena", "ABBA", "The Beatles",
           "Mark Forster", "Max Raabe", "Nirvana", "Beyoncé", "Udo Lindenberg",
           "Kraftwerk", "Billie Eilish", "Linkin Park", "Clueso", "Wincent Weiss",
           "Apache 207", "Mozart", "Ludovico Einaudi", "Silbermond", "Revolverheld",
           "Andrea Berg", "Pink Floyd", "Daft Punk", "Dua Lipa", "Rolf Zuckowski"]
TRACKS = [("Bohemian Rhapsody", "Queen"), ("Atemlos durch die Nacht", "Helene Fischer"),
          ("Yellow", "Coldplay"), ("Schrei nach Liebe", "Die Ärzte"),
          ("Tage wie diese", "Die Toten Hosen"), ("Shape of You", "Ed Sheeran"),
          ("Hello", "Adele"), ("Mensch", "Herbert Grönemeyer"), ("99 Luftballons", "Nena"),
          ("Dancing Queen", "ABBA"), ("Let It Be", "The Beatles"),
          ("Haus am See", "Peter Fox"), ("Chöre", "Mark Forster"), ("Bad Guy", "Billie Eilish"),
          ("In the End", "Linkin Park"), ("Roller", "Apache 207"),
          ("Das Model", "Kraftwerk"), ("Symphonie", "Silbermond"),
          ("Levitating", "Dua Lipa"), ("Get Lucky", "Daft Punk"),
          ("Wish You Were Here", "Pink Floyd"), ("Pocahontas", "AnnenMayKantereit")]
ALBUMS = [("A Night at the Opera", "Queen"), ("Parachutes", "Coldplay"),
          ("Stadtaffe", "Peter Fox"), ("Abbey Road", "The Beatles"),
          ("Divide", "Ed Sheeran"), ("21", "Adele"), ("Mutter", "Rammstein"),
          ("Nevermind", "Nirvana"), ("Random Access Memories", "Daft Punk"),
          ("The Dark Side of the Moon", "Pink Floyd"), ("Folklore", "Taylor Swift"),
          ("Farbenspiel", "Helene Fischer")]
PLAYLISTS = ["Chillout", "Kochen", "Party", "Workout", "Entspannung", "Sommerhits",
             "Lieblingssongs", "Einschlafen", "Rock Classics", "Kinderlieder",
             "Weihnachtslieder", "Fokus", "Discover Weekly", "Roadtrip", "Achtziger"]
STATIONS = ["1LIVE", "WDR 2", "Bayern 3", "SWR3", "NDR 2", "Deutschlandfunk",
            "Antenne Bayern", "radioeins", "hr3", "bigFM", "Klassik Radio",
            "Radio Bob", "Energy", "MDR Jump", "FFN", "Fritz"]
GENRES = ["Jazz", "Klassik", "Rockmusik", "Entspannungsmusik", "Kindermusik",
          "Weihnachtsmusik", "Lo-Fi", "Schlager", "Hip-Hop", "Musik zum Einschlafen",
          "Musik zum Kochen", "Achtziger-Hits", "Popmusik", "Meditationsmusik"]

SPEAKER_NAMES = ["Sonos {room}", "{room} Lautsprecher", "Echo {room}", "HomePod {room}",
                 "Lautsprecher {room}", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "{room} Speaker"]
MUSIC_LABELS = ("Lautsprecher", "Radio", "Soundbar")

GROCERIES = ["Milch", "Eier", "Butter", "Brot", "Käse", "Äpfel", "Bananen", "Tomaten",
             "Kaffee", "Nudeln", "Reis", "Joghurt", "Mehl", "Zucker", "Toilettenpapier",
             "Spülmittel", "Waschmittel", "Katzenfutter", "Hafermilch", "Zwiebeln",
             "Kartoffeln", "Mineralwasser", "Orangensaft", "Müllbeutel", "Haferflocken",
             "Gurken", "Paprika", "Hackfleisch", "Sahne", "Quark", "Olivenöl", "Salz"]
LISTS = {  # name: (accusative, dative, items, slugs)
    "Einkaufsliste": ("die Einkaufsliste", "der Einkaufsliste", GROCERIES,
                      ["shopping_list", "einkaufsliste"]),
    "Aufgaben": ("die Aufgabenliste", "der Aufgabenliste",
                 ["Müll rausbringen", "Steuererklärung machen", "Auto waschen",
                  "Rasen mähen", "Oma anrufen", "Fenster putzen", "Keller aufräumen",
                  "Paket abholen", "Arzttermin machen", "Blumen gießen"],
                 ["aufgaben", "tasks"]),
    "Baumarkt": ("die Baumarkt-Liste", "der Baumarkt-Liste",
                 ["Schrauben", "Dübel", "Wandfarbe", "Silikon", "Klebeband",
                  "Glühbirnen", "Blumenerde", "Kabelbinder", "Schleifpapier"],
                 ["baumarkt", "hardware_store"]),
    "Drogerie": ("die Drogerie-Liste", "der Drogerie-Liste",
                 ["Zahnpasta", "Shampoo", "Duschgel", "Sonnencreme", "Deo",
                  "Wattepads", "Taschentücher", "Handcreme"],
                 ["drogerie", "drugstore"]),
    "Packliste": ("die Packliste", "der Packliste",
                  ["Ladekabel", "Sonnenbrille", "Reisepass", "Badehose", "Zahnbürste",
                   "Regenjacke", "Kopfhörer", "Adapter"],
                  ["packliste", "packing_list"]),
}

CITIES = ["Münster", "Freiburg", "Kassel", "Bremen", "Augsburg", "Kiel", "Erfurt",
          "Potsdam", "Bonn", "Ulm"]
WEATHER_NAMES = ["Forecast Home", "Zuhause", "Wetter", "OpenWeatherMap", "DWD Wetter",
                 "Met.no", "Wettervorhersage"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "Morgen ..." + fragment
FORECAST_FRAG = {
    "sunny": "wird es sonnig", "clear-night": "bleibt es klar", "cloudy": "wird es bewölkt",
    "partlycloudy": "wird es teilweise bewölkt", "rainy": "soll es regnen",
    "pouring": "soll es stark regnen", "snowy": "soll es schneien",
    "snowy-rainy": "gibt es Schneeregen", "fog": "wird es neblig", "hail": "soll es hageln",
    "lightning": "gibt es Gewitter", "lightning-rainy": "gibt es Gewitter mit Regen",
    "windy": "wird es windig", "windy-variant": "wird es windig und bewölkt",
    "exceptional": "ist ein Unwetter angesagt",
}
NOW_SENT = {
    "sunny": "Es ist sonnig", "clear-night": "Der Himmel ist klar", "cloudy": "Es ist bewölkt",
    "partlycloudy": "Es ist teilweise bewölkt", "rainy": "Es regnet",
    "pouring": "Es regnet stark", "snowy": "Es schneit", "snowy-rainy": "Es gibt Schneeregen",
    "fog": "Es ist neblig", "hail": "Es hagelt", "lightning": "Es gewittert",
    "lightning-rainy": "Es gewittert und regnet", "windy": "Es ist windig",
    "windy-variant": "Es ist windig und bewölkt", "exceptional": "Es gibt ein Unwetter",
}

TIMER_NAMES = [("Pizza", "für die Pizza"), ("Nudeln", "für die Nudeln"),
               ("Eier", "für die Eier"), ("Tee", "für den Tee"), ("Kuchen", "für den Kuchen"),
               ("Wäsche", "für die Wäsche"), ("Brot", "für das Brot"), ("Reis", "für den Reis"),
               ("Kartoffeln", "für die Kartoffeln"), ("Ofen", "für den Ofen"),
               ("Pause", "für die Pause"), ("Parkuhr", "für die Parkuhr"),
               ("Hausaufgaben", "für die Hausaufgaben"), ("Steak", "für das Steak"),
               ("Lasagne", "für die Lasagne"), ("Pommes", "für die Pommes"),
               ("Waffeln", "für die Waffeln"), ("Gemüse", "für das Gemüse")]
NUM_WORDS = {2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs", 7: "sieben",
             8: "acht", 10: "zehn", 12: "zwölf", 15: "fünfzehn", 20: "zwanzig",
             30: "dreißig"}

BROADCASTS = [("das Essen fertig ist", "Das Essen ist fertig"),
              ("wir in fünf Minuten losfahren", "Wir fahren in fünf Minuten los"),
              ("es Zeit zum Schlafen ist", "Es ist Zeit zum Schlafen"),
              ("die Pizza da ist", "Die Pizza ist da"),
              ("jemand an der Tür ist", "Jemand ist an der Tür"),
              ("alle zum Abendessen kommen sollen", "Alle sollen zum Abendessen kommen"),
              ("der Film gleich anfängt", "Der Film fängt gleich an"),
              ("ich gleich nach Hause komme", "Ich komme gleich nach Hause"),
              ("der Müll noch raus muss", "Der Müll muss noch raus"),
              ("wir Besuch bekommen", "Wir bekommen Besuch"),
              ("das Frühstück fertig ist", "Das Frühstück ist fertig"),
              ("der Kuchen fertig ist", "Der Kuchen ist fertig"),
              ("alle runterkommen sollen", "Alle sollen runterkommen"),
              ("wir gleich spazieren gehen", "Wir gehen gleich spazieren"),
              ("die Hausaufgaben jetzt dran sind", "Die Hausaufgaben sind jetzt dran")]


DATIVE = [("eine Minute", "einer Minute"), ("eine halbe Stunde", "einer halben Stunde"),
          ("eine Viertelstunde", "einer Viertelstunde"),
          ("eine Dreiviertelstunde", "einer Dreiviertelstunde"), ("eine Stunde", "einer Stunde")]
# items that take a plural verb ("Bananen stehen")
PLURAL_ITEMS = {"Eier", "Äpfel", "Bananen", "Tomaten", "Nudeln", "Zwiebeln", "Kartoffeln",
                "Müllbeutel", "Haferflocken", "Gurken", "Schrauben", "Dübel", "Glühbirnen",
                "Kabelbinder", "Wattepads", "Taschentücher", "Paprika"}


def dat(spoken):
    """'in eine Minute' -> 'in einer Minute'."""
    for a, b in DATIVE:
        if spoken.startswith(a):
            return b + spoken[len(a):]
    return spoken


def with_target(sentence, where):
    """Put the player before a closing 'hören': '... auf Sonos hören'."""
    if where and sentence.endswith(" hören"):
        return sentence[:-6] + where + " hören"
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
    slug = name.lower().replace(" ", "_").replace(".", "")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        slug = slug.replace(a, b)
    return h.add("media_player", room, "Lautsprecher", "m", [slug, "speaker"], name=name,
                 state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs)


def add_list(h, name, items=None):
    acc, dat, pool, slugs = LISTS[name]
    if items is None:
        items = R.sample(pool, R.choice([0, 1, 2, 3, 3, 4, 5, 6]))
    e = h.add("todo", None, name, "f", slugs, name=name, state=str(len(items)),
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
    "sunny": "bleibt es sonnig", "clear-night": "bleibt es klar", "cloudy": "bleibt es bewölkt",
    "partlycloudy": "bleibt es teilweise bewölkt", "rainy": "regnet es weiter",
    "pouring": "regnet es weiter stark", "snowy": "schneit es weiter",
    "snowy-rainy": "bleibt es bei Schneeregen", "fog": "bleibt es neblig",
    "hail": "hagelt es weiter", "lightning": "bleibt es gewittrig",
    "lightning-rainy": "bleibt es gewittrig mit Regen", "windy": "bleibt es windig",
    "windy-variant": "bleibt es windig und bewölkt", "exceptional": "bleibt das Unwetter",
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
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Außenbereich", "Garten")])
    slug = "forecast_home" if name in ("Forecast Home", "Zuhause") else \
        name.lower().replace(".", "").replace(" ", "_").replace("ü", "ue")
    e = h.add("weather", room, "Wetter", "n", [slug], name=name, state=cond, attrs=attrs)
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
        add_list(h, "Einkaufsliste")
        for name in R.sample([n for n in LISTS if n != "Einkaufsliste"], R.choice([0, 0, 1, 2])):
            add_list(h, name)
    elif R.random() < 0.15:
        add_list(h, R.choice([n for n in LISTS if n != "Einkaufsliste"]))
    if R.random() < 0.12:
        rooms = [r for r in h.rooms if r in ("Schlafzimmer", "Kinderzimmer", "Wohnzimmer", "Büro")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "Luftbefeuchter", "m", ["luftbefeuchter", "humidifier"],
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
                  "radio", "genre", "plain"])
    if k == "artist":
        a = R.choice(ARTISTS)
        u = R.choice([f"Spiel Musik von {a}", f"Spiel was von {a}", f"Ich möchte {a} hören",
                      f"Spiel Lieder von {a}", f"Spiel Songs von {a}",
                      f"Ich will Musik von {a} hören"])
        return u, {"query": a, "media_type": "artist"}, f"Musik von {a}"
    if k == "track":
        t, a = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Spiel {t} von {a}", f"Spiel das Lied {t} von {a}",
                          f"Ich will {t} von {a} hören", f"Spiel den Song {t} von {a}"])
            return u, {"query": t, "media_type": "track", "artist": a}, f"{t} von {a}"
        u = R.choice([f"Spiel das Lied {t}", f"Spiel den Song {t}", f"Spiel den Titel {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al, a = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Spiel das Album {al} von {a}", f"Spiel {a}s Album {al}"
                              if not a.endswith(("s", "z")) else f"Spiel das Album {al} von {a}"]),
                    {"query": al, "media_type": "album", "artist": a}, f"das Album {al}")
        return f"Spiel das Album {al}", {"query": al, "media_type": "album"}, f"das Album {al}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Spiel meine Playlist {p}", f"Spiel die Playlist {p}",
                      f"Starte die Playlist {p}", f"Spiel die {p}-Playlist"])
        return u, {"query": p, "media_type": "playlist"}, f"die Playlist {p}"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Spiel den Sender {s}", f"Spiel Radio {s}", f"Mach Radio {s} an",
                      f"Schalte den Radiosender {s} ein", f"Ich will den Sender {s} hören"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        g = R.choice(GENRES)
        return f"Spiel {g}", {"query": g}, g
    # "Spiel Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a for a in ARTISTS])
    return R.choice([f"Spiel {t}", f"Spiel mal {t}"]), {"query": t}, t


def music_target(h, loc):
    """-> (player, target words) or None when the player is not clear."""
    ps = music_players(h)
    if not ps:
        return None
    opts = ["name"]
    single_rooms = [r for r in sorted({e.area for e in ps if e.area})
                    if len([e for e in ps if e.area == r]) == 1]
    if single_rooms:
        opts.append("room")
    in_loc = [e for e in ps if loc and e.area == loc]
    if len(in_loc) == 1:
        opts += ["loc", "loc"]
    if len(ps) == 1:
        opts += ["only", "only"]
    k = R.choice(opts)
    if k == "name":
        e = R.choice(ps)
        return e, R.choice([f" auf {e.name}", f" auf {e.name}", f" über {e.name}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.ROOM_BY_NAME[room][1]
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Ich spiele {desc} auf {e.name}.", f"Okay, {desc} läuft auf {e.name}.",
                     f"{cap(desc)} läuft jetzt auf {e.name}.", f"Ich spiele {desc}."])


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
         ["Schalte die Zufallswiedergabe ein{w}", "Zufallswiedergabe an{w}", "Mach Shuffle an{w}",
          "Spiel die Lieder in zufälliger Reihenfolge{w}"],
         ["Die Zufallswiedergabe ist an.", "Shuffle ist an."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Zufallswiedergabe aus{w}", "Schalte Shuffle aus{w}", "Mach die Zufallswiedergabe aus{w}"],
         ["Die Zufallswiedergabe ist aus."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Wiederhole das Lied{w}", "Spiel das Lied in Dauerschleife{w}",
          "Wiederhole den aktuellen Titel{w}"], ["Das Lied wird wiederholt."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Wiederhole alles{w}", "Wiederhole die Playlist{w}", "Alle Titel wiederholen{w}"],
         ["Alle Titel werden wiederholt."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Wiederholung aus{w}", "Schalte das Wiederholen aus{w}", "Hör auf zu wiederholen{w}"],
         ["Die Wiederholung ist aus."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Ton aus{w}", "Stummschalten{w}", "Schalte den Ton aus{w}", "Mach den Ton aus{w}"],
         [f"{e.name} ist stummgeschaltet.", "Der Ton ist aus."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Ton wieder an{w}", "Stummschaltung aus{w}", "Mach den Ton wieder an{w}"],
         ["Der Ton ist wieder an.", f"{e.name} ist nicht mehr stummgeschaltet."]),
        ("media_player.media_previous_track", None,
         ["Vorheriges Lied{w}", "Zurück zum letzten Lied{w}", "Spiel das vorherige Lied{w}",
          "Ein Lied zurück{w}"], ["Ich habe zum vorherigen Titel gewechselt.", "Vorheriger Titel."]),
        ("media_player.media_stop", None,
         ["Stopp die Musik{w}", "Beende die Wiedergabe{w}", "Musik stoppen{w}"],
         ["Die Wiedergabe ist gestoppt.", "Gestoppt."]),
    ]
    svc, data, users, confs = R.choice(opts)
    return (R.choice(users).format(w=where), [action(svc, [e], data)], R.choice(confs))


def s_music_ask(h, loc):
    """Several speakers, none named and none in the room of the voice device:
    ask which one; optionally the user answers (or gives up)."""
    while len(music_players(h)) < 2:
        if not add_speaker(h):
            return None
    ps = music_players(h)
    if loc and len([e for e in ps if e.area == loc]) == 1:
        loc = None
    u, data, desc = music_request()
    opts = ps[:4]
    ask = R.choice([f"Auf welchem Lautsprecher soll ich {desc} spielen: "
                    f"{join_de([e.name for e in opts], 'oder')}?",
                    f"Wo soll ich {desc} spielen? Zur Auswahl stehen "
                    f"{join_de([e.name for e in opts], 'und')}."])
    msgs = base_msgs(h, loc) + [user(u), say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", [], no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Vergiss es", "Doch nicht", "Egal"])), say("Okay, dann nicht.")]
        return sample(msgs, "conversation", [], gold_last=[])
    e = R.choice(ps)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"Auf {e.name}", f"Auf {e.name} bitte"]
    if e.area and len(same_room) == 1:
        replies += [cap(C.ROOM_BY_NAME[e.area][1]), e.area]
    acts = [action("media_player.search_and_play", [e], data)]
    msgs += [user(R.choice(replies))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), play_answer(desc, e))
    return sample(msgs, "conversation", [], gold_last=[["control", {"actions": acts}]])


def s_music_error(h, loc):
    """Search found nothing, or the player cannot search."""
    out = i_music_play(h, loc)
    if not out:
        if not add_speaker(h):
            return None
        out = i_music_play(h, loc)
        if not out:
            return None
    u, acts, _ = out
    e = next(x for x in h.ents if x.eid == acts[0]["entity_id"][0])
    q = acts[0]["data"]["query"]
    if R.random() < 0.75:
        err = f"media_player.search_and_play failed for {e.eid}: No results found for {q}"
        ans = R.choice([f"Ich habe {q} leider nicht gefunden.",
                        f"Leider habe ich zu {q} nichts gefunden.",
                        f"Zu {q} habe ich leider nichts gefunden. Soll ich etwas anderes spielen?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Auf {e.name} kann ich leider keine Musik suchen.",
                        f"{e.name} unterstützt die Musiksuche leider nicht."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


# --------------------------------------------------------------------------
# to-do lists
# --------------------------------------------------------------------------
def pick_list(h, with_items=False):
    lists = [e for e in h.of("todo") if not with_items or e.attrs.get("items")]
    if not lists:
        return None
    shop = [e for e in lists if e.name == "Einkaufsliste"]
    return shop[0] if shop and R.random() < 0.65 else R.choice(lists)


def i_todo_add(h, loc):
    lst = pick_list(h)
    if not lst:
        return None
    acc, dat, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return None
    items = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    it = join_de(items)
    zu = "zur " + dat[4:]
    users = [f"Setz {it} auf {acc}", f"Schreib {it} auf {acc}", f"Füge {it} {zu} hinzu",
             f"{it} auf {acc}", f"Pack {it} auf {acc}", f"Trag {it} in {acc} ein"]
    if lst.name == "Einkaufsliste":
        users += [f"Wir brauchen {it}", f"Ich muss {it} kaufen", f"Setz {it} auf die Liste"
                  if len(h.of("todo")) == 1 else f"Setz {it} auf die Einkaufsliste"]
    if lst.name == "Aufgaben":
        users += [f"Füge {it} zu meinen Aufgaben hinzu", f"Setz {it} auf meine Aufgaben"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    pl = len(items) > 1 or items[0] in PLURAL_ITEMS
    conf = R.choice([f"Ich habe {it} auf {acc} gesetzt.",
                     f"{it} {'stehen' if pl else 'steht'} jetzt auf {dat}.",
                     f"Okay, {it} {'sind' if pl else 'ist'} auf {dat}."])
    return R.choice(users), acts, conf


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    acc, dat, _, _ = LISTS[lst.name]
    item = R.choice(lst.attrs["items"])
    if R.random() < 0.6:
        users = [f"Hak {item} auf {dat} ab", f"Streich {item} von {dat}",
                 f"{item} von {dat} abhaken"]
        if lst.name == "Einkaufsliste":
            users += [f"Ich habe {item} gekauft, hak es ab"]
        if lst.name == "Aufgaben":
            users += [f"{item} ist erledigt, hak es auf {dat} ab"]
        svc = "todo.complete_item"
        conf = R.choice([f"Ich habe {item} abgehakt.",
                         f"{item} {'sind' if item in PLURAL_ITEMS else 'ist'} auf {dat} abgehakt."])
    else:
        users = [f"Lösch {item} von {dat}", f"Entferne {item} von {dat}",
                 f"Nimm {item} von {dat} runter"]
        svc = "todo.remove_item"
        conf = R.choice([f"{item} {'sind' if item in PLURAL_ITEMS else 'ist'} von {dat} gelöscht.",
                         f"Ich habe {item} von {dat} entfernt."])
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], conf


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, "Einkaufsliste")
    acc, dat, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [i for i in pool if i not in items] or items)
        u = f"{'Stehen' if item in PLURAL_ITEMS else 'Steht'} {item} auf {dat}?"
        yes = item in items
        st = "stehen" if item in PLURAL_ITEMS else "steht"
        ans = f"Ja, {item} {st} auf {dat}." if yes else f"Nein, {item} {st} nicht auf {dat}."
        must = [item]
    else:
        users = [f"Was steht auf {dat}?", f"Lies mir {acc} vor", f"Was ist auf {dat}?"]
        if lst.name == "Einkaufsliste":
            users += ["Was muss ich einkaufen?", "Was brauchen wir noch?"]
        if lst.name == "Aufgaben":
            users += ["Was steht heute an?", "Was muss ich noch erledigen?"]
        u = R.choice(users)
        if not items:
            ans = R.choice([f"{cap(acc)} ist leer.", f"Auf {dat} steht gerade nichts."])
            must = ["leer" if "leer" in ans else "nichts"]
        elif len(items) == 1:
            ans = f"Auf {dat} steht nur {items[0]}."
            must = items
        else:
            ans = f"Auf {dat} stehen {join_de(items)}."
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, "Einkaufsliste")
    acc, dat, pool, _ = LISTS[lst.name]
    missing = [i for i in pool if i not in lst.attrs["items"]]
    item = R.choice(missing)
    u = R.choice([f"Hak {item} auf {dat} ab", f"Streich {item} von {dat}",
                  f"Lösch {item} von {dat}"])
    svc = "todo.remove_item" if u.startswith("Lösch") else "todo.complete_item"
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"{item} {'stehen' if item in PLURAL_ITEMS else 'steht'} leider nicht auf {dat}.",
                    f"Auf {dat} habe ich {item} nicht gefunden."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    acc, dat, pool, _ = LISTS[name]
    item = R.choice(pool)
    u = R.choice([f"Setz {item} auf {acc}", f"Was steht auf {dat}?", f"Schreib {item} auf {acc}"])
    others = [e.name for e in h.of("todo")]
    ans = f"{cap(acc)} habe ich nicht gefunden."
    if others:
        ans += f" Es gibt nur {join_de(['die Liste ' + n for n in others])}."
    else:
        ans = R.choice([ans, f"Ich habe leider keine Liste namens {name} gefunden."])
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
    temp = de_num(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == "Außentemperatur" for x in h.of("sensor")):
            return None
        u = R.choice(["Wie warm ist es draußen?", "Wie kalt ist es draußen?",
                      "Wie viel Grad hat es draußen?", "Wie ist die Außentemperatur?"])
        ans, must = f"Draußen sind es {temp} Grad.", [temp]
    elif kind == "now":
        u = R.choice(["Wie ist das Wetter?", "Wie ist das Wetter gerade?",
                      "Wie ist das Wetter draußen?", "Was macht das Wetter?",
                      "Wie ist es draußen?", "Wie ist das Wetter heute?",
                      "Wetter?", "Was sagt das Wetter?", "Erzähl mir was zum Wetter"])
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
        day_cap = cap(day) if idx < 2 else f"Am {day}"
        day_q = day if idx < 2 else f"am {day}"
        frag = FORECAST_FRAG[raw["condition"]]
        hi, lo = f["temperature"], f.get("templow")
        if kind == "rain":
            u = R.choice([f"Wird es {day_q} regnen?", f"Regnet es {day_q}?",
                          f"Brauche ich {day_q} einen Regenschirm?"])
            p = f.get("precipitation_probability")
            if raw["condition"] in RAINY:
                ans = f"Ja, {day} {frag}" + (f", die Regenwahrscheinlichkeit liegt bei {p} Prozent."
                                             if p is not None else ".")
                must = ["Ja"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"Nein, aber {day} {frag}" + (
                    f", die Niederschlagswahrscheinlichkeit liegt bei {p} Prozent."
                    if p is not None else ".")
                must = ["Nein"]
            else:
                ans = f"Nein, {day} {frag}" + (f", die Regenwahrscheinlichkeit liegt bei {p} Prozent."
                                               if p is not None else ".")
                must = ["Nein"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"Wie warm wird es {day_q}?", f"Wie viel Grad wird es {day_q}?"])
                ans, must = f"{day_cap} werden es bis zu {hi} Grad.", [str(hi)]
            else:
                u = R.choice([f"Wie kalt wird es {day_q}?", f"Wie kalt wird es {day_q} nachts?"])
                ans, must = f"{day_cap} wird es bis zu {lo} Grad kalt.", [str(lo)]
        else:
            u = R.choice([f"Wie wird das Wetter {day_q}?", f"Was sagt der Wetterbericht für {day_q.replace('am ', '')}?"
                          if idx >= 2 else f"Wie ist der Wetterbericht für {day}?",
                          f"Wie wird {day_q} das Wetter?"])
            ans = f"{day_cap} {frag}"
            ans += f", mit {lo} bis {hi} Grad." if lo is not None else f", bis zu {hi} Grad."
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = f"{NOW_SENT[e.state]} bei {temp} Grad. " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Heute werden es zwischen {lo} und {hi} Grad."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" Die Regenwahrscheinlichkeit liegt bei {p} Prozent."
            if C.VERBOSITY == "short":
                ans = f"{day_cap} {frag}, bis {hi} Grad."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def im_moment(sentence):
    """'Es regnet stark' -> 'Im Moment regnet es stark'."""
    w = sentence.split(" ")
    if w[0] != "Es":
        return sentence
    return " ".join(["Im Moment", w[1], "es"] + w[2:])


def hour_word(time):
    return f"{int(time.split(':')[0])} Uhr"


def rest_of_day(e):
    """'Ab 18 Uhr soll es regnen, um 21 Uhr sind es noch 8 Grad.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    end = f"um {hour_word(last['time'])} sind es noch {last['temperature']} Grad"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        return cap(R.choice(["bis zum Abend", "für den Rest des Tages", "auch später"])) + \
            f" {STAY[e.state]}, {end}."
    f = later[change]
    out = f"Ab {hour_word(f['time'])} {FORECAST_FRAG[raw[change]]} bei {f['temperature']} Grad"
    if change < len(later) - 1:
        out += f", {end}"
    return out + "."


def weather_report(e, temp, cond_word):
    """'Wie ist das Wetter?' by answer length: short - now; medium - now and
    the rest of the day; long - plus today's range and rain chance;
    detailed - plus humidity, wind and tomorrow."""
    a, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    first = R.choice([f"{NOW_SENT[e.state]} bei {temp} Grad.",
                      f"Draußen sind es gerade {temp} Grad, {cond_word}.",
                      im_moment(NOW_SENT[e.state]) + f" bei {temp} Grad."])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {temp} Grad.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Heute werden es zwischen {t['templow']} und {t['temperature']} Grad.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"Die Luftfeuchtigkeit liegt bei {a['humidity']} Prozent und der Wind "
                     f"weht mit {de_num(a['wind_speed'])} {a['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([f"Heute werden es zwischen {t['templow']} und {t['temperature']} Grad.",
                                   f"Heute erwartet dich ein Höchstwert von {t['temperature']} Grad, "
                                   f"die Tiefsttemperatur liegt bei {t['templow']} Grad."]))
        else:
            parts.append(f"Heute werden es bis zu {t['temperature']} Grad.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"Die Regenwahrscheinlichkeit liegt bei {p} Prozent." if p else
                         "Regen ist heute nicht zu erwarten.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Morgen {FORECAST_FRAG[raw['condition']]}, bis zu {fc[1]['temperature']} Grad.")
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
        u = R.choice(["Wie spät ist es?", "Wie viel Uhr ist es?", "Wie spät haben wir?",
                      "Sag mir die Uhrzeit", "Was ist die Uhrzeit?", "Uhrzeit?",
                      "Wie viel Uhr haben wir?"])
        hm = now.strftime("%H:%M")
        ans, must = R.choice([f"Es ist {hm} Uhr.", f"Es ist jetzt {hm} Uhr."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm} Uhr."
        elif C.VERBOSITY == "detailed":
            ans = f"Es ist {hm} Uhr am {wd}, dem {date}."
    elif kind == "date":
        u = R.choice(["Welches Datum ist heute?", "Der Wievielte ist heute?",
                      "Welches Datum haben wir?", "Was ist heute für ein Datum?",
                      "Den Wievielten haben wir heute?", "Sag mir das Datum",
                      "Welches Datum ist es?", "Datum?", "Was ist das heutige Datum?"])
        ans, must = f"Heute ist {wd}, der {date}.", [date.split(" ")[0] + " " + date.split(" ")[1]]
    else:
        u = R.choice(["Welcher Tag ist heute?", "Welchen Wochentag haben wir?",
                      "Was ist heute für ein Tag?", "Welcher Wochentag ist heute?",
                      "Welchen Tag haben wir heute?", "Ist heute schon Freitag?",
                      "Was für ein Tag ist heute?", "Sag mir, welcher Tag heute ist",
                      "Welcher Wochentag ist es?", "Haben wir heute Montag?",
                      "Was ist heute?", "Ist heute Wochenende?"])
        ans, must = R.choice([f"Heute ist {wd}.", f"Heute ist {wd}, der {date}."]), [wd]
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
        return 0, m, 0, "eine Minute" if m == 1 else f"{numw(m)} Minuten"
    if k < 0.65:
        return R.choice([(0, 30, 0, "eine halbe Stunde"), (0, 15, 0, "eine Viertelstunde"),
                         (0, 45, 0, "eine Dreiviertelstunde"), (1, 0, 0, "eine Stunde"),
                         (1, 30, 0, "anderthalb Stunden"), (1, 30, 0, "eineinhalb Stunden"),
                         (2, 0, 0, "zwei Stunden"), (2, 0, 0, "2 Stunden"),
                         (3, 0, 0, "drei Stunden")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        return 0, 0, s, f"{numw(s)} Sekunden"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = "eine Minute" if m == 1 else f"{numw(m)} Minuten"
        return 0, m, s, f"{mw} und {s} Sekunden"
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = "eine Stunde" if hh == 1 else f"{numw(hh)} Stunden"
    return hh, m, 0, f"{hw} und {m} Minuten"


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
    return f"{name}-Timer" if name else "Timer"


def s_timer_start(h, loc):
    hh, m, s, spoken = duration()
    named = F("named", R.random() < 0.35)
    name, fuer = R.choice(TIMER_NAMES) if named else ("", "")
    if named:
        u = R.choice([f"Stell einen Timer {fuer} auf {spoken}", f"Stell einen {name}-Timer auf {spoken}",
                      f"{name}-Timer {spoken}", f"Timer {fuer}, {spoken}",
                      f"Stell einen Timer namens {name} auf {spoken}",
                      f"{cap(spoken)} Timer {fuer}", f"Starte einen Timer {fuer}, {spoken}"])
    else:
        u = R.choice([f"Stell einen Timer auf {spoken}", f"Timer {spoken}",
                      f"Stell einen Timer für {spoken}", f"Starte einen Timer über {spoken}",
                      f"Weck mich in {dat(spoken)}", f"{cap(spoken)} Timer",
                      f"Kannst du einen Timer auf {spoken} stellen?",
                      f"Erinnere mich in {dat(spoken)}", f"Timer auf {spoken}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Auf diesem Gerät kann ich leider keinen Timer stellen. Das geht "
                        "über ein Sprachgerät wie Home Assistant Voice.",
                        "Timer funktionieren leider nur auf einem Sprachgerät, nicht hier.",
                        "Hier kann ich leider keinen Timer stellen, nur auf einem Sprachgerät."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        ans = R.choice([f"Der {name}-Timer läuft: {dur}.", f"Der {name}-Timer ist auf {dur} gestellt.",
                        f"Okay, {dur} für {name}."])
    else:
        ans = R.choice([f"Der Timer läuft: {dur}.", f"Ich habe einen Timer auf {dur} gestellt.",
                        f"Timer für {dur} gestartet.", f"Okay, {dur}. Der Timer läuft."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Mach in 10 Minuten das Licht aus' -> timer with the command; the
    command runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = out[0].rstrip(".!?")
    hh, m, s, spoken = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2:
        return None       # "Würdest du ..." does not fit behind "In 10 Minuten"
    sp = dat(spoken)
    forms = [f"In {sp} {lower_first(cmd)}", f"{cmd} in {sp}",
             f"Stell einen Timer auf {spoken} und dann {lower_first(cmd)}",
             f"{words[0]} in {sp} {words[1]}", f"{words[0]} in {sp} {words[1]}"]
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"Okay, in {dur} führe ich das aus.", f"Alles klar, ich erledige das in {dur}.",
                    f"In {dur} erledige ich das."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"Ein {asked_name}-Timer läuft gerade nicht.", ["nicht"]
        return R.choice(["Es läuft gerade kein Timer.", "Gerade läuft kein Timer."]), ["kein"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        if e.get("paused"):
            return f"Der {lab} ist pausiert, es bleiben noch {e['remaining']}.", [e["remaining"]]
        return R.choice([f"Der {lab} läuft noch {e['remaining']}.",
                         f"Noch {e['remaining']}."]), [e["remaining"]]
    parts = [f"{e['name'] or 'Ein Timer'}: noch {e['remaining']}"
             + (", pausiert" if e.get("paused") else "") for e in entries]
    return f"Es laufen {len(entries)} Timer. " + ". ".join(parts) + ".", \
        [e["remaining"] for e in entries]


def s_timer_status(h, loc):
    n = R.choice([0, 1, 1, 1, 2, 2, 3])
    timers = running_timers(n, named=n > 1 or R.random() < 0.4)
    named = [t for t in timers if t[0]]
    args = {"action": "status"}
    asked = None
    if named and R.random() < 0.35:
        asked = R.choice(named)[0] if R.random() < 0.8 else R.choice(TIMER_NAMES)[0]
        fuer = dict(TIMER_NAMES)[asked]
        u = R.choice([f"Wie lange läuft der {asked}-Timer noch?", f"Wie lange noch {fuer}?",
                      f"Wie viel Zeit hat der Timer {fuer} noch?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["Wie lange läuft der Timer noch?", "Wie viel Zeit ist noch auf dem Timer?",
                      "Wie lange noch?", "Welche Timer laufen?", "Läuft noch ein Timer?",
                      "Wie lange dauert der Timer noch?", "Timer Status"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


TIMER_OPS = {
    "cancel": (["Brich den {t} ab", "Stopp den {t}", "{T} abbrechen", "Lösch den {t}",
                "Beende den {t}", "Den {t} bitte ausmachen"],
               ["Der {t} ist abgebrochen.", "Ich habe den {t} gelöscht.", "Okay, {t} abgebrochen."]),
    "pause": (["Pausiere den {t}", "Halte den {t} an", "{T} pausieren", "Stopp mal kurz den {t}"],
              ["Der {t} ist pausiert.", "Ich habe den {t} angehalten."]),
    "resume": (["Setz den {t} fort", "Lass den {t} weiterlaufen", "{T} weiter",
                "Starte den {t} wieder"],
               ["Der {t} läuft weiter.", "Okay, der {t} läuft wieder."]),
    "add": (["Füg {d} zum {t} hinzu", "Verlängere den {t} um {d}", "Noch {d} auf den {t}",
             "Gib dem {t} {d} mehr"],
            ["Ich habe {D} zum {t} hinzugefügt.", "Der {t} läuft jetzt {D} länger."]),
    "subtract": (["Zieh {d} vom {t} ab", "Verkürze den {t} um {d}", "Nimm {d} vom {t} weg"],
                 ["Der {t} ist jetzt {D} kürzer.", "Ich habe {D} vom {t} abgezogen."]),
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
        nm = R.choice([n for n, _ in TIMER_NAMES if n not in [t[0] for t in timers]])
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
        ans = R.choice(["Es läuft gerade kein Timer.", "Gerade läuft leider kein Timer."])
    else:
        others = [timer_label(x[0]) for x in timers]
        ans = f"Einen {t} gibt es nicht. Es läuft nur {join_de(['der ' + o for o in others])}."
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
    u = R.choice(users).format(t="Timer", T="Timer", d="")
    args = {"action": op}
    names = [t[0] for t in timers]
    ask = R.choice([f"Welchen Timer meinst du: {join_de(names, 'oder')}?",
                    f"Es laufen mehrere Timer: {join_de(names)}. Welchen meinst du?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["Alle", "Alle Timer", "Beide" if len(names) == 2 else "Alle"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          f"Ich habe alle {len(names)} Timer abgebrochen.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    fuer = dict(TIMER_NAMES)[pick]
    msgs += [user(R.choice([pick, f"Den {pick}-Timer", f"Den {fuer}", f"Den Timer {fuer}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      R.choice(answers).format(t=timer_label(pick), D=""))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Lösch alle Timer", "Brich alle Timer ab", "Stopp alle Timer",
                  "Alle Timer abbrechen", "Mach alle Timer aus", "Beende alle Timer",
                  "Alle Timer löschen", "Schalte alle Timer aus", "Hör mit allen Timern auf",
                  "Kannst du alle Timer abbrechen?", "Bitte alle Timer stoppen",
                  "Lösch sämtliche Timer", "Alle Timer aus"])
    args = {"action": "cancel_all"}
    ans = {0: "Es lief gerade kein Timer.", 1: "Ich habe den Timer abgebrochen."}.get(
        n, f"Ich habe alle {n} Timer abgebrochen.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh}:{mm:02d}" if mm else f"{hh}"
    u = R.choice([f"Weck mich um {t} Uhr", f"Stell einen Wecker auf {t} Uhr",
                  f"Stell den Wecker für morgen früh um {t} Uhr", f"Wecker {t} Uhr"])
    ans = R.choice(["Einen Wecker für eine bestimmte Uhrzeit kann ich leider nicht stellen. "
                    "Ich kann dir aber einen Timer stellen.",
                    "Wecker zu einer Uhrzeit gehen leider nicht, nur Timer. Soll ich einen "
                    "Timer stellen?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Mach eine Durchsage: {main}", f"Sag allen, dass {sub}",
                  f"Sag im ganzen Haus, dass {sub}", f"Durchsage: {main}",
                  f"Ruf durchs Haus: {main}", f"Gib allen Bescheid, dass {sub}",
                  f"Sag überall durch, dass {sub}", f"Mach eine Ansage, dass {sub}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Leider gibt es kein anderes Sprachgerät für eine Durchsage.",
                        "Eine Durchsage geht leider nicht, es gibt keine weiteren Sprachgeräte."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["Ich habe die Durchsage gemacht.", "Die Durchsage ist raus.",
                        "Erledigt, ich habe es im ganzen Haus durchgesagt."])
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
        users = [f"Stell {ref.acc} auf {v} Prozent", f"{cap(ref.acc)} auf {v} Prozent",
                 f"Stell die Luftfeuchtigkeit von {ref.acc} auf {v} Prozent"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Stell die Luftfeuchtigkeit {C.ROOM_BY_NAME[e.area][1]} auf {v} Prozent")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} ist auf {v} Prozent eingestellt."
    on = r < 0.8
    users = [f"Schalte {ref.acc} {'ein' if on else 'aus'}", f"Mach {ref.acc} {'an' if on else 'aus'}"]
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, "an" if on else "aus", "eingeschaltet" if on else "ausgeschaltet")


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "zur Hälfte"), (25, "auf 25 Prozent"), (30, "auf 30 Prozent"),
                        (75, "auf 75 Prozent"), (50, "auf 50 Prozent")])
    u = R.choice([f"Öffne {e.name} {word}", f"Stell {e.name} {word}"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"{e.name} ist zu {p} Prozent geöffnet."


def music_desc(data):
    """'Musik von Queen', 'das Album 21', 'WDR 2' for search_and_play data."""
    q, t, a = data.get("query", ""), data.get("media_type"), data.get("artist")
    if t == "artist":
        return f"Musik von {q}"
    if t == "album":
        return f"das Album {q}"
    if t == "playlist":
        return f"die Playlist {q}"
    return f"{q} von {a}" if a else q


def describe_assist(h, a, who):
    """describe() wording for the services of this module."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    if dom == "todo":
        lst = next((e for e in h.ents if e.eid in a["entity_id"]), None)
        acc, dat = LISTS[lst.name][:2] if lst and lst.name in LISTS else ("die Liste", "der Liste")
        item = join_de(data["items"]) if data.get("items") else data.get("item", "")
        return {"add_item": f"{item} auf {acc} gesetzt",
                "complete_item": f"{item} auf {dat} abgehakt",
                "remove_item": f"{item} von {dat} gelöscht"}.get(svc)
    if svc == "search_and_play":
        return f"auf {who} {music_desc(data)} gestartet"
    if svc == "shuffle_set":
        return f"bei {who} die Zufallswiedergabe {'eingeschaltet' if data.get('shuffle') else 'ausgeschaltet'}"
    if svc == "repeat_set":
        return {"one": f"bei {who} das Wiederholen des Titels eingeschaltet",
                "all": f"bei {who} das Wiederholen aller Titel eingeschaltet"}.get(
            data.get("repeat"), f"bei {who} das Wiederholen ausgeschaltet")
    if svc == "volume_mute":
        return f"{who} stummgeschaltet" if data.get("is_volume_muted") else \
            f"bei {who} den Ton wieder eingeschaltet"
    if svc == "set_humidity":
        return f"{who} auf {data.get('humidity')} Prozent Luftfeuchtigkeit gestellt"
    if svc == "set_valve_position":
        return f"{who} zu {data.get('position')} Prozent geöffnet"
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
    """'Spiel Queen in der Küche' -> 'Lauter' / 'Nächstes Lied': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Lauter", "Mach lauter", "Etwas lauter bitte"], "Lauter gestellt."),
        ("media_player.volume_down", ["Leiser", "Mach leiser", "Etwas leiser"], "Leiser gestellt."),
        ("media_player.media_next_track", ["Nächstes Lied", "Weiter", "Überspringen"], "Nächster Titel."),
        ("media_player.media_pause", ["Pause", "Mach mal Pause", "Halt an"], "Pausiert."),
        ("media_player.turn_off", ["Mach die Musik aus", "Aus"], "Die Musik ist aus."),
        ("media_player.shuffle_set", ["Und bitte zufällig"], "Die Zufallswiedergabe ist an."),
    ])
    data = {"shuffle": True} if svc.endswith("shuffle_set") else None
    c.user(R.choice(users))
    c.control([action(svc, [e], data)], ans)
    return True


def turn_timer_follow(c):
    """Start a timer, then ask how long it still runs or cancel it."""
    hh, m, s, spoken = duration()
    if not m and not hh:
        return False
    named = R.random() < 0.4
    name, fuer = R.choice(TIMER_NAMES) if named else ("", "")
    c.user(f"Stell einen Timer {fuer + ' ' if named else ''}auf {spoken}")
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"Der {timer_label(name)} läuft: {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["Wie lange noch?", "Wie lange läuft er noch?", "Und wie lange noch?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Brich ihn wieder ab", "Doch nicht, lösch den Timer", "Stopp ihn wieder"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"Der {timer_label(name)} ist abgebrochen.")
    return True


def turn_list_follow(c):
    """'Setz Milch auf die Einkaufsliste' -> 'und Eier'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    acc, dat, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item = R.choice(free)
    c.user(R.choice([f"Und {item}", f"{item} auch noch", f"Und noch {item}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    st = "stehen" if item in PLURAL_ITEMS else "steht"
    c.control(a2, R.choice([f"{item} {st} jetzt auch auf {dat}.", f"Ich habe {item} ergänzt."]))
    return True


C.NEXT += [(turn_music_follow, 2), (turn_timer_follow, 2), (turn_list_follow, 1)]


def cases():
    return [
        ("music:ask", s_music_ask, {}), ("music:ask_answer", s_music_ask, {"answer": True}),
        ("music:error", s_music_error, {}),
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
