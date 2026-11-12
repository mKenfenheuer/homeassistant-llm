"""English samples for the Home Assistant Assist built-ins.

Port of src/gen/de/assist.py: music (search and play, shuffle, repeat, mute),
timers on the voice device (start, named, delayed commands, status, cancel,
pause, resume, add, subtract), to-do lists, weather with forecast, time and
date, broadcasts, humidifiers and valve positions.

Registers itself in core: entities through POPULATE_HOOKS, device commands
through COMMANDS (they get a case each and take part in compound commands and
conversations), conversation turns through NEXT, the rest through CASE_HOOKS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import core as C
from .core import R, S, action, base_msgs, be, cap, en_num, join_en, tool_call, tool_msg, voice

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


VERBS = ("Turn", "Switch", "Open", "Close", "Set", "Lower", "Raise", "Start", "Pause",
         "Activate", "Dim", "Play", "Put", "Make", "Run", "Stop", "Send", "Move", "Shut",
         "Roll", "Brighten", "Let", "Have")


def lower_first(text):
    """Imperatives move behind 'In 10 minutes'; names keep their capital."""
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
ARTISTS = ["Queen", "Coldplay", "Ed Sheeran", "Adele", "Taylor Swift", "The Beatles",
           "Arctic Monkeys", "Oasis", "Bruce Springsteen", "Fleetwood Mac", "Harry Styles",
           "Olivia Rodrigo", "The Rolling Stones", "Elton John", "Kendrick Lamar",
           "Lady Gaga", "Bob Dylan", "Radiohead", "ABBA", "Ariana Grande", "Johnny Cash",
           "Miles Davis", "The Weeknd", "Nirvana", "Beyoncé", "Billie Eilish",
           "Linkin Park", "Mozart", "Ludovico Einaudi", "Pink Floyd", "Daft Punk",
           "Dua Lipa", "Sabrina Carpenter", "Dolly Parton", "Raffi"]
TRACKS = [("Bohemian Rhapsody", "Queen"), ("Yellow", "Coldplay"),
          ("Shape of You", "Ed Sheeran"), ("Hello", "Adele"),
          ("Anti-Hero", "Taylor Swift"), ("Let It Be", "The Beatles"),
          ("Do I Wanna Know?", "Arctic Monkeys"), ("Wonderwall", "Oasis"),
          ("Born to Run", "Bruce Springsteen"), ("Dreams", "Fleetwood Mac"),
          ("As It Was", "Harry Styles"), ("Dancing Queen", "ABBA"),
          ("Rocket Man", "Elton John"), ("Bad Guy", "Billie Eilish"),
          ("In the End", "Linkin Park"), ("Blinding Lights", "The Weeknd"),
          ("Jolene", "Dolly Parton"), ("Creep", "Radiohead"),
          ("Levitating", "Dua Lipa"), ("Get Lucky", "Daft Punk"),
          ("Wish You Were Here", "Pink Floyd"), ("Espresso", "Sabrina Carpenter")]
ALBUMS = [("A Night at the Opera", "Queen"), ("Parachutes", "Coldplay"),
          ("Rumours", "Fleetwood Mac"), ("Abbey Road", "The Beatles"),
          ("Divide", "Ed Sheeran"), ("21", "Adele"), ("AM", "Arctic Monkeys"),
          ("Nevermind", "Nirvana"), ("Random Access Memories", "Daft Punk"),
          ("The Dark Side of the Moon", "Pink Floyd"), ("Folklore", "Taylor Swift"),
          ("OK Computer", "Radiohead")]
PLAYLISTS = ["Chill", "Cooking", "Party", "Workout", "Relaxing", "Summer Hits",
             "Favorites", "Sleep", "Rock Classics", "Kids Songs",
             "Christmas Songs", "Focus", "Discover Weekly", "Road Trip", "80s Hits"]
STATIONS = ["BBC Radio 1", "BBC Radio 2", "BBC Radio 6 Music", "Capital FM", "Heart",
            "Classic FM", "Absolute Radio", "LBC", "KEXP", "NPR", "WNYC",
            "BBC World Service", "Kiss FM", "Magic Radio", "Radio X", "Smooth Radio"]
GENRES = ["jazz", "classical music", "rock music", "relaxing music", "kids' music",
          "Christmas music", "lo-fi", "country music", "hip-hop", "music to fall asleep to",
          "music for cooking", "80s hits", "pop music", "meditation music"]

SPEAKER_NAMES = ["Sonos {room}", "{room} Speaker", "Echo {room}", "HomePod {room}",
                 "{room} Echo", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "{room} HomePod"]
MUSIC_LABELS = ("Speaker", "Radio", "Soundbar")

GROCERIES = ["milk", "eggs", "butter", "bread", "cheese", "apples", "bananas", "tomatoes",
             "coffee", "pasta", "rice", "yogurt", "flour", "sugar", "toilet paper",
             "dish soap", "laundry detergent", "cat food", "oat milk", "onions",
             "potatoes", "sparkling water", "orange juice", "trash bags", "oats",
             "cucumbers", "peppers", "ground beef", "cream", "cottage cheese", "olive oil", "salt"]
TASKS = ["take out the trash", "do the taxes", "wash the car", "mow the lawn",
         "call grandma", "clean the windows", "tidy up the basement",
         "pick up the parcel", "book a doctor's appointment", "water the plants"]
LISTS = {  # name: (phrase, phrase, items, slugs)
    "Shopping List": ("the shopping list", "the shopping list", GROCERIES,
                      ["shopping_list", "groceries"]),
    "Tasks": ("the to-do list", "the to-do list", TASKS, ["tasks", "to_do"]),
    "Hardware Store": ("the hardware store list", "the hardware store list",
                       ["screws", "wall plugs", "wall paint", "silicone", "duct tape",
                        "light bulbs", "potting soil", "cable ties", "sandpaper"],
                       ["hardware_store", "diy"]),
    "Pharmacy": ("the pharmacy list", "the pharmacy list",
                 ["toothpaste", "shampoo", "shower gel", "sunscreen", "deodorant",
                  "cotton pads", "tissues", "hand cream"],
                 ["pharmacy", "drugstore"]),
    "Packing List": ("the packing list", "the packing list",
                     ["phone charger", "sunglasses", "passport", "swimsuit", "toothbrush",
                      "rain jacket", "headphones", "travel adapter"],
                     ["packing_list", "packing"]),
}
C.LIST_PHRASES.update({n: v[0] for n, v in LISTS.items()})
C.TASK_VERBS.update(t.split(" ", 1)[0] for t in TASKS)
# items that take a plural verb ("bananas are")
PLURAL_ITEMS = {"eggs", "apples", "bananas", "tomatoes", "onions", "potatoes",
                "trash bags", "oats", "cucumbers", "peppers", "screws", "wall plugs",
                "light bulbs", "cable ties", "cotton pads", "tissues", "sunglasses",
                "headphones"}


def it_(item):
    """Tasks are verb phrases: quote them in answers."""
    return f"“{item}”" if item in TASKS else item


def is_(item, neg=False):
    pl = item in PLURAL_ITEMS
    if neg:
        return "aren't" if pl else "isn't"
    return "are" if pl else "is"


CITIES = ["Leeds", "Bristol", "Portland", "Austin", "Denver", "Dublin", "Edinburgh",
          "Toronto", "Brisbane", "Boston"]
WEATHER_NAMES = ["Forecast Home", "Home", "Weather", "OpenWeatherMap", "Met Office",
                 "Met.no", "Weather Forecast"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "... tomorrow": what the day brings
FORECAST_FRAG = {
    "sunny": "it'll be sunny", "clear-night": "it'll be clear", "cloudy": "it'll be cloudy",
    "partlycloudy": "it'll be partly cloudy", "rainy": "it's going to rain",
    "pouring": "there'll be heavy rain", "snowy": "it's going to snow",
    "snowy-rainy": "there'll be sleet", "fog": "it'll be foggy", "hail": "there'll be hail",
    "lightning": "there'll be thunderstorms",
    "lightning-rainy": "there'll be thunderstorms with rain",
    "windy": "it'll be windy", "windy-variant": "it'll be windy and cloudy",
    "exceptional": "severe weather is expected",
}
NOW_SENT = {
    "sunny": "It's sunny", "clear-night": "It's a clear night", "cloudy": "It's cloudy",
    "partlycloudy": "It's partly cloudy", "rainy": "It's raining",
    "pouring": "It's pouring", "snowy": "It's snowing", "snowy-rainy": "It's sleeting",
    "fog": "It's foggy", "hail": "It's hailing", "lightning": "There's a thunderstorm",
    "lightning-rainy": "There's a thunderstorm with rain", "windy": "It's windy",
    "windy-variant": "It's windy and cloudy", "exceptional": "There's severe weather",
}

TIMER_NAMES = [("pizza", "for the pizza"), ("pasta", "for the pasta"),
               ("egg", "for the eggs"), ("tea", "for the tea"), ("cake", "for the cake"),
               ("laundry", "for the laundry"), ("bread", "for the bread"), ("rice", "for the rice"),
               ("potato", "for the potatoes"), ("oven", "for the oven"),
               ("break", "for my break"), ("parking", "for the parking meter"),
               ("homework", "for homework"), ("steak", "for the steak"),
               ("lasagna", "for the lasagna"), ("fries", "for the fries"),
               ("waffles", "for the waffles"), ("veggies", "for the veggies")]
NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 10: "ten", 12: "twelve", 15: "fifteen", 20: "twenty",
             30: "thirty"}

# (reported speech after "that", the announcement itself)
BROADCASTS = [("dinner is ready", "Dinner is ready"),
              ("we're leaving in five minutes", "We're leaving in five minutes"),
              ("it's time for bed", "It's time for bed"),
              ("the pizza is here", "The pizza is here"),
              ("someone is at the door", "Someone is at the door"),
              ("everyone should come to dinner", "Everyone, please come to dinner"),
              ("the movie is about to start", "The movie is about to start"),
              ("I'll be home soon", "I'll be home soon"),
              ("the trash still needs to go out", "The trash still needs to go out"),
              ("we're having guests", "We're having guests"),
              ("breakfast is ready", "Breakfast is ready"),
              ("the cake is ready", "The cake is ready"),
              ("everyone should come downstairs", "Everyone, please come downstairs"),
              ("we're going for a walk soon", "We're going for a walk soon"),
              ("it's time for homework", "It's time for homework")]


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
    return h.add("media_player", room, "Speaker", "s", [slug, "speaker"], name=name,
                 state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs)


def add_list(h, name, items=None):
    acc, dat, pool, slugs = LISTS[name]
    if items is None:
        items = R.sample(pool, R.choice([0, 1, 2, 3, 3, 4, 5, 6]))
    e = h.add("todo", None, name, "s", slugs, name=name, state=str(len(items)),
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
    "sunny": "It'll stay sunny", "clear-night": "It'll stay clear",
    "cloudy": "It'll stay cloudy", "partlycloudy": "It'll stay partly cloudy",
    "rainy": "It'll keep raining", "pouring": "It'll keep pouring",
    "snowy": "It'll keep snowing", "snowy-rainy": "The sleet will continue",
    "fog": "It'll stay foggy", "hail": "It'll keep hailing",
    "lightning": "It'll stay stormy", "lightning-rainy": "It'll stay stormy and rainy",
    "windy": "It'll stay windy", "windy-variant": "It'll stay windy and cloudy",
    "exceptional": "The severe weather will continue",
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
    cond = R.choice(fitting_conditions(temp) + ["clear-night", "sunny", "cloudy", "partlycloudy"]
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
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Outdoor", "Garden")])
    slug = "forecast_home" if name in ("Forecast Home", "Home") else \
        name.lower().replace(".", "").replace(" ", "_")
    e = h.add("weather", room, "Weather", "s", [slug], name=name, state=cond, attrs=attrs)
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
        add_list(h, "Shopping List")
        for name in R.sample([n for n in LISTS if n != "Shopping List"], R.choice([0, 0, 1, 2])):
            add_list(h, name)
    elif R.random() < 0.15:
        add_list(h, R.choice([n for n in LISTS if n != "Shopping List"]))
    if R.random() < 0.12:
        rooms = [r for r in h.rooms if r in ("Bedroom", "Kids Room", "Living Room", "Office")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "Humidifier", "s", ["humidifier", "air_humidifier"],
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
        u = R.choice([f"Play music by {a}", f"Play something by {a}", f"I want to listen to {a}",
                      f"Play songs by {a}", f"Put on {a}" if a.startswith("The ") else f"Put on some {a}",
                      f"I'd like to hear some music by {a}"])
        return u, {"query": a, "media_type": "artist"}, f"music by {a}"
    if k == "track":
        t, a = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Play {t} by {a}", f"Play the song {t} by {a}",
                          f"I want to hear {t} by {a}", f"Put on {t} by {a}"])
            return u, {"query": t, "media_type": "track", "artist": a}, f"{t} by {a}"
        u = R.choice([f"Play the song {t}", f"Play the track {t}", f"Put on the song {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al, a = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Play the album {al} by {a}", f"Play {a}'s album {al}"]),
                    {"query": al, "media_type": "album", "artist": a}, f"the album {al}")
        return f"Play the album {al}", {"query": al, "media_type": "album"}, f"the album {al}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Play my {p} playlist", f"Play the playlist {p}",
                      f"Start the {p} playlist", f"Put on my {p} playlist"])
        return u, {"query": p, "media_type": "playlist"}, f"the {p} playlist"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Play the radio station {s}",
                      f"Play {s}" if "Radio" in s else f"Play {s} radio", f"Put on {s}",
                      f"Tune in to {s}", f"I want to listen to {s}"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        g = R.choice(GENRES)
        return R.choice([f"Play {g}", f"Put on some {g}"]), {"query": g}, g
    # "Play Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a for a in ARTISTS])
    if t in ARTISTS and not t.startswith("The ") and R.random() < 0.5:
        return f"Play some {t}", {"query": t}, t
    return f"Play {t}", {"query": t}, t


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
        return e, R.choice([f" on {e.name}", f" on {e.name}", f" through {e.name}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.ROOM_BY_NAME[room][1]
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Playing {desc} on {e.name}.", f"Okay, {desc} is playing on {e.name}.",
                     f"{cap(desc)} is now playing on {e.name}.", f"Playing {desc}."])


def i_music_play(h, loc):
    tgt = music_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    u, data, desc = music_request()
    return u + where, [action("media_player.search_and_play", [e], data)], \
        play_answer(desc, e)


def i_media_extra(h, loc):
    """Shuffle, repeat, mute, previous track, stop."""
    tgt = music_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    opts = [
        ("media_player.shuffle_set", {"shuffle": True},
         ["Turn on shuffle{w}", "Shuffle on{w}", "Shuffle the music{w}",
          "Play the songs in random order{w}"],
         ["Shuffle is on.", "Shuffle is now on."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Shuffle off{w}", "Turn off shuffle{w}", "Stop shuffling{w}"],
         ["Shuffle is off."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Repeat this song{w}", "Put this song on repeat{w}",
          "Repeat the current track{w}"], ["The song is on repeat."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Repeat everything{w}", "Repeat the playlist{w}", "Repeat all tracks{w}"],
         ["All tracks are on repeat."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Repeat off{w}", "Turn off repeat{w}", "Stop repeating{w}"],
         ["Repeat is off."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Mute{w}", "Mute the sound{w}", "Turn the sound off{w}", "Mute it{w}"],
         [f"{e.name} is muted.", "The sound is off."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Unmute{w}", "Sound back on{w}", "Turn the sound back on{w}"],
         ["The sound is back on.", f"{e.name} is no longer muted."]),
        ("media_player.media_previous_track", None,
         ["Previous song{w}", "Go back to the last song{w}", "Play the previous song{w}",
          "One song back{w}"], ["I've gone back to the previous track.", "Previous track."]),
        ("media_player.media_stop", None,
         ["Stop the music{w}", "Stop playback{w}", "Music stop{w}"],
         ["Playback is stopped.", "Stopped."]),
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
    ask = R.choice([f"Which speaker should I play {desc} on: "
                    f"{join_en([e.name for e in opts], 'or')}?",
                    f"Where should I play {desc}? You can choose "
                    f"{join_en([e.name for e in opts], 'or')}."])
    msgs = base_msgs(h, loc) + [user(u), say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", [], no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Never mind", "Forget it", "Doesn't matter"])),
                 say("Okay, never mind.")]
        return sample(msgs, "conversation", [], gold_last=[])
    e = R.choice(ps)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"On {e.name}", f"On {e.name}, please"]
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
        ans = R.choice([f"Sorry, I couldn't find {q}.",
                        f"Sorry, I didn't find anything for {q}.",
                        f"I couldn't find anything for {q}. Should I play something else?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Sorry, I can't search for music on {e.name}.",
                        f"{e.name} doesn't support music search, unfortunately."])
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
    shop = [e for e in lists if e.name == "Shopping List"]
    return shop[0] if shop and R.random() < 0.65 else R.choice(lists)


def i_todo_add(h, loc):
    lst = pick_list(h)
    if not lst:
        return None
    acc, _, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return None
    items = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    it = join_en(items)
    users = [f"Put {it} on {acc}", f"Add {it} to {acc}", f"Write {it} on {acc}",
             f"{it} on {acc}", f"Can you add {it} to {acc}?", f"Stick {it} on {acc}"]
    if lst.name == "Shopping List":
        users += [f"We need {it}", f"I need to buy {it}", f"Add {it} to the list"
                  if len(h.of("todo")) == 1 else f"Add {it} to the shopping list",
                  f"We're out of {it}"]
    if lst.name == "Tasks":
        users += [f"Add {it} to my tasks", f"Put {it} on my to-do list"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    pl = len(items) > 1 or items[0] in PLURAL_ITEMS
    shown = join_en([it_(i) for i in items])
    conf = R.choice([f"I've added {shown} to {acc}.",
                     f"{cap(shown)} {be(pl)} now on {acc}.",
                     f"Okay, {shown} {be(pl)} on {acc}."])
    return R.choice(users), acts, conf


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    acc, _, _, _ = LISTS[lst.name]
    item = R.choice(lst.attrs["items"])
    shown = it_(item)
    if R.random() < 0.6:
        users = [f"Check off {item} on {acc}", f"Cross {item} off {acc}",
                 f"Mark {item} as done on {acc}"]
        if lst.name == "Shopping List":
            users += [f"I bought {item}, check {'them' if item in PLURAL_ITEMS else 'it'} off"]
        if lst.name == "Tasks":
            users += [f"Mark {item} as done", f"I'm done with {item}, check it off"]
        svc = "todo.complete_item"
        conf = R.choice([f"I've checked off {shown}.",
                         f"{cap(shown)} {is_(item)} checked off {acc}."])
    else:
        users = [f"Delete {item} from {acc}", f"Remove {item} from {acc}",
                 f"Take {item} off {acc}"]
        svc = "todo.remove_item"
        conf = R.choice([f"{cap(shown)} {is_(item)} off {acc}.",
                         f"I've removed {shown} from {acc}."])
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], conf


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, "Shopping List")
    acc, _, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [i for i in pool if i not in items] or items)
        u = f"{cap(is_(item))} {item} on {acc}?"
        yes = item in items
        ans = f"Yes, {it_(item)} {is_(item)} on {acc}." if yes else \
            f"No, {it_(item)} {is_(item, neg=True)} on {acc}."
        must = [item]
    else:
        users = [f"What's on {acc}?", f"Read me {acc}", f"What do I have on {acc}?"]
        if lst.name == "Shopping List":
            users += ["What do I need to buy?", "What do we still need?"]
        if lst.name == "Tasks":
            users += ["What's on for today?", "What do I still have to do?"]
        u = R.choice(users)
        shown = [it_(i) for i in items]
        if not items:
            ans = R.choice([f"{cap(acc)} is empty.", f"There's nothing on {acc} right now."])
            must = ["empty" if "empty" in ans else "nothing"]
        elif len(items) == 1:
            ans = f"The only thing on {acc} is {shown[0]}."
            must = items
        else:
            ans = R.choice([f"On {acc}: {join_en(shown)}.",
                            f"{cap(acc)} has {join_en(shown)}."])
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, "Shopping List")
    acc, _, pool, _ = LISTS[lst.name]
    missing = [i for i in pool if i not in lst.attrs["items"]]
    item = R.choice(missing)
    u = R.choice([f"Check off {item} on {acc}", f"Cross {item} off {acc}",
                  f"Delete {item} from {acc}"])
    svc = "todo.remove_item" if u.startswith("Delete") else "todo.complete_item"
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"Sorry, {it_(item)} {is_(item, neg=True)} on {acc}.",
                    f"I couldn't find {it_(item)} on {acc}."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    acc, _, pool, _ = LISTS[name]
    item = R.choice(pool)
    u = R.choice([f"Put {item} on {acc}", f"What's on {acc}?", f"Add {item} to {acc}"])
    others = [e.name for e in h.of("todo")]
    ans = f"I couldn't find {acc}."
    if others:
        ans += f" There's only {join_en([LISTS[n][0] for n in others])}."
    else:
        ans = R.choice([ans, f"Sorry, I couldn't find a list called {name}."])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def now_with_temp(cond, temp):
    """'It's sunny and 12 degrees.' / 'There's a thunderstorm and it's 18 degrees.'"""
    s = NOW_SENT[cond]
    if s.startswith("It's") and " and " not in s:
        return f"{s} and {temp} degrees."
    return f"{s}, at {temp} degrees."


def s_weather(h, loc):
    ws = [e for e in h.of("weather") if e.state != "unavailable"]
    if not ws:
        if not add_weather(h):
            return None
        ws = h.of("weather")
    e = R.choice(ws)
    fc = e.attrs["forecast"]
    cond_word = S.lang_texts(LANG)["conditions"][e.state]
    temp = en_num(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == "Outdoor Temperature" for x in h.of("sensor")):
            return None
        u = R.choice(["How warm is it outside?", "How cold is it outside?",
                      "How many degrees is it outside?", "What's the outside temperature?"])
        ans, must = f"It's {temp} degrees outside.", [temp]
    elif kind == "now":
        u = R.choice(["What's the weather like?", "What's the weather like right now?",
                      "How's the weather outside?", "What's the weather doing?",
                      "What's it like outside?", "What's the weather like today?",
                      "Weather?", "What's the weather?", "Tell me about the weather"])
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
        day_cap = cap(day) if idx < 2 else f"On {day}"
        day_q = day if idx < 2 else f"on {day}"
        frag = FORECAST_FRAG[raw["condition"]]
        hi, lo = f["temperature"], f.get("templow")
        if kind == "rain":
            u = R.choice([f"Will it rain {day_q}?", f"Is it going to rain {day_q}?",
                          f"Do I need an umbrella {day_q}?"])
            p = f.get("precipitation_probability")
            chance = f". The chance of rain is {p} percent." if p is not None else "."
            if raw["condition"] in RAINY:
                ans = f"Yes, {frag} {day_q}" + chance
                must = ["Yes"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"No, but {frag} {day_q}" + (
                    f". The chance of precipitation is {p} percent." if p is not None else ".")
                must = ["No"]
            else:
                ans = f"No, {frag} {day_q}" + chance
                must = ["No"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"How warm will it get {day_q}?",
                              f"How many degrees will it be {day_q}?"])
                ans, must = f"{day_cap} it'll get up to {hi} degrees.", [str(hi)]
            else:
                u = R.choice([f"How cold will it get {day_q}?",
                              f"How cold will it be {day_q} at night?"])
                ans, must = f"{day_cap} it'll get down to {lo} degrees.", [str(lo)]
        else:
            u = R.choice([f"What's the weather going to be like {day_q}?",
                          f"What's the forecast for {day}?",
                          f"What will the weather be like {day_q}?"])
            ans = f"{cap(frag)} {day_q}"
            ans += f", with temperatures from {lo} to {hi} degrees." if lo is not None \
                else f", with highs of {hi} degrees."
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = now_with_temp(e.state, temp) + " " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Today it'll be between {lo} and {hi} degrees."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" The chance of rain is {p} percent."
            if C.VERBOSITY == "short":
                ans = f"{cap(frag)} {day_q}, up to {hi} degrees."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def right_now(sentence):
    """'It's pouring' -> 'Right now, it's pouring'."""
    return "Right now, " + sentence[0].lower() + sentence[1:]


def rest_of_day(e):
    """'From 18:00 it's going to rain, at around 14 degrees, and at 21:00
    it'll be 8 degrees.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    end = f"at {last['time']} it'll be {last['temperature']} degrees"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        span = R.choice(["until the evening", "for the rest of the day", "later on as well"])
        return f"{STAY[e.state]} {span}, and {end}."
    f = later[change]
    out = f"From {f['time']} {FORECAST_FRAG[raw[change]]}, at around {f['temperature']} degrees"
    if change < len(later) - 1:
        out += f", and {end}"
    return out + "."


def weather_report(e, temp, cond_word):
    """'What's the weather like?' by answer length: short - now; medium - now
    and the rest of the day; long - plus today's range and rain chance;
    detailed - plus humidity, wind and tomorrow."""
    a, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    first = R.choice([now_with_temp(e.state, temp),
                      f"It's {temp} degrees outside right now. {NOW_SENT[e.state]}.",
                      right_now(now_with_temp(e.state, temp))])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {temp} degrees.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Today it'll be between {t['templow']} and {t['temperature']} degrees.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"The humidity is {a['humidity']} percent and the wind is blowing "
                     f"at {en_num(a['wind_speed'])} {a['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([f"Today it'll be between {t['templow']} and {t['temperature']} degrees.",
                                   f"Today's high is {t['temperature']} degrees "
                                   f"and the low is {t['templow']} degrees."]))
        else:
            parts.append(f"Today it'll get up to {t['temperature']} degrees.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"The chance of rain is {p} percent." if p else
                         "No rain is expected today.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Tomorrow {FORECAST_FRAG[raw['condition']]}, with highs of "
                     f"{fc[1]['temperature']} degrees.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# time and date
# --------------------------------------------------------------------------
def s_time(h, loc):
    now = rand_dt()
    res = S.time_result(now, LANG)
    t = S.lang_texts(LANG)
    wd = t["weekdays"][now.weekday()]
    month = t["months"][now.month - 1]
    date = t["date_format"].format(day=now.day, month=month, year=now.year)
    kind = F("tkind") or R.choice(["time", "date", "weekday"])
    if kind == "time":
        u = R.choice(["What time is it?", "What's the time?", "Do you have the time?",
                      "Tell me the time", "What time is it now?", "Time?",
                      "Got the time?", "Can you tell me the time?", "What's the time now?",
                      "Current time?"])
        hm = now.strftime("%H:%M")
        ans, must = R.choice([f"It's {hm}.", f"It's {hm} right now."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm}."
        elif C.VERBOSITY == "detailed":
            ans = f"It's {hm} on {wd}, {date}."
    elif kind == "date":
        u = R.choice(["What's the date today?", "What's today's date?",
                      "What date is it?", "What's the date?",
                      "Which day of the month is it?", "Tell me the date",
                      "What date is it today?", "Date?", "Can you tell me today's date?"])
        ans, must = f"Today is {wd}, {date}.", [f"{month} {now.day}"]
    else:
        # (question, weekdays that make the answer "yes")
        u, yes_days = R.choice([
            ("What day is it today?", None), ("What day of the week is it?", None),
            ("What day is it?", None), ("Which weekday is it today?", None),
            ("What day is today?", None), ("Is it Friday yet?", {"Friday"}),
            ("Tell me what day it is", None), ("What day of the week is today?", None),
            ("Which day is it?", None), ("Is it Monday today?", {"Monday"}),
            ("What's today?", None), ("Is it the weekend?", {"Saturday", "Sunday"})])
        ans = R.choice([f"Today is {wd}.", f"Today is {wd}, {date}."])
        if yes_days is not None:
            ans = ("Yes, " if wd in yes_days else "No, ") + ans[0].lower() + ans[1:]
        must = [wd]
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
    """-> (hours, minutes, seconds, spoken, adjective form or None)."""
    k = R.random()
    if k < 0.5:
        m = R.choice([1, 2, 3, 4, 5, 5, 6, 7, 8, 10, 10, 12, 15, 20, 25, 30, 40, 45, 50, 60, 90])
        if m == 1:
            return 0, 1, 0, R.choice(["a minute", "one minute"]), "one-minute"
        n = numw(m)
        return 0, m, 0, f"{n} minutes", f"{n}-minute"
    if k < 0.65:
        return R.choice([(0, 30, 0, "half an hour", "half-hour"),
                         (0, 15, 0, "a quarter of an hour", None),
                         (0, 45, 0, "three quarters of an hour", None),
                         (1, 0, 0, "an hour", "one-hour"),
                         (1, 30, 0, "an hour and a half", None),
                         (1, 30, 0, "one and a half hours", None),
                         (2, 0, 0, "two hours", "two-hour"), (2, 0, 0, "2 hours", "2-hour"),
                         (3, 0, 0, "three hours", "three-hour")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        n = numw(s)
        return 0, 0, s, f"{n} seconds", f"{n}-second"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = R.choice(["a minute", "one minute"]) if m == 1 else f"{numw(m)} minutes"
        return 0, m, s, f"{mw} and {s} seconds", None
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = R.choice(["an hour", "one hour"]) if hh == 1 else f"{numw(hh)} hours"
    return hh, m, 0, f"{hw} and {m} minutes", None


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
    return f"{name} timer" if name else "timer"


def s_timer_start(h, loc):
    hh, m, s, spoken, adj = duration()
    named = F("named", R.random() < 0.35)
    name, fuer = R.choice(TIMER_NAMES) if named else ("", "")
    if named:
        u = R.choice([f"Set a timer {fuer} for {spoken}",
                      f"Set {C.a_an(name)} {name} timer for {spoken}",
                      f"{cap(name)} timer, {spoken}", f"Timer {fuer}, {spoken}",
                      f"Set a timer called {name} for {spoken}",
                      f"Start {C.a_an(adj)} {adj} timer {fuer}" if adj else f"Start a timer {fuer} for {spoken}",
                      f"Start a timer {fuer}, {spoken}"])
    else:
        u = R.choice([f"Set a timer for {spoken}", f"Timer for {spoken}",
                      f"Start a timer for {spoken}",
                      f"Set {C.a_an(adj)} {adj} timer" if adj else f"Set a timer for {spoken}",
                      f"Wake me up in {spoken}", f"{cap(adj)} timer" if adj else f"Timer, {spoken}",
                      f"Can you set a timer for {spoken}?",
                      f"Remind me in {spoken}", f"Start {C.a_an(adj)} {adj} timer" if adj
                      else f"Put on a timer for {spoken}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Sorry, I can't set a timer on this device. That works on a voice "
                        "device like Home Assistant Voice.",
                        "Sorry, timers only work on a voice device, not here.",
                        "I can't set a timer here, only on a voice device."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        ans = R.choice([f"The {name} timer is running: {dur}.",
                        f"The {name} timer is set for {dur}.",
                        f"Okay, {dur} {fuer}."])
    else:
        ans = R.choice([f"Your timer for {dur} is running.", f"I've set a timer for {dur}.",
                        f"Timer for {dur} started.", f"Okay, {dur}. The timer is running."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Turn off the light in 10 minutes' -> timer with the command; the
    command runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = cap(out[0].rstrip(".!?"))
    hh, m, s, spoken, _ = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2:
        return None       # "Can you ..." does not fit behind "In 10 minutes"
    forms = [f"In {spoken}, {lower_first(cmd)}", f"{cmd} in {spoken}",
             f"Set a timer for {spoken} and then {lower_first(cmd)}",
             f"{cmd} in {spoken}, please", f"In {spoken} {lower_first(cmd)}"]
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"Okay, I'll do that in {dur}.", f"Sure, I'll take care of that in {dur}.",
                    f"In {dur}, I'll do that."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"There's no {asked_name} timer running right now.", ["no"]
        return R.choice(["No timer is running right now.", "There's no timer running."]), ["no"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        if e.get("paused"):
            return f"The {lab} is paused with {e['remaining']} left.", [e["remaining"]]
        return R.choice([f"The {lab} has {e['remaining']} left.",
                         f"{cap(e['remaining'])} left."]), [e["remaining"]]
    parts = [f"{cap(e['name']) if e['name'] else 'One timer'}: {e['remaining']} left"
             + (", paused" if e.get("paused") else "") for e in entries]
    return f"There are {len(entries)} timers running. " + ". ".join(parts) + ".", \
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
        u = R.choice([f"How long is left on the {asked} timer?", f"How much longer {fuer}?",
                      f"How much time does the timer {fuer} have left?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["How long is left on the timer?", "How much time is left on the timer?",
                      "How much longer?", "Which timers are running?", "Is a timer running?",
                      "How long until the timer goes off?", "Timer status"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


TIMER_OPS = {
    "cancel": (["Cancel the {t}", "Stop the {t}", "Cancel {t}", "Delete the {t}",
                "End the {t}", "Turn off the {t}, please"],
               ["The {t} is canceled.", "I've canceled the {t}.", "Okay, the {t} is canceled."]),
    "pause": (["Pause the {t}", "Hold the {t}", "Pause {t}", "Stop the {t} for a moment"],
              ["The {t} is paused.", "I've paused the {t}."]),
    "resume": (["Resume the {t}", "Let the {t} keep running", "Unpause the {t}",
                "Start the {t} again"],
               ["The {t} is running again.", "Okay, the {t} is running again."]),
    "add": (["Add {d} to the {t}", "Extend the {t} by {d}", "Put {d} more on the {t}",
             "Give the {t} {d} more"],
            ["I've added {D} to the {t}.", "The {t} now runs {D} longer."]),
    "subtract": (["Take {d} off the {t}", "Shorten the {t} by {d}", "Subtract {d} from the {t}"],
                 ["The {t} is now {D} shorter.", "I've taken {D} off the {t}."]),
}


def s_timer_op(h, loc):
    op = F("op") or R.choice(list(TIMER_OPS))
    users, answers = TIMER_OPS[op]
    hh, m, s, spoken, _ = duration() if op in ("add", "subtract") else (0, 0, 0, "", None)
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
        ans = R.choice(["No timer is running right now.", "Sorry, there's no timer running."])
    else:
        others = [timer_label(x[0]) for x in timers]
        ans = (f"There's no {t}. Only {join_en(['the ' + o for o in others])} "
               f"{be(len(others) > 1)} running.")
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
    u = R.choice([x for x in users if "{t}" in x]).format(t="timer", T="Timer", d="")
    args = {"action": op}
    names = [t[0] for t in timers]
    ask = R.choice([f"Which timer do you mean: {join_en([f'the {n} timer' for n in names], 'or')}?",
                    f"There are several timers running: {join_en(names)}. Which one do you mean?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["All of them", "All timers", "Both" if len(names) == 2 else "All"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          "I've canceled both timers." if len(names) == 2
                          else f"I've canceled all {len(names)} timers.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    fuer = dict(TIMER_NAMES)[pick]
    msgs += [user(R.choice([pick, f"The {pick} timer", f"The one {fuer}", f"The timer {fuer}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      R.choice(answers).format(t=timer_label(pick), D=""))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Cancel all timers", "Cancel all the timers", "Stop all timers",
                  "Delete all timers", "Turn off all timers", "End all timers",
                  "Clear all timers", "Switch off all the timers", "Get rid of all timers",
                  "Can you cancel all timers?", "Please stop all timers",
                  "Delete every timer", "All timers off", "Stop every timer",
                  "Cancel every timer, please"])
    args = {"action": "cancel_all"}
    ans = {0: "No timer was running.", 1: "I've canceled the timer.",
           2: "I've canceled both timers."}.get(n, f"I've canceled all {n} timers.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh}:{mm:02d}" if mm else R.choice([f"{hh}", f"{hh} o'clock", f"{hh} am"])
    u = R.choice([f"Wake me up at {t}", f"Set an alarm for {t}",
                  f"Set the alarm for {t} tomorrow morning", f"Alarm at {t}"])
    ans = R.choice(["Sorry, I can't set an alarm for a specific time. "
                    "But I can set a timer for you.",
                    "Alarms at a set time aren't possible, only timers. Should I set a "
                    "timer instead?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Make an announcement: {main}", f"Tell everyone that {sub}",
                  f"Tell the whole house that {sub}", f"Announcement: {main}",
                  f"Broadcast: {main}", f"Let everyone know that {sub}",
                  f"Announce that {sub}", f"Tell everybody {sub}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Sorry, there's no other voice device for an announcement.",
                        "I can't make an announcement, there are no other voice devices."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["I've made the announcement.", "The announcement is out.",
                        "Done, I've announced it throughout the house."])
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
        users = [f"Set {ref.acc} to {v} percent", f"{cap(ref.acc)} to {v} percent",
                 f"Set the target humidity of {ref.acc} to {v} percent"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Set the humidity {C.ROOM_BY_NAME[e.area][1]} to {v} percent")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} is set to {v} percent."
    on = r < 0.8
    users = [f"Turn {'on' if on else 'off'} {ref.acc}", f"Switch {ref.acc} {'on' if on else 'off'}"]
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, "on" if on else "off", "turned on" if on else "turned off")


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "halfway"), (25, "to 25 percent"), (30, "to 30 percent"),
                        (75, "to 75 percent"), (50, "to 50 percent")])
    u = R.choice([f"Open {e.name} {word}",
                  f"Set {e.name} {word}" if word != "halfway" else f"Open {e.name} halfway"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"{e.name} is {p} percent open."


C.COMMANDS += [(i_music_play, 6), (i_media_extra, 4), (i_todo_add, 4), (i_todo_complete, 2),
               (i_humidifier, 1), (i_valve_position, 1)]


# --------------------------------------------------------------------------
# conversation turns (each brings its own first step, so it can be forced)
# --------------------------------------------------------------------------
def conv_tool(c, name, args, result, answer):
    c.msgs += call_turn(name, args, result, answer)
    c.gold_last = [[name, args]]


def turn_music_follow(c):
    """'Play Queen in the kitchen' -> 'Louder' / 'Next song': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Louder", "Turn it up", "A bit louder, please"], "Volume up."),
        ("media_player.volume_down", ["Quieter", "Turn it down", "A bit quieter"], "Volume down."),
        ("media_player.media_next_track", ["Next song", "Skip", "Skip this one"], "Next track."),
        ("media_player.media_pause", ["Pause", "Pause it for a moment", "Hold on, pause"], "Paused."),
        ("media_player.turn_off", ["Turn the music off", "Off"], "The music is off."),
        ("media_player.shuffle_set", ["And shuffle, please"], "Shuffle is on."),
    ])
    data = {"shuffle": True} if svc.endswith("shuffle_set") else None
    c.user(R.choice(users))
    c.control([action(svc, [e], data)], ans)
    return True


def turn_timer_follow(c):
    """Start a timer, then ask how long it still runs or cancel it."""
    hh, m, s, spoken, _ = duration()
    if not m and not hh:
        return False
    named = R.random() < 0.4
    name, fuer = R.choice(TIMER_NAMES) if named else ("", "")
    c.user(f"Set {C.a_an(name)} {name} timer for {spoken}" if named else f"Set a timer for {spoken}")
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"The {timer_label(name)} is running: {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["How much longer?", "How long is left on it?", "And how much longer?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Cancel it again", "Actually, delete the timer", "Stop it again"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"The {timer_label(name)} is canceled.")
    return True


def turn_list_follow(c):
    """'Put milk on the shopping list' -> 'and eggs'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    acc, _, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item = R.choice(free)
    c.user(R.choice([f"And {item}", f"{item} too", f"And also {item}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    c.control(a2, R.choice([f"{cap(it_(item))} {is_(item)} on {acc} now too.",
                            f"I've added {it_(item)} as well."]))
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
