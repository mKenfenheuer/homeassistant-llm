"""Italian samples for the Home Assistant Assist built-ins.

Port of src/gen/de/assist.py: music (search and play, the media library of a
player, shuffle, repeat, mute),
timers on the voice device, to-do lists, weather with forecast, time and date,
broadcasts, humidifiers and valve positions. Registers itself in core through
POPULATE_HOOKS, COMMANDS, NEXT and CASE_HOOKS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import core as C
from .core import (R, S, action, agree, base_msgs, cap, essere, gradi, it_num, join_it,
                   lower_first, pa, pnum, the, tool_call, tool_msg, voice)

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


# imperatives that may move behind "Tra 10 minuti"
VERBS = ("Accendi", "Spegni", "Apri", "Chiudi", "Alza", "Abbassa", "Imposta", "Metti",
         "Avvia", "Attiva", "Disattiva", "Regola", "Porta", "Ferma", "Fai", "Riproduci",
         "Tira", "Solleva", "Arresta", "Rendi", "Colora", "Manda", "Riporta", "Rimanda",
         "Esegui", "Riprendi", "Continua", "Passa", "Stacca")
# verbs directly followed by their object: "Spegni tra 10 minuti la luce"
SPLIT_VERBS = ("Accendi", "Spegni", "Apri", "Chiudi", "Alza", "Abbassa", "Imposta",
               "Avvia", "Attiva", "Disattiva", "Regola", "Porta", "Ferma", "Stacca")


def sample(msgs, cat, gold, **meta):
    out = {"cat": cat, "gold": gold}
    out.update(meta)
    return msgs, out


def call_turn(name, args, result, answer):
    call, cid = tool_call(name, args)
    return [call, tool_msg(cid, result), {"role": "assistant", "content": answer}]


def user(text):
    return {"role": "user", "content": voice(text)}


def say(text):
    return {"role": "assistant", "content": text}


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
ARTISTS = ["Vasco Rossi", "Laura Pausini", "Eros Ramazzotti", "Måneskin", "Ligabue",
           "Jovanotti", "Zucchero", "Tiziano Ferro", "Mina", "Lucio Battisti",
           "Fabrizio De André", "Elisa", "Marco Mengoni", "Ultimo", "Annalisa", "Mahmood",
           "Andrea Bocelli", "Ludovico Einaudi", "Negramaro", "Cesare Cremonini",
           "Gianna Nannini", "Pino Daniele", "Lucio Dalla", "Franco Battiato", "Elodie",
           "Queen", "Coldplay", "Ed Sheeran", "Adele", "Taylor Swift", "ABBA",
           "The Beatles", "Pink Floyd", "Daft Punk", "Dua Lipa", "Mozart"]
TRACKS = [("Albachiara", "Vasco Rossi"), ("La solitudine", "Laura Pausini"),
          ("Più bella cosa", "Eros Ramazzotti"), ("Zitti e buoni", "Måneskin"),
          ("Certe notti", "Ligabue"), ("L'ombelico del mondo", "Jovanotti"),
          ("Senza una donna", "Zucchero"), ("Emozioni", "Lucio Battisti"),
          ("La cura", "Franco Battiato"), ("Soldi", "Mahmood"),
          ("Con te partirò", "Andrea Bocelli"), ("Caruso", "Lucio Dalla"),
          ("Sere nere", "Tiziano Ferro"), ("Due vite", "Marco Mengoni"),
          ("Bohemian Rhapsody", "Queen"), ("Yellow", "Coldplay"), ("Shape of You", "Ed Sheeran"),
          ("Hello", "Adele"), ("Dancing Queen", "ABBA"), ("Let It Be", "The Beatles"),
          ("Levitating", "Dua Lipa"), ("Get Lucky", "Daft Punk"),
          ("Wish You Were Here", "Pink Floyd"), ("Napule è", "Pino Daniele")]
ALBUMS = [("Vado al massimo", "Vasco Rossi"), ("Teatro d'ira", "Måneskin"),
          ("Buon compleanno Elvis", "Ligabue"), ("La voce del padrone", "Franco Battiato"),
          ("Anima latina", "Lucio Battisti"), ("A Night at the Opera", "Queen"),
          ("Parachutes", "Coldplay"), ("Abbey Road", "The Beatles"), ("21", "Adele"),
          ("Random Access Memories", "Daft Punk"), ("The Dark Side of the Moon", "Pink Floyd"),
          ("Folklore", "Taylor Swift")]
PLAYLISTS = ["Chill", "Cucina", "Festa", "Allenamento", "Relax", "Hit dell'estate",
             "Preferite", "Per dormire", "Rock Classics", "Canzoni per bambini",
             "Canzoni di Natale", "Concentrazione", "Discover Weekly", "Viaggio in macchina",
             "Anni Ottanta"]
STATIONS = ["RTL 102.5", "Radio Deejay", "Radio 105", "Rai Radio 1", "Rai Radio 2",
            "Rai Radio 3", "Radio Capital", "Virgin Radio", "Radio Kiss Kiss", "RDS",
            "Radio Italia", "m2o", "Radio Monte Carlo", "Radio 24", "Radio Subasio",
            "Radio Freccia"]
GENRES = ["jazz", "musica classica", "musica rock", "musica rilassante", "musica per bambini",
          "musica natalizia", "lo-fi", "musica italiana", "hip hop", "musica per dormire",
          "musica per cucinare", "successi anni Ottanta", "musica pop", "musica per meditare"]

SPEAKER_NAMES = ["Sonos {room}", "Altoparlante {room}", "Echo {room}", "HomePod {room}",
                 "Cassa {room}", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "{room} Speaker"]
MUSIC_LABELS = ("altoparlante", "radio", "soundbar")


def _items(pairs):
    for item, code in pairs:
        C.ITEM_G[item] = code
    return [i for i, _ in pairs]


GROCERIES = _items([
    ("latte", "m"), ("uova", "fp"), ("burro", "m"), ("pane", "m"), ("formaggio", "m"),
    ("mele", "fp"), ("banane", "fp"), ("pomodori", "mp"), ("caffè", "m"), ("pasta", "f"),
    ("riso", "m"), ("yogurt", "m"), ("farina", "f"), ("zucchero", "m"),
    ("carta igienica", "f"), ("detersivo per i piatti", "m"), ("detersivo per la lavatrice", "m"),
    ("cibo per gatti", "m"), ("latte d'avena", "m"), ("cipolle", "fp"), ("patate", "fp"),
    ("acqua minerale", "f"), ("succo d'arancia", "m"), ("sacchetti della spazzatura", "mp"),
    ("fiocchi d'avena", "mp"), ("zucchine", "fp"), ("peperoni", "mp"), ("carne macinata", "f"),
    ("panna", "f"), ("ricotta", "f"), ("olio d'oliva", "m"), ("sale", "m"),
    ("parmigiano", "m"), ("mozzarella", "f"), ("basilico", "m"), ("biscotti", "mp")])
LISTS = {  # name: (tail after "lista", items, slugs)
    "Lista della spesa": ("della spesa", GROCERIES, ["shopping_list", "lista_della_spesa"]),
    "Cose da fare": ("delle cose da fare", _items([
        ("portare fuori la spazzatura", "inf"), ("fare la dichiarazione dei redditi", "inf"),
        ("lavare la macchina", "inf"), ("tagliare l'erba", "inf"), ("chiamare la nonna", "inf"),
        ("pulire le finestre", "inf"), ("riordinare la cantina", "inf"),
        ("ritirare il pacco", "inf"), ("prenotare il medico", "inf"),
        ("innaffiare le piante", "inf")]), ["cose_da_fare", "tasks"]),
    "Ferramenta": ("della ferramenta", _items([
        ("viti", "fp"), ("tasselli", "mp"), ("vernice", "f"), ("silicone", "m"),
        ("nastro adesivo", "m"), ("lampadine", "fp"), ("terriccio", "m"),
        ("fascette", "fp"), ("carta vetrata", "f")]), ["ferramenta", "hardware_store"]),
    "Farmacia": ("della farmacia", _items([
        ("dentifricio", "m"), ("shampoo", "m"), ("bagnoschiuma", "m"), ("crema solare", "f"),
        ("deodorante", "m"), ("dischetti di cotone", "mp"), ("fazzoletti", "mp"),
        ("crema per le mani", "f"), ("cerotti", "mp")]), ["farmacia", "drugstore"]),
    "Valigia": ("per la valigia", _items([
        ("caricabatterie", "m"), ("occhiali da sole", "mp"), ("passaporto", "m"),
        ("costume da bagno", "m"), ("spazzolino", "m"), ("giacca a vento", "f"),
        ("cuffie", "fp"), ("adattatore", "m")]), ["valigia", "packing_list"]),
}
SHOPPING = "Lista della spesa"
TASKS = "Cose da fare"
C.LIST_TAILS.update({n: v[0] for n, v in LISTS.items()})


def lst_name(name, prep=None):
    """'la lista della spesa', with prep 'nella lista della spesa'."""
    word = f"lista {LISTS[name][0]}"
    return pa(prep, word, "f") if prep else the(word, "f")


def it_user(item):
    """How the user says an item: 'il latte', 'chiamare la nonna'."""
    return C.the_item(item, quote=False)


def is_pl(item):
    return C.ITEM_G.get(item, "m").endswith("p")


CITIES = ["Bologna", "Torino", "Verona", "Padova", "Bergamo", "Parma", "Trento", "Lecce",
          "Perugia", "Ancona"]
WEATHER_NAMES = ["Forecast Home", "Casa", "Meteo", "OpenWeatherMap", "Met.no",
                 "Previsioni meteo", "ilMeteo"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "Domani ..." + fragment
FORECAST_FRAG = {
    "sunny": "sarà soleggiato", "clear-night": "il cielo resterà sereno",
    "cloudy": "sarà nuvoloso", "partlycloudy": "sarà parzialmente nuvoloso",
    "rainy": "pioverà", "pouring": "pioverà forte", "snowy": "nevicherà",
    "snowy-rainy": "ci sarà nevischio", "fog": "ci sarà nebbia", "hail": "grandinerà",
    "lightning": "ci saranno temporali", "lightning-rainy": "ci saranno temporali con pioggia",
    "windy": "ci sarà vento", "windy-variant": "sarà ventoso e nuvoloso",
    "exceptional": "è previsto maltempo",
}
NOW_SENT = {
    "sunny": "C'è il sole", "clear-night": "Il cielo è sereno", "cloudy": "È nuvoloso",
    "partlycloudy": "È parzialmente nuvoloso", "rainy": "Piove",
    "pouring": "Piove forte", "snowy": "Nevica", "snowy-rainy": "C'è nevischio",
    "fog": "C'è nebbia", "hail": "Grandina", "lightning": "C'è un temporale",
    "lightning-rainy": "C'è un temporale con pioggia", "windy": "C'è vento",
    "windy-variant": "È ventoso e nuvoloso", "exceptional": "C'è maltempo",
}

# timer name, gender
TIMER_NAMES = [("Pizza", "f"), ("Pasta", "f"), ("Uova", "fp"), ("Tè", "m"), ("Torta", "f"),
               ("Bucato", "m"), ("Pane", "m"), ("Riso", "m"), ("Patate", "fp"), ("Forno", "m"),
               ("Pausa", "f"), ("Parcheggio", "m"), ("Compiti", "mp"), ("Bistecca", "f"),
               ("Lasagne", "fp"), ("Patatine", "fp"), ("Caffè", "m"), ("Verdure", "fp")]
TIMER_G = dict(TIMER_NAMES)
NUM_WORDS = {2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei", 7: "sette",
             8: "otto", 10: "dieci", 12: "dodici", 15: "quindici", 20: "venti",
             30: "trenta"}


def per(name):
    """'per la pizza'."""
    return "per " + the(name.lower(), TIMER_G[name])


BROADCASTS = [("la cena è pronta", "La cena è pronta"),
              ("partiamo tra cinque minuti", "Partiamo tra cinque minuti"),
              ("è ora di andare a letto", "È ora di andare a letto"),
              ("è arrivata la pizza", "È arrivata la pizza"),
              ("c'è qualcuno alla porta", "C'è qualcuno alla porta"),
              ("devono venire tutti a tavola", "Tutti a tavola"),
              ("il film sta per iniziare", "Il film sta per iniziare"),
              ("sto tornando a casa", "Sto tornando a casa"),
              ("bisogna ancora portare fuori la spazzatura",
               "Bisogna ancora portare fuori la spazzatura"),
              ("stanno arrivando gli ospiti", "Stanno arrivando gli ospiti"),
              ("la colazione è pronta", "La colazione è pronta"),
              ("la torta è pronta", "La torta è pronta"),
              ("devono scendere tutti", "Scendete tutti"),
              ("tra poco andiamo a fare una passeggiata",
               "Tra poco andiamo a fare una passeggiata"),
              ("adesso è ora di fare i compiti", "Adesso è ora di fare i compiti")]


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
    return h.add("media_player", room, "altoparlante", "m", [C.slugify(name), "speaker"],
                 name=name, state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs, ngender="f" if name.startswith("Cassa") else "m")


def add_list(h, name, items=None):
    _tail, pool, slugs = LISTS[name]
    if items is None:
        items = R.sample(pool, R.choice([0, 1, 2, 3, 3, 4, 5, 6]))
    e = h.add("todo", None, name, "fp" if name == TASKS else "f", slugs, name=name,
              state=str(len(items)), attrs={"items": list(items)})
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
    "sunny": "resterà soleggiato", "clear-night": "il cielo resterà sereno",
    "cloudy": "resterà nuvoloso", "partlycloudy": "resterà parzialmente nuvoloso",
    "rainy": "continuerà a piovere", "pouring": "continuerà a piovere forte",
    "snowy": "continuerà a nevicare", "snowy-rainy": "continuerà il nevischio",
    "fog": "resterà la nebbia", "hail": "continuerà a grandinare",
    "lightning": "resterà il rischio di temporali",
    "lightning-rainy": "continueranno i temporali con pioggia", "windy": "resterà ventoso",
    "windy-variant": "resterà ventoso e nuvoloso", "exceptional": "continuerà il maltempo",
}


def hourly_entries(now, cond, temp):
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
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Esterno", "Giardino")])
    slug = "forecast_home" if name in ("Forecast Home", "Casa") else C.slugify(name)
    e = h.add("weather", room, "meteo", "m", [slug], name=name, state=cond, attrs=attrs)
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
        rooms = [r for r in h.rooms if r in ("Camera da letto", "Cameretta", "Soggiorno",
                                             "Ufficio")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "umidificatore", "m", ["umidificatore", "humidifier"],
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
        return R.choice(["Metti della musica", "Metti qualcosa", "Fai partire un po' di musica",
                         "Metti un po' di musica", "Metti qualcosa dalla mia libreria",
                         "Voglio ascoltare musica", "Accendi la musica",
                         "Metti qualcosa di bello"]), {}, \
            "un po' di musica"
    if k == "artist":
        a = R.choice(ARTISTS)
        u = R.choice([f"Metti la musica di {a}", f"Metti qualcosa di {a}",
                      f"Voglio ascoltare {a}", f"Riproduci le canzoni di {a}",
                      f"Fai partire un po' di {a}", f"Ho voglia di sentire {a}"])
        return u, {"query": a, "media_type": "artist"}, f"la musica di {a}"
    if k == "track":
        t, a = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Metti {t} di {a}", f"Riproduci la canzone {t} di {a}",
                          f"Voglio sentire {t} di {a}", f"Fai partire il brano {t} di {a}"])
            return u, {"query": t, "media_type": "track", "artist": a}, f"{t} di {a}"
        u = R.choice([f"Metti la canzone {t}", f"Riproduci il brano {t}",
                      f"Fai partire la canzone {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al, a = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Metti l'album {al} di {a}", f"Riproduci l'album {al} di {a}"]),
                    {"query": al, "media_type": "album", "artist": a}, f"l'album {al}")
        return f"Metti l'album {al}", {"query": al, "media_type": "album"}, f"l'album {al}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Metti la mia playlist {p}", f"Riproduci la playlist {p}",
                      f"Avvia la playlist {p}", f"Fai partire la playlist {p}"])
        return u, {"query": p, "media_type": "playlist"}, f"la playlist {p}"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Metti la radio {s}", f"Sintonizza la radio su {s}",
                      f"Accendi la radio su {s}", f"Voglio ascoltare {s}",
                      f"Metti la stazione {s}"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        g = R.choice(GENRES)
        return R.choice([f"Metti un po' di {g}", f"Riproduci {g}"]), {"query": g}, f"un po' di {g}"
    # "Metti Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a for a in ARTISTS])
    return R.choice([f"Metti {t}", f"Fammi sentire {t}"]), {"query": t}, t


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
    single_rooms = [r for r in sorted({e.area for e in ps if e.area})
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
        return e, R.choice([f" su {e.name}", f" su {e.name}", f" da {e.name}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.ROOM_BY_NAME[room][1]
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Riproduco {desc} su {e.name}.", f"Ok, metto {desc} su {e.name}.",
                     f"Ecco {desc} su {e.name}.", f"Riproduco {desc}."])


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
         ["Attiva la riproduzione casuale{w}", "Riproduzione casuale{w}", "Attiva lo shuffle{w}",
          "Riproduci i brani in ordine casuale{w}"],
         ["La riproduzione casuale è attiva.", "Shuffle attivato."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Disattiva la riproduzione casuale{w}", "Togli lo shuffle{w}",
          "Basta con la riproduzione casuale{w}"],
         ["La riproduzione casuale è disattivata."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Ripeti questa canzone{w}", "Metti la canzone in loop{w}",
          "Ripeti il brano attuale{w}"], ["Ripeto la canzone.", "Il brano verrà ripetuto."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Ripeti tutto{w}", "Ripeti la playlist{w}", "Ripeti tutti i brani{w}"],
         ["Ripeto tutti i brani."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Disattiva la ripetizione{w}", "Smetti di ripetere{w}", "Basta ripetere{w}"],
         ["La ripetizione è disattivata."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Togli l'audio{w}", "Muto{w}", "Disattiva l'audio{w}", "Metti in muto{w}"],
         [f"Ho disattivato l'audio di {e.name}.", "L'audio è disattivato."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Riattiva l'audio{w}", "Togli il muto{w}", "Rimetti l'audio{w}"],
         ["L'audio è di nuovo attivo.", f"Ho riattivato l'audio di {e.name}."]),
        ("media_player.media_previous_track", None,
         ["Canzone precedente{w}", "Torna alla canzone di prima{w}",
          "Metti il brano precedente{w}", "Indietro di una canzone{w}"],
         ["Ho rimesso il brano precedente.", "Brano precedente."]),
        ("media_player.media_stop", None,
         ["Ferma la musica{w}", "Interrompi la riproduzione{w}", "Stop alla musica{w}", "Stop{w}"],
         ["Ho fermato la riproduzione.", "Fatto."]),
        ("media_player.media_play", None,
         ["Play{w}", "Riprendi la musica{w}", "Fai ripartire la musica{w}", "Continua{w}"],
         ["La riproduzione riprende.", "Si riparte."]),
        ("media_player.media_next_track", None,
         ["Prossima{w}", "Canzone successiva{w}", "Prossimo brano{w}"], ["Brano successivo."]),
        ("media_player.media_previous_track", None,
         ["Indietro{w}", "Brano precedente{w}"], ["Brano precedente."]),
        ("media_player.volume_up", None, ["Più forte{w}", "Alza il volume{w}"], ["Volume alzato."]),
        ("media_player.volume_down", None, ["Più piano{w}", "Abbassa il volume{w}"],
         ["Volume abbassato."]),
    ]
    sk = R.choice([10, 15, 30, 30, 60])
    sk_word = "un minuto" if sk == 60 else f"{sk} secondi"
    opts += [
        ("media_player.seek_relative", {"seconds": sk},
         [f"Vai avanti di {sk_word}{{w}}", "Avanti veloce{w}" if sk == 30 else
          f"Salta avanti di {sk_word}{{w}}", f"Manda avanti di {sk_word}{{w}}"],
         [f"Sono andato avanti di {sk_word}.", "Ok, avanti."]),
        ("media_player.seek_relative", {"seconds": -sk},
         [f"Torna indietro di {sk_word}{{w}}", "Riavvolgi{w}" if sk == 30 else
          f"Salta indietro di {sk_word}{{w}}", f"Riavvolgi di {sk_word}{{w}}"],
         [f"Sono tornato indietro di {sk_word}.", "Ok, indietro."]),
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
        msgs += [say(R.choice(["Al momento purtroppo nessun altoparlante è raggiungibile.",
                               "In questo momento nessuno dei tuoi altoparlanti è raggiungibile, "
                               "quindi non posso riprodurre musica.",
                               f"{join_it(gone[:4])} al momento "
                               f"{'non è raggiungibile' if len(gone) == 1 else 'non sono raggiungibili'}"
                               ", purtroppo non posso riprodurre niente."]))]
        return sample(msgs, "command", gold, no_history=True)
    if len(ps) == 1:
        e = ps[0]
        acts = [action("media_player.search_and_play", [e], data)]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Riproduco {desc} su {e.name}: al momento è l'unico "
                                    f"altoparlante disponibile.",
                                    play_answer(desc, e)]))
        return sample(msgs, "command", gold, no_history=True)
    opts = ps[:4]
    # numbers in the request (RTL 102.5, Rai Radio 2) are not in the tool result
    what = "la musica" if any(ch.isdigit() for ch in desc) else desc
    ask = R.choice([f"Su quale altoparlante devo riprodurre {what}: "
                    f"{join_it([e.name for e in opts], 'o')}?",
                    f"Dove vuoi che riproduca {what}? Puoi scegliere tra "
                    f"{join_it([e.name for e in opts], 'e')}."])
    gone = [e.name for e in ents if e.state == "unavailable"]
    if gone:
        ask = (f"{join_it(gone)} al momento "
               f"{'non è raggiungibile' if len(gone) == 1 else 'non sono raggiungibili'}. ") + ask
    msgs += [say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", gold, no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Lascia perdere", "No, niente", "Fa niente"])),
                 say("Ok, allora niente.")]
        return sample(msgs, "conversation", gold, gold_last=[])
    e = R.choice(opts)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"Su {e.name}", f"Su {e.name}, grazie"]
    if e.area and len(same_room) == 1:
        replies += [cap(C.ROOM_BY_NAME[e.area][1]), e.area]
    acts = [action("media_player.search_and_play", [e], data)]
    msgs += [user(R.choice(replies))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), play_answer(desc, e))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]])


def s_music_sat(h, loc):
    """'Metti della musica' on a voice device with its own speaker: play
    there, even when the room or the house has other speakers."""
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
    ("media_player.media_next_track", None, ["Canzone successiva", "Prossimo brano",
                                             "Passa alla prossima canzone", "Salta questa canzone",
                                             "Prossima"],
     "Su {n} parte il brano successivo."),
    ("media_player.media_previous_track", None, ["Canzone precedente", "Torna al brano di prima",
                                                 "Brano precedente", "Indietro"],
     "Su {n} ho rimesso il brano precedente."),
    ("media_player.media_pause", None, ["Pausa", "Metti in pausa la musica", "Ferma un attimo la musica"],
     "Ho messo in pausa la musica su {n}."),
    ("media_player.media_stop", None, ["Ferma la musica", "Spegni la musica", "Interrompi la riproduzione",
                                       "Stop"],
     "Ho fermato la riproduzione su {n}."),
    ("media_player.volume_up", None, ["Alza il volume", "Più forte, per favore", "Alza un po' la musica",
                                      "Più forte"],
     "Ho alzato il volume su {n}."),
    ("media_player.volume_down", None, ["Abbassa il volume", "Più piano, per favore",
                                        "Abbassa un po' la musica", "Più piano"],
     "Ho abbassato il volume su {n}."),
    ("media_player.seek_relative", {"seconds": 30}, ["Vai avanti", "Avanti veloce",
                                                     "Vai avanti di 30 secondi"],
     "Su {n} sono andato avanti di 30 secondi."),
    ("media_player.seek_relative", {"seconds": -30}, ["Torna indietro", "Riavvolgi",
                                                      "Torna indietro di 30 secondi"],
     "Su {n} sono tornato indietro di 30 secondi."),
    # play goes on with the paused title of the player
    ("media_player.media_play", None, ["Play", "Riprendi", "Continua", "Fai ripartire",
                                       "Continua la musica"],
     "Su {n} la riproduzione riprende."),
]


def s_media_unclear(h, loc):
    """'Canzone successiva' with several speakers, none named and none in the
    room of the voice device: read all players, act on the one that is
    playing ('Play': the one that is paused), ask when none or several are
    (v16 picked an unreachable player)."""
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
        ask = R.choice([f"Al momento non c'è niente in pausa. Cosa devo riprodurre e su quale "
                        f"altoparlante: {join_it(names, 'o')}?",
                        "Nessun altoparlante ha qualcosa in pausa. Cosa vuoi ascoltare?"])
    elif mode == "none":
        ask = R.choice([f"Al momento non c'è musica in riproduzione da nessuna parte. "
                        f"Quale altoparlante intendi: {join_it(names, 'o')}?",
                        "In questo momento nessun altoparlante sta suonando. Su quale devo farlo?"])
    elif resume:
        ask = f"La musica è in pausa su {join_it(names)}. Dove devo farla ripartire?"
    else:
        ask = f"La musica è in riproduzione su {join_it(names)}. Quale intendi?"
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
        ans = R.choice([f"Purtroppo non ho trovato {q}.",
                        f"Purtroppo non ho trovato niente per {q}.",
                        f"Per {q} non ho trovato niente. Vuoi che metta qualcos'altro?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Purtroppo su {e.name} non posso cercare musica.",
                        f"{e.name} purtroppo non supporta la ricerca musicale."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


# --------------------------------------------------------------------------
# media library of a player (music_library, media_player.browse_media)
# --------------------------------------------------------------------------
# mood -> (request words, playlist names that fit)
LIB_MOODS = {
    "relax": (["per rilassarmi", "per staccare un po'", "per un po' di relax",
               "per una serata tranquilla"],
              ["Chillout Lounge", "Relax totale", "Relax & Unwind", "Peaceful Piano",
               "Chill Out Music", "Totally Stress Free", "Serate tranquille"]),
    "sport": (["per fare sport", "per l'allenamento", "per la palestra", "per correre"],
              ["Gym Hits", "Workout Mix", "Running Beats", "Power Workout", "Allenamento 2026",
               "Beast Mode"]),
    "sleep": (["per dormire", "per addormentarmi", "per la notte"],
              ["Deep Sleep", "Musica per dormire", "Sleep Sounds", "Ninna nanna", "Buonanotte"]),
    "party": (["per la festa", "per fare festa", "per ballare"],
              ["Party Hits", "Hit da festa", "Dance Party", "Club Hits 2026", "Si balla!"]),
    "xmas": (["di Natale", "per Natale", "per le feste di Natale"],
             ["Canzoni di Natale", "Christmas Classics", "Natale 2025", "Xmas Hits"]),
    "cook": (["per cucinare", "per la cena", "da ascoltare a cena"],
             ["Cucinare con la musica", "Dinner Jazz", "Cooking Vibes", "Dinner for Two"]),
    "focus": (["per lavorare", "per concentrarmi", "per studiare", "per programmare"],
              ["Deep Focus", "Beats to think to", "Lofi Coders", "Concentrazione",
               "Programming Beats", "Musica per studiare"]),
    "rock": (["rock", "metal", "hard rock"],
             ["Rock Classics", "Inni rock", "Metal Love", "Hard Rock Hits"]),
    "kids": (["per i bambini", "per i bimbi"],
             ["Canzoni per bambini", "Baby Dance", "Zecchino d'Oro"]),
    "eighties": (["anni Ottanta", "degli anni Ottanta"],
                 ["Hit anni 80", "Best of the 80s", "Festa anni 80"]),
    "car": (["per il viaggio in macchina", "per il viaggio"],
            ["Roadtrip", "In viaggio", "Driving Hits"]),
}
FAV_NAMES = ["Preferiti", "Liked Songs", "Brani che ti piacciono", "I miei preferiti",
             "Canzoni preferite", "Favs", "Liked Songs Max", "Liked Songs Anna",
             "Le mie canzoni preferite", "Musica preferita"]
FAV_USERS = ["Metti le mie canzoni preferite", "Metti i miei preferiti",
             "Metti la mia musica preferita", "Voglio ascoltare le mie canzoni preferite",
             "Metti le canzoni che mi piacciono", "Metti la mia playlist preferita",
             "Fammi sentire i miei brani preferiti"]
NEUTRAL_PLAYLISTS = ["Discover Weekly", "Release Radar", "La mia playlist n. 3",
                     "La mia playlist n. 7", "Compleanno di Alex", "Vacanze 2024", "Estate 2025",
                     "Matrimonio", "Mix della settimana", "Nuove scoperte", "Brani di Shazam",
                     "Teen Wolf Soundtrack", "Electro Swing", "Deep House 2026",
                     "Psytrance Supernova", "Goa Mix", "Recently added tracks",
                     "Dobby + Maximilian", "In bici", "Festa dei boomer", "Welcome Back 20s"]
LIB_KINDS = {  # media_type -> (plural, user question words, pool)
    "playlist": ("playlist", ["Quali playlist ho", "Che playlist ho", "Quali playlist ci sono",
                              "Che playlist ho salvato", "Quali sono le mie playlist"], None),
    "album": ("album", ["Quali album ho", "Che album ho",
                        "Quali album ci sono nella mia libreria"], None),
    "artist": ("artisti", ["Quali artisti ho", "Che cantanti ho",
                           "Quali artisti ci sono nella mia libreria"], None),
    "radio": ("stazioni radio", ["Quali stazioni radio ho", "Che stazioni ho salvato",
                                 "Quali radio ci sono"], None),
    "podcast": ("podcast", ["Quali podcast ho", "A quali podcast sono iscritto"], None),
}
PODCASTS = ["Muschio Selvaggio", "Tintoria", "Morning", "Il Mondo", "Stories", "Indagini",
            "Elisa True Crime", "Daily Cogito", "Globo", "Non hanno un amico"]


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
        return R.choice([f"Su {who} non ho trovato {plural}.",
                         f"Nella libreria di {who} non ci sono {plural}."])
    k = {"short": 3, "medium": 5, "long": 6, "detailed": 8}[C.VERBOSITY]
    if total <= k:
        if total == 1:
            return f"C'è un solo elemento: {names[0]}."
        return f"Hai {total} {plural}: {join_it(names)}."
    shown = R.sample(names, k) if R.random() < 0.3 else names[:k]
    return f"Hai {total} {plural}, per esempio {join_it(shown)}."


def lib_target(h, loc):
    tgt = music_target(h, loc)
    if not tgt:
        if not add_speaker(h):
            return None
        tgt = music_target(h, loc)
    return tgt


def lib_where(sentence, where):
    """'Quali playlist ho' + ' su Sonos Cucina' + '?'."""
    return sentence + (where.replace(" da ", " su ") if where else "") + "?"


def s_library_list(h, loc):
    """'Quali playlist ho?' -> music_library -> names; optionally
    'Metti <name>' afterwards plays one of them."""
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
        # 'Quali album dei Queen ho?': albums are listed as 'Queen - Innuendo'
        a = R.choice(sorted({a for _, a in ALBUMS}))
        mine = [f"{a} - {al}" for al, aa in ALBUMS if aa == a]
        names = make_library("album", must=mine if R.random() < 0.85 else [], dash=True,
                             avoid=[] if R.random() < 0.85 else mine)
        hits = [n for n in names if a.casefold() in n.casefold()]
        args["search"] = a
        u = lib_where(R.choice([f"Quali album di {a} ho",
                                f"Che album di {a} ci sono",
                                f"Quali album ho di {a}"]), where)
        if hits:
            titles = [n.split(" - ", 1)[1] for n in hits]
            ans = (f"Di {a} hai l'album {titles[0]}." if len(titles) == 1 else
                   f"Di {a} hai {len(titles)} album: {join_it(titles)}.")
        else:
            ans = f"Nella tua libreria non ho trovato album di {a}."
        msgs = base_msgs(h, loc) + [user(u)]
        msgs += call_turn("music_library", args, S.library_result(hits), ans)
        return sample(msgs, "question", [["music_library", args]], no_history=True)
    msgs = base_msgs(h, loc) + [user(u)]
    gold = [["music_library", args]]
    if R.random() < 0.08:
        res = S.library_result(error=f"media library is not supported by {e.eid}")
        msgs += call_turn("music_library", args, res,
                          R.choice([f"Purtroppo {e.name} non permette di sfogliare la libreria.",
                                    f"Purtroppo non riesco a leggere la libreria di {e.name}."]))
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
    msgs += [user(R.choice([f"Metti {pick}", f"Allora metti {pick}", f"Metti {pick}, per favore",
                            f"{pick}, grazie"]))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                      R.choice([f"Riproduco {pick} su {e.name}.", f"Ecco {pick}."]))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]],
                  no_history=True)


def s_library_search(h, loc):
    """'Ho una playlist che si chiama Festa?' -> music_library with search."""
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
    q = {"playlist": [f"Ho una playlist che si chiama {word}", f"C'è una playlist {word}",
                      f"La playlist {word} è nella mia libreria"],
         "album": [f"Ho l'album {word}", f"L'album {word} è nella mia libreria"],
         "artist": [f"Ho della musica di {word}", f"{word} è nella mia libreria"],
         "radio": [f"Ho salvato la stazione {word}", f"C'è la stazione {word}"]}[kind]
    args = {"entity_id": e.eid, "media_type": kind, "search": word}
    res = S.library_result(hits)
    if hits:
        ans = R.choice([f"Sì, {hits[0]} è nella tua libreria." if len(hits) == 1 else
                        f"Sì, ho trovato {len(hits)} risultati: {join_it(hits[:5])}.",
                        f"Sì, {join_it(hits[:5])}."])
    else:
        ans = R.choice([f"No, nella tua libreria non ho trovato {word}.",
                        f"No, {word} non è nella libreria di {e.name}."])
    msgs = base_msgs(h, loc) + [user(lib_where(R.choice(q), where))]
    msgs += call_turn("music_library", args, res, ans)
    return sample(msgs, "question", [["music_library", args]], no_history=True)


def s_library_play(h, loc, favorites=False):
    """'Metti una playlist per rilassarmi' / 'Metti le mie canzoni preferite':
    read the playlists, play the one that fits; none fits -> say so and
    name a few."""
    tgt = lib_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    if favorites:
        u = R.choice(FAV_USERS)
        fit_pool, avoid = FAV_NAMES, FAV_NAMES
        what = "le tue canzoni preferite"
    else:
        mood = R.choice(list(LIB_MOODS))
        words, fit_pool = LIB_MOODS[mood]
        w = R.choice(words)
        u = R.choice([f"Metti una playlist {w}", f"Metti qualcosa {w}", f"Metti musica {w}",
                      f"Cercami una playlist {w}", f"Metti una delle mie playlist {w}"])
        avoid = fit_pool
        what = f"musica {w}"
    fits = R.random() < 0.8
    must = [R.choice(fit_pool)] if fits else []
    names = make_library("playlist", must=must, avoid=[n for n in avoid if n not in must])
    if not names:
        return None
    args = {"entity_id": e.eid, "media_type": "playlist"}
    msgs = base_msgs(h, loc) + [user(u + where)]
    call, cid = tool_call("music_library", args)
    msgs += [call, tool_msg(cid, S.library_result(names))]
    gold = [["music_library", args]]
    if fits:
        pick = must[0]
        acts = [action("media_player.search_and_play", [e],
                       {"query": pick, "media_type": "playlist"})]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Riproduco la playlist {pick} su {e.name}.",
                                    f"Ok, su {e.name} parte {pick}.",
                                    f"Ho avviato {pick}."]))
        return sample(msgs, "command", gold, no_history=True)
    some = R.sample(names, min(3, len(names)))
    ans = R.choice([f"Non ho trovato una playlist con {what}. Hai per esempio "
                    f"{join_it(some)}.",
                    f"Nessuna delle tue playlist è adatta. Vuoi che ne metta una di queste: "
                    f"{join_it(some, 'o')}?" if len(some) > 1 else
                    f"Nessuna delle tue playlist è adatta. Vuoi che metta {some[0]}?"])
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
    _tail, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return None
    items = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    it = join_it(items)
    nella, alla, sulla = (lst_name(lst.name, p) for p in ("in", "a", "su"))
    users = [f"Aggiungi {it} {alla}", f"Metti {it} {nella}", f"Scrivi {it} {nella}",
             f"Segna {it} {sulla}", f"Inserisci {it} {nella}", f"{cap(it)} {nella}"]
    if lst.name == SHOPPING:
        users += [f"Dobbiamo comprare {it}", f"Devo comprare {it}",
                  f"Aggiungi {it} alla lista" if len(h.of("todo")) == 1
                  else f"Aggiungi {it} alla lista della spesa"]
    if lst.name == TASKS:
        users += [f"Aggiungi {it} alle mie cose da fare", f"Metti {it} tra le cose da fare"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    pl = len(items) > 1 or is_pl(items[0])
    confs = [f"Ho aggiunto {it} {alla}.", f"Ok, ho messo {it} {nella}."]
    if lst.name != TASKS:
        confs.append(f"Ora {nella} {C.c_e(pl)} "
                     + (join_it(items) if len(items) > 1 else C.the_item(items[0])) + ".")
    return R.choice(users), acts, R.choice(confs)


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    item = R.choice(lst.attrs["items"])
    code = C.ITEM_G.get(item, "m")
    iu, ia = it_user(item), C.the_item(item)
    nella, dalla = lst_name(lst.name, "in"), lst_name(lst.name, "da")
    if R.random() < 0.6:
        users = [f"Spunta {iu} {nella}", f"Depenna {iu} {dalla}", f"Spunta {iu} {dalla}"]
        if lst.name == SHOPPING:
            users += [f"Ho comprato {iu}, spunta{C.clitic(code)}"]
        if lst.name == TASKS:
            users += [f"Ho finito di {item}, spuntalo {dalla}"]
        svc = "todo.complete_item"
        confs = [f"Ho spuntato {ia}."]
        confs.append(f"{cap(ia)} {essere(is_pl(item))} {agree('spuntato', code)} {nella}."
                     if code != "inf" else f"Ho segnato {ia} come fatto.")
    else:
        users = [f"Cancella {iu} {dalla}", f"Rimuovi {iu} {dalla}", f"Togli {iu} {dalla}"]
        svc = "todo.remove_item"
        confs = [f"Ho tolto {ia} {dalla}.", f"Ho cancellato {ia} {dalla}."]
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], R.choice(confs)


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOPPING)
    _tail, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    L, nella = lst_name(lst.name), lst_name(lst.name, "in")
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [i for i in pool if i not in items] or items)
        pl = is_pl(item)
        yes = item in items
        if C.ITEM_G.get(item) == "inf":
            u = f"Devo ancora {item}?"
            ans = (f"Sì, {C.the_item(item)} è ancora {nella}." if yes
                   else f"No, {C.the_item(item)} non è {nella}.")
        else:
            ti = C.the_item(item)
            u = R.choice([f"{cap(C.c_e(pl))} {ti} {nella}?", f"{cap(ti)} {essere(pl)} {nella}?"])
            ans = (f"Sì, {ti} {essere(pl)} {nella}." if yes
                   else f"No, {ti} non {essere(pl)} {nella}.")
        must = [item]
    else:
        users = [f"Cosa c'è {nella}?", f"Leggimi {L}", f"Cosa c'è scritto {nella}?"]
        if lst.name == SHOPPING:
            users += ["Cosa devo comprare?", "Cosa ci serve ancora?"]
        if lst.name == TASKS:
            users += ["Cosa c'è da fare oggi?", "Cosa devo ancora fare?"]
        u = R.choice(users)
        if not items:
            ans = R.choice([f"{cap(L)} è vuota.", f"{cap(nella)} al momento non c'è niente."])
            must = ["vuota" if "vuota" in ans else "niente"]
        elif len(items) == 1:
            ans = f"{cap(nella)} {C.c_e(is_pl(items[0]))} solo {C.the_item(items[0], quote=False)}."
            must = items
        else:
            ans = f"{cap(nella)} ci sono: {join_it(items)}."
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOPPING)
    _tail, pool, _ = LISTS[lst.name]
    missing = [i for i in pool if i not in lst.attrs["items"]]
    item = R.choice(missing)
    iu, ia = it_user(item), C.the_item(item)
    nella, dalla = lst_name(lst.name, "in"), lst_name(lst.name, "da")
    u = R.choice([f"Spunta {iu} {nella}", f"Depenna {iu} {dalla}",
                  f"Cancella {iu} {dalla}"])
    svc = "todo.remove_item" if u.startswith("Cancella") else "todo.complete_item"
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"Purtroppo {ia} non {essere(is_pl(item))} {nella}.",
                    f"Non ho trovato {ia} {nella}."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    _tail, pool, _ = LISTS[name]
    item = R.choice(pool)
    iu = it_user(item)
    u = R.choice([f"Aggiungi {iu} {lst_name(name, 'a')}", f"Cosa c'è {lst_name(name, 'in')}?",
                  f"Scrivi {iu} {lst_name(name, 'in')}"])
    others = [e.name for e in h.of("todo")]
    ans = f"Non ho trovato {lst_name(name)}."
    if others:
        ans += f" {cap(C.c_e(len(others) > 1))} solo {join_it([lst_name(n) for n in others])}."
    else:
        ans = R.choice([ans, f"Purtroppo non ho trovato nessuna lista chiamata «{name}»."])
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
    temp = it_num(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == C.OUT_LABEL for x in h.of("sensor")):
            return None
        u = R.choice(["Che temperatura c'è fuori?", "Quanto fa freddo fuori?",
                      "Quanti gradi ci sono fuori?", "Qual è la temperatura esterna?",
                      "Fa caldo fuori?"])
        ans, must = f"Fuori {C.ci_sono_gradi(temp)}.", [temp]
    elif kind == "now":
        u = R.choice(["Che tempo fa?", "Com'è il tempo adesso?", "Che tempo fa fuori?",
                      "Com'è il meteo?", "Com'è fuori?", "Che tempo fa oggi?",
                      "Meteo?", "Cosa dice il meteo?", "Dimmi qualcosa sul tempo"])
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
        day_q = day
        frag = FORECAST_FRAG[raw["condition"]]
        hi, lo = f["temperature"], f.get("templow")
        if kind == "rain":
            u = R.choice([f"Pioverà {day_q}?", f"{cap(day_q)} piove?",
                          f"Mi serve l'ombrello {day_q}?"])
            p = f.get("precipitation_probability")
            tail = (f", con una probabilità di pioggia {pnum('di', p)}."
                    if p is not None else ".")
            if raw["condition"] in RAINY:
                ans = f"Sì, {day} {frag}" + tail
                must = ["Sì"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"No, ma {day} {frag}" + (
                    f", la probabilità di precipitazioni è {pnum('di', p)}."
                    if p is not None else ".")
                must = ["No"]
            else:
                ans = f"No, {day} {frag}" + tail
                must = ["No"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"Quanto farà caldo {day_q}?", f"Quanti gradi ci saranno {day_q}?",
                              f"Che temperatura farà {day_q}?"])
                ans, must = R.choice([f"{day_cap} si arriverà fino a {gradi(hi)}.",
                                      f"{day_cap} la massima sarà di {gradi(hi)}."]), [str(hi)]
            else:
                u = R.choice([f"Quanto farà freddo {day_q}?",
                              f"Quanto farà freddo {day_q} di notte?"])
                ans, must = f"{day_cap} la minima scenderà a {gradi(lo)}.", [str(lo)]
        else:
            u = R.choice([f"Che tempo farà {day_q}?", f"Cosa dicono le previsioni per {day_q}?",
                          f"Come sarà il tempo {day_q}?", f"Che tempo fa {day_q}?"
                          if idx == 0 else f"Com'è il meteo per {day_q}?"])
            ans = f"{day_cap} {frag}"
            ans += f", con temperature tra {lo} e {gradi(hi)}." if lo is not None \
                else f", fino a {gradi(hi)}."
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = f"{NOW_SENT[e.state]} e la temperatura è di {gradi(temp)}. " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Oggi le temperature saranno tra {lo} e {gradi(hi)}."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" La probabilità di pioggia è {pnum('di', p)}."
            if C.VERBOSITY == "short":
                ans = f"{day_cap} {frag}, fino a {gradi(hi)}."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def al_momento(sentence):
    """'Piove forte' -> 'Al momento piove forte'."""
    return "Al momento " + lower_first(sentence)


def alle(time, prep="a"):
    """'21:00' -> 'alle 21', 'dall'una'."""
    hr = int(time.split(":")[0])
    if hr == 1:
        return {"a": "all'una", "da": "dall'una"}[prep]
    return f"{'alle' if prep == 'a' else 'dalle'} {hr}"


def rest_of_day(e):
    """'Dalle 18 pioverà con 9 gradi, alle 21 ci saranno ancora 8 gradi.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    t_last = last["temperature"]
    end = f"{alle(last['time'])} {'ci sarà' if str(t_last) in ('1', '-1') else 'ci saranno'} " \
          f"ancora {gradi(t_last)}"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        return cap(R.choice(["fino a sera", "per il resto della giornata", "anche più tardi"])) + \
            f" {STAY[e.state]}, {end}."
    f = later[change]
    out = f"{cap(alle(f['time'], 'da'))} {FORECAST_FRAG[raw[change]]}, con {gradi(f['temperature'])}"
    if change < len(later) - 1:
        out += f"; {end}"
    return out + "."


def weather_report(e, temp, cond_word):
    """'Che tempo fa?' by answer length: short - now; medium - now and the
    rest of the day; long - plus today's range and rain chance; detailed -
    plus humidity, wind and tomorrow."""
    a, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    now = NOW_SENT[e.state]
    first = R.choice([f"{now} e la temperatura è di {gradi(temp)}.",
                      f"Fuori {C.ci_sono_gradi(temp)}. {now}.",
                      al_momento(now) + f", con {gradi(temp)}."])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {gradi(temp)}.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Oggi le temperature saranno tra {t['templow']} e "
                         f"{gradi(t['temperature'])}.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"L'umidità è {pnum('di', a['humidity'])} e il vento soffia a "
                     f"{it_num(a['wind_speed'])} {a['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([f"Oggi le temperature saranno tra {t['templow']} e "
                                   f"{gradi(t['temperature'])}.",
                                   f"Oggi la massima sarà di {gradi(t['temperature'])} e la "
                                   f"minima di {gradi(t['templow'])}."]))
        else:
            parts.append(f"Oggi si arriverà fino a {gradi(t['temperature'])}.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"La probabilità di pioggia è {pnum('di', p)}." if p else
                         "Oggi non è prevista pioggia.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Domani {FORECAST_FRAG[raw['condition']]}, fino a "
                     f"{gradi(fc[1]['temperature'])}.")
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
        u = R.choice(["Che ore sono?", "Che ora è?", "Mi dici che ore sono?",
                      "Dimmi l'ora", "Sai che ore sono?", "Ora?", "Che ora abbiamo?"])
        hm = now.strftime("%H:%M")
        sono = "Sono le ore" if now.hour == 1 else "Sono le"
        ans, must = R.choice([f"{sono} {hm}.", f"Adesso {lower_first(sono)} {hm}."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm}."
        elif C.VERBOSITY == "detailed":
            ans = f"{sono} {hm} di {wd} {date}."
    elif kind == "date":
        u = R.choice(["Che data è oggi?", "Quanti ne abbiamo oggi?", "Che data abbiamo?",
                      "Qual è la data di oggi?", "Mi dici la data di oggi?", "Dimmi la data",
                      "Che giorno del mese è oggi?", "Data?", "Oggi che data è?"])
        ans, must = f"Oggi è {wd} {date}.", [date.split(" ")[0] + " " + date.split(" ")[1]]
    else:
        u = R.choice(["Che giorno è oggi?", "Che giorno della settimana è?",
                      "Oggi che giorno è?", "Che giorno della settimana è oggi?",
                      "In che giorno siamo?", "È già venerdì?",
                      "Dimmi che giorno è oggi", "Che giorno è?",
                      "Oggi è lunedì?", "È già il fine settimana?", "Che giorno abbiamo oggi?",
                      "Sai che giorno è oggi?"])
        ans, must = R.choice([f"Oggi è {wd}.", f"Oggi è {wd} {date}."]), [wd]
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
        return 0, m, 0, "un minuto" if m == 1 else f"{numw(m)} minuti"
    if k < 0.65:
        return R.choice([(0, 30, 0, "mezz'ora"), (0, 15, 0, "un quarto d'ora"),
                         (0, 45, 0, "tre quarti d'ora"), (1, 0, 0, "un'ora"),
                         (1, 30, 0, "un'ora e mezza"), (1, 30, 0, "un'ora e mezzo"),
                         (2, 0, 0, "due ore"), (2, 0, 0, "2 ore"),
                         (3, 0, 0, "tre ore")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        return 0, 0, s, f"{numw(s)} secondi"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = "un minuto" if m == 1 else f"{numw(m)} minuti"
        return 0, m, s, f"{mw} e {s} secondi"
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = "un'ora" if hh == 1 else f"{numw(hh)} ore"
    return hh, m, 0, f"{hw} e {m} minuti"


def dur_args(hh, m, s):
    return {k: v for k, v in (("hours", hh), ("minutes", m), ("seconds", s)) if v}


def running_timers(n, named=True, unique=True):
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
    """'timer della pizza', 'timer'."""
    return f"timer {pa('di', name.lower(), TIMER_G[name])}" if name else "timer"


def manca(remaining):
    """'manca 1 minuto', 'mancano 5 minuti'."""
    return "manca" if remaining.startswith("1 ") and " e " not in remaining \
        and "," not in remaining else "mancano"


def s_timer_start(h, loc):
    hh, m, s, spoken = duration()
    named = F("named", R.random() < 0.35)
    name = R.choice(TIMER_NAMES)[0] if named else ""
    if named:
        p = per(name)
        u = R.choice([f"Imposta un timer {p} di {spoken}", f"Timer {name.lower()} {spoken}",
                      f"Timer {p}, {spoken}", f"Imposta un timer chiamato {name} di {spoken}",
                      f"{cap(spoken)} {p}", f"Avvia un timer {p} da {spoken}",
                      f"Metti un timer di {spoken} {p}"])
    else:
        u = R.choice([f"Imposta un timer di {spoken}", f"Timer {spoken}",
                      f"Metti un timer di {spoken}", f"Avvia un timer da {spoken}",
                      f"Svegliami tra {spoken}", f"Timer da {spoken}",
                      f"Puoi impostare un timer di {spoken}?",
                      f"Avvisami tra {spoken}", f"Fai partire un timer di {spoken}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Su questo dispositivo purtroppo non posso impostare timer. Serve un "
                        "dispositivo vocale come Home Assistant Voice.",
                        "Purtroppo i timer funzionano solo su un dispositivo vocale, non qui.",
                        "Qui purtroppo non posso impostare un timer, solo su un dispositivo "
                        "vocale."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        lab = timer_label(name)
        ans = R.choice([f"Il {lab} è partito: {dur}.", f"Il {lab} è impostato su {dur}.",
                        f"Ok, {dur} {per(name)}."])
    else:
        ans = R.choice([f"Il timer è partito: {dur}.", f"Ho impostato un timer di {dur}.",
                        f"Timer di {dur} avviato.", f"Ok, {dur}. Il timer è partito."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Spegni la luce tra 10 minuti' -> timer with the command; the command
    runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = out[0].rstrip(".!?")
    hh, m, s, spoken = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2 or ", " in cmd:
        return None       # "Puoi accendere ...?" does not fit behind "Tra 10 minuti"
    forms = [f"Tra {spoken} {lower_first(cmd)}", f"{cmd} tra {spoken}",
             f"Imposta un timer di {spoken} e poi {lower_first(cmd)}",
             f"Fra {spoken} {lower_first(cmd)}"]
    if words[0] in SPLIT_VERBS:
        forms += [f"{words[0]} tra {spoken} {words[1]}"] * 2
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"Ok, tra {dur} lo faccio.", f"Va bene, me ne occupo tra {dur}.",
                    f"Tra {dur} ci penso io."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"Al momento non c'è nessun {timer_label(asked_name)} attivo.", ["non"]
        return R.choice(["Al momento non c'è nessun timer attivo.",
                         "Non c'è nessun timer attivo."]), ["nessun"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        rem = e["remaining"]
        if e.get("paused"):
            return f"Il {lab} è in pausa, {manca(rem)} ancora {rem}.", [rem]
        return R.choice([f"Al {lab} {manca(rem)} ancora {rem}.",
                         f"{cap(manca(rem))} ancora {rem}."]), [rem]
    parts = [f"{e['name'] or 'Un timer'}: ancora {e['remaining']}"
             + (", in pausa" if e.get("paused") else "") for e in entries]
    return f"Ci sono {len(entries)} timer attivi. " + ". ".join(parts) + ".", \
        [e["remaining"] for e in entries]


def s_timer_status(h, loc):
    n = R.choice([0, 1, 1, 1, 2, 2, 3])
    timers = running_timers(n, named=n > 1 or R.random() < 0.4)
    named = [t for t in timers if t[0]]
    args = {"action": "status"}
    asked = None
    if named and R.random() < 0.35:
        asked = R.choice(named)[0] if R.random() < 0.8 else R.choice(TIMER_NAMES)[0]
        u = R.choice([f"Quanto manca al {timer_label(asked)}?", f"Quanto manca {per(asked)}?",
                      f"Quanto tempo resta sul {timer_label(asked)}?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["Quanto manca al timer?", "Quanto tempo resta sul timer?",
                      "Quanto manca?", "Quali timer sono attivi?", "C'è ancora un timer attivo?",
                      "Quanto dura ancora il timer?", "Stato del timer"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


TIMER_OPS = {
    "cancel": (["Annulla il {t}", "Ferma il {t}", "Cancella il {t}", "Elimina il {t}",
                "Spegni il {t}", "Togli il {t}, per favore"],
               ["Il {t} è annullato.", "Ho cancellato il {t}.", "Ok, {t} annullato."]),
    "pause": (["Metti in pausa il {t}", "Ferma un attimo il {t}", "Pausa al {t}",
               "Sospendi il {t}"],
              ["Il {t} è in pausa.", "Ho messo in pausa il {t}."]),
    "resume": (["Riprendi il {t}", "Fai ripartire il {t}", "Continua il {t}",
                "Riavvia il {t}"],
               ["Il {t} riparte.", "Ok, il {t} è ripartito."]),
    "add": (["Aggiungi {d} al {t}", "Allunga il {t} di {d}", "Ancora {d} sul {t}",
             "Dai {d} in più al {t}"],
            ["Ho aggiunto {D} al {t}.", "Ora il {t} dura {D} in più."]),
    "subtract": (["Togli {d} dal {t}", "Accorcia il {t} di {d}", "Leva {d} dal {t}"],
                 ["Ora il {t} dura {D} in meno.", "Ho tolto {D} dal {t}."]),
}


def s_timer_op(h, loc):
    op = F("op") or R.choice(list(TIMER_OPS))
    users, answers = TIMER_OPS[op]
    hh, m, s, spoken = duration() if op in ("add", "subtract") else (0, 0, 0, "")
    if op in ("add", "subtract") and hh > 1:
        return None
    r = R.random()
    if r < 0.6:
        timers = running_timers(1, named=R.random() < 0.4)
        nm = timers[0][0] if timers[0][0] and R.random() < 0.6 else ""
        result = S.timer_result()
    elif r < 0.8:
        timers, nm = [], ""
        result = S.timer_result("no matching timer", [])
    else:
        timers = running_timers(R.choice([1, 2]))
        nm = R.choice([n for n, _ in TIMER_NAMES if n not in [t[0] for t in timers]])
        result = S.timer_result("no matching timer", status_entries(timers))
    t = timer_label(nm)
    u = R.choice(users).format(t=t, d=spoken)
    args = {"action": op}
    if op in ("add", "subtract"):
        args.update(dur_args(hh, m, s))
    if nm:
        args["name"] = nm
    if r < 0.6:
        ans = R.choice(answers).format(t=t, D=S.fmt_duration(hh, m, s, LANG))
    elif r < 0.8:
        ans = R.choice(["Al momento non c'è nessun timer attivo.",
                        "Purtroppo non c'è nessun timer attivo."])
    else:
        others = ["il " + timer_label(x[0]) for x in timers]
        ans = f"Non c'è nessun {t}. " + (f"È attivo solo {others[0]}." if len(others) == 1
                                          else f"Sono attivi solo {join_it(others)}.")
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
    u = R.choice(users).format(t="timer", d="")
    args = {"action": op}
    names = [t[0] for t in timers]
    ask = R.choice([f"Quale timer intendi: {join_it(names, 'o')}?",
                    f"Ci sono più timer attivi: {join_it(names)}. Quale intendi?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["Tutti", "Tutti i timer",
                                "Entrambi" if len(names) == 2 else "Tutti"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          f"Ho annullato tutti e {NUM_WORDS[len(names)]} i timer.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    msgs += [user(R.choice([pick, f"Il {timer_label(pick)}", f"Quello {per(pick)}",
                            f"Il timer {per(pick)}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      R.choice(answers).format(t=timer_label(pick), D=""))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Cancella tutti i timer", "Annulla tutti i timer", "Ferma tutti i timer",
                  "Elimina tutti i timer", "Spegni tutti i timer", "Togli tutti i timer",
                  "Puoi annullare tutti i timer?", "Per favore ferma tutti i timer",
                  "Via tutti i timer", "Basta timer, cancellali tutti", "Tutti i timer off",
                  "Annulla ogni timer", "Elimina i timer"])
    args = {"action": "cancel_all"}
    ans = {0: "Al momento non c'era nessun timer attivo.", 1: "Ho annullato il timer."}.get(
        n, f"Ho annullato tutti e {NUM_WORDS.get(n, n)} i timer.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh}:{mm:02d}" if mm else f"{hh}"
    u = R.choice([f"Svegliami alle {t}", f"Imposta una sveglia alle {t}",
                  f"Metti la sveglia per domattina alle {t}", f"Sveglia alle {t}"])
    ans = R.choice(["Purtroppo non posso impostare una sveglia a un orario preciso. Posso però "
                    "impostarti un timer.",
                    "Le sveglie a un orario purtroppo non sono supportate, solo i timer. Vuoi "
                    "che imposti un timer?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Fai un annuncio: {main}", f"Di' a tutti che {sub}",
                  f"Annuncia in tutta la casa che {sub}", f"Annuncio: {main}",
                  f"Comunica a tutti: {main}", f"Avvisa tutti che {sub}",
                  f"Fai sapere a tutti che {sub}", f"Annuncia che {sub}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Purtroppo non c'è nessun altro dispositivo vocale per un annuncio.",
                        "Un annuncio purtroppo non è possibile, non ci sono altri dispositivi "
                        "vocali."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["Ho fatto l'annuncio.", "L'annuncio è partito.",
                        "Fatto, l'ho annunciato in tutta la casa."])
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
        al = pnum("a", v)
        users = [f"Imposta {ref.acc} {al}", f"{cap(ref.acc)} {al}",
                 f"Imposta l'umidità {ref.p('di')} {al}"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Porta l'umidità {C.ROOM_BY_NAME[e.area][1]} {al}")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} è {ref.adj('impostato')} {al}."
    on = r < 0.8
    users = [f"{'Accendi' if on else 'Spegni'} {ref.acc}",
             f"Mi {'accendi' if on else 'spegni'} {ref.acc}?"]
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, "on" if on else "off")


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "a metà"), (25, pnum("a", 25)), (30, pnum("a", 30)),
                        (75, pnum("a", 75)), (50, pnum("a", 50))])
    u = R.choice([f"Apri {e.name} {word}", f"Imposta {e.name} {word}"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"Ho aperto {e.name} {pnum('a', p)}."


def describe_assist(h, a, who):
    """describe() wording for the Assist services (after 'Ho')."""
    svc = a["service"].split(".", 1)[1]
    data = a.get("data") or {}
    items = data.get("items") or ([data["item"]] if "item" in data else [])
    if svc == "add_item":
        return f"aggiunto {join_it(items)} {who.p('a')}"
    if svc == "complete_item":
        return f"spuntato {join_it(C.the_item(i) for i in items)} {who.p('in')}"
    if svc == "remove_item":
        return f"tolto {join_it(C.the_item(i) for i in items)} {who.p('da')}"
    if svc == "search_and_play":
        return f"messo {data.get('query', 'la musica')} {who.p('su')}"
    if svc == "seek_relative":
        n = int(data.get("seconds") or 0)
        word = "un minuto" if abs(n) == 60 else f"{abs(n)} secondi"
        return f"{'mandato avanti' if n >= 0 else 'riportato indietro'} di {word} " \
               f"la riproduzione {who.p('su')}"
    if svc == "shuffle_set":
        return f"{'attivato' if data.get('shuffle') else 'disattivato'} la riproduzione " \
               f"casuale {who.p('su')}"
    if svc == "repeat_set":
        rep = data.get("repeat")
        return {"one": "impostato la ripetizione del brano ",
                "all": "impostato la ripetizione di tutti i brani ",
                "off": "disattivato la ripetizione "}.get(rep, "cambiato la ripetizione ") \
            + who.p("su")
    if svc == "volume_mute":
        return f"{'disattivato' if data.get('is_volume_muted') else 'riattivato'} " \
               f"l'audio {who.p('di')}"
    if svc == "set_humidity":
        return f"impostato l'umidità {who.p('di')} {pnum('a', data['humidity'])}"
    if svc == "set_valve_position":
        return f"aperto {who.acc} {pnum('a', data['position'])}"
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
    """'Metti Vasco in cucina' -> 'Più forte' / 'Prossima canzone': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Più forte", "Alza il volume", "Alza un po'"],
         "Volume alzato."),
        ("media_player.volume_down", ["Più piano", "Abbassa il volume", "Abbassa un po'"],
         "Volume abbassato."),
        ("media_player.media_next_track", ["Prossima canzone", "Avanti", "Salta"],
         "Brano successivo."),
        ("media_player.media_pause", ["Pausa", "Metti in pausa", "Fermati un attimo"],
         "In pausa."),
        ("media_player.turn_off", ["Spegni la musica", "Spegni"], "La musica è spenta."),
        ("media_player.shuffle_set", ["E in ordine casuale, per favore"],
         "La riproduzione casuale è attiva."),
        ("media_player.media_previous_track", ["Indietro", "Brano precedente",
                                               "Rimetti quella di prima"], "Brano precedente."),
        ("media_player.seek_relative", ["Vai avanti", "Manda avanti un pochino"], "Ok, avanti."),
        ("media_player.seek_relative-", ["Torna indietro", "Riavvolgi un pochino"], "Ok, indietro."),
        ("media_player.media_stop", ["Stop", "Ferma la musica"], "Riproduzione fermata."),
    ])
    data = {"shuffle": True} if svc.endswith("shuffle_set") else None
    if svc.startswith("media_player.seek_relative"):
        data = {"seconds": -30 if svc.endswith("-") else 30}
        svc = "media_player.seek_relative"
    c.user(R.choice(users))
    c.control([action(svc, [e], data)], ans)
    if svc == "media_player.media_pause" and R.random() < 0.6:
        c.user(R.choice(["Play", "Riprendi", "Continua"]))
        c.control([action("media_player.media_play", [e])], "La riproduzione riprende.")
    return True


def turn_timer_follow(c):
    """Start a timer, then ask how long it still runs or cancel it."""
    hh, m, s, spoken = duration()
    if not m and not hh:
        return False
    named = R.random() < 0.4
    name = R.choice(TIMER_NAMES)[0] if named else ""
    c.user(f"Imposta un timer {per(name) + ' ' if named else ''}di {spoken}")
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"Il {timer_label(name)} è partito: {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["Quanto manca?", "Quanto manca ancora?", "E quanto manca adesso?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Annullalo", "Anzi no, cancella il timer", "Fermalo"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"Il {timer_label(name)} è annullato.")
    return True


def turn_list_follow(c):
    """'Aggiungi il latte alla lista della spesa' -> 'E le uova'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    _tail, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item = R.choice(free)
    c.user(R.choice([f"E {item}", f"Anche {item}", f"E poi {item}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    ans = [f"Ho aggiunto anche {item}."]
    if C.ITEM_G.get(item) != "inf":
        ans.append(f"Ora {lst_name(lst.name, 'in')} {C.c_e(is_pl(item))} anche "
                   f"{C.the_item(item)}.")
    c.control(a2, R.choice(ans))
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
