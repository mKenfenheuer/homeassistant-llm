"""Spanish samples for the Home Assistant Assist built-ins.

Port of src/gen/de/assist.py: music (search and play, the media library of
a player, shuffle, repeat, mute), timers on the voice device (start, named, delayed commands, status,
cancel, pause, resume, add, subtract), to-do lists, weather with forecast,
time and date, broadcasts, humidifiers and valve positions.

Registers itself in core: entities through POPULATE_HOOKS, device commands
through COMMANDS (they get a case each and take part in compound commands and
conversations), conversation turns through NEXT, the rest through CASE_HOOKS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import core as C
from .. import playlists as PL
from .core import (ART, R, S, action, al, base_msgs, cap, de, es_num, esta, join_es,
                   low, slugify, the, tool_call, tool_msg, voice)

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


VERBS = C.VERBS


def lower_first(text):
    """Imperatives move behind 'En 10 minutos'; names keep their capital."""
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
ARTISTS = ["Queen", "Rosalía", "Coldplay", "Alejandro Sanz", "Shakira", "Bad Bunny",
           "Estopa", "Joaquín Sabina", "Ed Sheeran", "Adele", "Taylor Swift", "Mecano",
           "Juanes", "Manu Chao", "ABBA", "The Beatles", "Pablo Alborán", "Vetusta Morla",
           "Nirvana", "Beyoncé", "Héroes del Silencio", "Billie Eilish", "Linkin Park",
           "Aitana", "Karol G", "C. Tangana", "Mozart", "Ludovico Einaudi",
           "La Oreja de Van Gogh", "Maná", "Pink Floyd", "Daft Punk", "Dua Lipa",
           "Camilo", "CantaJuego"]
TRACKS = [("Bohemian Rhapsody", "Queen"), ("Despechá", "Rosalía"),
          ("Yellow", "Coldplay"), ("Corazón partío", "Alejandro Sanz"),
          ("Hips Don't Lie", "Shakira"), ("Shape of You", "Ed Sheeran"),
          ("Hello", "Adele"), ("Y nos dieron las diez", "Joaquín Sabina"),
          ("Hijo de la Luna", "Mecano"), ("Dancing Queen", "ABBA"),
          ("Let It Be", "The Beatles"), ("Entre dos tierras", "Héroes del Silencio"),
          ("La camisa negra", "Juanes"), ("Bad Guy", "Billie Eilish"),
          ("In the End", "Linkin Park"), ("Tití me preguntó", "Bad Bunny"),
          ("Clandestino", "Manu Chao"), ("Puedes contar conmigo", "La Oreja de Van Gogh"),
          ("Levitating", "Dua Lipa"), ("Get Lucky", "Daft Punk"),
          ("Wish You Were Here", "Pink Floyd"), ("Como Camarón", "Estopa")]
ALBUMS = [("A Night at the Opera", "Queen"), ("Parachutes", "Coldplay"),
          ("El mal querer", "Rosalía"), ("Abbey Road", "The Beatles"),
          ("Divide", "Ed Sheeran"), ("21", "Adele"), ("Más", "Alejandro Sanz"),
          ("Nevermind", "Nirvana"), ("Random Access Memories", "Daft Punk"),
          ("The Dark Side of the Moon", "Pink Floyd"), ("Folklore", "Taylor Swift"),
          ("Un verano sin ti", "Bad Bunny")]
PLAYLISTS = ["Chill", "Cocina", "Fiesta", "Gimnasio", "Relax", "Éxitos del verano",
             "Mis favoritas", "Dormir", "Rock clásico", "Infantil", "Villancicos",
             "Concentración", "Discover Weekly", "Road Trip", "Años 80"]
STATIONS = ["Cadena SER", "Los 40", "COPE", "Onda Cero", "RNE", "Radio 3",
            "Europa FM", "Cadena 100", "Rock FM", "Kiss FM", "Cadena Dial",
            "Radio Clásica", "Los 40 Classic", "Radio Marca", "Catalunya Ràdio", "Radio Nacional"]
GENRES = ["jazz", "música clásica", "rock", "música relajante", "música infantil",
          "villancicos", "lo-fi", "flamenco", "reguetón", "hip-hop", "música para dormir",
          "música para cocinar", "éxitos de los 80", "pop", "música de meditación"]

# {room}: area name, {de}: "del salón"
SPEAKER_NAMES = ["Sonos {room}", "Altavoz {room}", "Echo {room}", "HomePod {room}",
                 "Altavoz {de}", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "{room} Speaker"]
MUSIC_LABELS = ("Altavoz", "Radio", "Barra de sonido")

GROCERIES = ["leche", "huevos", "mantequilla", "pan", "queso", "manzanas", "plátanos",
             "tomates", "café", "pasta", "arroz", "yogures", "harina", "azúcar",
             "papel higiénico", "lavavajillas", "detergente", "comida para gatos",
             "leche de avena", "cebollas", "patatas", "agua mineral", "zumo de naranja",
             "bolsas de basura", "copos de avena", "pepinos", "pimientos", "carne picada",
             "nata", "queso fresco", "aceite de oliva", "sal"]
LISTS = {  # name: (phrase with article, items, slugs)
    "Lista de la compra": ("la lista de la compra", GROCERIES,
                           ["shopping_list", "lista_de_la_compra"]),
    "Tareas": ("la lista de tareas",
               ["sacar la basura", "hacer la declaración de la renta", "lavar el coche",
                "cortar el césped", "llamar a la abuela", "limpiar las ventanas",
                "ordenar el trastero", "recoger el paquete", "pedir cita con el médico",
                "regar las plantas"],
               ["tareas", "tasks"]),
    "Ferretería": ("la lista de la ferretería",
                   ["tornillos", "tacos", "pintura", "silicona", "cinta americana",
                    "bombillas", "tierra para macetas", "bridas", "papel de lija"],
                   ["ferreteria", "hardware_store"]),
    "Droguería": ("la lista de la droguería",
                  ["pasta de dientes", "champú", "gel de ducha", "crema solar",
                   "desodorante", "discos de algodón", "pañuelos", "crema de manos"],
                  ["drogueria", "drugstore"]),
    "Equipaje": ("la lista del equipaje",
                 ["cargador", "gafas de sol", "pasaporte", "bañador", "cepillo de dientes",
                  "chubasquero", "auriculares", "adaptador"],
                 ["equipaje", "packing_list"]),
}
SHOP = "Lista de la compra"
TASKS = "Tareas"
# items that take a plural verb ("los huevos están ...")
PLURAL_ITEMS = {"huevos", "manzanas", "plátanos", "tomates", "yogures", "cebollas",
                "patatas", "bolsas de basura", "copos de avena", "pepinos", "pimientos",
                "tornillos", "tacos", "bombillas", "bridas", "discos de algodón",
                "pañuelos", "gafas de sol", "auriculares"}

CITIES = ["Valencia", "Sevilla", "Zaragoza", "Bilbao", "Málaga", "Granada", "Salamanca",
          "Valladolid", "Oviedo", "Murcia"]
WEATHER_NAMES = ["Forecast Home", "Casa", "Tiempo", "OpenWeatherMap", "AEMET",
                 "Met.no", "Previsión del tiempo"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "Mañana ..." + fragment (future)
FORECAST_FRAG = {
    "sunny": "hará sol", "clear-night": "estará despejado", "cloudy": "estará nublado",
    "partlycloudy": "estará parcialmente nublado", "rainy": "lloverá",
    "pouring": "lloverá con fuerza", "snowy": "nevará",
    "snowy-rainy": "caerá aguanieve", "fog": "habrá niebla", "hail": "granizará",
    "lightning": "habrá tormenta", "lightning-rainy": "habrá tormenta con lluvia",
    "windy": "hará viento", "windy-variant": "hará viento y estará nublado",
    "exceptional": "se espera un temporal",
}
NOW_SENT = {
    "sunny": "Hace sol", "clear-night": "El cielo está despejado", "cloudy": "Está nublado",
    "partlycloudy": "Está parcialmente nublado", "rainy": "Está lloviendo",
    "pouring": "Llueve con fuerza", "snowy": "Está nevando", "snowy-rainy": "Cae aguanieve",
    "fog": "Hay niebla", "hail": "Está granizando", "lightning": "Hay tormenta",
    "lightning-rainy": "Hay tormenta y llueve", "windy": "Hace viento",
    "windy-variant": "Hace viento y está nublado", "exceptional": "Hay un temporal",
}

# timer name, article ("el temporizador de la pizza", "para la pizza")
TIMER_NAMES = [("pizza", "la"), ("pasta", "la"), ("huevos", "los"), ("té", "el"),
               ("bizcocho", "el"), ("colada", "la"), ("pan", "el"), ("arroz", "el"),
               ("patatas", "las"), ("horno", "el"), ("descanso", "el"),
               ("parquímetro", "el"), ("deberes", "los"), ("filete", "el"),
               ("lasaña", "la"), ("patatas fritas", "las"), ("gofres", "los"),
               ("verduras", "las")]
TIMER_ART = dict(TIMER_NAMES)
NUM_WORDS = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete",
             8: "ocho", 10: "diez", 12: "doce", 15: "quince", 20: "veinte",
             30: "treinta"}

# (reported "diles que ...", direct message)
BROADCASTS = [("la comida está lista", "La comida está lista"),
              ("salimos en cinco minutos", "Salimos en cinco minutos"),
              ("es hora de dormir", "Es hora de dormir"),
              ("ha llegado la pizza", "Ha llegado la pizza"),
              ("hay alguien en la puerta", "Hay alguien en la puerta"),
              ("vengan a cenar", "Todos a cenar"),
              ("la película empieza enseguida", "La película empieza enseguida"),
              ("llego a casa enseguida", "Llego a casa enseguida"),
              ("todavía hay que sacar la basura", "Todavía hay que sacar la basura"),
              ("viene visita", "Viene visita"),
              ("el desayuno está listo", "El desayuno está listo"),
              ("el bizcocho ya está hecho", "El bizcocho ya está hecho"),
              ("bajen", "Todos abajo"),
              ("salimos a pasear enseguida", "Salimos a pasear enseguida"),
              ("toca hacer los deberes", "Toca hacer los deberes")]


def de_t(name):
    """'de la pizza' / 'del té'."""
    return de(f"{TIMER_ART[name]} {name}")


def para_t(name):
    return f"para {TIMER_ART[name]} {name}"


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
        name = R.choice(SPEAKER_NAMES).format(room=room, de=C.de_room(room))
        if name not in h.used_names:
            break
    else:
        return None
    playing = R.random() < 0.3
    attrs = {"volume_level": R.choice([0.2, 0.3, 0.4, 0.5, 0.6])}
    if playing:
        title, artist = R.choice(TRACKS)
        attrs.update(media_title=title, media_artist=artist)
    return h.add("media_player", room, "Altavoz", "m", [slugify(name), "speaker"], name=name,
                 state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs)


def add_list(h, name, items=None):
    _, pool, slugs = LISTS[name]
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
    "sunny": "seguirá haciendo sol", "clear-night": "seguirá despejado",
    "cloudy": "seguirá nublado", "partlycloudy": "seguirá parcialmente nublado",
    "rainy": "seguirá lloviendo", "pouring": "seguirá lloviendo con fuerza",
    "snowy": "seguirá nevando", "snowy-rainy": "seguirá cayendo aguanieve",
    "fog": "seguirá habiendo niebla", "hail": "seguirá granizando",
    "lightning": "seguirá habiendo tormenta",
    "lightning-rainy": "seguirá habiendo tormenta con lluvia",
    "windy": "seguirá haciendo viento",
    "windy-variant": "seguirá haciendo viento y estará nublado",
    "exceptional": "seguirá el temporal",
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
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Exterior", "Jardín")])
    slug = "forecast_home" if name in ("Forecast Home", "Casa") else slugify(name)
    e = h.add("weather", room, "Tiempo", "m", [slug], name=name, state=cond, attrs=attrs)
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
        add_list(h, SHOP)
        for name in R.sample([n for n in LISTS if n != SHOP], R.choice([0, 0, 1, 2])):
            add_list(h, name)
    elif R.random() < 0.15:
        add_list(h, R.choice([n for n in LISTS if n != SHOP]))
    if R.random() < 0.12:
        rooms = [r for r in h.rooms if r in ("Dormitorio", "Habitación infantil", "Salón", "Oficina")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "Humidificador", "m", ["humidificador", "humidifier"],
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
        return R.choice(["Pon música", "Pon algo", "Pon algo de música",
                         "Ponme algo de música", "Pon algo de mi biblioteca",
                         "Quiero escuchar música", "Enciende la música",
                         "Pon algo bonito"]), {}, "música"
    if k == "artist":
        a = R.choice(ARTISTS)
        u = R.choice([f"Pon música de {a}", f"Pon algo de {a}", f"Quiero escuchar a {a}",
                      f"Pon canciones de {a}", f"Reproduce música de {a}",
                      f"Me apetece escuchar a {a}"])
        return u, {"query": a, "media_type": "artist"}, f"música de {a}"
    if k == "track":
        t, a = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Pon {t} de {a}", f"Pon la canción {t} de {a}",
                          f"Quiero escuchar {t} de {a}", f"Reproduce el tema {t} de {a}"])
            return u, {"query": t, "media_type": "track", "artist": a}, f"{t} de {a}"
        u = R.choice([f"Pon la canción {t}", f"Reproduce el tema {t}", f"Pon el tema {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al_, a = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Pon el disco {al_} de {a}", f"Reproduce el álbum {al_} de {a}"]),
                    {"query": al_, "media_type": "album", "artist": a}, f"el disco {al_}")
        return R.choice([f"Pon el disco {al_}", f"Pon el álbum {al_}"]), \
            {"query": al_, "media_type": "album"}, f"el disco {al_}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Pon mi lista {p}", f"Pon la playlist {p}",
                      f"Reproduce la lista de reproducción {p}", f"Pon mi playlist {p}"])
        return u, {"query": p, "media_type": "playlist"}, f"la playlist {p}"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Pon la emisora {s}", f"Ponme la emisora {s}", f"Sintoniza {s}",
                      f"Pon {s} en la radio", f"Quiero escuchar la emisora {s}"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        g = R.choice(GENRES)
        return R.choice([f"Pon {g}", f"Pon algo de {g}"]), {"query": g}, g
    # "Pon Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a for a in ARTISTS])
    return R.choice([f"Pon {t}", f"Ponme {t}"]), {"query": t}, t


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
        return e, R.choice([f" en {e.name}", f" en {the(e)}", f" en {the(e)}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.en_room(room)
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Reproduzco {desc} en {e.name}.", f"Vale, ya suena {desc} en {e.name}.",
                     f"{cap(desc)} ya suena en {e.name}.", f"Pongo {desc}."])


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
         ["Activa el modo aleatorio{w}", "Reproducción aleatoria{w}", "Pon el shuffle{w}",
          "Reproduce las canciones en orden aleatorio{w}"],
         ["El modo aleatorio está activado.", "Aleatorio activado."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Desactiva el modo aleatorio{w}", "Quita el shuffle{w}", "Quita el aleatorio{w}"],
         ["El modo aleatorio está desactivado."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Repite esta canción{w}", "Pon la canción en bucle{w}",
          "Repite la canción actual{w}"], ["Se repetirá la canción."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Repite todo{w}", "Repite la lista{w}", "Repite todas las canciones{w}"],
         ["Se repetirán todas las canciones."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Quita la repetición{w}", "Desactiva la repetición{w}", "Deja de repetir{w}"],
         ["La repetición está desactivada."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Quita el sonido{w}", "Silencia el audio{w}", "Pon el sonido en silencio{w}",
          "Mute{w}"],
         [f"He silenciado {the(e)}.", "El sonido está desactivado."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Vuelve a poner el sonido{w}", "Quita el silencio{w}", "Activa el sonido{w}"],
         ["El sonido vuelve a estar activado.", f"{cap(the(e))} ya no está en silencio."]),
        ("media_player.media_previous_track", None,
         ["Canción anterior{w}", "Vuelve a la canción anterior{w}", "Pon la canción anterior{w}",
          "Una canción atrás{w}"], ["He vuelto a la canción anterior.", "Canción anterior."]),
        ("media_player.media_stop", None,
         ["Para la música{w}", "Detén la reproducción{w}", "Parar música{w}", "Para{w}"],
         ["La reproducción está detenida.", "Detenido."]),
        ("media_player.media_play", None,
         ["Play{w}", "Sigue{w}", "Continúa con la música{w}", "Reanuda la música{w}"],
         ["La reproducción continúa.", "Seguimos."]),
        ("media_player.media_next_track", None,
         ["Siguiente{w}", "Siguiente canción{w}", "Pasa a la siguiente{w}"], ["Siguiente canción."]),
        ("media_player.media_previous_track", None,
         ["Atrás{w}", "La anterior{w}"], ["Canción anterior."]),
        ("media_player.volume_up", None, ["Más alto{w}", "Sube el volumen{w}"], ["Volumen subido."]),
        ("media_player.volume_down", None, ["Más bajo{w}", "Baja el volumen{w}"], ["Volumen bajado."]),
    ]
    sk = R.choice([10, 15, 30, 30, 60])
    sk_word = "un minuto" if sk == 60 else f"{sk} segundos"
    opts += [
        ("media_player.seek_relative", {"seconds": sk},
         [f"Adelanta {sk_word}{{w}}", "Adelanta{w}" if sk == 30 else f"Avanza {sk_word}{{w}}",
          f"Pasa {sk_word} hacia delante{{w}}"],
         [f"He adelantado {sk_word}.", "Adelantado."]),
        ("media_player.seek_relative", {"seconds": -sk},
         [f"Retrocede {sk_word}{{w}}", "Rebobina{w}" if sk == 30 else f"Vuelve {sk_word} atrás{{w}}",
          f"Atrasa {sk_word}{{w}}"],
         [f"He retrocedido {sk_word}.", "Retrocedido."]),
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
        msgs += [say(R.choice(["Lo siento, ahora mismo no hay ningún altavoz disponible.",
                               "En este momento ninguno de tus altavoces está disponible, así "
                               "que no puedo poner música.",
                               f"{join_es(gone[:4])} "
                               f"{'no está disponible' if len(gone) == 1 else 'no están disponibles'}"
                               " ahora mismo, así que no puedo reproducir nada."]))]
        return sample(msgs, "command", gold, no_history=True)
    if len(ps) == 1:
        e = ps[0]
        acts = [action("media_player.search_and_play", [e], data)]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Pongo {desc} en {e.name}, que ahora mismo es el único "
                                    f"altavoz que responde.",
                                    play_answer(desc, e)]))
        return sample(msgs, "command", gold, no_history=True)
    opts = ps[:4]
    # numbers in the request (Los 40, Radio 3) are not in the tool result
    what = "la música" if any(ch.isdigit() for ch in desc) else desc
    ask = R.choice([f"¿En qué altavoz pongo {what}: "
                    f"{join_es([e.name for e in opts], 'o')}?",
                    f"¿Dónde quieres que ponga {what}? Puedo usar "
                    f"{join_es([e.name for e in opts], 'y')}."])
    if gone:
        ask = (f"{join_es(gone)} {'no está disponible' if len(gone) == 1 else 'no están disponibles'}"
               f" ahora mismo. ") + ask
    msgs += [say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", gold, no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Olvídalo", "Mejor no", "Da igual"])), say("Vale, entonces no.")]
        return sample(msgs, "conversation", gold, gold_last=[])
    e = R.choice(opts)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"En {e.name}", f"En {e.name}, por favor"]
    if e.area and len(same_room) == 1:
        replies += [cap(C.en_room(e.area)), e.area]
    acts = [action("media_player.search_and_play", [e], data)]
    msgs += [user(R.choice(replies))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts), play_answer(desc, e))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]])


def s_music_sat(h, loc):
    """'Pon música' on a voice device with its own speaker: play there,
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
    ("media_player.media_next_track", None, ["Siguiente canción", "Pasa a la siguiente canción",
                                             "Pon la siguiente", "Salta esta canción", "Siguiente"],
     "Siguiente canción en {n}."),
    ("media_player.media_previous_track", None, ["Canción anterior", "Vuelve a la canción anterior",
                                                 "La anterior", "Atrás"],
     "Canción anterior en {n}."),
    ("media_player.media_pause", None, ["Pausa", "Pausa la música", "Pon la música en pausa"],
     "La música de {n} está en pausa."),
    ("media_player.media_stop", None, ["Para la música", "Quita la música", "Detén la reproducción",
                                       "Para"],
     "He detenido la reproducción en {n}."),
    ("media_player.volume_up", None, ["Sube el volumen", "Más alto, por favor", "Sube la música",
                                      "Más alto"],
     "He subido el volumen de {n}."),
    ("media_player.volume_down", None, ["Baja el volumen", "Más bajo, por favor", "Baja la música",
                                        "Más bajo"],
     "He bajado el volumen de {n}."),
    ("media_player.seek_relative", {"seconds": 30}, ["Adelanta", "Avanza", "Adelanta 30 segundos"],
     "He adelantado 30 segundos en {n}."),
    ("media_player.seek_relative", {"seconds": -30}, ["Retrocede", "Rebobina",
                                                      "Retrocede 30 segundos"],
     "He retrocedido 30 segundos en {n}."),
    # play goes on with the paused title of the player
    ("media_player.media_play", None, ["Play", "Sigue", "Continúa", "Reanuda",
                                       "Sigue con la música"],
     "La reproducción continúa en {n}."),
]


def s_media_unclear(h, loc):
    """'Siguiente canción' with several speakers, none named and none in the
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
        ask = R.choice([f"Ahora mismo no hay nada en pausa. ¿Qué pongo y en qué altavoz: "
                        f"{join_es(names, 'o')}?",
                        "No hay nada en pausa en ningún altavoz. ¿Qué quieres escuchar?"])
    elif mode == "none":
        ask = R.choice([f"Ahora mismo no suena música en ningún sitio. ¿Qué altavoz quieres decir: "
                        f"{join_es(names, 'o')}?",
                        "En este momento no está sonando nada en ningún altavoz. ¿En cuál lo hago?"])
    elif resume:
        ask = f"Hay música en pausa en {join_es(names)}. ¿Dónde quieres que siga?"
    else:
        ask = f"Hay música en {join_es(names)}. ¿Cuál quieres decir?"
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
        ans = R.choice([f"Lo siento, no he encontrado {q}.",
                        f"No he encontrado nada de {q}.",
                        f"No he encontrado nada para {q}. ¿Quieres que ponga otra cosa?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Lo siento, en {e.name} no puedo buscar música.",
                        f"{e.name} no admite la búsqueda de música."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


# --------------------------------------------------------------------------
# media library of a player (music_library, media_player.browse_media)
# --------------------------------------------------------------------------
# mood -> (request words, playlist names that fit)
LIB_MOODS = {
    "relax": (["para relajarme", "para desconectar", "para descansar", "para una noche tranquila"],
              ["Relax total", "Relax & Unwind", "Peaceful Piano", "Noches tranquilas",
               "Chill Vibes", "Domingo por la mañana", "Calma", "Easy Listening", "Para desconectar",
               "Tarde de playa", "Slow Evening"]),
    "sport": (["para hacer deporte", "para entrenar", "para el gimnasio", "para correr"],
              ["Workout Mix", "Running Beats", "Power Workout", "Entreno 2026", "Beast Mode",
               "Correr 10 km", "Motivación gym", "Pump It Up", "Cardio Hits", "A tope en el gym"]),
    "sleep": (["para dormir", "para dormirme", "para la noche"],
              ["Música para dormir", "Sleep Sounds", "A dormir", "Buenas noches", "Sleepy Time",
               "Lluvia de noche", "Dulces sueños", "Dormir bien", "Nanas para adultos",
               "Sleep Well"]),
    "party": (["para la fiesta", "para celebrar", "para bailar"],
              ["Fiestón", "Club Hits 2025", "¡A bailar!", "Dance Floor Fillers", "Playlist fiestera",
               "Viernes de fiesta", "Disco Fever", "Let's Dance", "Reggaetón para el finde"]),
    "xmas": (["de villancicos", "para Navidad", "de Navidad"],
             ["Villancicos", "Christmas Classics", "Navidad 2025", "Xmas Hits",
              "Nochebuena en casa", "Jingle Bells & Co", "Christmas Pop"]),
    "cook": (["para cocinar", "para la cena", "para comer"],
             ["Cocinando con música", "Dinner Jazz", "Cooking Vibes", "Dinner for Two",
              "Radio cocina", "Cena con amigos", "Kitchen Grooves"]),
    "focus": (["para trabajar", "para concentrarme", "para estudiar", "para programar"],
              ["Deep Focus", "Concentración", "Programming Beats", "Música para estudiar",
               "Coding Mode", "Focus Flow", "Teletrabajo", "Brain Food", "Code & Coffee",
               "Study Session"]),
    "rock": (["de rock", "de rock duro", "de metal"],
             ["Rock Classics", "Himnos del rock", "Hard Rock Hits", "Metal Mayhem", "Headbanger",
              "Rock Anthems", "Noche de heavy"]),
    "kids": (["para los niños", "de canciones infantiles"],
             ["Canciones infantiles", "Fiesta infantil", "Música para niños", "Kids Party",
              "Cantajuegos y más"]),
    "eighties": (["de los ochenta", "con éxitos de los ochenta"],
                 ["Éxitos de los 80", "Best of the 80s", "Fiesta ochentera", "80s Forever",
                  "La Movida"]),
    "car": (["para el coche", "para el viaje"],
            ["Roadtrip", "De viaje", "Driving Hits", "Carretera y manta", "On the Road",
             "Highway Songs"]),
}
# favorites playlists; {name}: a first name
FAV_NAMES = ["Mis canciones favoritas", "Liked Songs", "Favoritos", "Mis favoritos",
             "Liked Songs {name}", "Canciones que me gustan", "Música favorita",
             "Favoritas de {name}", "Lo mejor de {name}", "Liked Songs ❤️", "My Favorites",
             "Favorites", "Mis favoritas", "Favoritos de {name}", "Las de {name}"]
FAV_USERS = ["Pon mis canciones favoritas", "Pon mis favoritos", "Pon mi música favorita",
             "Pon mis favoritas", "Quiero escuchar mis canciones favoritas",
             "Pon las canciones que me gustan", "Pon mi playlist favorita"]
NEUTRAL_PLAYLISTS = ["Mi playlist n.º 3", "Boda", "Mix de la semana", "Descubrimientos",
                     "Mis Shazam", "Vacaciones 2024", "Verano 2025", "Variado", "Playlist 1",
                     "Joyas antiguas", "Encontradas por ahí", "Para luego", "Colección 2023"]
# personal playlist names for the generated part of a library (gen/playlists.py)
PL_LOCAL = {
    "personal": ["Cumple de {name}", "{name} + {name2}", "Vacaciones {place} {year}",
                 "Boda {name} y {name2}", "Playlist de {name}", "Mi playlist n.º {n}",
                 "Mix de {name}", "Verano {year}", "Selectividad {year}", "Nochevieja {year}",
                 "Compartida por {name}", "Nuevas {year}", "Viaje de fin de curso {place}",
                 "Piso {place}"],
    "names": ["Lucía", "Javi", "Carmen", "Pablo", "Marta", "Sergio", "Laura", "Dani", "Paula",
              "Hugo", "Sara", "Álvaro", "Elena", "Diego", "Irene", "Nacho", "Cristina", "Raúl"],
    "places": ["Ibiza", "Cádiz", "Asturias", "Lisboa", "Menorca", "Galicia", "Roma",
               "Tenerife", "Valencia", "Granada"],
}
LIB_KINDS = {  # media_type -> (plural, user question words, pool)
    "playlist": ("playlists", ["Qué playlists tengo", "Qué listas de reproducción tengo",
                               "Qué playlists hay", "Cuáles son mis playlists",
                               "Qué tengo en playlists"], None),
    "album": ("álbumes", ["Qué álbumes tengo", "Qué discos tengo",
                          "Qué álbumes hay en mi biblioteca"], None),
    "artist": ("artistas", ["Qué artistas tengo", "Qué intérpretes tengo",
                            "Qué artistas hay en mi biblioteca"], None),
    "radio": ("emisoras", ["Qué emisoras tengo", "Qué emisoras tengo guardadas",
                           "Qué emisoras de radio hay"], None),
    "podcast": ("podcasts", ["Qué podcasts tengo", "A qué podcasts estoy suscrito"],
                None),
}
PODCASTS = ["Nadie Sabe Nada", "Estirando el chicle", "La Pija y la Quinqui", "Todopoderosos",
            "Entiende tu mente", "La Escóbula de la Brújula", "Leyendas Legendarias",
            "Carne Cruda", "Un tema al día", "Cuarto Milenio"]


def lib_pool(kind, dash=False):
    """dash: albums and tracks as 'Artist - Title' (Music Assistant style)."""
    if kind == "playlist":
        return NEUTRAL_PLAYLISTS + [n for _, ns in LIB_MOODS.values() for n in ns] + \
            list(dict.fromkeys(fav_name() for _ in FAV_NAMES))
    if kind == "album":
        return [f"{a} - {al_}" if dash else al_ for al_, a in ALBUMS]
    if kind == "track":
        return [f"{a} - {t}" if dash else t for t, a in TRACKS]
    if kind == "artist":
        return list(ARTISTS)
    if kind == "radio":
        return list(STATIONS)
    return list(PODCASTS)


def fav_name():
    return R.choice(FAV_NAMES).format(name=R.choice(PL_LOCAL["names"]))


def make_library(kind, must=(), avoid=(), dash=False, avoid_tags=(), size=None):
    """Names in the library of one kind, sorted like a browse result.
    Playlists: fixed names plus generated real-world ones (gen/playlists.py),
    none of them fitting avoid_tags (moods)."""
    n = size or (PL.library_size(R) if kind == "playlist" else
                 R.choice([R.randint(1, 4), R.randint(4, 9), R.randint(8, 20),
                           R.randint(15, 40), R.randint(40, 80)]))
    pool = [n_ for n_ in lib_pool(kind, dash) if n_ not in avoid and n_ not in must]
    if kind == "playlist":
        fixed = [n_ for n_ in NEUTRAL_PLAYLISTS if n_ not in avoid]
        moods = [x for m, (_, ns) in LIB_MOODS.items() if m not in avoid_tags for x in ns
                 if x not in avoid and x not in must]
        k_fixed = min(len(fixed), R.randint(0, max(1, n // 6)))
        k_mood = min(len(moods), R.randint(0, max(1, n // 5)))
        names = list(must) + R.sample(fixed, k_fixed) + R.sample(moods, k_mood)
        if R.random() < 0.5 and "fav" not in avoid_tags:
            names.append(fav_name())
        names += PL.noise(R, max(0, n - len(names)), ARTISTS, TRACKS, PL_LOCAL,
                          avoid_tags=avoid_tags, avoid=list(avoid) + names)
        names = names[:max(n, len(must))]
    else:
        names = list(must) + R.sample(pool, min(n, len(pool)))
    return sorted(dict.fromkeys(names), key=str.casefold)


def lib_answer(kind, names, total, who):
    plural = LIB_KINDS[kind][0]
    if not names:
        return R.choice([f"No he encontrado {plural} en {who}.",
                         f"En la biblioteca de {who} no hay {plural}."])
    k = {"short": 3, "medium": 5, "long": 6, "detailed": 8}[C.VERBOSITY]
    if total <= k:
        if total == 1:
            return f"Solo tienes una entrada: {names[0]}."
        return f"Tienes {total} {plural}: {join_es(names)}."
    shown = R.sample(names, k) if R.random() < 0.3 else names[:k]
    return f"Tienes {total} {plural}, por ejemplo {join_es(shown)}."


def lib_target(h, loc):
    tgt = music_target(h, loc)
    if not tgt:
        if not add_speaker(h):
            return None
        tgt = music_target(h, loc)
    return tgt


def lib_where(sentence, where):
    """'Qué playlists tengo' + ' en Sonos Cocina' -> '¿Qué playlists tengo en Sonos Cocina?'."""
    return "¿" + sentence + where + "?"


def you_form(words):
    """'para relajarme' -> 'para relajarte' when the assistant repeats it."""
    return " ".join(w[:-3] + "rte" if w.endswith("rme") else w for w in words.split())


def s_library_list(h, loc):
    """'¿Qué playlists tengo?' -> music_library -> names; optionally
    'Pon <name>' afterwards plays one of them."""
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
        # '¿Qué álbumes de Queen tengo?': albums are listed as 'Queen - Innuendo'
        a = R.choice(sorted({a for _, a in ALBUMS}))
        mine = [f"{a} - {al_}" for al_, aa in ALBUMS if aa == a]
        names = make_library("album", must=mine if R.random() < 0.85 else [], dash=True,
                             avoid=[] if R.random() < 0.85 else mine)
        hits = [n for n in names if a.casefold() in n.casefold()]
        args["search"] = a
        u = lib_where(R.choice([f"Qué álbumes de {a} tengo",
                                f"Qué discos de {a} tengo",
                                f"Tengo algún álbum de {a}"]), where)
        if hits:
            titles = [n.split(" - ", 1)[1] for n in hits]
            ans = (f"De {a} tienes el álbum {titles[0]}." if len(titles) == 1 else
                   f"De {a} tienes {len(titles)} álbumes: {join_es(titles)}.")
        else:
            ans = f"No he encontrado álbumes de {a} en tu biblioteca."
        msgs = base_msgs(h, loc) + [user(u)]
        msgs += call_turn("music_library", args, S.library_result(hits), ans)
        return sample(msgs, "question", [["music_library", args]], no_history=True)
    msgs = base_msgs(h, loc) + [user(u)]
    gold = [["music_library", args]]
    if R.random() < 0.08:
        res = S.library_result(error=f"media library is not supported by {e.eid}")
        msgs += call_turn("music_library", args, res,
                          R.choice([f"Lo siento, {e.name} no permite explorar la biblioteca.",
                                    f"Lo siento, no puedo leer la biblioteca de {e.name}."]))
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
    said = PL.spoken(pick)
    msgs += [user(R.choice([f"Pon {said}", f"Pues pon {said}", f"Pon {said}, por favor",
                            f"{said}, por favor"]))]
    msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                      R.choice([f"Pongo {pick} en {e.name}.", f"Ya suena {pick}."]))
    return sample(msgs, "conversation", gold, gold_last=[["control", {"actions": acts}]],
                  no_history=True)


def s_library_search(h, loc):
    """'¿Tengo una playlist llamada Fiestón?' -> music_library with search."""
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
    q = {"playlist": [f"Tengo una playlist llamada {word}", f"Hay una playlist {word}",
                      f"Está la playlist {word} en mi biblioteca"],
         "album": [f"Tengo el álbum {word}", f"Está el disco {word} en mi biblioteca"],
         "artist": [f"Tengo música de {word}", f"Está {word} en mi biblioteca"],
         "radio": [f"Tengo guardada la emisora {word}", f"Tengo la emisora {word}"]}[kind]
    args = {"entity_id": e.eid, "media_type": kind, "search": word}
    res = S.library_result(hits)
    if hits:
        ans = R.choice([f"Sí, {hits[0]} está en tu biblioteca." if len(hits) == 1 else
                        f"Sí, hay {len(hits)} resultados: {join_es(hits[:5])}.",
                        f"Sí: {join_es(hits[:5])}."])
    else:
        ans = R.choice([f"No, no he encontrado {word} en tu biblioteca.",
                        f"No, {word} no está en la biblioteca de {e.name}."])
    msgs = base_msgs(h, loc) + [user(lib_where(R.choice(q), where))]
    msgs += call_turn("music_library", args, res, ans)
    return sample(msgs, "question", [["music_library", args]], no_history=True)


def s_library_play(h, loc, favorites=False):
    """'Pon una playlist para relajarme' / 'Pon mis canciones favoritas':
    read the playlists, play the one that fits; none fits -> say so and
    name a few."""
    tgt = lib_target(h, loc)
    if not tgt:
        return None
    e, where = tgt
    if favorites:
        u = R.choice(FAV_USERS)
        fit, tags = fav_name(), ("fav",)
        what = "con tus canciones favoritas"
    else:
        mood = R.choice(list(LIB_MOODS))
        words, fit_pool = LIB_MOODS[mood]
        w = R.choice(words)
        u = R.choice([f"Pon una playlist {w}", f"Pon algo {w}", f"Pon música {w}",
                      f"Búscame una playlist {w}", f"Pon una de mis playlists {w}"])
        fit, tags = PL.decorate(R, R.choice(fit_pool))[0], PL.mood_tags(mood)
        what = you_form(w)
    # mostly long libraries: the fitting playlist has to be found among many
    fits = R.random() < 0.88
    size = R.choice([R.randint(3, 15), R.randint(15, 40), R.randint(40, 80),
                     R.randint(60, 100), R.randint(80, 100)])
    names = make_library("playlist", must=[fit] if fits else [], avoid_tags=tags, size=size)
    if not names:
        return None
    args = {"entity_id": e.eid, "media_type": "playlist"}
    msgs = base_msgs(h, loc) + [user(u + where)]
    call, cid = tool_call("music_library", args)
    msgs += [call, tool_msg(cid, S.library_result(names))]
    gold = [["music_library", args]]
    if fits:
        pick = fit
        acts = [action("media_player.search_and_play", [e],
                       {"query": pick, "media_type": "playlist"})]
        msgs += call_turn("control", {"actions": acts}, C.ok_result(acts),
                          R.choice([f"Pongo la playlist {pick} en {e.name}.",
                                    f"Vale, ya suena {pick} en {e.name}.",
                                    f"He puesto {pick}."]))
        return sample(msgs, "command", gold, no_history=True)
    some = R.sample(names, min(3, len(names)))
    offer = (f"¿Quieres que ponga {some[0]}?" if len(some) == 1 else
             f"¿Quieres que ponga alguna de estas: {join_es(some, 'o')}?")
    ans = R.choice([f"No he encontrado ninguna playlist {what}. Tienes, por ejemplo, "
                    f"{join_es(some)}.",
                    f"Ninguna de tus playlists encaja. {offer}"])
    msgs += [say(ans)]
    return sample(msgs, "command", gold, no_history=True)


# --------------------------------------------------------------------------
# to-do lists
# --------------------------------------------------------------------------
def pick_list(h, with_items=False):
    lists = [e for e in h.of("todo") if not with_items or e.attrs.get("items")]
    if not lists:
        return None
    shop = [e for e in lists if e.name == SHOP]
    return shop[0] if shop and R.random() < 0.65 else R.choice(lists)


def is_plural(items):
    return len(items) > 1 or items[0] in PLURAL_ITEMS


def i_todo_add(h, loc):
    lst = pick_list(h)
    if not lst:
        return None
    acc, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return None
    items = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    it = join_es(items)
    users = [f"Añade {it} {al(acc)}", f"Apunta {it} en {acc}", f"Pon {it} en {acc}",
             f"{cap(it)} {al(acc)}", f"Mete {it} en {acc}", f"Añade {it} en {acc}"]
    if lst.name == SHOP:
        users += [f"Necesitamos {it}", f"Tengo que comprar {it}", f"Apunta {it} en la lista"
                  if len(h.of("todo")) == 1 else f"Apunta {it} en la lista de la compra"]
    if lst.name == TASKS:
        users += [f"Añade {it} a mis tareas", f"Apunta {it} en mis tareas"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    conf = R.choice([f"He añadido {it} {al(acc)}.",
                     f"Listo, {it} ya {esta(is_plural(items))} en {acc}.",
                     f"Vale, he apuntado {it} en {acc}."])
    return R.choice(users), acts, conf


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    acc, _, _ = LISTS[lst.name]
    item = R.choice(lst.attrs["items"])
    if R.random() < 0.6:
        users = [f"Tacha {item} de {acc}", f"Marca {item} como hecho en {acc}",
                 f"Tachar {item} de {acc}"]
        if lst.name == SHOP:
            users += [f"Ya he comprado {item}, táchalo de la lista"]
        if lst.name == TASKS:
            users += [f"Ya he hecho lo de {item}, táchalo de {acc}"]
        svc = "todo.complete_item"
        conf = R.choice([f"He tachado {item}.", f"Listo, he tachado {item} de {acc}."])
    else:
        users = [f"Borra {item} de {acc}", f"Elimina {item} de {acc}",
                 f"Quita {item} de {acc}"]
        svc = "todo.remove_item"
        conf = R.choice([f"Hecho, {item} ya no {esta(item in PLURAL_ITEMS)} en {acc}.",
                         f"He borrado {item} de {acc}."])
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], conf


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOP)
    acc, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [i for i in pool if i not in items] or items)
        yes = item in items
        if lst.name == TASKS:
            u = f"¿Está «{item}» en {acc}?"
            ans = f"Sí, «{item}» está en {acc}." if yes else f"No, «{item}» no está en {acc}."
        else:
            u = R.choice([f"¿Hay {item} en {acc}?", f"¿Tengo apuntado {item} en {acc}?"])
            ans = f"Sí, hay {item} en {acc}." if yes else f"No, no hay {item} en {acc}."
        must = [item]
    else:
        users = [f"¿Qué hay en {acc}?", f"Léeme {acc}", f"¿Qué pone en {acc}?"]
        if lst.name == SHOP:
            users += ["¿Qué tengo que comprar?", "¿Qué nos falta?"]
        if lst.name == TASKS:
            users += ["¿Qué tengo pendiente hoy?", "¿Qué me queda por hacer?"]
        u = R.choice(users)
        if not items:
            ans = R.choice([f"{cap(acc)} está vacía.", f"Ahora mismo no hay nada en {acc}."])
            must = ["vacía" if "vacía" in ans else "nada"]
        elif len(items) == 1:
            ans = f"En {acc} solo hay {items[0]}."
            must = items
        else:
            ans = f"En {acc} hay {join_es(items)}."
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOP)
    acc, pool, _ = LISTS[lst.name]
    missing = [i for i in pool if i not in lst.attrs["items"]]
    item = R.choice(missing)
    u = R.choice([f"Tacha {item} de {acc}", f"Marca {item} como hecho en {acc}",
                  f"Borra {item} de {acc}"])
    svc = "todo.remove_item" if u.startswith("Borra") else "todo.complete_item"
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"Lo siento, no hay {item} en {acc}.",
                    f"No he encontrado {item} en {acc}."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    acc, pool, _ = LISTS[name]
    item = R.choice(pool)
    u = R.choice([f"Añade {item} {al(acc)}", f"¿Qué hay en {acc}?", f"Apunta {item} en {acc}"])
    others = [e.name for e in h.of("todo")]
    ans = f"No he encontrado {acc}."
    if others:
        ans += f" Solo tienes {join_es([LISTS[n][0] for n in others])}."
    else:
        ans = R.choice([ans, f"Lo siento, no he encontrado ninguna lista llamada {name}."])
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
    temp = es_num(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == "Temperatura exterior" for x in h.of("sensor")):
            return None
        u = R.choice(["¿Qué temperatura hace fuera?", "¿Cuánto frío hace fuera?",
                      "¿Cuántos grados hace afuera?", "¿Cuál es la temperatura exterior?"])
        ans, must = f"Fuera hay {temp} grados.", [temp]
    elif kind == "now":
        u = R.choice(["¿Qué tiempo hace?", "¿Qué tiempo hace ahora?",
                      "¿Qué tiempo hace fuera?", "¿Cómo está el tiempo?",
                      "¿Cómo está el día?", "¿Qué tiempo hace hoy?",
                      "¿El tiempo?", "¿Qué dice el tiempo?", "Cuéntame qué tiempo hace"])
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
        day_q = day if idx < 2 else f"el {day}"
        day_cap = cap(day_q)
        frag = FORECAST_FRAG[raw["condition"]]
        hi, lo = f["temperature"], f.get("templow")
        if kind == "rain":
            u = R.choice([f"¿Va a llover {day_q}?", f"¿Lloverá {day_q}?",
                          f"¿Necesito paraguas {day_q}?"])
            p = f.get("precipitation_probability")
            tail = f"; la probabilidad de lluvia es del {p} %." if p is not None else "."
            if raw["condition"] in RAINY:
                ans = f"Sí, {day_q} {frag}" + tail
                must = ["Sí"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"No, pero {day_q} {frag}" + (
                    f"; la probabilidad de precipitación es del {p} %." if p is not None else ".")
                must = ["No"]
            else:
                ans = f"No, {day_q} {frag}" + tail
                must = ["No"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"¿Cuánto calor hará {day_q}?", f"¿Cuántos grados hará {day_q}?",
                              f"¿Qué temperatura hará {day_q}?"])
                ans, must = f"{day_cap} la máxima será de {gr(hi)}.", [str(hi)]
            else:
                u = R.choice([f"¿Cuánto frío hará {day_q}?",
                              f"¿Cuánto frío hará {day_q} por la noche?"])
                ans, must = f"{day_cap} la mínima será de {gr(lo)}.", [str(lo)]
        else:
            u = R.choice([f"¿Qué tiempo hará {day_q}?", f"¿Qué dice la previsión para {day_q}?",
                          f"¿Cómo será el tiempo {day_q}?"])
            ans = f"{day_cap} {frag}"
            ans += (f", con temperaturas de entre {lo} y {hi} grados." if lo is not None
                    else f", con una máxima de {gr(hi)}.")
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = f"{NOW_SENT[e.state]}, con {temp} grados. " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Hoy las temperaturas irán de {lo} a {hi} grados."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" La probabilidad de lluvia es del {p} %."
            if C.VERBOSITY == "short":
                ans = f"{day_cap} {frag}, máxima de {gr(hi)}."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def gr(n):
    """'1 grado' / '-1 grado' / '5 grados' for whole degrees."""
    return f"{n} grado" if abs(n) == 1 else f"{n} grados"


def ahora_mismo(sentence):
    """'Hace sol' -> 'Ahora mismo hace sol'."""
    return "Ahora mismo " + sentence[:1].lower() + sentence[1:]


def hour_word(time):
    """'18:00' -> 'las 18:00', '01:00' -> 'la 1:00'."""
    hh = int(time.split(":")[0])
    return f"la {hh}:00" if hh == 1 else f"las {hh}:00"


def rest_of_day(e):
    """'A partir de las 18:00 lloverá, con 12 grados, y a las 21:00 habrá 8 grados.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    end = f"a {hour_word(last['time'])} habrá {gr(last['temperature'])}"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        return cap(R.choice(["hasta la noche", "el resto del día", "también más tarde"])) + \
            f" {STAY[e.state]}, y {end}."
    f = later[change]
    out = f"A partir de {hour_word(f['time'])} {FORECAST_FRAG[raw[change]]}, con {gr(f['temperature'])}"
    if change < len(later) - 1:
        out += f", y {end}"
    return out + "."


def weather_report(e, temp, cond_word):
    """'¿Qué tiempo hace?' by answer length: short - now; medium - now and
    the rest of the day; long - plus today's range and rain chance;
    detailed - plus humidity, wind and tomorrow."""
    a, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    first = R.choice([f"{NOW_SENT[e.state]}, con {temp} grados.",
                      f"Ahora mismo hay {temp} grados fuera: {cond_word}.",
                      ahora_mismo(NOW_SENT[e.state]) + f", con {temp} grados."])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {temp} grados.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Hoy las temperaturas irán de {t['templow']} a {t['temperature']} grados.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"La humedad es del {a['humidity']} % y el viento sopla a "
                     f"{es_num(a['wind_speed'])} {a['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([f"Hoy las temperaturas irán de {t['templow']} a {t['temperature']} grados.",
                                   f"Hoy se espera una máxima de {t['temperature']} grados "
                                   f"y una mínima de {t['templow']} grados."]))
        else:
            parts.append(f"Hoy la máxima será de {gr(t['temperature'])}.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"La probabilidad de lluvia es del {p} %." if p else
                         "Hoy no se espera lluvia.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Mañana {FORECAST_FRAG[raw['condition']]}, con una máxima de {gr(fc[1]['temperature'])}.")
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
        u = R.choice(["¿Qué hora es?", "¿Me dices la hora?", "¿Qué hora tenemos?",
                      "Dime la hora", "¿Tienes hora?", "¿La hora?",
                      "¿Qué hora es ahora?"])
        hm = now.strftime("%H:%M")
        son = "Es la" if now.hour == 1 else "Son las"
        ans, must = R.choice([f"{son} {hm}.", f"Ahora mismo {lower_first_word(son)} {hm}."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm}."
        elif C.VERBOSITY == "detailed":
            ans = f"{son} {hm} del {wd} {date}."
    elif kind == "date":
        u = R.choice(["¿Qué fecha es hoy?", "¿A qué fecha estamos?",
                      "¿A cuántos estamos hoy?", "¿Qué día del mes es hoy?",
                      "¿A qué día estamos?", "Dime la fecha",
                      "¿Cuál es la fecha de hoy?", "¿La fecha?", "¿Qué fecha tenemos?"])
        ans, must = f"Hoy es {wd}, {date}.", [" ".join(date.split(" ")[:3])]
    else:
        u = R.choice(["¿Qué día de la semana es hoy?", "¿Qué día es hoy?",
                      "¿Hoy qué día es?", "¿En qué día de la semana estamos?",
                      "¿Qué día es?", "¿Hoy es viernes?",
                      "¿Ya es viernes?", "Dime qué día es hoy",
                      "¿Estamos a lunes?", "¿A qué día de la semana estamos?",
                      "¿Hoy es fin de semana?", "¿Qué día tenemos hoy?"])
        ans, must = R.choice([f"Hoy es {wd}.", f"Hoy es {wd}, {date}."]), [wd]
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("get_time", {}, res, ans)
    return sample(msgs, "question", [["get_time", {}]], answer_contains=must)


def lower_first_word(s):
    return s[:1].lower() + s[1:]


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
        return 0, m, 0, "un minuto" if m == 1 else f"{numw(m)} minutos"
    if k < 0.65:
        return R.choice([(0, 30, 0, "media hora"), (0, 15, 0, "un cuarto de hora"),
                         (0, 45, 0, "tres cuartos de hora"), (1, 0, 0, "una hora"),
                         (1, 30, 0, "hora y media"), (1, 30, 0, "una hora y media"),
                         (2, 0, 0, "dos horas"), (2, 0, 0, "2 horas"),
                         (3, 0, 0, "tres horas")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        return 0, 0, s, f"{numw(s)} segundos"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = "un minuto" if m == 1 else f"{numw(m)} minutos"
        return 0, m, s, f"{mw} y {s} segundos"
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = "una hora" if hh == 1 else f"{numw(hh)} horas"
    return hh, m, 0, f"{hw} y {m} minutos"


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
    """'temporizador de la pizza' / 'temporizador'."""
    return f"temporizador {de_t(name)}" if name else "temporizador"


def s_timer_start(h, loc):
    hh, m, s, spoken = duration()
    named = F("named", R.random() < 0.35)
    name = R.choice(TIMER_NAMES)[0] if named else ""
    if named:
        para, dt = para_t(name), de_t(name)
        u = R.choice([f"Pon un temporizador de {spoken} {para}",
                      f"Pon el temporizador {dt} a {spoken}",
                      f"Temporizador {dt}, {spoken}", f"Temporizador {para}, {spoken}",
                      f"Pon un temporizador de {spoken} llamado {name}",
                      f"{cap(spoken)} {para}", f"Inicia un temporizador {para} de {spoken}"])
    else:
        u = R.choice([f"Pon un temporizador de {spoken}", f"Temporizador de {spoken}",
                      f"Programa un temporizador de {spoken}", f"Inicia un temporizador de {spoken}",
                      f"Despiértame dentro de {spoken}", f"Temporizador {spoken}",
                      f"¿Puedes poner un temporizador de {spoken}?",
                      f"Avísame en {spoken}", f"Cuenta atrás de {spoken}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Lo siento, en este dispositivo no puedo poner temporizadores. Eso "
                        "funciona en un dispositivo de voz como Home Assistant Voice.",
                        "Los temporizadores solo funcionan en un dispositivo de voz, no aquí.",
                        "Aquí no puedo poner temporizadores, solo en un dispositivo de voz."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        ans = R.choice([f"El temporizador {de_t(name)} está en marcha: {dur}.",
                        f"He puesto un temporizador de {dur} {para_t(name)}.",
                        f"Vale, {dur} {para_t(name)}."])
    else:
        ans = R.choice([f"El temporizador está en marcha: {dur}.",
                        f"He puesto un temporizador de {dur}.",
                        f"Temporizador de {dur} iniciado.",
                        f"Vale, {dur}. El temporizador está en marcha."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Apaga la luz dentro de 10 minutos' -> timer with the command; the
    command runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = out[0].rstrip(".!?")
    if cmd.endswith(", por favor"):
        cmd = cmd[:-len(", por favor")]
    hh, m, s, spoken = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2 or words[0].endswith("r"):
        return None       # "¿Puedes ...?" or "Encender ..." do not fit behind "En 10 minutos"
    forms = [f"En {spoken} {lower_first(cmd)}", f"{cmd} en {spoken}",
             f"{cmd} dentro de {spoken}",
             f"Pon un temporizador de {spoken} y luego {lower_first(cmd)}",
             f"{words[0]} en {spoken} {words[1]}", f"Dentro de {spoken} {lower_first(cmd)}"]
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"Vale, lo haré dentro de {dur}.", f"De acuerdo, me encargo dentro de {dur}.",
                    f"En {dur} lo hago."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"No hay ningún {timer_label(asked_name)} en marcha.", ["no"]
        return R.choice(["Ahora mismo no hay ningún temporizador en marcha.",
                         "No hay ningún temporizador activo."]), ["ningún"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        if e.get("paused"):
            return f"El {lab} está en pausa; quedan {e['remaining']}.", [e["remaining"]]
        return R.choice([f"Al {lab} le quedan {e['remaining']}.",
                         f"Quedan {e['remaining']}."]), [e["remaining"]]
    parts = [(f"el {de_t(e['name'])}" if e["name"] else "uno sin nombre")
             + f", con {e['remaining']}" + (" (en pausa)" if e.get("paused") else "")
             for e in entries]
    return f"Hay {len(entries)} temporizadores en marcha: " + join_es(parts) + ".", \
        [e["remaining"] for e in entries]


def s_timer_status(h, loc):
    n = R.choice([0, 1, 1, 1, 2, 2, 3])
    timers = running_timers(n, named=n > 1 or R.random() < 0.4)
    named = [t for t in timers if t[0]]
    args = {"action": "status"}
    asked = None
    if named and R.random() < 0.35:
        asked = R.choice(named)[0] if R.random() < 0.8 else R.choice(TIMER_NAMES)[0]
        u = R.choice([f"¿Cuánto le queda al {timer_label(asked)}?", f"¿Cuánto falta {para_t(asked)}?",
                      f"¿Cuánto tiempo le queda al temporizador {de_t(asked)}?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["¿Cuánto le queda al temporizador?", "¿Cuánto tiempo queda en el temporizador?",
                      "¿Cuánto falta?", "¿Qué temporizadores hay?", "¿Hay algún temporizador en marcha?",
                      "¿Cuánto queda?", "Estado del temporizador"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


TIMER_OPS = {
    "cancel": (["Cancela el {t}", "Detén el {t}", "Cancelar el {t}", "Borra el {t}",
                "Quita el {t}", "Apaga el {t}, por favor"],
               ["He cancelado el {t}.", "El {t} está cancelado.", "Vale, {t} cancelado."]),
    "pause": (["Pausa el {t}", "Pon en pausa el {t}", "Pausar el {t}", "Para un momento el {t}"],
              ["El {t} está en pausa.", "He pausado el {t}."]),
    "resume": (["Reanuda el {t}", "Continúa con el {t}", "Que siga el {t}",
                "Vuelve a poner en marcha el {t}"],
               ["El {t} vuelve a estar en marcha.", "Vale, el {t} sigue."]),
    "add": (["Añade {d} al {t}", "Alarga el {t} {d}", "Ponle {d} más al {t}",
             "Dale {d} más al {t}"],
            ["He añadido {D} al {t}.", "El {t} durará {D} más."]),
    "subtract": (["Quita {d} al {t}", "Acorta el {t} {d}", "Réstale {d} al {t}"],
                 ["El {t} durará {D} menos.", "He quitado {D} al {t}."]),
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
    # "Alarga el temporizador 5 minutos" reads better with "en" for the amount
    u = R.choice(users).format(t=t, T=cap(t), d=spoken)
    u = u.replace(f"Alarga el {t} {spoken}", f"Alarga el {t} en {spoken}") \
         .replace(f"Acorta el {t} {spoken}", f"Acorta el {t} en {spoken}")
    args = {"action": op}
    if op in ("add", "subtract"):
        args.update(dur_args(hh, m, s))
    if nm:
        args["name"] = nm
    if r < 0.6:
        ans = R.choice(answers).format(t=t, D=S.fmt_duration(hh, m, s, LANG))
    elif r < 0.8:
        ans = R.choice(["Ahora mismo no hay ningún temporizador en marcha.",
                        "Lo siento, no hay ningún temporizador en marcha."])
    else:
        others = [timer_label(x[0]) for x in timers]
        ans = (f"No hay ningún {t}. Solo {esta(len(others) > 1)} en marcha "
               f"{join_es(['el ' + o for o in others])}.")
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
    u = R.choice(users).format(t="temporizador", T="Temporizador", d="")
    args = {"action": op}
    names = [t[0] for t in timers]
    labels = [f"el {de_t(n)}" for n in names]
    ask = R.choice([f"¿Qué temporizador quieres decir: {join_es(labels, 'o')}?",
                    f"Hay varios temporizadores en marcha: {join_es(labels)}. ¿Cuál quieres decir?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["Todos", "Todos los temporizadores",
                                "Los dos" if len(names) == 2 else "Todos"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          f"He cancelado los {len(names)} temporizadores.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    msgs += [user(R.choice([pick, f"El {de_t(pick)}", f"El temporizador {de_t(pick)}",
                            f"El temporizador {para_t(pick)}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      R.choice(answers).format(t=timer_label(pick), D=""))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Cancela todos los temporizadores", "Borra todos los temporizadores",
                  "Para todos los temporizadores", "Cancelar todos los temporizadores",
                  "Quita todos los temporizadores", "Apaga todos los temporizadores",
                  "Elimina todos los temporizadores", "Detén todos los temporizadores",
                  "¿Puedes cancelar todos los temporizadores?",
                  "Borra todos los temporizadores, por favor",
                  "Fuera todos los temporizadores", "Cancela todos los timers",
                  "Quita todas las cuentas atrás"])
    args = {"action": "cancel_all"}
    ans = {0: "No había ningún temporizador en marcha.", 1: "He cancelado el temporizador."}.get(
        n, f"He cancelado los {n} temporizadores.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh}:{mm:02d}" if mm else f"{hh}"
    u = R.choice([f"Despiértame a las {t}", f"Pon una alarma a las {t}",
                  f"Pon el despertador para mañana a las {t}", f"Alarma a las {t}"])
    ans = R.choice(["Lo siento, no puedo poner una alarma a una hora concreta. "
                    "Pero sí puedo ponerte un temporizador.",
                    "Las alarmas a una hora fija no las puedo poner, solo temporizadores. "
                    "¿Quieres que ponga uno?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Haz un anuncio: {main}", f"Diles a todos que {sub}",
                  f"Avisa en toda la casa de que {sub}", f"Anuncio: {main}",
                  f"Di por toda la casa: {main}", f"Avisa a todos de que {sub}",
                  f"Anuncia que {sub}", f"Di en todos los altavoces que {sub}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Lo siento, no hay ningún otro dispositivo de voz para hacer el anuncio.",
                        "No puedo hacer el anuncio: no hay más dispositivos de voz en casa."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["He hecho el anuncio.", "Anuncio enviado.",
                        "Hecho, lo he anunciado en toda la casa."])
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
        users = [f"Pon {ref.acc} al {v} %", f"{cap(ref.acc)} al {v} %",
                 f"Ajusta la humedad {de(ref.acc)} al {v} %"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Pon la humedad {C.en_room(e.area)} al {v} %")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} {esta(ref.g)} {C.agr('ajustado', ref.g)} al {v} %."
    on = r < 0.8
    users = [f"{'Enciende' if on else 'Apaga'} {ref.acc}",
             f"{'Encender' if on else 'Apagar'} {ref.acc}"]
    adj = "encendido" if on else "apagado"
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, adj, adj)


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "a la mitad"), (25, "al 25 %"), (30, "al 30 %"),
                        (75, "al 75 %"), (50, "al 50 %")])
    u = R.choice([f"Abre {e.name} {word}", f"Pon {e.name} {word}"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"He abierto {e.name} al {p} %."


def describe_assist(h, a, who):
    """Wording behind "He ..." for the Assist services."""
    dom, svc = a["service"].split(".", 1)
    data = a.get("data") or {}
    by_id = {e.eid: e for e in h.ents}
    if dom == "todo" and "item" in data:
        lst = by_id.get(a["entity_id"][0]) if a.get("entity_id") else None
        acc = LISTS[lst.name][0] if lst is not None and lst.name in LISTS else who
        it = join_es(data.get("items") or [data["item"]])
        return {"add_item": f"añadido {it} {al(acc)}",
                "complete_item": f"tachado {it} de {acc}",
                "remove_item": f"borrado {it} de {acc}"}.get(svc)
    if svc == "search_and_play":
        what = data.get("query") or "música"
        if data.get("media_type") == "artist":
            what = f"música de {what}"
        elif data.get("artist"):
            what = f"{what} de {data['artist']}"
        return f"puesto {what} en {who}"
    if svc == "seek_relative":
        n = int(data.get("seconds") or 0)
        word = "un minuto" if abs(n) == 60 else f"{abs(n)} segundos"
        return f"{'adelantado' if n >= 0 else 'retrocedido'} {word} en {who}"
    if svc == "shuffle_set":
        return f"{'activado' if data.get('shuffle') else 'desactivado'} el modo aleatorio en {who}"
    if svc == "repeat_set":
        return {"one": f"puesto la canción en bucle en {who}",
                "all": f"activado la repetición de todas las canciones en {who}",
                "off": f"desactivado la repetición en {who}"}.get(data.get("repeat"))
    if svc == "volume_mute":
        return (f"silenciado {who}" if data.get("is_volume_muted")
                else f"vuelto a activar el sonido {de(who)}")
    if svc == "set_humidity":
        return f"puesto {who} al {data.get('humidity')} % de humedad"
    if svc == "set_valve_position":
        return f"abierto {who} al {data.get('position')} %"
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
    """'Pon Queen en la cocina' -> 'Más alto' / 'Siguiente': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Más alto", "Sube el volumen", "Sube un poco, por favor"],
         "Volumen subido."),
        ("media_player.volume_down", ["Más bajo", "Baja el volumen", "Un poco más bajo"],
         "Volumen bajado."),
        ("media_player.media_next_track", ["Siguiente canción", "Siguiente", "Pasa de canción"],
         "Siguiente canción."),
        ("media_player.media_pause", ["Pausa", "Para un momento", "Pausa la música"], "En pausa."),
        ("media_player.turn_off", ["Apaga la música", "Apágalo"], "La música está apagada."),
        ("media_player.shuffle_set", ["Y en aleatorio, por favor"], "El modo aleatorio está activado."),
        ("media_player.media_previous_track", ["Atrás", "La anterior", "Vuelve a la anterior"],
         "Canción anterior."),
        ("media_player.seek_relative", ["Adelanta", "Adelanta un poco"], "Adelantado."),
        ("media_player.seek_relative-", ["Retrocede", "Rebobina un poco"], "Retrocedido."),
        ("media_player.media_stop", ["Para", "Para la música"], "Detenido."),
    ])
    data = {"shuffle": True} if svc.endswith("shuffle_set") else None
    if svc.startswith("media_player.seek_relative"):
        data = {"seconds": -30 if svc.endswith("-") else 30}
        svc = "media_player.seek_relative"
    c.user(R.choice(users))
    c.control([action(svc, [e], data)], ans)
    if svc == "media_player.media_pause" and R.random() < 0.6:
        c.user(R.choice(["Play", "Sigue", "Reanuda", "Continúa"]))
        c.control([action("media_player.media_play", [e])], "La reproducción continúa.")
    return True


def turn_timer_follow(c):
    """Start a timer, then ask how long it still runs or cancel it."""
    hh, m, s, spoken = duration()
    if not m and not hh:
        return False
    named = R.random() < 0.4
    name = R.choice(TIMER_NAMES)[0] if named else ""
    c.user(f"Pon un temporizador de {spoken}" + (f" {para_t(name)}" if named else ""))
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"El {timer_label(name)} está en marcha: {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["¿Cuánto queda?", "¿Cuánto le queda?", "¿Y cuánto falta?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Cancélalo", "Mejor no, borra el temporizador", "Quítalo otra vez"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"He cancelado el {timer_label(name)}.")
    return True


def turn_list_follow(c):
    """'Añade leche a la lista de la compra' -> 'y huevos'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    acc, pool, _ = LISTS[lst.name]
    free = [i for i in pool if i not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item = R.choice(free)
    c.user(R.choice([f"{cap(C.y_word(item))} {item}", f"{cap(item)} también",
                     f"Y también {item}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    c.control(a2, R.choice([f"He añadido también {item} {al(acc)}.",
                            f"Apuntado: {item} también {esta(item in PLURAL_ITEMS)} en {acc}."]))
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
        # finding one playlist among many is the hardest library task: twice the share
        ("music:library_play_long", s_library_play, {}),
        ("music:library_favorites_long", lambda h, loc: s_library_play(h, loc, True), {}),
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
