"""French samples for the Home Assistant Assist built-ins.

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
from .core import (R, S, a, action, agr, base_msgs, cap, de, deg, est, fr_num, low, nest,
                   que, slugify, the, tool_call, tool_msg, voice)

LANG = C.LANG


def F(key, default=None):
    return C.FORCE.get(key, default)


def join_fr(items, word="et"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


VERBS = C.VERBS


def lower_first(text):
    """Imperatives move behind 'Dans 10 minutes, '; names keep their capital."""
    return low(text) if text.split(" ", 1)[0] in VERBS else text


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
ARTISTS = ["Stromae", "Angèle", "Indochine", "Johnny Hallyday", "Édith Piaf", "Daft Punk",
           "Céline Dion", "Jean-Jacques Goldman", "Christine and the Queens", "Aya Nakamura",
           "Orelsan", "Zaz", "Vianney", "Louane", "Mylène Farmer", "Francis Cabrel",
           "Coldplay", "Queen", "Adele", "Ed Sheeran", "Taylor Swift", "ABBA", "Dua Lipa",
           "Billie Eilish", "Phoenix", "Jul", "Calogero", "Renaud", "Serge Gainsbourg",
           "Charles Aznavour", "Mozart", "Ludovico Einaudi", "Pink Floyd", "Clara Luciani",
           "Julien Doré"]
TRACKS = [("Alors on danse", "Stromae"), ("Papaoutai", "Stromae"),
          ("Balance ton quoi", "Angèle"), ("L'aventurier", "Indochine"),
          ("La vie en rose", "Édith Piaf"), ("Djadja", "Aya Nakamura"), ("Je veux", "Zaz"),
          ("Get Lucky", "Daft Punk"), ("Bohemian Rhapsody", "Queen"), ("Yellow", "Coldplay"),
          ("Shape of You", "Ed Sheeran"), ("Hello", "Adele"), ("Dancing Queen", "ABBA"),
          ("Levitating", "Dua Lipa"), ("Bad Guy", "Billie Eilish"),
          ("Pour que tu m'aimes encore", "Céline Dion"), ("Désenchantée", "Mylène Farmer"),
          ("La Bohème", "Charles Aznavour"), ("Basique", "Orelsan"),
          ("La grenade", "Clara Luciani"), ("Wish You Were Here", "Pink Floyd"),
          ("Je te promets", "Johnny Hallyday")]
ALBUMS = [("Racine carrée", "Stromae"), ("Brol", "Angèle"),
          ("Random Access Memories", "Daft Punk"), ("A Night at the Opera", "Queen"),
          ("Parachutes", "Coldplay"), ("Civilisation", "Orelsan"), ("D'eux", "Céline Dion"),
          ("Divide", "Ed Sheeran"), ("21", "Adele"), ("The Dark Side of the Moon", "Pink Floyd"),
          ("Folklore", "Taylor Swift"), ("Nakamura", "Aya Nakamura")]
PLAYLISTS = ["Chill", "Cuisine", "Soirée", "Sport", "Détente", "Tubes de l'été",
             "Mes préférées", "Dodo", "Rock Classics", "Comptines",
             "Chansons de Noël", "Concentration", "Discover Weekly", "Road trip", "Années 80"]
STATIONS = ["France Inter", "RTL", "Europe 1", "NRJ", "Fun Radio", "RMC", "France Info",
            "Nostalgie", "Skyrock", "Chérie FM", "RFM", "France Culture", "FIP",
            "Radio Nova", "Virgin Radio", "France Musique"]
# (query, with partitive article)
GENRES = [("jazz", "du jazz"), ("musique classique", "de la musique classique"),
          ("rock", "du rock"), ("musique relaxante", "de la musique relaxante"),
          ("comptines", "des comptines"), ("chants de Noël", "des chants de Noël"),
          ("lo-fi", "du lo-fi"), ("variété française", "de la variété française"),
          ("hip-hop", "du hip-hop"), ("musique pour dormir", "de la musique pour dormir"),
          ("musique pour cuisiner", "de la musique pour cuisiner"),
          ("tubes des années 80", "des tubes des années 80"), ("pop", "de la pop"),
          ("musique de méditation", "de la musique de méditation")]

SPEAKER_NAMES = ["Sonos {room}", "Enceinte {room}", "Echo {room}", "HomePod {room}",
                 "Enceinte {room}", "{room} Sonos", "Nest Audio {room}",
                 "Music Assistant {room}", "Haut-parleur {room}"]
MUSIC_LABELS = ("Enceinte", "Radio", "Barre de son")


def item_forms(item, kind, g):
    """-> (partitive 'du lait', definite 'le lait')."""
    if kind == "t":                              # a task: the words themselves
        return item, f"« {item} »"
    if kind == "d":                              # always with the definite article
        return the(g, item), the(g, item)
    if C.plural(g):
        part = f"des {item}"
    elif C.elides(item):
        part = f"de l'{item}"
    else:
        part = f"{'de la' if C.fem(g) else 'du'} {item}"
    return part, the(g, item)


def _items(kind, rows):
    return [(i, *item_forms(i, kind, g), g) for i, g in rows]


# list items: (item, partitive, definite, gender)
GROCERIES = _items("p", [
    ("lait", "m"), ("œufs", "mp"), ("beurre", "m"), ("pain", "m"), ("fromage", "m"),
    ("pommes", "fp"), ("bananes", "fp"), ("tomates", "fp"), ("café", "m"), ("pâtes", "fp"),
    ("riz", "m"), ("yaourts", "mp"), ("farine", "f"), ("sucre", "m"),
    ("papier toilette", "m"), ("liquide vaisselle", "m"), ("lessive", "f"),
    ("croquettes pour chat", "fp"), ("lait d'avoine", "m"), ("oignons", "mp"),
    ("pommes de terre", "fp"), ("eau gazeuse", "f"), ("jus d'orange", "m"),
    ("sacs poubelle", "mp"), ("flocons d'avoine", "mp"), ("concombres", "mp"),
    ("poivrons", "mp"), ("viande hachée", "f"), ("crème fraîche", "f"),
    ("huile d'olive", "f"), ("sel", "m"), ("jambon", "m")])
LISTS = {  # name: (phrase, items, slugs)
    "Liste de courses": ("la liste de courses", GROCERIES, ["shopping_list", "liste_de_courses"]),
    "Tâches": ("la liste de tâches", _items("t", [
        ("sortir les poubelles", "m"), ("faire la déclaration d'impôts", "m"),
        ("laver la voiture", "m"), ("tondre la pelouse", "m"), ("appeler mamie", "m"),
        ("nettoyer les vitres", "m"), ("ranger la cave", "m"), ("récupérer le colis", "m"),
        ("prendre rendez-vous chez le médecin", "m"), ("arroser les plantes", "m")]),
        ["taches", "tasks"]),
    "Bricolage": ("la liste bricolage", _items("p", [
        ("vis", "fp"), ("chevilles", "fp"), ("peinture", "f"), ("silicone", "m"),
        ("ruban adhésif", "m"), ("ampoules", "fp"), ("terreau", "m"), ("serre-câbles", "mp"),
        ("papier de verre", "m")]), ["bricolage", "hardware_store"]),
    "Pharmacie": ("la liste pharmacie", _items("p", [
        ("dentifrice", "m"), ("shampoing", "m"), ("gel douche", "m"), ("crème solaire", "f"),
        ("déodorant", "m"), ("cotons démaquillants", "mp"), ("mouchoirs", "mp"),
        ("crème pour les mains", "f")]), ["pharmacie", "drugstore"]),
    "Valise": ("la liste valise", _items("d", [
        ("chargeur", "m"), ("lunettes de soleil", "fp"), ("passeport", "m"),
        ("maillot de bain", "m"), ("brosse à dents", "f"), ("imperméable", "m"),
        ("écouteurs", "mp"), ("adaptateur", "m")]), ["valise", "packing_list"]),
}
SHOP = "Liste de courses"
TASKS = "Tâches"


def item_info(lst_name, item):
    return next(x for x in LISTS[lst_name][1] if x[0] == item)


CITIES = ["Lyon", "Nantes", "Bordeaux", "Lille", "Rennes", "Grenoble", "Strasbourg",
          "Dijon", "Tours", "Annecy"]
WEATHER_NAMES = ["Forecast Home", "Maison", "Météo", "OpenWeatherMap", "Météo-France",
                 "Met.no", "Prévisions"]
RAINY = ("rainy", "pouring", "lightning-rainy", "snowy-rainy")
# "Demain, " + fragment
FORECAST_FRAG = {
    "sunny": "il fera beau", "clear-night": "le ciel sera dégagé",
    "cloudy": "le ciel sera couvert", "partlycloudy": "il y aura quelques nuages",
    "rainy": "il va pleuvoir", "pouring": "il va pleuvoir fort", "snowy": "il va neiger",
    "snowy-rainy": "il y aura de la neige fondue", "fog": "il y aura du brouillard",
    "hail": "il va grêler", "lightning": "il y aura de l'orage",
    "lightning-rainy": "il y aura des orages et de la pluie",
    "windy": "il y aura du vent", "windy-variant": "il y aura du vent et des nuages",
    "exceptional": "des intempéries sont annoncées",
}
NOW_SENT = {
    "sunny": "Il fait beau", "clear-night": "Le ciel est dégagé", "cloudy": "Le ciel est couvert",
    "partlycloudy": "Le ciel est partiellement nuageux", "rainy": "Il pleut",
    "pouring": "Il pleut à verse", "snowy": "Il neige", "snowy-rainy": "Il tombe de la neige fondue",
    "fog": "Il y a du brouillard", "hail": "Il grêle", "lightning": "Il y a de l'orage",
    "lightning-rainy": "Il y a de l'orage et il pleut", "windy": "Il y a du vent",
    "windy-variant": "Il y a du vent et des nuages", "exceptional": "Il y a des intempéries",
}

# (name, "pour ..." phrase, "de ..." phrase)
TIMER_NAMES = [("Pizza", "pour la pizza", "de la pizza"), ("Pâtes", "pour les pâtes", "des pâtes"),
               ("Œufs", "pour les œufs", "des œufs"), ("Thé", "pour le thé", "du thé"),
               ("Gâteau", "pour le gâteau", "du gâteau"),
               ("Lessive", "pour la lessive", "de la lessive"),
               ("Pain", "pour le pain", "du pain"), ("Riz", "pour le riz", "du riz"),
               ("Pommes de terre", "pour les pommes de terre", "des pommes de terre"),
               ("Four", "pour le four", "du four"), ("Pause", "pour la pause", "de la pause"),
               ("Parking", "pour le parking", "du parking"),
               ("Devoirs", "pour les devoirs", "des devoirs"),
               ("Steak", "pour le steak", "du steak"),
               ("Lasagnes", "pour les lasagnes", "des lasagnes"),
               ("Frites", "pour les frites", "des frites"),
               ("Gaufres", "pour les gaufres", "des gaufres"),
               ("Légumes", "pour les légumes", "des légumes")]
TIMER_POUR = {n: p for n, p, _ in TIMER_NAMES}
TIMER_DE = {n: d for n, _, d in TIMER_NAMES}
NUM_WORDS = {2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept",
             8: "huit", 10: "dix", 12: "douze", 15: "quinze", 20: "vingt",
             30: "trente"}

# (reported speech after "que", the announcement itself)
BROADCASTS = [("le repas est prêt", "Le repas est prêt"),
              ("on part dans cinq minutes", "On part dans cinq minutes"),
              ("c'est l'heure d'aller au lit", "C'est l'heure d'aller au lit"),
              ("la pizza est arrivée", "La pizza est arrivée"),
              ("il y a quelqu'un à la porte", "Il y a quelqu'un à la porte"),
              ("il faut venir à table", "Tout le monde à table"),
              ("le film va commencer", "Le film va commencer"),
              ("je rentre bientôt", "Je rentre bientôt"),
              ("il faut encore sortir les poubelles", "Il faut encore sortir les poubelles"),
              ("on attend de la visite", "On attend de la visite"),
              ("le petit-déjeuner est prêt", "Le petit-déjeuner est prêt"),
              ("le gâteau est prêt", "Le gâteau est prêt"),
              ("tout le monde doit descendre", "Tout le monde descend"),
              ("on va bientôt se promener", "On va bientôt se promener"),
              ("c'est l'heure des devoirs", "C'est l'heure des devoirs")]


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
    g = "m" if name.startswith(("Haut-parleur", "HomePod", "Echo", "Nest", "Music")) else "f"
    return h.add("media_player", room, "Enceinte", g, [slugify(name), "speaker"], name=name,
                 state="playing" if playing else R.choice(["idle", "paused", "off"]),
                 attrs=attrs)


def add_list(h, name, items=None):
    phrase, pool, slugs = LISTS[name]
    if items is None:
        items = [x[0] for x in R.sample(pool, R.choice([0, 1, 2, 3, 3, 4, 5, 6]))]
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
    "sunny": "le temps restera ensoleillé", "clear-night": "le ciel restera dégagé",
    "cloudy": "le ciel restera couvert", "partlycloudy": "le ciel restera partiellement nuageux",
    "rainy": "la pluie va continuer", "pouring": "il va continuer à pleuvoir fort",
    "snowy": "il va continuer à neiger", "snowy-rainy": "la neige fondue va continuer",
    "fog": "le brouillard va persister", "hail": "la grêle va continuer",
    "lightning": "le temps restera orageux",
    "lightning-rainy": "le temps restera orageux et pluvieux", "windy": "le vent va persister",
    "windy-variant": "le temps restera venteux et nuageux",
    "exceptional": "les intempéries vont continuer",
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
    room = R.choice([None, None] + [r for r in h.rooms if r in ("Extérieur", "Jardin")])
    slug = "forecast_home" if name in ("Forecast Home", "Maison") else slugify(name)
    e = h.add("weather", room, "Météo", "f", [slug], name=name, state=cond, attrs=attrs)
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
        rooms = [r for r in h.rooms if r in ("Chambre", "Chambre d'enfant", "Salon", "Bureau")]
        if rooms:
            room = R.choice(rooms)
            on = R.random() < 0.5
            h.add("humidifier", room, "Humidificateur", "m", ["humidificateur", "humidifier"],
                  state="on" if on else "off",
                  attrs={"humidity": R.choice([40, 45, 50, 55]),
                         "current_humidity": R.randint(30, 60)})


C.POPULATE_HOOKS.append(populate)


def describe_assist(h, x, who):
    """Descriptions for the summary of a conversation ('J'ai ...')."""
    dom, svc = x["service"].split(".", 1)
    data = x.get("data") or {}
    if svc == "search_and_play":
        q = data.get("query", "de la musique")
        kind = {"artist": f"de la musique {de(q)}", "album": f"l'album {q}",
                "playlist": f"la playlist {q}", "radio": f"la radio {q}"}.get(data.get("media_type"), q)
        return f"lancé {kind} sur {who}"
    if dom == "todo":
        lst = next((e for e in h.ents if e.eid in x["entity_id"]), None)
        if lst is None or lst.name not in LISTS or "item" not in data:
            return None
        phrase = LISTS[lst.name][0]
        infos = [next((i for i in LISTS[lst.name][1] if i[0] == it), (it, it, it, "m"))
                 for it in data.get("items") or [data["item"]]]
        defi = join_fr([i[2] for i in infos])
        if svc == "add_item":
            return f"ajouté {join_fr([i[1] for i in infos])} {a(phrase)}"
        if svc == "complete_item":
            return f"coché {defi} sur {phrase}"
        return f"retiré {defi} {de(phrase)}"
    if svc == "shuffle_set":
        return f"{'activé' if data.get('shuffle') else 'désactivé'} la lecture aléatoire sur {who}"
    if svc == "repeat_set":
        return {"one": f"mis la chanson en boucle sur {who}",
                "all": f"activé la répétition de tous les titres sur {who}"}.get(
            data.get("repeat"), f"désactivé la répétition sur {who}")
    if svc == "volume_mute":
        return f"{'coupé' if data.get('is_volume_muted') else 'remis'} le son {de(who)}"
    if svc == "set_humidity":
        return f"réglé {who} sur {data.get('humidity')} %"
    if svc == "set_valve_position":
        return f"ouvert {who} à {data.get('position')} %"
    return None


C.DESCRIBE_HOOKS.append(describe_assist)


# --------------------------------------------------------------------------
# music
# --------------------------------------------------------------------------
def music_request():
    """-> (sentence without target, data, description for the answer)."""
    k = R.choice(["artist", "artist", "track", "track", "album", "playlist", "radio",
                  "radio", "genre", "plain"])
    if k == "artist":
        a_ = R.choice(ARTISTS)
        u = R.choice([f"Mets de la musique {de(a_)}", f"Joue des chansons {de(a_)}",
                      f"Lance des morceaux {de(a_)}", f"J'ai envie d'écouter des chansons {de(a_)}",
                      f"Mets un peu de musique {de(a_)}", f"Joue l'artiste {a_}"])
        return u, {"query": a_, "media_type": "artist"}, f"de la musique {de(a_)}"
    if k == "track":
        t, a_ = R.choice(TRACKS)
        if R.random() < 0.5:
            u = R.choice([f"Joue {t} {de(a_)}", f"Mets la chanson {t} {de(a_)}",
                          f"Je veux écouter {t} {de(a_)}", f"Lance le morceau {t} {de(a_)}"])
            return u, {"query": t, "media_type": "track", "artist": a_}, f"{t} {de(a_)}"
        u = R.choice([f"Mets la chanson {t}", f"Joue le morceau {t}", f"Lance le titre {t}"])
        return u, {"query": t, "media_type": "track"}, t
    if k == "album":
        al, a_ = R.choice(ALBUMS)
        if R.random() < 0.5:
            return (R.choice([f"Mets l'album {al} {de(a_)}", f"Joue l'album {al} {de(a_)}"]),
                    {"query": al, "media_type": "album", "artist": a_}, f"l'album {al}")
        return R.choice([f"Mets l'album {al}", f"Lance l'album {al}"]), \
            {"query": al, "media_type": "album"}, f"l'album {al}"
    if k == "playlist":
        p = R.choice(PLAYLISTS)
        u = R.choice([f"Mets ma playlist {p}", f"Lance la playlist {p}",
                      f"Joue la playlist {p}", f"Mets la playlist {p}"])
        return u, {"query": p, "media_type": "playlist"}, f"la playlist {p}"
    if k == "radio":
        s = R.choice(STATIONS)
        u = R.choice([f"Mets la radio {s}", f"Mets la station {s}", f"Lance la radio {s}",
                      f"Allume la radio sur {s}", f"Je veux écouter la radio {s}"])
        return u, {"query": s, "media_type": "radio"}, s
    if k == "genre":
        q, part = R.choice(GENRES)
        return R.choice([f"Mets {part}", f"Joue {part}"]), {"query": q}, part
    # "Mets Yellow": the kind is not said, so no media_type
    t = R.choice([t for t, _ in TRACKS] + [a_ for a_ in ARTISTS])
    return R.choice([f"Mets {t}", f"Joue {t}"]), {"query": t}, t


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
        return e, R.choice([f" sur {e.name}", f" sur {e.name}", f" via {e.name}"])
    if k == "room":
        room = R.choice(single_rooms)
        e = next(e for e in ps if e.area == room)
        return e, " " + C.prep(room)
    if k == "loc":
        return in_loc[0], ""
    return ps[0], ""


def play_answer(desc, e):
    return R.choice([f"Je lance {desc} sur {e.name}.", f"D'accord, {desc} sur {e.name}.",
                     f"C'est parti : {desc} sur {e.name}.", f"Je lance {desc}."])


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
         ["Active la lecture aléatoire{w}", "Lecture aléatoire{w}", "Mets le mode aléatoire{w}",
          "Joue les chansons dans le désordre{w}"],
         ["La lecture aléatoire est activée.", "Le mode aléatoire est activé."]),
        ("media_player.shuffle_set", {"shuffle": False},
         ["Désactive la lecture aléatoire{w}", "Arrête le mode aléatoire{w}",
          "Plus de lecture aléatoire{w}"],
         ["La lecture aléatoire est désactivée."]),
        ("media_player.repeat_set", {"repeat": "one"},
         ["Répète cette chanson{w}", "Mets la chanson en boucle{w}",
          "Répète le titre en cours{w}"], ["La chanson passe en boucle."]),
        ("media_player.repeat_set", {"repeat": "all"},
         ["Répète tout{w}", "Répète la playlist{w}", "Répète tous les titres{w}"],
         ["Tous les titres seront répétés."]),
        ("media_player.repeat_set", {"repeat": "off"},
         ["Désactive la répétition{w}", "Arrête de répéter{w}", "Plus de répétition{w}"],
         ["La répétition est désactivée."]),
        ("media_player.volume_mute", {"is_volume_muted": True},
         ["Coupe le son{w}", "Mets en sourdine{w}", "Mets le son en muet{w}", "Silence{w}"],
         [f"{e.name} est en sourdine.", "Le son est coupé."]),
        ("media_player.volume_mute", {"is_volume_muted": False},
         ["Remets le son{w}", "Enlève la sourdine{w}", "Réactive le son{w}"],
         ["Le son est rétabli.", f"{e.name} n'est plus en sourdine."]),
        ("media_player.media_previous_track", None,
         ["Chanson précédente{w}", "Reviens à la chanson d'avant{w}",
          "Remets le titre précédent{w}", "Morceau précédent{w}"],
         ["Je suis revenu au titre précédent.", "Titre précédent."]),
        ("media_player.media_stop", None,
         ["Arrête la musique{w}", "Stoppe la lecture{w}", "Coupe la musique{w}"],
         ["La lecture est arrêtée.", "C'est arrêté."]),
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
    ask = R.choice([f"Sur quel appareil dois-je lancer {desc} : "
                    f"{join_fr([e.name for e in opts], 'ou')} ?",
                    f"Où veux-tu écouter {desc} ? Tu as le choix entre "
                    f"{join_fr([e.name for e in opts], 'et')}."])
    msgs = base_msgs(h, loc) + [user(u), say(ask)]
    if not F("answer"):
        return sample(msgs, "clarify", [], no_history=True)
    if R.random() < 0.12:
        msgs += [user(R.choice(["Laisse tomber", "Finalement non", "Peu importe"])),
                 say("D'accord, on laisse tomber.")]
        return sample(msgs, "conversation", [], gold_last=[])
    e = R.choice(ps)
    same_room = [x for x in ps if x.area == e.area]
    replies = [e.name, f"Sur {e.name}", f"Sur {e.name}, s'il te plaît"]
    if e.area and len(same_room) == 1:
        replies += [cap(C.prep(e.area)), e.area]
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
        ans = R.choice([f"Je n'ai malheureusement pas trouvé {q}.",
                        f"Désolé, je n'ai rien trouvé pour « {q} ».",
                        f"Je n'ai rien trouvé pour « {q} ». Tu veux que je mette autre chose ?"])
    else:
        err = (f"media_player.search_and_play failed for {e.eid}: "
               f"music search is not supported by {e.eid}")
        ans = R.choice([f"Je ne peux malheureusement pas chercher de musique sur {e.name}.",
                        f"{e.name} ne prend malheureusement pas en charge la recherche de musique."])
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
    shop = [e for e in lists if e.name == SHOP]
    return shop[0] if shop and R.random() < 0.65 else R.choice(lists)


def i_todo_add(h, loc):
    lst = pick_list(h)
    if not lst:
        return None
    phrase, pool, _ = LISTS[lst.name]
    free = [x for x in pool if x[0] not in lst.attrs["items"]]
    if not free:
        return None
    picked = R.sample(free, min(len(free), R.choice([1, 1, 1, 2, 2, 3])))
    items = [x[0] for x in picked]
    it = join_fr([x[1] for x in picked])
    defs = join_fr([x[2] for x in picked])
    users = [f"Ajoute {it} {a(phrase)}", f"Mets {it} sur {phrase}", f"Note {it} sur {phrase}",
             f"Rajoute {it} {a(phrase)}", f"Inscris {it} sur {phrase}"]
    if lst.name == SHOP:
        users += [f"Il nous faut {it}", f"Je dois acheter {it}", f"Mets {it} sur la liste"
                  if len(h.of("todo")) == 1 else f"Ajoute {it} aux courses"]
    if lst.name == TASKS:
        users += [f"Ajoute {it} à mes tâches", f"Mets {it} dans mes tâches"]
    acts = [action("todo.add_item", [lst], {"item": i}) for i in items]
    lst.attrs["items"] = lst.attrs["items"] + items
    lst.state = str(len(lst.attrs["items"]))
    pl = len(items) > 1 or C.plural(picked[0][3])
    conf = R.choice([f"J'ai ajouté {it} {a(phrase)}.",
                     f"{cap(defs)} {est(pl)} maintenant sur {phrase}.",
                     f"D'accord, {defs} {est(pl)} sur {phrase}."])
    return R.choice(users), acts, conf


def i_todo_complete(h, loc):
    lst = pick_list(h, with_items=True)
    if not lst:
        return None
    phrase = LISTS[lst.name][0]
    item = R.choice(lst.attrs["items"])
    _, part, defi, g = item_info(lst.name, item)
    pron = C.pronoun_obj(g)
    if R.random() < 0.6:
        users = [f"Coche {defi} sur {phrase}", f"Raye {defi} {de(phrase)}",
                 f"Coche {defi} {de(phrase)}"]
        if lst.name == SHOP:
            users += [f"J'ai acheté {defi}, coche-{pron}"]
        if lst.name == TASKS:
            users += [f"C'est fait pour {defi}, coche-{pron} sur {phrase}"]
        svc = "todo.complete_item"
        conf = R.choice([f"J'ai coché {defi}.",
                         f"{cap(defi)} {est(g)} {agr('coché', g)} sur {phrase}."])
    else:
        users = [f"Supprime {defi} {de(phrase)}", f"Enlève {defi} {de(phrase)}",
                 f"Retire {defi} {de(phrase)}"]
        svc = "todo.remove_item"
        conf = R.choice([f"{cap(defi)} {est(g)} {agr('supprimé', g)} {de(phrase)}.",
                         f"J'ai retiré {defi} {de(phrase)}."])
    lst.attrs["items"] = [i for i in lst.attrs["items"] if i != item]
    lst.state = str(len(lst.attrs["items"]))
    return R.choice(users), [action(svc, [lst], {"item": item})], conf


def s_todo_question(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOP)
    phrase, pool, _ = LISTS[lst.name]
    items = lst.attrs["items"]
    if items and R.random() < 0.3:
        item = R.choice(items) if R.random() < 0.5 else R.choice(
            [x[0] for x in pool if x[0] not in items] or items)
        _, part, defi, g = item_info(lst.name, item)
        u = R.choice([f"Est-ce {que(defi)} {est(g)} sur {phrase} ?",
                      f"Il y a {part} sur {phrase} ?" if lst.name != TASKS
                      else f"Est-ce {que(defi)} {est(g)} sur {phrase} ?"])
        yes = item in items
        ans = f"Oui, {defi} {est(g)} sur {phrase}." if yes else \
            f"Non, {defi} {nest(g)} pas sur {phrase}."
        must = [item]
    else:
        users = [f"Qu'est-ce qu'il y a sur {phrase} ?", f"Lis-moi {phrase}",
                 f"Que contient {phrase} ?"]
        if lst.name == SHOP:
            users += ["Qu'est-ce que je dois acheter ?", "Qu'est-ce qu'il nous faut encore ?"]
        if lst.name == TASKS:
            users += ["Qu'est-ce que j'ai à faire aujourd'hui ?", "Qu'est-ce qu'il me reste à faire ?"]
        u = R.choice(users)
        if not items:
            ans = R.choice([f"{cap(phrase)} est vide.", f"Il n'y a rien sur {phrase} en ce moment."])
            must = ["vide" if "vide" in ans else "rien"]
        elif len(items) == 1:
            ans = f"Sur {phrase}, il n'y a que {items[0]}."
            must = items
        else:
            ans = f"Sur {phrase}, il y a : {join_fr(items)}."
            must = items[:3]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [lst.eid]}
    msgs += call_turn("get_state", args, h.state_of([lst]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def s_todo_missing(h, loc):
    lst = pick_list(h)
    if not lst:
        lst = add_list(h, SHOP)
    phrase, pool, _ = LISTS[lst.name]
    missing = [x for x in pool if x[0] not in lst.attrs["items"]]
    item, part, defi, g = R.choice(missing)
    u = R.choice([f"Coche {defi} sur {phrase}", f"Raye {defi} {de(phrase)}",
                  f"Supprime {defi} {de(phrase)}"])
    svc = "todo.remove_item" if u.startswith("Supprime") else "todo.complete_item"
    acts = [action(svc, [lst], {"item": item})]
    err = f"{svc} failed for {lst.eid}: Unable to find to-do list item: {item}"
    ans = R.choice([f"{cap(defi)} {nest(g)} malheureusement pas sur {phrase}.",
                    f"Je n'ai pas trouvé {defi} sur {phrase}."])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("control", {"actions": acts}, S.control_result([err]), ans)
    return sample(msgs, "command", [["control", {"actions": acts}]])


def s_todo_no_list(h, loc):
    """The list does not exist: say so, no tool call, no other list."""
    name = R.choice(list(LISTS))
    for e in [e for e in h.ents if e.domain == "todo" and e.name == name]:
        h.ents.remove(e)
    phrase, pool, _ = LISTS[name]
    item, part, defi, g = R.choice(pool)
    u = R.choice([f"Ajoute {part} {a(phrase)}", f"Qu'est-ce qu'il y a sur {phrase} ?",
                  f"Mets {part} sur {phrase}"])
    others = [LISTS[e.name][0] for e in h.of("todo") if e.name in LISTS]
    ans = f"Je n'ai pas trouvé {phrase}."
    if others:
        ans += f" Il y a seulement {join_fr(others)}."
    else:
        ans = R.choice([ans, f"Je n'ai malheureusement trouvé aucune liste nommée « {name} »."])
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
    temp = fr_num(e.attrs["temperature"])
    tdeg = deg(e.attrs["temperature"])
    kind = F("wkind") or R.choice(["now", "rain", "day", "temp"])
    if kind == "outdoor":
        if any(x.label == C.OUTDOOR_LABEL for x in h.of("sensor")):
            return None
        u = R.choice(["Quelle température fait-il dehors ?", "Il fait froid dehors ?",
                      "Il fait combien dehors ?", "Quelle est la température extérieure ?"])
        ans, must = f"Il fait {tdeg} dehors.", [temp]
    elif kind == "now":
        u = R.choice(["Quel temps fait-il ?", "Il fait quel temps ?",
                      "Quel temps fait-il dehors ?", "C'est quoi la météo ?",
                      "Il fait beau dehors ?", "Quel temps fait-il aujourd'hui ?",
                      "Météo ?", "Que dit la météo ?", "Donne-moi la météo"])
        ans, must = weather_report(e, tdeg, cond_word), [temp]
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
            u = R.choice([f"Est-ce qu'il va pleuvoir {day_q} ?", f"Il pleut {day_q} ?",
                          f"Il me faut un parapluie {day_q} ?"])
            p = f.get("precipitation_probability")
            tail = f", la probabilité de pluie est de {p} %." if p is not None else "."
            if raw["condition"] in RAINY:
                ans = f"Oui, {day} {frag}" + tail
                must = ["Oui"]
            elif raw["condition"] in SNOW:
                # snow or hail: no rain, but precipitation
                ans = f"Non, mais {day} {frag}" + (
                    f", la probabilité de précipitations est de {p} %." if p is not None else ".")
                must = ["Non"]
            else:
                ans = f"Non, {day} {frag}" + tail
                must = ["Non"]
        elif kind == "temp":
            if R.random() < 0.6 or lo is None:
                u = R.choice([f"Il va faire combien {day_q} ?",
                              f"Quelle température fera-t-il {day_q} ?"])
                ans, must = f"{day_cap}, il fera jusqu'à {deg(hi)}.", [str(hi)]
            else:
                u = R.choice([f"Il va faire froid {day_q} ?",
                              f"Quelle sera la température minimale {day_q} ?"])
                ans, must = f"{day_cap}, les températures descendront jusqu'à {deg(lo)}.", [str(lo)]
        else:
            u = R.choice([f"Quel temps fera-t-il {day_q} ?", f"Quelle est la météo pour {day_q} ?",
                          f"Il va faire quel temps {day_q} ?"])
            ans = f"{day_cap}, {frag}"
            ans += f", avec des températures de {lo} à {deg(hi)}." if lo is not None \
                else f", jusqu'à {deg(hi)}."
            p = f.get("precipitation_probability")
            if idx == 0 and C.VERBOSITY != "short" and rest_of_day(e):
                ans = f"{NOW_SENT[e.state]}, avec {tdeg}. " + rest_of_day(e)
                if lo is not None and C.VERBOSITY in ("long", "detailed"):
                    ans += f" Aujourd'hui, les températures iront de {lo} à {deg(hi)}."
            if p is not None and C.VERBOSITY in ("long", "detailed"):
                ans += f" La probabilité de pluie est de {p} %."
            if C.VERBOSITY == "short":
                ans = f"{day_cap}, {frag}, jusqu'à {deg(hi)}."
            must = [str(hi)] if idx else [temp]
    msgs = base_msgs(h, loc) + [user(u)]
    args = {"entity_ids": [e.eid]}
    msgs += call_turn("get_state", args, h.state_of([e]), ans)
    return sample(msgs, "question", [["get_state", args]], answer_contains=must)


def im_moment(sentence):
    """'Il pleut à verse' -> 'En ce moment, il pleut à verse'."""
    return "En ce moment, " + low(sentence)


def hour_word(time):
    return f"{int(time.split(':')[0])} h"


def rest_of_day(e):
    """'À partir de 18 h, il va pleuvoir, avec 12 degrés, et vers 21 h, on
    attend 8 degrés.'"""
    later, raw = e.attrs.get("later_today") or [], e.attrs.get("_later_raw") or []
    if not later:
        return ""
    last = later[-1]
    end = f"vers {hour_word(last['time'])}, on attend {deg(last['temperature'])}"
    change = next((i for i, c in enumerate(raw) if c != e.state
                   and not {c, e.state} <= {"sunny", "clear-night"}), None)
    if change is None:
        return cap(R.choice(["jusqu'au soir", "pour le reste de la journée", "jusqu'à ce soir"])) + \
            f", {STAY[e.state]}, et {end}."
    f = later[change]
    out = f"À partir de {hour_word(f['time'])}, {FORECAST_FRAG[raw[change]]}, avec {deg(f['temperature'])}"
    if change < len(later) - 1:
        out += f", et {end}"
    return out + "."


def weather_report(e, tdeg, cond_word):
    """'Quel temps fait-il ?' by answer length: short - now; medium - now and
    the rest of the day; long - plus today's range and rain chance;
    detailed - plus humidity, wind and tomorrow."""
    a_, fc = e.attrs, e.attrs["forecast"]
    v = C.VERBOSITY
    first = R.choice([f"{NOW_SENT[e.state]}, avec {tdeg}.",
                      f"Dehors, il fait {tdeg} ({cond_word}).",
                      im_moment(NOW_SENT[e.state]) + f", avec {tdeg}."])
    if v == "short":
        return R.choice([f"{cap(cond_word)}, {tdeg}.", first])
    parts = [first]
    rest = rest_of_day(e)
    if v == "medium":
        if rest:
            return " ".join(parts + [rest])
        t = fc[0] if fc else None
        if t and t.get("templow") is not None:
            parts.append(f"Aujourd'hui, les températures iront de {t['templow']} à {deg(t['temperature'])}.")
        return " ".join(parts)
    if v == "detailed":
        parts.append(f"L'humidité est de {a_['humidity']} % et le vent souffle à "
                     f"{fr_num(a_['wind_speed'])} {a_['wind_speed_unit']}.")
    if rest:
        parts.append(rest)
    if fc:
        t = fc[0]
        if t.get("templow") is not None:
            parts.append(R.choice([
                f"Aujourd'hui, les températures iront de {t['templow']} à {deg(t['temperature'])}.",
                f"Aujourd'hui, on attend un maximum de {deg(t['temperature'])} "
                f"et un minimum de {deg(t['templow'])}."]))
        else:
            parts.append(f"Aujourd'hui, il fera jusqu'à {deg(t['temperature'])}.")
        p = t.get("precipitation_probability")
        if p is not None:
            parts.append(f"La probabilité de pluie est de {p} %." if p else
                         "Aucune pluie n'est prévue aujourd'hui.")
    if len(fc) > 1 and v == "detailed":
        raw = e.attrs["_entries"][1]
        parts.append(f"Demain, {FORECAST_FRAG[raw['condition']]}, jusqu'à {deg(fc[1]['temperature'])}.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# time and date
# --------------------------------------------------------------------------
def spoken_time(now):
    """'14 h 05', '9 h'."""
    return f"{now.hour} h {now.minute:02d}" if now.minute else f"{now.hour} h"


def s_time(h, loc):
    now = rand_dt()
    res = S.time_result(now, LANG)
    t = S.lang_texts(LANG)
    wd = t["weekdays"][now.weekday()]
    day = "1er" if now.day == 1 else str(now.day)
    month = t["months"][now.month - 1]
    date = f"{day} {month} {now.year}"
    kind = F("tkind") or R.choice(["time", "date", "weekday"])
    if kind == "time":
        u = R.choice(["Quelle heure est-il ?", "Il est quelle heure ?", "Tu as l'heure ?",
                      "Donne-moi l'heure", "Quelle heure il est ?", "L'heure ?",
                      "Il est quelle heure, là ?"])
        hm = spoken_time(now)
        ans, must = R.choice([f"Il est {hm}.", f"Il est actuellement {hm}."]), [hm]
        if C.VERBOSITY == "short":
            ans = f"{hm}."
        elif C.VERBOSITY == "detailed":
            ans = f"Il est {hm}, nous sommes le {wd} {date}."
    elif kind == "date":
        u = R.choice(["Quelle est la date d'aujourd'hui ?", "On est le combien aujourd'hui ?",
                      "Quelle date sommes-nous ?", "C'est quoi la date ?",
                      "On est le combien ?", "Donne-moi la date",
                      "Quelle est la date ?", "Date ?", "Nous sommes le combien ?"])
        ans, must = f"Aujourd'hui, nous sommes le {wd} {date}.", [f"{day} {month}"]
    else:
        u = R.choice(["Quel jour sommes-nous ?", "On est quel jour ?",
                      "Quel jour on est aujourd'hui ?", "C'est quel jour aujourd'hui ?",
                      "Quel jour de la semaine sommes-nous ?", "On est déjà vendredi ?",
                      "Dis-moi quel jour on est", "C'est quoi aujourd'hui ?",
                      "Quel jour de la semaine on est ?", "On est lundi aujourd'hui ?",
                      "Aujourd'hui, c'est quel jour ?", "C'est le week-end ?"])
        ans, must = R.choice([f"Aujourd'hui, c'est {wd}.", f"Nous sommes le {wd} {date}."]), [wd]
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
        return 0, m, 0, "une minute" if m == 1 else f"{numw(m)} minutes"
    if k < 0.65:
        return R.choice([(0, 30, 0, "une demi-heure"), (0, 15, 0, "un quart d'heure"),
                         (0, 45, 0, "trois quarts d'heure"), (1, 0, 0, "une heure"),
                         (1, 30, 0, "une heure et demie"), (1, 30, 0, "une heure trente"),
                         (2, 0, 0, "deux heures"), (2, 0, 0, "2 heures"),
                         (3, 0, 0, "trois heures")])
    if k < 0.8:
        s = R.choice([10, 15, 20, 30, 40, 45, 90])
        return 0, 0, s, f"{numw(s)} secondes"
    if k < 0.9:
        m, s = R.choice([1, 2, 3, 4, 5, 8]), R.choice([10, 15, 20, 30, 45])
        mw = "une minute" if m == 1 else f"{numw(m)} minutes"
        return 0, m, s, f"{mw} et {s} secondes"
    hh, m = R.choice([1, 2]), R.choice([10, 15, 20, 45])
    hw = "une heure" if hh == 1 else f"{numw(hh)} heures"
    return hh, m, 0, R.choice([f"{hw} et {m} minutes", f"{hw} {m}"])


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
    """'le minuteur des pâtes' / 'le minuteur'."""
    return f"le minuteur {TIMER_DE[name]}" if name else "le minuteur"


def s_timer_start(h, loc):
    hh, m, s, spoken = duration()
    named = F("named", R.random() < 0.35)
    name, pour, _ = R.choice(TIMER_NAMES) if named else ("", "", "")
    if named:
        u = R.choice([f"Mets un minuteur {pour} {de(spoken)}", f"Minuteur {pour}, {spoken}",
                      f"Lance un minuteur {de(spoken)} {pour}",
                      f"Règle un minuteur {pour} sur {spoken}",
                      f"{cap(spoken)} {pour}", f"Mets un minuteur {de(spoken)} {pour}",
                      f"Minuteur {TIMER_DE[name]}, {spoken}"])
    else:
        u = R.choice([f"Mets un minuteur {de(spoken)}", f"Minuteur {spoken}",
                      f"Lance un minuteur {de(spoken)}", f"Réveille-moi dans {spoken}",
                      f"Minuteur {de(spoken)}", f"Tu peux mettre un minuteur {de(spoken)} ?",
                      f"Rappelle-moi dans {spoken}", f"Mets un timer {de(spoken)}",
                      f"Compte à rebours {de(spoken)}"])
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    msgs = base_msgs(h, loc) + [user(u)]
    if F("unsupported"):
        ans = R.choice(["Je ne peux malheureusement pas régler de minuteur sur cet appareil. "
                        "Il faut passer par un appareil vocal comme Home Assistant Voice.",
                        "Les minuteurs ne fonctionnent malheureusement que sur un appareil vocal, pas ici.",
                        "Ici, je ne peux pas mettre de minuteur, seulement sur un appareil vocal."])
        msgs += call_turn("timer", args, S.timer_result("timers are not supported on this device"),
                          ans)
        return sample(msgs, "timer", [["timer", args]])
    if name:
        t = timer_label(name)
        ans = R.choice([f"{cap(t)} est lancé pour {dur}.", f"{cap(t)} est réglé sur {dur}.",
                        f"D'accord, {dur} {pour}."])
    else:
        ans = R.choice([f"Le minuteur est lancé pour {dur}.", f"J'ai mis un minuteur {de(dur)}.",
                        f"Minuteur {de(dur)} lancé.", f"D'accord, {dur}. Le minuteur tourne."])
    msgs += call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


DELAY_INTENTS = ["i_light_onoff", "i_light_room_phrase", "i_switch", "i_cover", "i_climate",
                 "i_media", "i_fan_pct", "i_vacuum", "i_scene"]


def s_timer_delayed(h, loc):
    """'Éteins la lumière dans 10 minutes' -> timer with the command; the
    command runs through the agent when the timer ends."""
    fn = getattr(C, R.choice(DELAY_INTENTS))
    out = fn(h, loc)
    if not out:
        return None
    cmd = out[0].replace(", s'il te plaît", "").replace(" stp", "").rstrip(" .!?")
    hh, m, s, spoken = duration()
    if s and not m and not hh and s < 30:
        return None
    words = cmd.split(" ", 1)
    if words[0] not in VERBS or len(words) < 2 or "?" in cmd:
        return None       # "Tu peux ..." does not fit behind "Dans 10 minutes"
    sp = spoken
    forms = [f"Dans {sp}, {lower_first(cmd)}", f"{cmd} dans {sp}",
             f"Mets un minuteur {de(spoken)} et ensuite {lower_first(cmd)}",
             f"{words[0]} dans {sp} {words[1]}", f"{cmd} dans {sp}"]
    u = R.choice(forms)
    args = {"action": "start", **dur_args(hh, m, s), "command": cmd}
    dur = S.fmt_duration(hh, m, s, LANG)
    ans = R.choice([f"D'accord, je le ferai dans {dur}.", f"Très bien, je m'en occupe dans {dur}.",
                    f"C'est noté, ce sera fait dans {dur}."])
    msgs = base_msgs(h, loc) + [user(u)] + call_turn("timer", args, S.timer_result(), ans)
    return sample(msgs, "timer", [["timer", args]])


def status_answer(entries, asked_name=None):
    if not entries:
        if asked_name:
            return f"Aucun minuteur {TIMER_DE[asked_name]} n'est en cours.", ["aucun"]
        return R.choice(["Aucun minuteur n'est en cours.",
                         "Il n'y a aucun minuteur en cours."]), ["aucun"]
    if len(entries) == 1:
        e = entries[0]
        lab = timer_label(e["name"])
        if e.get("paused"):
            return f"{cap(lab)} est en pause, il reste encore {e['remaining']}.", [e["remaining"]]
        return R.choice([f"Il reste {e['remaining']} sur {lab}.",
                         f"Encore {e['remaining']}."]), [e["remaining"]]
    parts = [f"{e['name'] or 'Un minuteur'} : encore {e['remaining']}"
             + (", en pause" if e.get("paused") else "") for e in entries]
    return f"Il y a {len(entries)} minuteurs en cours. " + ". ".join(parts) + ".", \
        [e["remaining"] for e in entries]


def s_timer_status(h, loc):
    n = R.choice([0, 1, 1, 1, 2, 2, 3])
    timers = running_timers(n, named=n > 1 or R.random() < 0.4)
    named = [t for t in timers if t[0]]
    args = {"action": "status"}
    asked = None
    if named and R.random() < 0.35:
        asked = R.choice(named)[0] if R.random() < 0.8 else R.choice(TIMER_NAMES)[0]
        u = R.choice([f"Combien de temps reste-t-il sur {timer_label(asked)} ?",
                      f"Il reste combien de temps {TIMER_POUR[asked]} ?",
                      f"Où en est {timer_label(asked)} ?"])
        args["name"] = asked
        timers = [t for t in timers if t[0] == asked]
    else:
        u = R.choice(["Combien de temps reste-t-il sur le minuteur ?", "Il reste combien de temps ?",
                      "Combien de temps encore ?", "Quels minuteurs sont en cours ?",
                      "Il y a encore un minuteur en cours ?", "Le minuteur en est où ?",
                      "État du minuteur"])
    entries = status_entries(timers)
    ans, must = status_answer(entries, asked)
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(timers=entries), ans)
    return sample(msgs, "question", [["timer", args]], answer_contains=must)


# {t} timer with article, {T} capitalised, {at} "au minuteur ...",
# {dt} "du minuteur ...", {d} spoken duration, {dd} "de" + spoken duration,
# {D} formatted duration, {dD} "de" + formatted duration
TIMER_OPS = {
    "cancel": (["Annule {t}", "Arrête {t}", "Supprime {t}", "Stoppe {t}",
                "Coupe {t}, s'il te plaît", "Tu peux annuler {t} ?"],
               ["{T} est annulé.", "J'ai supprimé {t}.", "D'accord, {t} est annulé."]),
    "pause": (["Mets {t} en pause", "Pause sur {t}", "Suspends {t}", "Arrête {t} un instant"],
              ["{T} est en pause.", "J'ai mis {t} en pause."]),
    "resume": (["Relance {t}", "Reprends {t}", "Fais repartir {t}", "Redémarre {t}"],
               ["{T} repart.", "D'accord, {t} tourne à nouveau."]),
    "add": (["Ajoute {d} {at}", "Prolonge {t} {dd}", "Rajoute {d} {at}",
             "Mets {d} de plus sur {t}"],
            ["J'ai ajouté {D} {at}.", "{T} tourne maintenant {D} de plus."]),
    "subtract": (["Enlève {d} {at}", "Raccourcis {t} {dd}", "Retire {d} {dt}"],
                 ["{T} est raccourci {dD}.", "J'ai retiré {D} {dt}."]),
}


def fmt_op(text, t, d="", dur=""):
    return text.format(t=t, T=cap(t), at=a(t), dt=de(t), d=d, dd=de(d) if d else "",
                       D=dur, dD=de(dur) if dur else "")


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
    u = fmt_op(R.choice(users), t, spoken)
    args = {"action": op}
    if op in ("add", "subtract"):
        args.update(dur_args(hh, m, s))
    if nm:
        args["name"] = nm
    if r < 0.6:
        ans = fmt_op(R.choice(answers), t, dur=S.fmt_duration(hh, m, s, LANG))
    elif r < 0.8:
        ans = R.choice(["Aucun minuteur n'est en cours.",
                        "Il n'y a malheureusement aucun minuteur en cours."])
    else:
        others = [timer_label(x[0]) for x in timers]
        ans = f"Il n'y a pas de minuteur {TIMER_DE[nm]}. " + (
            f"Seul {others[0]} est en cours." if len(others) == 1 else
            f"Seuls {join_fr(others)} sont en cours.")
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
    u = fmt_op(R.choice(users), "le minuteur")
    args = {"action": op}
    names = [t[0] for t in timers]
    ask = R.choice([f"Quel minuteur veux-tu dire : "
                    f"{join_fr(['celui ' + TIMER_DE[n] for n in names], 'ou')} ?",
                    f"Plusieurs minuteurs sont en cours : {join_fr([timer_label(n) for n in names])}. "
                    f"Lequel veux-tu dire ?"])
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result("several timers match, name one", entries), ask)
    if R.random() < 0.3:
        return sample(msgs, "timer", [["timer", args]], no_history=True)
    pick = R.choice(names)
    if op == "cancel" and R.random() < 0.2:
        msgs += [user(R.choice(["Tous", "Tous les minuteurs", "Les deux" if len(names) == 2 else "Tous"]))]
        a2 = {"action": "cancel_all"}
        msgs += call_turn("timer", a2, S.timer_result(canceled=len(names)),
                          f"J'ai annulé les {len(names)} minuteurs.")
        return sample(msgs, "conversation", [], gold_last=[["timer", a2]])
    msgs += [user(R.choice([pick, f"Celui {TIMER_DE[pick]}", f"Le minuteur {TIMER_DE[pick]}",
                            f"Le minuteur {TIMER_POUR[pick]}"]))]
    a2 = {"action": op, "name": pick}
    msgs += call_turn("timer", a2, S.timer_result(),
                      fmt_op(R.choice(answers), timer_label(pick)))
    return sample(msgs, "conversation", [], gold_last=[["timer", a2]])


def s_timer_cancel_all(h, loc):
    n = R.choice([0, 1, 2, 2, 3])
    u = R.choice(["Supprime tous les minuteurs", "Annule tous les minuteurs",
                  "Arrête tous les minuteurs", "Annule tous les timers",
                  "Coupe tous les minuteurs", "Stoppe tous les minuteurs",
                  "Efface tous les minuteurs", "Tu peux annuler tous les minuteurs ?",
                  "Arrête tous les minuteurs, s'il te plaît", "Supprime l'ensemble des minuteurs",
                  "Plus aucun minuteur", "Annule-moi tous les minuteurs", "Tous les minuteurs, stop"])
    args = {"action": "cancel_all"}
    ans = {0: "Aucun minuteur n'était en cours.", 1: "J'ai annulé le minuteur."}.get(
        n, f"J'ai annulé les {n} minuteurs.")
    msgs = base_msgs(h, loc) + [user(u)]
    msgs += call_turn("timer", args, S.timer_result(canceled=n), ans)
    return sample(msgs, "timer", [["timer", args]])


def s_alarm_clock(h, loc):
    """Alarms at a clock time are no Assist feature: say so, offer a timer."""
    hh, mm = R.choice([6, 7, 7, 8, 9]), R.choice([0, 0, 15, 30, 45])
    t = f"{hh} h {mm:02d}" if mm else f"{hh} h"
    u = R.choice([f"Réveille-moi à {t}", f"Mets un réveil à {t}",
                  f"Règle le réveil pour demain matin à {t}", f"Réveil à {t}"])
    ans = R.choice(["Je ne peux malheureusement pas régler de réveil à une heure précise. "
                    "Par contre, je peux te mettre un minuteur.",
                    "Les réveils à une heure fixe ne sont pas possibles, seulement les minuteurs. "
                    "Tu veux que je mette un minuteur ?"])
    msgs = base_msgs(h, loc) + [user(u), say(ans)]
    return sample(msgs, "unknown", [])


# --------------------------------------------------------------------------
# broadcast and small talk
# --------------------------------------------------------------------------
def s_broadcast(h, loc):
    sub, main = R.choice(BROADCASTS)
    u = R.choice([f"Fais une annonce : {main}", f"Dis à tout le monde {que(sub)}",
                  f"Annonce dans toute la maison {que(sub)}", f"Annonce : {main}",
                  f"Diffuse dans la maison : {main}", f"Préviens tout le monde {que(sub)}",
                  f"Fais savoir à tout le monde {que(sub)}", f"Fais une annonce pour dire {que(sub)}"])
    args = {"message": main}
    if F("error"):
        res = S.dumps({"success": False, "error": "no other voice devices for a broadcast"})
        ans = R.choice(["Il n'y a malheureusement aucun autre appareil vocal pour faire une annonce.",
                        "Je ne peux pas faire d'annonce, il n'y a pas d'autre appareil vocal."])
    else:
        res = S.dumps({"success": True})
        ans = R.choice(["J'ai fait l'annonce.", "L'annonce est partie.",
                        "C'est fait, je l'ai annoncé dans toute la maison."])
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
        users = [f"Règle {ref.acc} sur {v} %", f"{cap(ref.acc)} à {v} pour cent",
                 f"Mets {ref.acc} à {v} % d'humidité"]
        if e.area and len(h.of("humidifier", e.area)) == 1:
            users.append(f"Règle l'humidité {C.droom(e.area)} sur {v} %")
        return R.choice(users), [action("humidifier.set_humidity", [e], {"humidity": v})], \
            f"{cap(ref.nom)} {est(ref.g)} {agr('réglé', ref.g)} sur {v} %."
    on = r < 0.8
    users = [f"{'Allume' if on else 'Éteins'} {ref.acc}",
             f"{'Mets en marche' if on else 'Coupe'} {ref.acc}"]
    return R.choice(users), [action(f"humidifier.turn_{'on' if on else 'off'}", [e])], \
        C.confirm(ref, "allumé" if on else "éteint", "allumé" if on else "éteint")


def i_valve_position(h, loc):
    ents = h.of("valve")
    if not ents:
        return None
    e = R.choice(ents)
    p, word = R.choice([(50, "à moitié"), (25, "à 25 %"), (30, "à 30 %"),
                        (75, "à 75 %"), (50, "à 50 pour cent")])
    u = R.choice([f"Ouvre {e.name} {word}", f"Règle {e.name} {word}" if word != "à moitié"
                  else f"Ouvre {e.name} à moitié", f"Mets {e.name} {word}"])
    return u, [action("valve.set_valve_position", [e], {"position": p})], \
        f"{e.name} {est(e.gender)} {agr('ouvert', e.gender)} à {p} %."


C.COMMANDS += [(i_music_play, 6), (i_media_extra, 4), (i_todo_add, 4), (i_todo_complete, 2),
               (i_humidifier, 1), (i_valve_position, 1)]


# --------------------------------------------------------------------------
# conversation turns (each brings its own first step, so it can be forced)
# --------------------------------------------------------------------------
def conv_tool(c, name, args, result, answer):
    c.msgs += call_turn(name, args, result, answer)
    c.gold_last = [[name, args]]


def turn_music_follow(c):
    """'Mets Queen dans la cuisine' -> 'Plus fort' / 'Chanson suivante': same player."""
    out = i_music_play(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    c.user(u)
    c.control(acts, conf)
    e = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    svc, users, ans = R.choice([
        ("media_player.volume_up", ["Plus fort", "Monte le son", "Un peu plus fort, s'il te plaît"],
         "Volume augmenté."),
        ("media_player.volume_down", ["Moins fort", "Baisse le son", "Un peu moins fort"],
         "Volume baissé."),
        ("media_player.media_next_track", ["Chanson suivante", "Suivant", "Passe à la suivante"],
         "Titre suivant."),
        ("media_player.media_pause", ["Pause", "Mets en pause", "Arrête un instant"], "En pause."),
        ("media_player.turn_off", ["Éteins la musique", "Coupe tout"], "La musique est coupée."),
        ("media_player.shuffle_set", ["Et en aléatoire, s'il te plaît"],
         "La lecture aléatoire est activée."),
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
    name, pour, _ = R.choice(TIMER_NAMES) if named else ("", "", "")
    c.user(f"Mets un minuteur {pour + ' ' if named else ''}{de(spoken)}")
    args = {"action": "start", **dur_args(hh, m, s)}
    if name:
        args["name"] = name
    dur = S.fmt_duration(hh, m, s, LANG)
    conv_tool(c, "timer", args, S.timer_result(), f"{cap(timer_label(name))} est lancé pour {dur}.")
    if R.random() < 0.5:
        total = hh * 3600 + m * 60 + s - R.randint(10, max(11, (hh * 3600 + m * 60 + s) // 2))
        mm, ss = divmod(max(total, 5), 60)
        h2, mm = divmod(mm, 60)
        entries = status_entries([(name, *round_time(h2, mm, ss), True)])
        c.user(R.choice(["Il reste combien de temps ?", "Combien de temps encore ?",
                         "Et il reste combien ?"]))
        ans, _ = status_answer(entries)
        conv_tool(c, "timer", {"action": "status"}, S.timer_result(timers=entries), ans)
    else:
        c.user(R.choice(["Annule-le", "Finalement non, supprime le minuteur", "Arrête-le"]))
        conv_tool(c, "timer", {"action": "cancel"}, S.timer_result(),
                  f"{cap(timer_label(name))} est annulé.")
    return True


def turn_list_follow(c):
    """'Ajoute du lait à la liste de courses' -> 'et des œufs'."""
    out = i_todo_add(c.h, c.loc)
    if not out:
        return False
    u, acts, conf = out
    lst = next(x for x in c.h.ents if x.eid == acts[0]["entity_id"][0])
    phrase, pool, _ = LISTS[lst.name]
    free = [x for x in pool if x[0] not in lst.attrs["items"]]
    if not free:
        return False
    c.user(u)
    c.control(acts, conf)
    item, part, defi, g = R.choice(free)
    c.user(R.choice([f"Et {part}", f"{cap(part)} aussi", f"Et aussi {part}"]))
    a2 = [action("todo.add_item", [lst], {"item": item})]
    lst.attrs["items"] = lst.attrs["items"] + [item]
    c.control(a2, R.choice([f"{cap(defi)} {est(g)} aussi sur {phrase} maintenant.",
                            f"J'ai aussi ajouté {part}."]))
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
