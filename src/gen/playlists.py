"""Realistic playlist libraries for the music_library samples of all languages.

A real media library (Spotify via Music Assistant, Plex, Jellyfin) holds
dozens of playlists whose names say little: built-in mixes, artist radios,
"This Is ..." playlists, soundtracks, festival sets, personal names with
emojis. The model has to find the one playlist that fits a request among
them, so the samples mix the fitting playlist into long lists of such names,
generated freshly for every sample instead of drawn from a small fixed pool.

Language-neutral parts live here; every language passes its own personal
templates, first names and places (`local`).

Tags: a generated name that could fit a mood carries that mood as tag, and
samples for this mood leave it out, so exactly one playlist fits.
"""

from __future__ import annotations

# built-in playlists of Music Assistant and streaming services
BUILTIN = ["Random tracks (from library)", "250 Random tracks (from library)",
           "Infinite Mix (library)", "Random Album (from library)",
           "Random Artist (from library)", "Recently played tracks", "Daily Mix 1",
           "Daily Mix 2", "Daily Mix 4", "On Repeat", "Repeat Rewind", "Time Capsule",
           "Your Top Songs 2024", "Your Top Songs 2025", "Top Songs 2023", "Release Mix",
           "Mix of the Week", "New Music Mix", "Replay 2025", "My Mix 3"]

# genre -> moods it could fit
GENRES = {
    "Techno": ("party", "sport"), "Hardstyle": ("party", "sport"), "Trance": ("party", "sport"),
    "Melodic Techno": ("party",), "Drum & Bass": ("party", "sport"), "Hip-Hop": ("sport",),
    "Deutschrap": ("sport",), "Indie": (), "Indie Pop": (), "Pop": (), "K-Pop": (),
    "Jazz": ("relax", "cook", "focus"), "Soul": ("cook",), "Funk": ("party",),
    "Reggae": ("relax",), "Latin": ("party",), "Country": ("car",), "Ambient": ("relax", "sleep", "focus"),
    "Piano": ("relax", "sleep", "focus"), "Lofi": ("relax", "focus", "sleep"),
    "Metalcore": ("rock", "sport"), "Power Metal": ("rock", "sport"), "Punk": ("rock",),
    "Grunge": ("rock",), "Synthwave": ("eighties", "focus"), "Dubstep": ("party", "sport"),
    "Psytrance": ("party",), "Minimal": ("focus",), "Acoustic": ("relax",),
    "Singer-Songwriter": ("relax",), "Blues": (), "Afrobeats": ("party",), "Bossa Nova": ("cook", "relax"),
}
YEARS = ["2014", "2016", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
SOUNDTRACKS = ["Interstellar Soundtrack", "The Witcher 3 OST", "Mario Kart Soundtrack",
               "GTA Vice City Radio", "Stranger Things Soundtrack", "Gladiator OST",
               "Lord of the Rings Soundtrack", "Skyrim OST", "Zelda Soundtrack",
               "Inception Soundtrack", "Pirates of the Caribbean OST", "Halo Soundtrack",
               "Cyberpunk 2077 Radio", "Top Gun Soundtrack", "Dune Soundtrack",
               "The Last of Us OST", "Fast & Furious Soundtrack", "Guardians of the Galaxy Mix"]
FESTIVALS = ["Parookaville", "Rock am Ring", "Wacken", "Fusion", "Nature One", "Mayday",
             "Lollapalooza", "Glastonbury", "Primavera", "Sziget", "Hurricane", "Melt"]
EMOJI = ["🔥", "✨", "🎧", "🌴", "⚡", "🖤", "🎸", "🌊", "🚀", "😎", "🪩", "🌙"]
# emojis that already say something about a mood
EMOJI_TAGS = {"🪩": ("party",), "🌙": ("sleep", "relax"), "🌴": ("relax",), "🎸": ("rock",)}


# moods whose playlists could also be meant: a request for one leaves the
# playlists of the related moods out ("Ruhige Abende" would do for sleep)
RELATED = {"relax": ("sleep", "cook", "focus"), "sleep": ("relax",), "cook": ("relax",),
           "focus": ("relax",), "party": ("eighties", "kids", "sport"), "eighties": ("party",),
           "kids": ("party",), "sport": ("party", "rock"), "rock": ("sport",), "car": (),
           "xmas": (), "fav": ()}


def mood_tags(mood):
    return (mood,) + RELATED.get(mood, ())


def _artist_radio(rng, artists, tracks):
    a, remixer = rng.sample(artists, 2)
    t, ta = rng.choice(tracks)
    return rng.choice([f"{a} Radio", f"This Is {a}", f"Best of {a}", f"{a} Mix",
                       f"{t} Radio", f"{t} ({remixer} Remix) Radio", f"{a} - Live",
                       f"Spotalike: {ta} - {t}", f"{a} Essentials", f"{a} Complete"]), ()


def _genre(rng, artists, tracks):
    g = rng.choice(list(GENRES))
    y = rng.choice(YEARS)
    name = rng.choice([f"{g} {y}", f"Best of {g}", f"{g} Classics", f"{g} Essentials",
                       f"New {g}", f"{g.upper()} BUNKER", f"{g} Hits {y}", f"{g} & Chill"
                       if not GENRES[g] else f"{g} Session", f"{g} Deep Cuts", f"{g} Mix {y}"])
    tags = GENRES[g] + (("relax",) if "Chill" in name else ())
    return name, tags


def _soundtrack(rng, artists, tracks):
    return rng.choice(SOUNDTRACKS), ()


def _festival(rng, artists, tracks):
    f = rng.choice(FESTIVALS)
    y = rng.choice(YEARS)
    return rng.choice([f"{f} {y}", f"{f} {y} Setlist", f"Warm-up {f} {y}", f"{f} Lineup {y}"]), \
        ("party",)


def _builtin(rng, artists, tracks):
    return rng.choice(BUILTIN), ()


def _personal(rng, local):
    names, places = local["names"], local["places"]
    a, b = rng.sample(names, 2)
    t = rng.choice(local["personal"])
    return t.format(name=a, name2=b, place=rng.choice(places), year=rng.choice(YEARS),
                    n=rng.randint(1, 14)), ()


def decorate(rng, name):
    """Real names carry emojis, capitals and years now and then."""
    r = rng.random()
    if r < 0.12:
        e = rng.choice(EMOJI)
        return f"{name} {e}", EMOJI_TAGS.get(e, ())
    if r < 0.17:
        return name.upper(), ()
    if r < 0.21:
        return f"{name}!!!", ()
    return name, ()


def noise(rng, k, artists, tracks, local, avoid_tags=(), avoid=()):
    """k distinct playlist names that fit none of avoid_tags."""
    makers = [(_artist_radio, 4), (_genre, 4), (_soundtrack, 1), (_festival, 1),
              (_builtin, 2), (None, 3)]
    out: dict[str, None] = {}
    bad = {a.casefold() for a in avoid}
    tries = 0
    while len(out) < k and tries < k * 20:
        tries += 1
        maker = rng.choices([m for m, _ in makers], [w for _, w in makers])[0]
        name, tags = maker(rng, artists, tracks) if maker else _personal(rng, local)
        name, etags = decorate(rng, name)
        if set(tags + etags) & set(avoid_tags) or name.casefold() in bad:
            continue
        bad.add(name.casefold())
        out[name] = None
    return list(out)


def library_size(rng):
    """Number of playlists: small libraries as well as long real ones."""
    return rng.choice([rng.randint(2, 8), rng.randint(8, 25), rng.randint(25, 50),
                       rng.randint(50, 80), rng.randint(70, 100)])


def spoken(name):
    """How a user says a playlist name: without emojis and '!!!', capitals
    as in normal text."""
    words = [w for w in name.replace("!!!", "").split() if any(c.isalnum() for c in w)]
    text = " ".join(words)
    return text.title() if text.isupper() and len(text) > 4 else text
