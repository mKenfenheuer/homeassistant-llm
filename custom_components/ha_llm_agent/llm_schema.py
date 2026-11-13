"""Prompt, tool schema and state format shared by training and runtime.

This module is the single source of truth. The training pipeline (src/)
imports it, and the Home Assistant integration uses it at runtime. It has no
Home Assistant imports so it can run in both environments.

Every language has its own model. Service names, tool names and JSON keys are
the same for all languages; only the texts in LANGUAGES differ (system prompt,
tool descriptions, weekdays, weather conditions, durations). A new language
needs an entry in LANGUAGES and its own dataset generator (src/gen/<lang>/).

Any change to the system prompt, TOOLS or format_state() changes what the
model sees and requires a new training run.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Iterable

SCHEMA_VERSION = 7
DEFAULT_LANGUAGE = "de"
# languages with a released model; the integration offers only these. The
# other entries of LANGUAGES are ready for training (src/gen/<lang>/).
MODEL_LANGUAGES = ["de"]

# --- services the model may call, with the data keys it may pass ------------
# Services marked "virtual" are not Home Assistant services; the integration
# maps them (see conversation.py):
#   media_player.search_and_play: a title from music_library, otherwise
#                                 media_player.search_media; without query the
#                                 loaded title goes on, else the favorites
#                                 playlist of the library (FAVORITE_WORDS), else
#                                 a random playlist
#   media_player.seek_relative:   media_seek to the current position + seconds
#                                 (negative: back)
#   todo.complete_item:           todo.update_item with status completed
SERVICES: dict[str, list[str]] = {
    "light.turn_on": ["brightness_pct", "brightness_step_pct", "color_name",
                      "color_temp_kelvin"],
    "light.turn_off": [],
    "switch.turn_on": [],
    "switch.turn_off": [],
    "input_boolean.turn_on": [],
    "input_boolean.turn_off": [],
    "fan.turn_on": ["percentage"],
    "fan.turn_off": [],
    "fan.set_percentage": ["percentage"],
    "cover.open_cover": [],
    "cover.close_cover": [],
    "cover.stop_cover": [],
    "cover.set_cover_position": ["position"],
    "climate.set_temperature": ["temperature"],
    "climate.turn_on": [],
    "climate.turn_off": [],
    "humidifier.turn_on": [],
    "humidifier.turn_off": [],
    "humidifier.set_humidity": ["humidity"],
    "lock.lock": [],
    "lock.unlock": [],
    "alarm_control_panel.alarm_arm_home": ["code"],
    "alarm_control_panel.alarm_arm_away": ["code"],
    "alarm_control_panel.alarm_arm_night": ["code"],
    "alarm_control_panel.alarm_disarm": ["code"],
    "media_player.turn_on": [],
    "media_player.turn_off": [],
    "media_player.media_play": [],
    "media_player.media_pause": [],
    "media_player.media_stop": [],
    "media_player.media_next_track": [],
    "media_player.media_previous_track": [],
    "media_player.volume_set": ["volume_level"],
    "media_player.volume_up": [],
    "media_player.volume_down": [],
    "media_player.volume_mute": ["is_volume_muted"],
    "media_player.shuffle_set": ["shuffle"],
    "media_player.repeat_set": ["repeat"],
    "media_player.search_and_play": ["query", "media_type", "artist"],   # virtual
    "media_player.seek_relative": ["seconds"],                           # virtual
    "todo.add_item": ["item"],
    "todo.complete_item": ["item"],                                      # virtual
    "todo.remove_item": ["item"],
    "vacuum.start": [],
    "vacuum.pause": [],
    "vacuum.return_to_base": [],
    "scene.turn_on": [],
    "script.turn_on": [],
    "button.press": [],
    "lawn_mower.start_mowing": [],
    "lawn_mower.pause": [],
    "lawn_mower.dock": [],
    "valve.open_valve": [],
    "valve.close_valve": [],
    "valve.set_valve_position": ["position"],
}
# media_type values of media_player.search_and_play (Music Assistant names)
MEDIA_TYPES = ["track", "album", "artist", "playlist", "radio", "podcast", "audiobook"]
REPEAT_MODES = ["off", "one", "all"]
# media commands whose success is heard: the answer is not spoken, whatever
# the answer length ("Lauter", "Nächster Titel", "Pause", "Play", "Vorspulen",
# "Spiel Queen"); a failure is always spoken
QUIET_SERVICES = ["media_player.search_and_play",
                  "media_player.volume_up", "media_player.volume_down",
                  "media_player.media_next_track", "media_player.media_previous_track",
                  "media_player.media_play", "media_player.media_pause",
                  "media_player.media_stop", "media_player.seek_relative"]
LIBRARY_ITEMS = 100        # entries of a music_library result passed to the model
# names of a favorites playlist in a media library ("Liked Songs Max")
FAVORITE_WORDS = ["liked songs", "lieblingssongs", "lieblingslieder", "lieblingsmusik",
                  "meine favoriten", "my favorites", "favorite songs", "favoritos",
                  "canciones que me gustan", "titres likés", "favoris", "preferiti",
                  "brani che ti piacciono", "favorieten", "leuke nummers", "favoriten",
                  "favorites", "favourites", "favs", "all favorited tracks"]
# a query that names no title ("Spiel Musik" as query "Musik") plays the
# favorites like an empty query
GENERIC_QUERIES = ["musik", "irgendwas", "irgendeine musik", "etwas musik", "was",
                   "music", "something", "some music", "anything",
                   "música", "algo", "algo de música", "musique", "de la musique",
                   "quelque chose", "musica", "qualcosa", "un po' di musica",
                   "muziek", "iets", "wat muziek"]
# a favorites word inside these is no favorites playlist ("Radio-Favoriten")
FAVORITE_SKIP = ["radio", "mix"]


def favorites_playlist(names: Iterable[str]) -> str | None:
    """The favorites playlist among playlist names, by FAVORITE_WORDS order."""
    names = [n for n in names if n and not any(w in n.casefold() for w in FAVORITE_SKIP)]
    for word in FAVORITE_WORDS:
        for n in names:
            if word in n.casefold():
                return n
    return None


# media_type of music_library -> media_class of media_player.browse_media
LIBRARY_TYPES = ["playlist", "artist", "album", "track", "radio", "podcast"]
LIBRARY_CLASSES = {"playlist": ["playlist"], "artist": ["artist"], "album": ["album"],
                   "track": ["track", "music"], "radio": ["channel", "music"],
                   "podcast": ["podcast"]}

# domains that appear in the device list (controllable + read-only)
CONTROL_DOMAINS = sorted({s.split(".", 1)[0] for s in SERVICES})
READ_DOMAINS = ["sensor", "binary_sensor", "weather"]
EXPOSED_DOMAINS = CONTROL_DOMAINS + READ_DOMAINS
# domains that can be addressed by area ("das Licht in der Küche", "alle
# Lichter"). Covers are left out: "die Rollläden auf der Terrasse" must not
# move the awning in the same area.
AREA_DOMAINS = ["light"]
ALL_AREAS = "*"

TIMER_ACTIONS = ["start", "cancel", "cancel_all", "pause", "resume", "add",
                 "subtract", "status"]
WEATHER_CONDITIONS = ["clear-night", "cloudy", "exceptional", "fog", "hail",
                      "lightning", "lightning-rainy", "partlycloudy", "pouring",
                      "rainy", "snowy", "snowy-rainy", "sunny", "windy",
                      "windy-variant"]
FORECAST_DAYS = 4          # forecast entries passed to the model
HOURLY_STEP = 3            # hours between the entries of "later_today"
HOURLY_MAX = 5
# answer length, set in the integration options; the system prompt carries
# the matching sentence of LANGUAGES[lang]["verbosity"]
VERBOSITY = ["short", "medium", "long", "detailed"]
DEFAULT_VERBOSITY = "medium"
TODO_ITEMS = 25            # open to-do items passed to the model

# --- language-specific texts ------------------------------------------------
LANGUAGES: dict[str, dict[str, Any]] = {
    "de": {
        "name": "Deutsch",
        "system": (
            "Du bist der Sprachassistent eines Smart Homes mit Home Assistant. "
            "Antworte immer auf Deutsch und freundlich. {verbosity}\n"
            "Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte "
            "mit get_state ab, bevor du Fragen dazu beantwortest.\n"
            "Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät "
            "nicht, sage das, statt eine entity_id zu erfinden.\n"
            "Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den "
            "Bereich an (area), für das ganze Haus area \"*\".\n"
            "Timer stellst du mit timer, Uhrzeit und Datum liefert get_time, "
            "Durchsagen an alle Lautsprecher machst du mit broadcast.\n\n"
            "Geräte (entity_id | Name | Bereich):\n{devices}"
        ),
        "location": "\n\nDas Sprachgerät steht im Bereich: {area}",
        "satellite_player": "\nDer Lautsprecher des Sprachgeräts: {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Sprachbefehle für das Smart Home: Musik pausieren, Pause, Play, weiter, zurück, lauter, leiser, nächster Titel, vorspulen, zurückspulen, Licht an, Licht aus, Rollläden hoch, Timer, Wetter.",
        "verbosity": {
            "short": "Antworte so knapp wie möglich, mit wenigen Worten.",
            "medium": "Antworte kurz, meist in einem Satz.",
            "long": "Antworte in ganzen Sätzen und nenne die wichtigsten Details.",
            "detailed": "Antworte ausführlich und nenne alle verfügbaren Details.",
        },
        "verbosity_labels": {"short": "kurz", "medium": "mittel", "long": "lang",
                             "detailed": "ausführlich"},
        "default_system": ("Du bist der Sprachassistent eines Smart Homes mit "
                           "Home Assistant. Antworte immer auf Deutsch, kurz und freundlich."),
        "tool_control": (
            "Steuert ein oder mehrere Geräte. Jede Aktion ruft einen Home-Assistant-Dienst für eine oder mehrere entity_ids auf, bei Lichtern auch für alle Lichter eines Bereichs (area). Musik sucht und spielt media_player.search_and_play (query, media_type, artist; ohne query weiter mit dem geladenen Titel, sonst die Lieblingssongs), Spulen media_player.seek_relative (seconds, negativ zurück), Listen ändert todo.add_item, todo.complete_item und todo.remove_item (item). Erlaubte Dienste: {services}."
        ),
        "tool_control_service": "Dienst im Format domain.dienst",
        "tool_control_area": ("Statt entity_id: alle Lichter dieses Bereichs, "
                              "\"*\" für das ganze Haus"),
        "tool_control_data": (
            "Optionale Parameter, z. B. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Liefert den aktuellen Zustand und wichtige Attribute von Geräten "
            "und Sensoren, entweder für entity_ids oder für alle Geräte einer "
            "domain in einem Bereich (domain und area, \"*\" für das ganze Haus, "
            "optional device_class, z. B. window). Wetter-Entitäten liefern den "
            "Verlauf des restlichen Tages (later_today) und die Vorhersage der "
            "nächsten Tage (forecast), Listen (todo) ihre offenen Einträge."
        ),
        "tool_timer": (
            "Timer auf dem Sprachgerät: starten (start mit hours, minutes, "
            "seconds, optional name und command für einen Befehl, der nach "
            "Ablauf ausgeführt wird), abbrechen (cancel, cancel_all), "
            "pausieren (pause), fortsetzen (resume), verlängern (add), "
            "verkürzen (subtract) und abfragen (status). name wählt einen "
            "bestimmten Timer."
        ),
        "tool_get_time": "Liefert die aktuelle Uhrzeit, den Wochentag und das Datum.",
        "tool_broadcast": "Spricht eine Durchsage auf allen Sprachgeräten im Haus.",
        "tool_music_library": (
            'Liest die Medienbibliothek eines Players (entity_id): Playlists, Künstler, Alben, Titel, Sender oder Podcasts (media_type), optional gefiltert nach search. Liefert die Namen, die media_player.search_and_play als query abspielt.'
        ),
        "weekdays": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag",
                     "Samstag", "Sonntag"],
        "months": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
                   "August", "September", "Oktober", "November", "Dezember"],
        "date_format": "{day}. {month} {year}",
        "today": "heute",
        "tomorrow": "morgen",
        "conditions": {
            "clear-night": "klar", "cloudy": "bewölkt", "exceptional": "Unwetter",
            "fog": "Nebel", "hail": "Hagel", "lightning": "Gewitter",
            "lightning-rainy": "Gewitter mit Regen", "partlycloudy": "teilweise bewölkt",
            "pouring": "starker Regen", "rainy": "Regen", "snowy": "Schnee",
            "snowy-rainy": "Schneeregen", "sunny": "sonnig", "windy": "windig",
            "windy-variant": "windig und bewölkt",
        },
        # (singular, plural) per unit
        "units": {"hours": ("Stunde", "Stunden"), "minutes": ("Minute", "Minuten"),
                  "seconds": ("Sekunde", "Sekunden")},
        "and": "und",
        "warmup_text": "Hallo",
        "error_timeout": "Der Sprachassistent hat nicht rechtzeitig geantwortet.",
        "error_unreachable": "Der Sprachassistent ist gerade nicht erreichbar.",
        "error_incomplete": "Das konnte ich leider nicht vollständig ausführen.",
        "error_area_domain": "area wird für {domain} nicht unterstützt",
        "error_area_unknown": "unbekannter Bereich {area}",
        "error_entity_unknown": "unbekannte entity_id",
    },
    "en": {
        "name": "English",
        "system": (
            "You are the voice assistant of a smart home with Home Assistant. "
            "Always answer in English and in a friendly way. {verbosity}\n"
            "Control devices with the tool control. Read states and measurements "
            "with get_state before you answer questions about them.\n"
            "Only use entity_ids from the following list. If a device does not "
            "exist, say so instead of making up an entity_id.\n"
            "If the user means all lights of an area, pass the area (area) instead "
            "of entity_id, area \"*\" for the whole house.\n"
            "Set timers with timer, get_time returns the time and date, make "
            "announcements on all speakers with broadcast.\n\n"
            "Devices (entity_id | name | area):\n{devices}"
        ),
        "location": "\n\nThe voice device is in the area: {area}",
        "satellite_player": "\nThe speaker of the voice device: {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Smart home voice commands: pause the music, pause, play, next track, previous track, louder, quieter, fast forward, rewind, lights on, lights off, open the blinds, timer, weather.",
        "verbosity": {
            "short": "Answer as briefly as possible, in a few words.",
            "medium": "Answer briefly, usually in one sentence.",
            "long": "Answer in full sentences and give the most important details.",
            "detailed": "Answer in detail and give all available details.",
        },
        "verbosity_labels": {"short": "short", "medium": "medium", "long": "long",
                             "detailed": "detailed"},
        "default_system": ("You are the voice assistant of a smart home with Home "
                           "Assistant. Always answer in English, briefly and in a friendly way."),
        "tool_control": (
            "Controls one or more devices. Every action calls a Home Assistant service for one or more entity_ids, for lights also for all lights of an area (area). media_player.search_and_play (query, media_type, artist; without query the loaded title goes on, otherwise the favorite songs), seeking media_player.seek_relative (seconds, negative back) searches and plays music, todo.add_item, todo.complete_item and todo.remove_item (item) change lists. Allowed services: {services}."
        ),
        "tool_control_service": "Service in the format domain.service",
        "tool_control_area": "Instead of entity_id: all lights of this area, \"*\" for the whole house",
        "tool_control_data": (
            "Optional parameters, e.g. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Returns the current state and important attributes of devices and "
            "sensors, either for entity_ids or for all devices of a domain in an "
            "area (domain and area, \"*\" for the whole house, optionally "
            "device_class, e.g. window). Weather entities return the course of the rest of "
            "the day (later_today) and the forecast of the next days (forecast), "
            "lists (todo) their open items."
        ),
        "tool_timer": (
            "Timers on the voice device: start (start with hours, minutes, seconds, "
            "optionally name and command for a command that runs when the timer "
            "ends), cancel (cancel, cancel_all), pause (pause), resume (resume), "
            "extend (add), shorten (subtract) and query (status). name selects a "
            "specific timer."
        ),
        "tool_get_time": "Returns the current time, the weekday and the date.",
        "tool_broadcast": "Speaks an announcement on all voice devices in the house.",
        "tool_music_library": (
            'Reads the music library of a player (entity_id): playlists, artists, albums, tracks, stations or podcasts (media_type), optionally filtered by search. Returns the names that media_player.search_and_play plays as query.'
        ),
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"],
        "months": ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"],
        "date_format": "{month} {day}, {year}",
        "today": "today",
        "tomorrow": "tomorrow",
        "conditions": {
            "clear-night": "clear", "cloudy": "cloudy", "exceptional": "severe weather",
            "fog": "fog", "hail": "hail", "lightning": "thunderstorm",
            "lightning-rainy": "thunderstorm with rain", "partlycloudy": "partly cloudy",
            "pouring": "heavy rain", "rainy": "rain", "snowy": "snow",
            "snowy-rainy": "sleet", "sunny": "sunny", "windy": "windy",
            "windy-variant": "windy and cloudy",
        },
        "units": {"hours": ("hour", "hours"), "minutes": ("minute", "minutes"),
                  "seconds": ("second", "seconds")},
        "and": "and",
        "warmup_text": "Hello",
        "error_timeout": "The voice assistant did not answer in time.",
        "error_unreachable": "The voice assistant is not reachable right now.",
        "error_incomplete": "Sorry, I could not complete that.",
        "error_area_domain": "area is not supported for {domain}",
        "error_area_unknown": "unknown area {area}",
        "error_entity_unknown": "unknown entity_id",
    },
    "es": {
        "name": "Español",
        "system": (
            "Eres el asistente de voz de un hogar inteligente con Home Assistant. "
            "Responde siempre en español y de forma amable. {verbosity}\n"
            "Controla los dispositivos con la herramienta control. Consulta estados "
            "y mediciones con get_state antes de responder preguntas sobre ellos.\n"
            "Usa solo entity_ids de la siguiente lista. Si un dispositivo no existe, "
            "dilo en lugar de inventar un entity_id.\n"
            "Si el usuario se refiere a todas las luces de una zona, indica la zona "
            "(area) en lugar de entity_id, area \"*\" para toda la casa.\n"
            "Los temporizadores se ponen con timer, get_time da la hora y la fecha, "
            "los avisos en todos los altavoces se hacen con broadcast.\n\n"
            "Dispositivos (entity_id | nombre | zona):\n{devices}"
        ),
        "location": "\n\nEl dispositivo de voz está en la zona: {area}",
        "satellite_player": "\nEl altavoz del dispositivo de voz: {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Órdenes de voz para la casa: pausa la música, pausa, play, siguiente canción, anterior, más alto, más bajo, adelanta, retrocede, enciende la luz, apaga la luz, sube las persianas, temporizador, tiempo.",
        "verbosity": {
            "short": "Responde lo más breve posible, con pocas palabras.",
            "medium": "Responde de forma breve, normalmente en una frase.",
            "long": "Responde con frases completas y da los detalles más importantes.",
            "detailed": "Responde de forma detallada y da todos los detalles disponibles.",
        },
        "verbosity_labels": {"short": "corta", "medium": "media", "long": "larga",
                             "detailed": "detallada"},
        "default_system": ("Eres el asistente de voz de un hogar inteligente con Home "
                           "Assistant. Responde siempre en español, de forma breve y amable."),
        "tool_control": (
            "Controla uno o varios dispositivos. Cada acción llama a un servicio de Home Assistant para uno o varios entity_ids, para luces también para todas las luces de una zona (area). media_player.search_and_play (query, media_type, artist; sin query sigue el título cargado, si no las canciones favoritas), avanzar o retroceder media_player.seek_relative (seconds, negativo atrás) busca y reproduce música, todo.add_item, todo.complete_item y todo.remove_item (item) cambian listas. Servicios permitidos: {services}."
        ),
        "tool_control_service": "Servicio en el formato dominio.servicio",
        "tool_control_area": "En lugar de entity_id: todas las luces de esta zona, \"*\" para toda la casa",
        "tool_control_data": (
            "Parámetros opcionales, p. ej. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Devuelve el estado actual y los atributos importantes de dispositivos y "
            "sensores, para entity_ids o para todos los dispositivos de un domain en "
            "una zona (domain y area, \"*\" para toda la casa, opcionalmente "
            "device_class, p. ej. window). Las entidades del tiempo devuelven la evolución del resto "
            "del día (later_today) y la previsión de los próximos días (forecast), "
            "las listas (todo) sus elementos pendientes."
        ),
        "tool_timer": (
            "Temporizadores del dispositivo de voz: iniciar (start con hours, minutes, "
            "seconds, opcionalmente name y command para una orden que se ejecuta al "
            "terminar), cancelar (cancel, cancel_all), pausar (pause), reanudar "
            "(resume), alargar (add), acortar (subtract) y consultar (status). name "
            "elige un temporizador concreto."
        ),
        "tool_get_time": "Devuelve la hora actual, el día de la semana y la fecha.",
        "tool_broadcast": "Dice un aviso en todos los dispositivos de voz de la casa.",
        "tool_music_library": (
            'Lee la biblioteca de música de un reproductor (entity_id): listas, artistas, álbumes, canciones, emisoras o podcasts (media_type), opcionalmente filtrada por search. Devuelve los nombres que media_player.search_and_play reproduce como query.'
        ),
        "weekdays": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado",
                     "domingo"],
        "months": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                   "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        "date_format": "{day} de {month} de {year}",
        "today": "hoy",
        "tomorrow": "mañana",
        "conditions": {
            "clear-night": "despejado", "cloudy": "nublado", "exceptional": "temporal",
            "fog": "niebla", "hail": "granizo", "lightning": "tormenta",
            "lightning-rainy": "tormenta con lluvia", "partlycloudy": "parcialmente nublado",
            "pouring": "lluvia fuerte", "rainy": "lluvia", "snowy": "nieve",
            "snowy-rainy": "aguanieve", "sunny": "soleado", "windy": "viento",
            "windy-variant": "viento y nubes",
        },
        "units": {"hours": ("hora", "horas"), "minutes": ("minuto", "minutos"),
                  "seconds": ("segundo", "segundos")},
        "and": "y",
        "warmup_text": "Hola",
        "error_timeout": "El asistente de voz no ha respondido a tiempo.",
        "error_unreachable": "El asistente de voz no está disponible ahora mismo.",
        "error_incomplete": "Lo siento, no he podido hacerlo del todo.",
        "error_area_domain": "area no se admite para {domain}",
        "error_area_unknown": "zona desconocida {area}",
        "error_entity_unknown": "entity_id desconocido",
    },
    "fr": {
        "name": "Français",
        "system": (
            "Tu es l'assistant vocal d'une maison connectée avec Home Assistant. "
            "Réponds toujours en français et de manière aimable. {verbosity}\n"
            "Commande les appareils avec l'outil control. Consulte les états et les "
            "mesures avec get_state avant de répondre aux questions à leur sujet.\n"
            "N'utilise que des entity_ids de la liste suivante. Si un appareil "
            "n'existe pas, dis-le au lieu d'inventer un entity_id.\n"
            "Si l'utilisateur parle de toutes les lumières d'une pièce, indique la "
            "pièce (area) au lieu de entity_id, area \"*\" pour toute la maison.\n"
            "Les minuteurs se règlent avec timer, get_time donne l'heure et la date, "
            "les annonces sur tous les haut-parleurs se font avec broadcast.\n\n"
            "Appareils (entity_id | nom | pièce) :\n{devices}"
        ),
        "location": "\n\nL'appareil vocal se trouve dans la pièce : {area}",
        "satellite_player": "\nLe haut-parleur de l'appareil vocal : {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Commandes vocales de la maison : mets la musique en pause, pause, play, titre suivant, titre précédent, plus fort, moins fort, avance, recule, allume la lumière, éteins la lumière, ouvre les volets, minuteur, météo.",
        "verbosity": {
            "short": "Réponds le plus brièvement possible, en quelques mots.",
            "medium": "Réponds brièvement, le plus souvent en une phrase.",
            "long": "Réponds par des phrases complètes et donne les détails les plus importants.",
            "detailed": "Réponds de manière détaillée et donne tous les détails disponibles.",
        },
        "verbosity_labels": {"short": "courte", "medium": "moyenne", "long": "longue",
                             "detailed": "détaillée"},
        "default_system": ("Tu es l'assistant vocal d'une maison connectée avec Home "
                           "Assistant. Réponds toujours en français, brièvement et aimablement."),
        "tool_control": (
            "Commande un ou plusieurs appareils. Chaque action appelle un service de Home Assistant pour un ou plusieurs entity_ids, pour les lumières aussi pour toutes les lumières d'une pièce (area). media_player.search_and_play (query, media_type, artist ; sans query le titre chargé reprend, sinon les titres favoris), avancer ou reculer media_player.seek_relative (seconds, négatif en arrière) cherche et joue de la musique, todo.add_item, todo.complete_item et todo.remove_item (item) modifient les listes. Services autorisés : {services}."
        ),
        "tool_control_service": "Service au format domaine.service",
        "tool_control_area": "Au lieu de entity_id : toutes les lumières de cette pièce, \"*\" pour toute la maison",
        "tool_control_data": (
            "Paramètres facultatifs, p. ex. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Renvoie l'état actuel et les attributs importants des appareils et "
            "capteurs, soit pour des entity_ids, soit pour tous les appareils d'un "
            "domain dans une pièce (domain et area, \"*\" pour toute la maison, "
            "éventuellement device_class, p. ex. window). Les entités météo renvoient l'évolution du "
            "reste de la journée (later_today) et les prévisions des prochains jours "
            "(forecast), les listes (todo) leurs éléments ouverts."
        ),
        "tool_timer": (
            "Minuteurs de l'appareil vocal : démarrer (start avec hours, minutes, "
            "seconds, en option name et command pour une commande exécutée à la "
            "fin), annuler (cancel, cancel_all), mettre en pause (pause), reprendre "
            "(resume), prolonger (add), raccourcir (subtract) et interroger (status). "
            "name choisit un minuteur précis."
        ),
        "tool_get_time": "Renvoie l'heure actuelle, le jour de la semaine et la date.",
        "tool_broadcast": "Diffuse une annonce sur tous les appareils vocaux de la maison.",
        "tool_music_library": (
            "Lit la bibliothèque musicale d'un lecteur (entity_id) : playlists, artistes, albums, titres, stations ou podcasts (media_type), filtrée en option par search. Renvoie les noms que media_player.search_and_play joue comme query."
        ),
        "weekdays": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
                     "dimanche"],
        "months": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                   "août", "septembre", "octobre", "novembre", "décembre"],
        "date_format": "{day} {month} {year}",
        "today": "aujourd'hui",
        "tomorrow": "demain",
        "conditions": {
            "clear-night": "ciel dégagé", "cloudy": "nuageux", "exceptional": "intempéries",
            "fog": "brouillard", "hail": "grêle", "lightning": "orage",
            "lightning-rainy": "orage avec pluie", "partlycloudy": "partiellement nuageux",
            "pouring": "forte pluie", "rainy": "pluie", "snowy": "neige",
            "snowy-rainy": "neige fondue", "sunny": "ensoleillé", "windy": "venteux",
            "windy-variant": "venteux et nuageux",
        },
        "units": {"hours": ("heure", "heures"), "minutes": ("minute", "minutes"),
                  "seconds": ("seconde", "secondes")},
        "and": "et",
        "warmup_text": "Bonjour",
        "error_timeout": "L'assistant vocal n'a pas répondu à temps.",
        "error_unreachable": "L'assistant vocal n'est pas joignable pour le moment.",
        "error_incomplete": "Désolé, je n'ai pas pu le faire entièrement.",
        "error_area_domain": "area n'est pas pris en charge pour {domain}",
        "error_area_unknown": "pièce inconnue {area}",
        "error_entity_unknown": "entity_id inconnu",
    },
    "it": {
        "name": "Italiano",
        "system": (
            "Sei l'assistente vocale di una casa intelligente con Home Assistant. "
            "Rispondi sempre in italiano e in modo cortese. {verbosity}\n"
            "Controlla i dispositivi con lo strumento control. Leggi stati e misure "
            "con get_state prima di rispondere a domande su di essi.\n"
            "Usa solo entity_id della lista seguente. Se un dispositivo non esiste, "
            "dillo invece di inventare un entity_id.\n"
            "Se l'utente intende tutte le luci di una zona, indica la zona (area) al "
            "posto di entity_id, area \"*\" per tutta la casa.\n"
            "I timer si impostano con timer, get_time fornisce ora e data, gli "
            "annunci su tutti gli altoparlanti si fanno con broadcast.\n\n"
            "Dispositivi (entity_id | nome | zona):\n{devices}"
        ),
        "location": "\n\nIl dispositivo vocale si trova nella zona: {area}",
        "satellite_player": "\nL'altoparlante del dispositivo vocale: {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Comandi vocali per la casa: metti in pausa la musica, pausa, play, brano successivo, brano precedente, più forte, più piano, vai avanti, torna indietro, accendi la luce, spegni la luce, alza le tapparelle, timer, meteo.",
        "verbosity": {
            "short": "Rispondi nel modo più breve possibile, con poche parole.",
            "medium": "Rispondi in breve, di solito con una frase.",
            "long": "Rispondi con frasi complete e indica i dettagli più importanti.",
            "detailed": "Rispondi in modo dettagliato e indica tutti i dettagli disponibili.",
        },
        "verbosity_labels": {"short": "breve", "medium": "media", "long": "lunga",
                             "detailed": "dettagliata"},
        "default_system": ("Sei l'assistente vocale di una casa intelligente con Home "
                           "Assistant. Rispondi sempre in italiano, in breve e in modo cortese."),
        "tool_control": (
            "Controlla uno o più dispositivi. Ogni azione chiama un servizio di Home Assistant per uno o più entity_id, per le luci anche per tutte le luci di una zona (area). media_player.search_and_play (query, media_type, artist; senza query riprende il brano caricato, altrimenti i brani preferiti), avanti o indietro media_player.seek_relative (seconds, negativo indietro) cerca e riproduce musica, todo.add_item, todo.complete_item e todo.remove_item (item) modificano le liste. Servizi consentiti: {services}."
        ),
        "tool_control_service": "Servizio nel formato dominio.servizio",
        "tool_control_area": "Al posto di entity_id: tutte le luci di questa zona, \"*\" per tutta la casa",
        "tool_control_data": (
            "Parametri facoltativi, ad es. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Restituisce lo stato attuale e gli attributi importanti di dispositivi e "
            "sensori, per entity_id o per tutti i dispositivi di un domain in una "
            "zona (domain e area, \"*\" per tutta la casa, facoltativamente "
            "device_class, ad es. window). Le entità meteo restituiscono l'andamento del resto della "
            "giornata (later_today) e le previsioni dei prossimi giorni (forecast), "
            "le liste (todo) le voci aperte."
        ),
        "tool_timer": (
            "Timer del dispositivo vocale: avviare (start con hours, minutes, seconds, "
            "facoltativi name e command per un comando eseguito alla scadenza), "
            "annullare (cancel, cancel_all), mettere in pausa (pause), riprendere "
            "(resume), allungare (add), accorciare (subtract) e chiedere lo stato "
            "(status). name sceglie un timer preciso."
        ),
        "tool_get_time": "Restituisce l'ora attuale, il giorno della settimana e la data.",
        "tool_broadcast": "Pronuncia un annuncio su tutti i dispositivi vocali della casa.",
        "tool_music_library": (
            'Legge la libreria musicale di un lettore (entity_id): playlist, artisti, album, brani, stazioni o podcast (media_type), filtrata facoltativamente con search. Restituisce i nomi che media_player.search_and_play riproduce come query.'
        ),
        "weekdays": ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato",
                     "domenica"],
        "months": ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                   "agosto", "settembre", "ottobre", "novembre", "dicembre"],
        "date_format": "{day} {month} {year}",
        "today": "oggi",
        "tomorrow": "domani",
        "conditions": {
            "clear-night": "sereno", "cloudy": "nuvoloso", "exceptional": "maltempo",
            "fog": "nebbia", "hail": "grandine", "lightning": "temporale",
            "lightning-rainy": "temporale con pioggia", "partlycloudy": "parzialmente nuvoloso",
            "pouring": "pioggia forte", "rainy": "pioggia", "snowy": "neve",
            "snowy-rainy": "nevischio", "sunny": "soleggiato", "windy": "ventoso",
            "windy-variant": "ventoso e nuvoloso",
        },
        "units": {"hours": ("ora", "ore"), "minutes": ("minuto", "minuti"),
                  "seconds": ("secondo", "secondi")},
        "and": "e",
        "warmup_text": "Ciao",
        "error_timeout": "L'assistente vocale non ha risposto in tempo.",
        "error_unreachable": "L'assistente vocale non è raggiungibile in questo momento.",
        "error_incomplete": "Mi dispiace, non sono riuscito a farlo del tutto.",
        "error_area_domain": "area non è supportato per {domain}",
        "error_area_unknown": "zona sconosciuta {area}",
        "error_entity_unknown": "entity_id sconosciuto",
    },
    "nl": {
        "name": "Nederlands",
        "system": (
            "Je bent de spraakassistent van een slim huis met Home Assistant. "
            "Antwoord altijd in het Nederlands en vriendelijk. {verbosity}\n"
            "Bedien apparaten met de tool control. Vraag toestanden en meetwaarden op "
            "met get_state voordat je vragen erover beantwoordt.\n"
            "Gebruik alleen entity_ids uit de volgende lijst. Bestaat een apparaat "
            "niet, zeg dat dan in plaats van een entity_id te verzinnen.\n"
            "Bedoelt de gebruiker alle lampen van een ruimte, geef dan in plaats van "
            "entity_id de ruimte op (area), voor het hele huis area \"*\".\n"
            "Timers zet je met timer, get_time geeft de tijd en de datum, "
            "omroepberichten op alle speakers doe je met broadcast.\n\n"
            "Apparaten (entity_id | naam | ruimte):\n{devices}"
        ),
        "location": "\n\nHet spraakapparaat staat in de ruimte: {area}",
        "satellite_player": "\nDe luidspreker van het spraakapparaat: {entity_id}",
        # vocabulary hint for speech-to-text (whisper prompt)
        "stt_prompt": "Spraakopdrachten voor het slimme huis: pauzeer de muziek, pauze, play, volgende nummer, vorige nummer, harder, zachter, vooruitspoelen, terugspoelen, licht aan, licht uit, rolluiken omhoog, timer, weer.",
        "verbosity": {
            "short": "Antwoord zo kort mogelijk, met een paar woorden.",
            "medium": "Antwoord kort, meestal in één zin.",
            "long": "Antwoord in hele zinnen en noem de belangrijkste details.",
            "detailed": "Antwoord uitgebreid en noem alle beschikbare details.",
        },
        "verbosity_labels": {"short": "kort", "medium": "gemiddeld", "long": "lang",
                             "detailed": "uitgebreid"},
        "default_system": ("Je bent de spraakassistent van een slim huis met Home "
                           "Assistant. Antwoord altijd in het Nederlands, kort en vriendelijk."),
        "tool_control": (
            "Bedient een of meer apparaten. Elke actie roept een Home Assistant-dienst aan voor een of meer entity_ids, bij lampen ook voor alle lampen van een ruimte (area). media_player.search_and_play (query, media_type, artist; zonder query gaat het geladen nummer verder, anders de favoriete nummers), spoelen media_player.seek_relative (seconds, negatief terug) zoekt en speelt muziek, todo.add_item, todo.complete_item en todo.remove_item (item) wijzigen lijsten. Toegestane diensten: {services}."
        ),
        "tool_control_service": "Dienst in het formaat domein.dienst",
        "tool_control_area": "In plaats van entity_id: alle lampen van deze ruimte, \"*\" voor het hele huis",
        "tool_control_data": (
            "Optionele parameters, bijv. brightness_pct, brightness_step_pct, "
            "color_name, color_temp_kelvin, position, temperature, humidity, "
            "percentage, volume_level, is_volume_muted, shuffle, repeat "
            "(off, one, all), code, query, media_type ({media_types}), artist, seconds, item"
        ),
        "tool_get_state": (
            "Geeft de huidige toestand en belangrijke attributen van apparaten en "
            "sensoren, voor entity_ids of voor alle apparaten van een domain in een "
            "ruimte (domain en area, \"*\" voor het hele huis, optioneel "
            "device_class, bijv. window). Weer-entiteiten geven het verloop van de rest van de dag "
            "(later_today) en de verwachting voor de komende dagen (forecast), lijsten "
            "(todo) hun open items."
        ),
        "tool_timer": (
            "Timers op het spraakapparaat: starten (start met hours, minutes, seconds, "
            "optioneel name en command voor een opdracht die na afloop wordt "
            "uitgevoerd), annuleren (cancel, cancel_all), pauzeren (pause), hervatten "
            "(resume), verlengen (add), verkorten (subtract) en opvragen (status). "
            "name kiest een bepaalde timer."
        ),
        "tool_get_time": "Geeft de huidige tijd, de weekdag en de datum.",
        "tool_broadcast": "Spreekt een omroepbericht uit op alle spraakapparaten in huis.",
        "tool_music_library": (
            'Leest de muziekbibliotheek van een speler (entity_id): afspeellijsten, artiesten, albums, nummers, zenders of podcasts (media_type), optioneel gefilterd op search. Geeft de namen die media_player.search_and_play als query afspeelt.'
        ),
        "weekdays": ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag",
                     "zaterdag", "zondag"],
        "months": ["januari", "februari", "maart", "april", "mei", "juni", "juli",
                   "augustus", "september", "oktober", "november", "december"],
        "date_format": "{day} {month} {year}",
        "today": "vandaag",
        "tomorrow": "morgen",
        "conditions": {
            "clear-night": "helder", "cloudy": "bewolkt", "exceptional": "noodweer",
            "fog": "mist", "hail": "hagel", "lightning": "onweer",
            "lightning-rainy": "onweer met regen", "partlycloudy": "half bewolkt",
            "pouring": "zware regen", "rainy": "regen", "snowy": "sneeuw",
            "snowy-rainy": "natte sneeuw", "sunny": "zonnig", "windy": "winderig",
            "windy-variant": "winderig en bewolkt",
        },
        "units": {"hours": ("uur", "uur"), "minutes": ("minuut", "minuten"),
                  "seconds": ("seconde", "seconden")},
        "and": "en",
        "warmup_text": "Hallo",
        "error_timeout": "De spraakassistent heeft niet op tijd geantwoord.",
        "error_unreachable": "De spraakassistent is nu niet bereikbaar.",
        "error_incomplete": "Sorry, dat kon ik niet helemaal uitvoeren.",
        "error_area_domain": "area wordt niet ondersteund voor {domain}",
        "error_area_unknown": "onbekende ruimte {area}",
        "error_entity_unknown": "onbekende entity_id",
    },
}


def pick_language(code: str | None) -> str:
    """Released model language for a code such as "de-CH", else the default."""
    code = (code or "").split("-")[0].lower()
    return code if code in MODEL_LANGUAGES else DEFAULT_LANGUAGE


def lang_texts(lang: str | None) -> dict[str, Any]:
    return LANGUAGES.get(lang or DEFAULT_LANGUAGE) or LANGUAGES[DEFAULT_LANGUAGE]


def build_tools(lang: str = DEFAULT_LANGUAGE) -> list[dict[str, Any]]:
    t = lang_texts(lang)
    duration = {"type": "integer", "minimum": 0}
    return [
        {
            "type": "function",
            "function": {
                "name": "control",
                "description": t["tool_control"].format(services=", ".join(SERVICES)),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "service": {"type": "string",
                                                "description": t["tool_control_service"]},
                                    "entity_id": {"type": "array",
                                                  "items": {"type": "string"}},
                                    "area": {"type": "string",
                                             "description": t["tool_control_area"]},
                                    "data": {"type": "object",
                                             "description": t["tool_control_data"].format(
                                                 media_types=", ".join(MEDIA_TYPES))},
                                },
                                "required": ["service"],
                            },
                        }
                    },
                    "required": ["actions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_state",
                "description": t["tool_get_state"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": {"type": "array", "items": {"type": "string"}},
                        "domain": {"type": "string"},
                        "area": {"type": "string"},
                        "device_class": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "music_library",
                "description": t["tool_music_library"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "media_type": {"type": "string", "enum": LIBRARY_TYPES},
                        "search": {"type": "string"},
                    },
                    "required": ["entity_id", "media_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timer",
                "description": t["tool_timer"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": TIMER_ACTIONS},
                        "hours": duration, "minutes": duration, "seconds": duration,
                        "name": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": t["tool_get_time"],
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "broadcast",
                "description": t["tool_broadcast"],
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        },
    ]


TOOLS: list[dict[str, Any]] = build_tools(DEFAULT_LANGUAGE)
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
# kept for scripts that only use the default language
SYSTEM_TEMPLATE = LANGUAGES[DEFAULT_LANGUAGE]["system"]
LOCATION_TEMPLATE = LANGUAGES[DEFAULT_LANGUAGE]["location"]

# attributes passed back to the model by get_state
STATE_ATTRS = [
    "device_class", "unit_of_measurement", "current_temperature", "temperature",
    "hvac_action", "current_position", "percentage", "current_humidity", "humidity",
    "wind_speed", "wind_speed_unit", "volume_level", "is_volume_muted", "shuffle", "repeat",
    "media_title", "media_artist", "items", "later_today", "forecast",
]


def device_line(entity_id: str, name: str, area: str | None) -> str:
    return f"{entity_id} | {name} | {area or '-'}"


def sort_key(entity: tuple[str, str, str | None]) -> tuple[str, str]:
    """Entities are listed grouped by area, then by entity_id."""
    return (entity[2] or "~", entity[0])


def build_system_prompt(entities: Iterable[tuple[str, str, str | None]],
                        location: str | None = None,
                        lang: str = DEFAULT_LANGUAGE,
                        verbosity: str = DEFAULT_VERBOSITY,
                        player: str | None = None) -> str:
    """entities: (entity_id, name, area_name) tuples. player: the media
    player of the voice device ("Spiel Musik" plays there first)."""
    t = lang_texts(lang)
    lines = "\n".join(device_line(*e) for e in sorted(entities, key=sort_key))
    style = t["verbosity"].get(verbosity) or t["verbosity"][DEFAULT_VERBOSITY]
    prompt = t["system"].format(devices=lines, verbosity=style)
    if location:
        prompt += t["location"].format(area=location)
    if player:
        prompt += ("\n" if not location else "") + t["satellite_player"].format(entity_id=player)
    return prompt


def format_state(entity_id: str, name: str, state: str,
                 attrs: dict[str, Any], lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """One entry of a get_state result. Weather conditions are translated;
    `forecast` must already be built with format_forecast()."""
    domain = entity_id.split(".", 1)[0]
    if domain == "weather":
        state = lang_texts(lang)["conditions"].get(state, state)
    out: dict[str, Any] = {"entity_id": entity_id, "name": name, "state": state}
    if attrs.get("brightness") is not None and state == "on":
        out["brightness_pct"] = round(attrs["brightness"] / 255 * 100)
    for key in STATE_ATTRS:
        val = attrs.get(key)
        if val is None or val == "":
            continue
        if domain != "media_player" and key in ("is_volume_muted", "shuffle", "repeat"):
            continue
        if isinstance(val, float):
            val = round(val, 2)
        out[key] = val
    return out


def format_forecast(entries: Iterable[dict[str, Any]], today: date,
                    lang: str = DEFAULT_LANGUAGE) -> list[dict[str, Any]]:
    """Daily forecast entries of weather.get_forecasts -> model format: the
    day as "heute", "morgen" or weekday, the condition translated."""
    t = lang_texts(lang)
    out = []
    for f in entries:
        try:
            day = datetime.fromisoformat(str(f["datetime"])).date()
        except (KeyError, ValueError):
            continue
        delta = (day - today).days
        if delta < 0:
            continue
        label = t["today"] if delta == 0 else t["tomorrow"] if delta == 1 \
            else t["weekdays"][day.weekday()]
        item: dict[str, Any] = {"day": label}
        if f.get("condition"):
            item["condition"] = t["conditions"].get(f["condition"], f["condition"])
        for key in ("temperature", "templow", "precipitation_probability"):
            if f.get(key) is not None:
                v = f[key]
                item[key] = round(v) if isinstance(v, float) else v
        out.append(item)
        if len(out) >= FORECAST_DAYS:
            break
    return out


def format_hourly(entries: Iterable[dict[str, Any]], now: datetime,
                  lang: str = DEFAULT_LANGUAGE) -> list[dict[str, Any]]:
    """Hourly forecast of weather.get_forecasts -> the rest of today in
    HOURLY_STEP steps: [{"time": "15:00", "condition", "temperature",
    "precipitation_probability"}]. `now` must be local time with tzinfo or
    naive local time like the entries after conversion."""
    t = lang_texts(lang)
    out: list[dict[str, Any]] = []
    last_hour = None
    for f in entries:
        try:
            when = datetime.fromisoformat(str(f["datetime"]))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is not None and now.tzinfo is not None:
            when = when.astimezone(now.tzinfo)
        elif when.tzinfo is not None:
            when = when.replace(tzinfo=None)
        if when <= now or when.date() != now.date():
            continue
        if last_hour is not None and when.hour - last_hour < HOURLY_STEP:
            continue
        last_hour = when.hour
        item: dict[str, Any] = {"time": when.strftime("%H:00")}
        if f.get("condition"):
            item["condition"] = t["conditions"].get(f["condition"], f["condition"])
        for key in ("temperature", "precipitation_probability"):
            if f.get(key) is not None:
                v = f[key]
                item[key] = round(v) if isinstance(v, float) else v
        out.append(item)
        if len(out) >= HOURLY_MAX:
            break
    return out


def time_result(now: datetime, lang: str = DEFAULT_LANGUAGE) -> str:
    """Result of get_time."""
    t = lang_texts(lang)
    return dumps({
        "time": now.strftime("%H:%M"),
        "weekday": t["weekdays"][now.weekday()],
        "date": t["date_format"].format(day=now.day, month=t["months"][now.month - 1],
                                        year=now.year),
    })


def fmt_duration(hours: int = 0, minutes: int = 0, seconds: int = 0,
                 lang: str = DEFAULT_LANGUAGE) -> str:
    """'1 Stunde und 30 Minuten'."""
    t = lang_texts(lang)
    parts = []
    for key, val in (("hours", hours), ("minutes", minutes), ("seconds", seconds)):
        if val:
            sing, plur = t["units"][key]
            parts.append(f"{val} {sing if val == 1 else plur}")
    if not parts:
        sing, plur = t["units"]["seconds"]
        return f"0 {plur}"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" {t['and']} " + parts[-1]


def timer_status_entry(name: str, h: int, m: int, s: int, active: bool,
                       lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """One running timer in the result of timer status (rounded time left)."""
    out: dict[str, Any] = {"name": name or "", "remaining": fmt_duration(h, m, s, lang)}
    if not active:
        out["paused"] = True
    return out


def timer_result(error: str | None = None, timers: list[dict] | None = None,
                 **extra: Any) -> str:
    """Result of the timer tool. `timers` (status, or the candidates when a
    name was ambiguous) lists timer_status_entry() dicts."""
    out: dict[str, Any] = {"success": error is None}
    if error:
        out["error"] = error
    if timers is not None:
        out["timers"] = timers
    out.update(extra)
    return dumps(out)


def library_result(names: list[str] | None = None, error: str | None = None) -> str:
    """Result of music_library: {"items": [names], "total": n}. At most
    LIBRARY_ITEMS names; "total" counts all matches."""
    if error:
        return dumps({"success": False, "error": error})
    names = [n for n in (names or []) if n]
    return dumps({"items": names[:LIBRARY_ITEMS], "total": len(names)})


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def control_result(errors: list[str] | None = None,
                   done: list[str] | None = None) -> str:
    """Result of a control call. `done` lists the entity_ids that were
    switched: on failure the successful part, on success the entities an
    area resolved to (callers pass it on success only for area actions)."""
    if errors:
        out: dict[str, Any] = {"success": False, "error": "; ".join(errors)}
        if done:
            out["done"] = done
        return dumps(out)
    if done:
        return dumps({"success": True, "done": done})
    return dumps({"success": True})


def area_targets(domain: str, area: str,
                 entities: Iterable[tuple[str, str, str | None]],
                 groups: Iterable[str] = ()) -> list[str] | None:
    """entity_ids of `domain` in `area` ("*" = whole house), in device-list
    order. Group entities are left out, their members are listed anyway.
    None if the area does not exist."""
    ents = sorted(entities, key=sort_key)
    groups = set(groups)
    if area != ALL_AREAS:
        names = {a for _, _, a in ents if a}
        match = next((a for a in names if a == area), None) or \
            next((a for a in names if a.casefold() == area.strip().casefold()), None)
        if match is None:
            return None
        area = match
    return [e for e, _, a in ents
            if e.split(".", 1)[0] == domain and e not in groups
            and (area == ALL_AREAS or a == area)]
