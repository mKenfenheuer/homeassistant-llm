"""Words used by src/check_dataset.py to check Spanish answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("no", "error", "disponible", "en parte", "lo siento", "ningún", "ninguna",
             "tampoco"),
    # word for "unreachable": required when a device is unavailable, forbidden
    # after a success
    "unreachable": "disponible",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Sí\. |No\. )?(?:Está|Están) (?:encendidas?|abiertas?): (.+)\.$",
    "list_split": r", | y | e ",
}
