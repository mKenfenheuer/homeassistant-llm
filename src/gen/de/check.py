"""Words used by src/check_dataset.py to check German answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("nicht", "fehler", "erreichbar", "teilweise", "leider", "kein"),
    # word for "unreachable": required when a device is unavailable, forbidden
    # after a success
    "unreachable": "erreichbar",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Ja\. |Nein\. )?(?:Offen|An) sind?: (.+)\.$",
    "list_split": r", | und ",
}
