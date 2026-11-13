"""Words used by src/check_dataset.py to check French answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("pas", "erreur", "joignable", "partie", "malheureusement", "aucun",
             "impossible", "échoué", "rien", "désolé"),
    # word for "unreachable": required when a device is unavailable, forbidden
    # after a success
    "unreachable": "joignable",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Oui\. |Non\. )?(?:Les (?:lumières allumées|fenêtres ouvertes|volets ouverts|stores ouverts) sont"
               r"|La seule fenêtre ouverte est) : (.+)\.$",
    "list_split": r", | et ",
}
