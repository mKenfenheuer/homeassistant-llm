"""Words used by src/check_dataset.py to check Italian answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("non", "errore", "raggiungibil", "parzialmente", "in parte", "purtroppo",
             "nessun"),
    # stem of "raggiungibile/raggiungibili": required when a device is
    # unavailable, forbidden after a success
    "unreachable": "raggiungibil",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Sì\. |No\. )?(?:Sono accese|Sono aperte|Sono aperti|È aperta|È accesa): (.+)\.$",
    "list_split": r", | e ",
}
