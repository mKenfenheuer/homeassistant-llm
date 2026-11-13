"""Words used by src/check_dataset.py to check Dutch answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("niet", "fout", "bereikbaar", "deels", "helaas", "geen"),
    # word for "unreachable": required when a device is unavailable, forbidden
    # after a success
    "unreachable": "bereikbaar",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Ja\. |Nee\. )?(?:Deze lampen zijn aan|Dit raam staat open|"
               r"Deze ramen staan open|Deze zijn open): (.+)\.$",
    "list_split": r", | en ",
}
