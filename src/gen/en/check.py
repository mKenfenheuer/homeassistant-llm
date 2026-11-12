"""Words used by src/check_dataset.py to check English answers."""

WORDS = {
    # an answer after a failed tool call must contain one of these
    "fail": ("not", "n't", "error", "reach", "partly", "sorry", "unfortunately", "no "),
    # word for "unreachable": required when a device is unavailable, forbidden
    # after a success
    "unreachable": "reachable",
    # answer that lists the devices that are on or open, and its separators
    "listing": r"(?:Yes\. |No\. )?Currently (?:open|on): (.+)\.$",
    "list_split": r", | and ",
}
