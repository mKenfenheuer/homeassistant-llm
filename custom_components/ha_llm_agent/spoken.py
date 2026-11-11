"""Numbers as words for text-to-speech (German and English).

TTS models misread digits, e.g. "21,5" or "45". Integers up to 999,999 and
decimals are written out, other numbers stay as they are.
"""

from __future__ import annotations

import re

_DE_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
            "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
            "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig",
            "siebzig", "achtzig", "neunzig"]
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]


def _de_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    out = ""
    if hundreds:
        out = ("ein" if hundreds == 1 else _DE_ONES[hundreds]) + "hundert"
    if rest == 0:
        return out
    if rest < 20:
        word = _DE_ONES[rest]
        # "einhunderteins", but "eins" alone
        return out + word
    tens, ones = divmod(rest, 10)
    if ones == 0:
        return out + _DE_TENS[tens]
    return out + ("ein" if ones == 1 else _DE_ONES[ones]) + "und" + _DE_TENS[tens]


def de_integer(n: int) -> str:
    if n == 0:
        return "null"
    thousands, rest = divmod(n, 1000)
    out = ""
    if thousands:
        out = ("ein" if thousands == 1 else _de_below_1000(thousands)) + "tausend"
    if rest:
        out += _de_below_1000(rest)
    return out


def _en_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(_EN_ONES[hundreds] + " hundred")
    if rest:
        if rest < 20:
            parts.append(_EN_ONES[rest])
        else:
            tens, ones = divmod(rest, 10)
            parts.append(_EN_TENS[tens] + ("-" + _EN_ONES[ones] if ones else ""))
    return " ".join(parts)


def en_integer(n: int) -> str:
    if n == 0:
        return "zero"
    thousands, rest = divmod(n, 1000)
    parts = []
    if thousands:
        parts.append(_en_below_1000(thousands) + " thousand")
    if rest:
        parts.append(_en_below_1000(rest))
    return " ".join(parts)


# language: (integer to words, decimal separator, word for it, minus)
_LANGUAGES = {
    "de": (de_integer, ",", "Komma", "minus"),
    "en": (en_integer, ".", "point", "minus"),
}


_DE_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?:\s?Uhr)?(?![\d:])")


def _de_time(match: re.Match[str]) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    words = f"{de_integer(hours) if hours != 1 else 'ein'} Uhr"
    return f"{words} {de_integer(minutes)}" if minutes else words


def numbers_to_words(text: str, language: str) -> str:
    """Write times (German), integers below one million and decimals as words."""
    code = language.split("-")[0].lower()
    lang = _LANGUAGES.get(code)
    if lang is None:
        return text
    if code == "de":
        text = _DE_TIME.sub(_de_time, text)
    to_words, separator, separator_word, minus = lang
    digit_words = [to_words(d) for d in range(10)]
    pattern = re.compile(
        rf"(?<![\w.,:])(-)?(\d{{1,6}})(?:{re.escape(separator)}(\d{{1,4}}))?(?![\w.,:]?\d)")

    def replace(match: re.Match[str]) -> str:
        sign, integer, decimals = match.groups()
        words = to_words(int(integer))
        if decimals:
            words += f" {separator_word} " + " ".join(digit_words[int(d)] for d in decimals)
        return f"{minus} {words}" if sign else words

    return pattern.sub(replace, text)
