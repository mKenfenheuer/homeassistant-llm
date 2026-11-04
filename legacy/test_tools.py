"""Quick sanity check against a llama-server endpoint: does the model emit
valid tool calls in German-speak commands?  Run while llama-server is up."""

import json
import sys
import urllib.request
import ha_config

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1"
MODEL = "qwen2.5-3b-ha"

TESTS = [
    "schlight the kitchen lights",
    "close all shutters",
    "set the living room lights to 40 percent",
    "deactivate the alarm!",
    "what temperature is it in the bedroom?",
]


def ask(user):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content":
                                    ha_config.SYSTEM_PROMPT},
                                    {"role": "user", "content": user}],
                       "tools": ha_config.TOOLS,
                       "max_tokens": 256, "temperature": 0.1})
    req = urllib.request.Request(URL + "/chat/completions",
                                 data=body.encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]


def main():
    names = set(t["function"]["name"] for t in ha_config.TOOLS)
    ok = 0
    for t in TESTS:
        m = ask(t)
        calls = m.get("tool_calls") or []
        if calls:
            good = all(c["function"]["name"] in names for c in calls)
            for c in calls:
                try:
                    json.loads(c["function"]["arguments"])
                except Exception:
                    good = False
            print(("OK  " if good else "BAD "), t, "->",
                  [(c["function"]["name"], c["function"]["arguments"])
                   for c in calls])
            ok += good
        else:
            print("--  ", t, "-> text only:", (m.get("content") or "")[:80])
    print(ok, "/", len(TESTS), "produced valid tool calls")


if __name__ == "__main__":
    main()
