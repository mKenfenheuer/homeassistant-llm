"""Latency of a llama-server for a house with N entities: cold vs. cached prompt.
Usage: python src/bench_cpu.py http://127.0.0.1:18090 95"""
import json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_dataset as G
S = G.S
URL, N = sys.argv[1], int(sys.argv[2])
G.R.seed(5)
h = G.House(N)
while len(h.ents) < N:
    h = G.House(N)
system = h.system_prompt()

def ask(user, max_tokens=400):
    body = json.dumps({"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "tools": S.TOOLS, "temperature": 0, "max_tokens": max_tokens}).encode()
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=3600).read())
    tm = r.get("timings", {})
    print(f"{user[:40]!r:44} {time.time()-t:6.1f}s  prompt {tm.get('prompt_n')} tok "
          f"@ {tm.get('prompt_per_second',0):.1f}/s, gen {tm.get('predicted_n')} tok "
          f"@ {tm.get('predicted_per_second',0):.1f}/s", flush=True)

print(f"{len(h.ents)} entities")
ask("Hallo", max_tokens=1)                         # = warm-up (cold cache)
light = h.of("light")[0]
ask(f"Mach {light.name} an")                       # tool call, cached prefix
ask("Wie warm ist es draußen?")
