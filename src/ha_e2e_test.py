"""End-to-end test: German sentences -> HA conversation API -> entity states.

Runs against the HA test instance (demo integration, German names, see
README "Test"). Each case resets the relevant entities, sends the sentence to
the agent and checks the resulting state or the spoken answer.

Usage: HA_URL=http://localhost:18123 HA_TOKEN=... python3 src/ha_e2e_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HA = os.environ.get("HA_URL", "http://localhost:18123")
TOKEN = os.environ["HA_TOKEN"]
AGENT = os.environ.get("AGENT_ID", "conversation.lokales_llm")


def api(method, path, body=None):
    req = urllib.request.Request(
        HA + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + TOKEN,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read() or b"null")


def state(eid):
    return api("GET", f"/api/states/{eid}")


def svc(service, **data):
    d, s = service.split(".")
    try:
        api("POST", f"/api/services/{d}/{s}", data)
    except urllib.error.HTTPError as err:
        print(f"  reset {service} {data.get('entity_id')}: HTTP {err.code}")


def say(text, conv_id=None):
    t0 = time.time()
    body = {"text": text, "language": "de", "agent_id": AGENT}
    if conv_id:
        body["conversation_id"] = conv_id
    r = api("POST", "/api/conversation/process", body)
    speech = r["response"]["speech"].get("plain", {}).get("speech", "")
    return speech, r.get("conversation_id"), time.time() - t0


def bri_pct(eid):
    b = state(eid)["attributes"].get("brightness")
    return None if b is None else round(b / 255 * 100)


def reset():
    svc("light.turn_on", entity_id=["light.kitchen_lights", "light.ceiling_lights",
                                    "light.living_room_rgbww_lights",
                                    "light.office_rgbw_lights",
                                    "light.entrance_color_white_lights"],
        brightness_pct=100, color_name="white")
    svc("light.turn_off", entity_id="light.bed_light")
    svc("cover.close_cover", entity_id="cover.kitchen_window")
    svc("cover.open_cover", entity_id=["cover.living_room_window", "cover.hall_window"])
    svc("climate.set_temperature", entity_id="climate.heatpump", temperature=20)
    svc("lock.unlock", entity_id="lock.kitchen_door")
    svc("lock.lock", entity_id="lock.front_door")
    svc("alarm_control_panel.alarm_disarm", entity_id="alarm_control_panel.security",
        code="1234")
    svc("switch.turn_off", entity_id="switch.ac")
    svc("switch.turn_on", entity_id="switch.decorative_lights")
    svc("fan.turn_off", entity_id="fan.ceiling_fan")
    svc("vacuum.return_to_base", entity_id="vacuum.demo_vacuum_0_ground_floor")
    svc("media_player.volume_set", entity_id="media_player.living_room", volume_level=0.5)
    time.sleep(12)   # demo covers move 10 %/s


CASES = [
    ("Mach das Küchenlicht aus",
     lambda: state("light.kitchen_lights")["state"] == "off"),
    ("Schalte die Nachttischlampe ein",
     lambda: state("light.bed_light")["state"] == "on"),
    ("Dimme die Stehlampe im Wohnzimmer auf 30 Prozent",
     lambda: bri_pct("light.living_room_rgbww_lights") == 30),
    ("Mach die Schreibtischlampe blau",
     lambda: state("light.office_rgbw_lights")["attributes"].get("rgb_color") in ([0, 0, 255], (0, 0, 255))),
    ("Fahr den Rollladen in der Küche hoch",
     lambda: state("cover.kitchen_window")["state"] in ("opening", "open")),
    ("Schließe alle Rollläden",
     lambda: all(state(c)["state"] in ("closing", "closed")
                 for c in ("cover.living_room_window", "cover.hall_window"))),
    ("Stell die Heizung im Wohnzimmer auf 22 Grad",
     lambda: state("climate.heatpump")["attributes"].get("temperature") == 22),
    ("Schließ die Kellertür ab",
     lambda: state("lock.kitchen_door")["state"] in ("locking", "locked")),
    ("Kaffeemaschine an",
     lambda: state("switch.ac")["state"] == "on"
     and state("switch.decorative_lights")["state"] == "on"),
    ("Stell den Deckenventilator auf 50 Prozent",
     lambda: state("fan.ceiling_fan")["attributes"].get("percentage") == 50),
    ("Starte den Saugroboter",
     lambda: state("vacuum.demo_vacuum_0_ground_floor")["state"] == "cleaning"),
    ("Mach den Fernseher im Wohnzimmer leiser",
     lambda: state("media_player.living_room")["attributes"].get("volume_level", 1) < 0.5),
    ("Aktiviere die Szene Filmabend",
     lambda: state("light.ceiling_lights")["state"] == "off"),
    ("Mach das Licht im Wohnzimmer aus",
     lambda: state("light.ceiling_lights")["state"] == "off"
     and state("light.living_room_rgbww_lights")["state"] == "off"
     and state("light.office_rgbw_lights")["state"] == "on"),   # no side effects
    ("Mach das Licht im Flur aus und die Nachttischlampe an",
     lambda: state("light.entrance_color_white_lights")["state"] == "off"
     and state("light.bed_light")["state"] == "on"),
    ("Schalte die Alarmanlage scharf, Code 1234",
     lambda: state("alarm_control_panel.security")["state"] in ("arming", "armed_away")),
]

QUESTIONS = [
    ("Wie warm ist es draußen?", lambda s: "15,6" in s or "15.6" in s),
    ("Wie hoch ist die Luftfeuchtigkeit draußen?", lambda s: "54" in s),
    ("Ist die Haustür abgeschlossen?", lambda s: "abgeschlossen" in s.lower()
     and "nicht" not in s.lower()),
    ("Ist das Küchenlicht an?", lambda s: " an" in s.lower()),
    ("Schalte die Sauna ein", lambda s: "sauna" in s.lower()),
    ("Was kannst du?", lambda s: len(s) > 20),
]


def main():
    ok = n = 0
    times = []
    print(f"agent {AGENT} @ {HA}")
    reset()
    for text, check in CASES:
        speech, _, dt = say(text)
        time.sleep(3)
        good = bool(check())
        ok += good
        n += 1
        times.append(dt)
        print(f"{'OK ' if good else 'BAD'} {dt:5.1f}s  {text!r} -> {speech!r}")
    svc("light.turn_on", entity_id="light.kitchen_lights")
    for text, check in QUESTIONS:
        speech, _, dt = say(text)
        good = bool(check(speech))
        ok += good
        n += 1
        times.append(dt)
        print(f"{'OK ' if good else 'BAD'} {dt:5.1f}s  {text!r} -> {speech!r}")
    # unreachable device: the answer must report the failure, not success
    api("POST", "/api/states/light.bed_light",
        {"state": "unavailable", "attributes": {"friendly_name": "Nachttischlampe"}})
    speech, _, dt = say("Schalte die Nachttischlampe ein")
    good = "nicht" in speech.lower() and state("light.bed_light")["state"] == "unavailable"
    ok += good
    n += 1
    times.append(dt)
    print(f"{'OK ' if good else 'BAD'} {dt:5.1f}s  unavailable -> {speech!r}")
    svc("light.turn_off", entity_id="light.bed_light")   # demo light writes its state again
    # follow-up within one conversation
    svc("light.turn_on", entity_id="light.kitchen_lights")
    s1, cid, dt1 = say("Mach das Küchenlicht aus")
    time.sleep(1)
    s2, _, dt2 = say("Und wieder an", cid)
    time.sleep(1)
    good = state("light.kitchen_lights")["state"] == "on"
    ok += good
    n += 1
    times += [dt1, dt2]
    print(f"{'OK ' if good else 'BAD'} {dt1 + dt2:5.1f}s  follow-up -> {s1!r} / {s2!r}")
    times.sort()
    print(f"\n{ok}/{n} passed, latency median {times[len(times) // 2]:.1f}s, "
          f"max {times[-1]:.1f}s")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
