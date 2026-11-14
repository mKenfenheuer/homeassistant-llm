"""Write <lang>/layout.json and <lang>/scenarios.json (src/agent_sim.py
format) for the acceptance house: house.py holds the devices, <lang>.py the
names and what the user says.

    python tests/acceptance/build.py en
    python src/agent_sim.py URL tests/acceptance/en/scenarios.json
"""
import importlib
import json
import sys
from pathlib import Path

from house import GROUPS, HOUSE

HERE = Path(__file__).parent
# device classes as Home Assistant reports them, so get_state(device_class=...) works
CLASSES = (("_occupancy", "occupancy"), ("french_door", "door"), ("flat_door", "door"),
           ("window_", "window"), ("_motion", "motion"), ("_contact", "opening"),
           ("_temperature", "temperature"), ("_humidity", "humidity"),
           ("_battery", "battery"), ("_moisture", "moisture"), ("state_of_charge", "battery"))

TRV = "climate.hmip_etrv_b_2_r4m_00a1b2c3d4e5%02d"


def C(svc, eid=None, area=None, **data):
    spec = {"service": svc}
    if eid:
        spec["entity_id"] = [eid]
    if area:
        spec["area"] = area
    if data:
        spec["data"] = data
    return ["control", spec]


def G(*ids):
    return ["get_state", {"entity_ids": list(ids)}]


# (key, [expectation per turn]); an expectation is {"calls"}, {"any_of"},
# {"no_tool"}, plus "contains". Area names are area keys, filled in per language.
SCENARIOS = [
    ("galley light", [{"any_of": [[C("light.turn_on", "light.galley")], [C("light.turn_on", area="@gal")]]}]),
    ("master suite shutter", [{"calls": [C("cover.close_cover", "cover.master_suite_2")]}]),
    ("sitting room radiator", [{"calls": [C("climate.set_temperature", TRV % 3, temperature=22)]}]),
    ("settee lamp, then dim it", [{"calls": [C("light.turn_on", "light.sitting_room_sofa")]},
                                  {"calls": [C("light.turn_on", "light.sitting_room_sofa", brightness_pct=30)]}]),
    ("vanity cabinet", [{"calls": [C("light.turn_on", "light.shellydimmer2_a1b2c3d4e5f6")]}]),
    ("banister", [{"calls": [C("light.turn_on", "light.banister")]}]),
    ("bookcase glow", [{"calls": [C("light.turn_on", "light.smpw701e_07_relay")]}]),
    ("twig tree", [{"calls": [C("light.turn_on", "light.twig_tree")]}]),
    ("yard lanterns", [{"calls": [C("light.turn_on", "light.hama_outdoor_01_relay")]}]),
    ("every lamp off", [{"any_of": [[C("light.turn_off", "light.every_lamp")], [C("light.turn_off", area="*")]]}]),
    ("every shutter down", [{"any_of": [[C("cover.close_cover", "cover.every_shutter")],
                                        [C("cover.close_cover", area="*")]]}]),
    ("downstairs shutters half", [{"calls": [C("cover.set_cover_position", "cover.downstairs_shutters",
                                               position=50)]}]),
    ("sunshade", [{"calls": [C("cover.open_cover", "cover.somfy_io")]}]),
    ("split unit", [{"calls": [C("climate.turn_on", "climate.sitting_room_split_unit")]}]),
    ("cylinder boost", [{"calls": [C("switch.turn_on", "switch.jane_example_org_home_hot_water_0_boost")]}]),
    ("booster pump", [{"calls": [C("switch.turn_off", "switch.tasmota_plug_04_relay")]}]),
    ("clothes dryer", [{"calls": [C("switch.turn_off", "switch.clothes_dryer_power")]}]),
    ("holiday mode", [{"calls": [C("input_boolean.turn_on", "input_boolean.holiday")]}]),
    ("drip line", [{"calls": [C("valve.open_valve", "valve.hama_outdoor_plug_02_relay")]}]),
    ("mowbot", [{"calls": [C("lawn_mower.start_mowing", "lawn_mower.yard_mowbot")]}]),
    ("echo do not disturb", [{"calls": [C("switch.turn_on", "switch.echo_show_3f2a_do_not_disturb")]}]),
    # offline devices: trying and reporting it, or looking first, are both fine
    ("star lantern offline", [{"any_of": [[C("light.turn_on", "light.star_lantern")], [G("light.star_lantern")]],
                               "contains": "@offline"}]),
    ("vacuum offline", [{"any_of": [[C("vacuum.start", "vacuum.q_revo")], [G("vacuum.q_revo")]],
                         "contains": "@offline"}]),
    ("french door", [{"calls": [G("binary_sensor.french_door_breakfast")]}]),
    ("flat door", [{"calls": [G("binary_sensor.flat_door_state")]}]),
    ("washroom occupancy", [{"calls": [G("binary_sensor.mmwave_03_occupancy")]}]),
    ("outside", [{"any_of": [[G("sensor.gw2000a_outdoor_temperature")],
                             [G("sensor.jane_example_org_home_outdoor_temperature")], [G("weather.millbrook")]]}]),
    ("mowbot battery", [{"calls": [G("sensor.yard_mowbot_battery")], "contains": ["97"]}]),
    ("rack humidity", [{"calls": [G("sensor.gw2000a_indoor_humidity")], "contains": ["34"]}]),
    ("mint", [{"calls": [G("sensor.plant_sensor_1a2b_moisture")]}]),
    ("ioniq range", [{"calls": [G("sensor.ioniq_remaining_range")], "contains": ["241"]}]),
    ("ioniq charge", [{"calls": [G("sensor.ioniq_hv_state_of_charge")], "contains": ["81"]}]),
    ("galley radiator", [{"calls": [G(TRV % 1)]}]),
    ("weather", [{"calls": [G("weather.millbrook")]}]),
    ("letterbox offline", [{"calls": [G("binary_sensor.hmip_swdm_2_00a1b2c3d4e5f6_contact")]}]),
    ("no dishwasher", [{"calls": []}]),
    ("timer", [{"calls": [["timer", {"action": "start", "minutes": 10}]]}]),
    ("broadcast", [{"calls": [["broadcast", {}]]}]),
    ("time", [{"calls": [["get_time", {}]]}]),
    ("thanks", [{"no_tool": True}]),
]


def fill(v, L):
    """Area keys ("@gal") and words ("@offline") of the language."""
    if isinstance(v, str) and v.startswith("@"):
        return L.AREA.get(v[1:]) or L.WORDS[v[1:]]
    if isinstance(v, list):
        return [fill(x, L) for x in v]
    if isinstance(v, dict):
        return {k: fill(x, L) for k, x in v.items()}
    return v


def main(lang):
    L = importlib.import_module(lang)
    out = HERE / lang
    out.mkdir(exist_ok=True)
    layout = [[e, L.NAME[e], L.AREA.get(a, "") if a else ""] for e, a, _ in HOUSE]
    unavailable = [e for e, _, s in HOUSE if s == "unavailable"]
    states = {e: s for e, _, s in HOUSE if s != "unavailable"}
    attrs = {e: {"device_class": c} for e, *_ in HOUSE if e.split(".")[0] in ("sensor", "binary_sensor")
             for k, c in CLASSES if k in e}
    scenarios = []
    for key, turns in SCENARIOS:
        said = L.SAY[key]
        said = [said] if isinstance(said, str) else said
        assert len(said) == len(turns), key
        scenarios.append({"name": key, "unavailable": unavailable, "location": L.AREA["stu"],
                          "player": "media_player.echo_show_3f2a_media_player_2",
                          "turns": [{"text": t, **fill(x, L)} for t, x in zip(said, turns)]})
    json.dump(layout, open(out / "layout.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump({"lang": L.LANG, "house": "layout.json", "groups": GROUPS, "states": states,
               "attrs": attrs, "scenarios": scenarios},
              open(out / "scenarios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{lang}: {len(layout)} entities, {len(scenarios)} scenarios")


if __name__ == "__main__":
    for lang in sys.argv[1:]:
        main(lang)
