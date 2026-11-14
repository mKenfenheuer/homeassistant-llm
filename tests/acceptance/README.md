# Acceptance test

A house built like a real installation: 107 entities in 15 areas, cryptic
entity ids, groups, ten offline devices, plants, a car, a mower, a 3D printer
and a weather station. Every device and area name is new to the training
data of its language (`check_names.py`), so the test shows how a model does
in a house it has never seen.

- `house.py`: devices, areas and states, the same in every language
- `<lang>.py`: names, area names and what the user says
- `build.py`: writes `<lang>/layout.json` and `<lang>/scenarios.json`

```sh
python tests/acceptance/check_names.py en de
python tests/acceptance/build.py en de
python src/agent_sim.py http://localhost:8080 tests/acceptance/en/scenarios.json
```

Results are on the model card.
