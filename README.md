# ha-llm-training

Dieses Projekt trainiert ein kleines deutsches Sprachmodell für Home Assistant.
Das Modell steuert Geräte und beantwortet Fragen zu ihrem Zustand. Basis ist
Qwen2.5-3B-Instruct mit LoRA (Built with Qwen, Qwen RESEARCH LICENSE,
nur nicht-kommerzielle Nutzung). Für den Betrieb läuft es als GGUF (Q4_K_M,
1,8 GB) in einem Home-Assistant-Add-on auf der CPU.

## Aufbau

| Pfad | Inhalt |
|------|--------|
| `custom_components/ha_llm_agent/` | Integration „HA LLM Agent (lokal)“, Gesprächsagent für Assist |
| `custom_components/ha_llm_agent/llm_schema.py` | Prompt, Werkzeuge und Zustandsformat, gemeinsam für Training und Laufzeit |
| `addons/ha_llm_inference/` | Add-on „HA LLM Inference“ mit llama.cpp-Server (CPU) |
| `src/gen_dataset.py` | Generator für Trainings- und Testdaten |
| `src/train_sft.py` | LoRA-Training, speichert Adapter und zusammengeführtes Modell |
| `src/eval_model.py` | Bewertung eines llama-servers auf dem Testdatensatz |
| `src/check_dataset.py` | Prüft jede Antwort im Datensatz gegen die vorherigen Werkzeugergebnisse |
| `src/agent_sim.py` | Spielt Gespräche über mehrere Runden gegen ein simuliertes Haus durch |
| `src/ha_e2e_test.py` | Ende-zu-Ende-Test gegen eine Home-Assistant-Instanz |
| `legacy/` | Erster Versuch (Dataset, Add-on, Integration), nur zur Ansicht |

## Wie das Modell arbeitet

Die Integration baut bei jeder Anfrage einen Systemprompt mit allen Entitäten,
die für Assist freigegeben sind (`entity_id | Name | Bereich`). Steht das
Sprachgerät in einem Bereich, ergänzt sie diesen Bereich. Das Modell hat zwei
Werkzeuge:

- `control` ruft echte Home-Assistant-Dienste auf, z. B.
  `{"service": "light.turn_on", "entity_id": ["light.kueche"], "data": {"brightness_pct": 30}}`.
- `get_state` liefert Zustand und wichtige Attribute, z. B. Temperatur,
  Position oder Helligkeit.

Lichter spricht das Modell auch über den Bereich an: „Mach das Licht in der
Küche an“ wird zu `{"service": "light.turn_on", "area": "Küche"}`, „alle
Lichter aus“ zu `"area": "*"`. Die Integration löst den Bereich in die
freigegebenen Lampen auf und meldet sie im Ergebnis zurück (`"done": [...]`).
Das Modell muss bei vier Lampen in der Küche keine vier entity_ids mehr
abschreiben. Ebenso fragt `get_state` mit `domain` und `area` alle Lichter
eines Bereichs ab.

Die Integration prüft jeden Aufruf. Sie erlaubt nur Dienste aus
`llm_schema.SERVICES`, nur freigegebene Entitäten und nur die dort genannten
Parameter.

Der Trainingsdatensatz entsteht vollständig aus Vorlagen. Jede Probe nutzt ein
neu gewürfeltes Haus mit eigenen Räumen, Namen und teils kryptischen
`entity_id`s wie `light.shelly1_3c61a5`. Das Modell lernt dadurch die Zuordnung
über die Geräteliste und merkt sich kein festes Haus. Die Daten decken Befehle,
Gruppen („alle Rollläden“), kombinierte Befehle, Fragen mit `get_state`,
Gespräche über mehrere Runden, unbekannte Geräte, nicht erreichbare Geräte
und Teilerfolge ab. Jeder der 64 Fälle hat dieselbe Quote, auch seltene wie
„Im Schlafzimmer gibt es keinen Ventilator“ oder ein Haus ohne Bereiche.

Ehrlichkeit kommt aus den Daten, nicht aus Regeln in der Integration. Meldet
`control` einen Fehler, z. B. `{"success": false, "error": "Entity light.garten
is unavailable", "done": ["light.flur"]}`, lernt das Modell, genau das zu
sagen. `src/check_dataset.py` verwirft jeden Datensatz, in dem eine Antwort
einen Erfolg behauptet, den das Werkzeugergebnis nicht hergibt.

## Training

```bash
./run_train.sh       # 8.000 Trainings- und 300 Testproben, LoRA r=32, 2,5 Epochen
./run_export.sh output/merged qwen2.5-3b-ha-v15  # -> model/qwen2.5-3b-ha-v15-Q4_K_M.gguf
./run_test.sh        # Bewertung auf data/eval.jsonl (GPU, Docker)
```

Das Training braucht rund 17 GB VRAM und auf der RTX 3090 etwa 5,5 Stunden.
Konvertierung und Quantisierung laufen im offiziellen Image
`ghcr.io/ggml-org/llama.cpp:full-cuda`, ein eigener llama.cpp-Build ist nicht
nötig.

**Quantisierung** Liegt `model/calib.txt` vor, berechnet `run_export.sh` vor
der Quantisierung eine Importance Matrix. Die Datei entsteht aus 300
Trainingsgesprächen:
`.venv/bin/python src/make_calib.py data/train.jsonl model/calib.txt 300`.
Die Testdaten fließen nicht ein. `QUANTS="Q4_K_M Q5_K_M"` erzeugt mehrere
Formate in einem Lauf.

**Model Soup** `src/soup.py` mittelt die LoRA-Änderungen mehrerer Läufe auf
demselben Basismodell, ohne neues Training:
`.venv/bin/python src/soup.py output/merged-soup output/lora-v13 output/lora-v14 output/lora-v15`.
Das veröffentlichte Modell v15.1 entstand so aus drei Läufen mit den Seeds
13, 14 und 15.

**Chat-Vorlage** `run_export.sh` schreibt `templates/chat_template.jinja` in
jede GGUF-Datei. Die Vorlage entspricht der von Qwen2.5, nur der
Standard-Systemprompt ohne Systemnachricht ist deutsch statt „You are Qwen“.
Für alle 8.300 Trainings- und Testgespräche erzeugt sie denselben Text wie
die Vorlage im Training.

Wer `llm_schema.py` ändert, muss neu trainieren. Training und Integration
lesen dieselbe Datei, damit der Prompt zur Laufzeit exakt dem Training
entspricht.

## Installation in Home Assistant

Das Git-Repository dient gleichzeitig als Add-on-Repository und als
HACS-Repository. Das Modell liegt auf Hugging Face, das Add-on lädt es beim
ersten Start.

**Add-on** Einstellungen → Apps → App-Store → Menü → Repositories, die URL
dieses Git-Repositorys eintragen. Danach „HA LLM Inference“ installieren und
starten. Der erste Start lädt das Modell (1,9 GB) von
[mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF)
nach `/share/ha_llm/`. Im Log erscheint danach `listening on http://0.0.0.0:8080`.

**Modell-Updates** Mit `auto_update` (Standard: an) liest das Add-on beim
Start und alle 6 Stunden (`update_interval`) die Datei
[`latest.json`](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF/blob/main/latest.json)
im Modell-Repository. Nennt sie ein neues Modell, lädt das Add-on die Datei im
laufenden Betrieb, prüft die SHA-256-Summe, wechselt auf das neue Modell und
löscht das alte. Ein Modell für ein neueres Prompt-Schema überspringt es, bis
ein Add-on-Update die passende Integration mitbringt. Ist Hugging Face nicht
erreichbar, startet es mit dem zuletzt geladenen Modell. Eine eigene
`model_url` oder `model_path` schaltet die Updates ab.

**Integration** In HACS → Menü → Benutzerdefinierte Repositories die URL
dieses Git-Repositorys mit Typ „Integration“ eintragen, „HA LLM Agent (lokal)“
herunterladen und Home Assistant neu starten. Ohne HACS den Ordner
`custom_components/ha_llm_agent` nach `/config/custom_components/` kopieren.

**Einrichtung** „HA LLM Agent (lokal)“ hinzufügen. Die Integration erkennt das
installierte Add-on und trägt dessen URL vor, z. B.
`http://a1b2c3d4-ha-llm-inference:8080`. Läuft llama-server woanders, die URL
von Hand eintragen. Eine spätere Änderung, etwa vom Add-on auf einen GPU-Rechner,
läuft über das Menü des Eintrags → „Neu konfigurieren“. Der Agent und seine
Optionen bleiben dabei erhalten.

**Assist** Unter Einstellungen → Sprachassistenten den Agenten „Lokales LLM“
als Gesprächsagenten wählen. Welche Geräte das Modell sieht, steuert die
Freigabe unter „Entitäten freigeben“.

## Hugging Face

| Inhalt | Repository |
|--------|------------|
| Modell (GGUF, Q4_K_M) | [mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF) |
| Datensatz | [mkenfenheuer/homeassistant-tool-calling-de](https://huggingface.co/datasets/mkenfenheuer/homeassistant-tool-calling-de) |

Upload eines neuen Stands: `HF_TOKEN` in `.env` eintragen, dann
`set -a; . ./.env; set +a; .venv/bin/python src/upload_hf.py
mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF mkenfenheuer/homeassistant-tool-calling-de`.
Mit `--gguf model/<datei>.gguf` lädt das Skript ein neues Modell hoch und
löscht im selben Commit alle anderen GGUF-Dateien. Es schreibt `latest.json`
aus `hf/model/latest.json` (Version, Schema, Hinweis) und Name, SHA-256 und
Größe der Datei. Ab diesem Moment aktualisieren sich alle Add-ons mit
`auto_update`. Für Neuinstallationen ohne Netz zu Hugging Face beim Start
zusätzlich `DEFAULT_URL` und `DEFAULT_SHA` in `addons/ha_llm_inference/run.sh`
anpassen und die alte URL in `LEGACY_URLS` eintragen. Ändert sich das
Prompt-Schema, `schema` in `hf/model/latest.json` und `SUPPORTED_SCHEMA` in
`run.sh` gemeinsam mit der Integration erhöhen.

## Test

`src/ha_e2e_test.py` schickt deutsche Sätze an die Conversation-API und prüft
danach die echten Zustände. Für den Test lief Home Assistant 2026.10.4 im
Container mit der Demo-Integration. Die Demo-Geräte haben deutsche Namen und
Bereiche erhalten.

```bash
HA_URL=http://localhost:18123 HA_TOKEN=... python3 src/ha_e2e_test.py
```

## Ergebnisse (Stand 09.11.2026)

**Testdatensatz** 300 Proben aus Häusern, die im Training nicht vorkamen
(`./run_test.sh`):

| Modell | Werkzeugaufrufe korrekt | Antworten korrekt |
|--------|------------------------|-------------------|
| Qwen2.5-3B-Instruct ohne Training (60 Proben, v1-Testdaten) | 61,7 % | 50,0 % |
| v1 (LoRA, 2 Epochen), v1-Testdaten | 98,3 % | 100 % |
| v1, v2-Testdaten | 89,0 % | 98,4 % |
| v2 (1 weitere Epoche auf v2-Daten), v2-Testdaten | 94,3 % | 100 % |
| v2, v3-Testdaten | 93,7 % | 93,3 % |
| v3 (LoRA, 2 Epochen ab Basismodell), v3-Testdaten | 93,3 % | 100 % |
| v3, v4-Testdaten | 93,3 % | 96,8 % |
| v4 (LoRA, 2,5 Epochen ab Basismodell), v4-Testdaten | 96,0 % | 100 % |
| v4, v7-Testdaten | 96,3 % | 100 % |
| v7 (LoRA, 2,5 Epochen ab Basismodell), v7-Testdaten | 98,0 % | 100 % |
| v11 (LoRA, 2,5 Epochen ab Basismodell), v11-Testdaten | 98,7 % | 98,7 % |
| v11, v15-Testdaten | 98,0 % | 97,1 % |
| v15-soup (Mittel aus v13, v14, v15), v15-Testdaten | 99,7 % | 100 % |

Die v2-Testdaten sind schwerer als die v1-Testdaten: Häuser mit bis zu 110
Entitäten, Gruppen, Räume ohne Licht und Geräte, die nicht in der Liste
stehen. Ab v3 kommen Gespräche über mehrere Runden, nicht erreichbare Geräte
und Geräte mit Status-Sensor und Steckdose hinzu, ab v4 Lampen mit dem Namen
ihres Raums und mehrere Bewässerungen, ab v7 Räume mit teilweise nicht
erreichbaren Lampen, mehrfaches Beharren auf fehlenden Geräten und
Modus-Schalter der Heizung. Die v11-Testdaten nutzen Schema 5 mit Bereichen
und enthalten Türkontakte, Elektroautos sowie Außentemperatur neben
Vorhersage-Sensoren. v7 lässt sich darauf nicht direkt messen, weil es
`area` nicht kennt. v11 lehnt alle 15 Anfragen nach fehlenden Geräten ab und
beantwortet 78 von 81 Zustandsfragen mit dem richtigen Aufruf. Auf den
v15-Testdaten löst v15-soup alle 118 Befehle, alle 17 Anfragen nach fehlenden
Geräten und 71 von 72 Zustandsfragen.

**Echtes Haus** Der Abnahmetest läuft auf einer realen Installation mit über
100 freigegebenen Entitäten: 26 Sätze auf einem älteren Stand der
Geräteliste, 32 Sätze auf dem aktuellen Stand und vier Varianten des
aktuellen Stands mit je 8 fehlenden Geräten.

| Modell | Älterer Stand | Aktueller Stand | Varianten |
|--------|---------------|-----------------|-----------|
| v7 | 24/26 | 32/32 | 123/128 |
| v11 | 26/26 | 32/32 | 128/128 |
| v15-soup | 26/26 | 32/32 | 128/128 |

v7 fragte bei „Ist die Balkontür im Wohnzimmer offen?“ die Wohnungstür ab und
bei der Reichweite des Elektroautos den Ladestand. v11 löst beide. Fragen
nach Geräten, die in einem Raum fehlen, beantwortet v15-soup in allen 30
Fällen auf dem echten Haus und seinen Varianten mit dem Raum, ohne ein
anderes Gerät zu nennen.

**Dialoge** `src/agent_sim.py` spielt Gespräche über mehrere Runden gegen ein
simuliertes Haus durch. In 19 Szenarien meldet v11 nicht erreichbare Geräte in
jeder Runde als Fehler. Sind zwei von vier Küchenlampen nicht erreichbar,
schaltet v11 „das Licht in der Küche“ über den Bereich und nennt die
geschalteten und die nicht erreichbaren Lampen getrennt. Die Korrektur „Nein,
ich meinte im Esszimmer“ schaltet jetzt alle Küchenlampen aus, v7 nur eine.
v15-soup behebt die beiden Schwächen von v11: Beim fehlenden Fernseher im
Schlafzimmer nennt es den Raum statt des Media Players eines
Sprachassistenten, und zur teilweise erreichbaren Küche nennt es beide nicht
erreichbaren Lampen. „Bewässere die Balkonpflanzen“ öffnet das Balkon-Ventil,
und „Welche Heizungen hast du verstellt?“ nennt beide verstellten Heizungen.
Gerätenamen gibt es in Antworten nicht immer in der Schreibweise der Liste
wieder, etwa mit Leerzeichen in einem zusammengesetzten Namen.

**Home Assistant** Der Ende-zu-Ende-Test mit Add-on (CPU, 8 Threads) und
Integration besteht mit v15-soup 24 von 24 Fällen. Dazu gehören eine nicht
erreichbare Lampe und eine Rückfrage im selben Gespräch. Die Antwortzeit
liegt im Median bei 2,9 Sekunden, die längste Anfrage braucht 8,6 Sekunden
(erste Anfrage nach dem Start).

## Hardware für das Add-on

Das Add-on rechnet auf der CPU. Entscheidend ist AVX2: Ohne AVX2 nutzt
llama.cpp für Q4_K langsamen generischen Code. Messung mit 95 Entitäten
(2.342 Tokens Systemprompt) auf 4 Kernen (`src/bench_cpu.py`):

| Befehlssatz | Erste Anfrage (kalter Cache) | Folgende Befehle |
|-------------|------------------------------|------------------|
| AVX2 (z. B. Intel N100, Core i3/i5, Ryzen) | 34 s | 2–3 s |
| nur SSE4.2 (z. B. Atom C3000, Celeron J4125) | 19 min | 30–40 s |

Die Integration schickt den Systemprompt beim Start und nach jeder Änderung
der Geräteliste vorab an llama-server (Warm-up). Danach liegt er im Cache, und
nur die Frage selbst muss neu verarbeitet werden. Dauert das Warm-up länger als
60 Sekunden, schreibt die Integration eine Warnung ins Log.

Hat der Home-Assistant-Host kein AVX2, sollte llama-server auf einem anderen
Rechner laufen, etwa per Docker auf einem Server mit GPU. Eine GTX 1060 (6 GB,
Treiber 580) verarbeitet den Prompt mit 540 Tokens/s und erzeugt 45 Tokens/s.
Der Systemprompt mit 100 Entitäten liegt damit nach etwa 10 Sekunden im Cache.

```bash
docker run -d --name ha-llm --restart unless-stopped --gpus all \
  -p 8080:8080 -v $PWD/model:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12063 \
  -m /models/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
  --alias ha-llm --jinja -ngl 99 -fa on -c 16384 -np 1 --cache-reuse 256
```

In der Integration dann `http://<server>:8080` als URL eintragen oder einen
bestehenden Eintrag über „Neu konfigurieren“ auf diese URL umstellen.

## Grenzen

- Das Modell kennt nur die Dienste in `llm_schema.SERVICES`. Automationen
  anlegen oder Timer beherrscht es nicht. Wettervorhersagen beantwortet es
  nur über freigegebene Vorhersage-Sensoren.
- Integration und Modell gehören zusammen. Integration 0.7.0 braucht Modell
  v11 oder neuer, ältere Modelle kennen `area` nicht.
- Ab etwa 250 Entitäten reicht die Kontextlänge von 8.192 Tokens nicht mehr.
  Die Integration kürzt dann die Liste (Option „Maximale Anzahl Entitäten“).
- Die Option „Bereich des Sprachgeräts verwenden“ ist standardmäßig aus. Der
  Bereich steht im Prompt vor den Werkzeugen, deshalb verwirft jeder Wechsel
  des Sprachgeräts rund 600 Tokens Cache.
