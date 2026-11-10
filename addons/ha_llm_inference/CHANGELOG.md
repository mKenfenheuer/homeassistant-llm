# Changelog

## 0.9.1

- Die AVX2-Warnung erscheint nur noch auf x86-CPUs, nicht mehr auf ARM.
- Die GPU-Add-ons laufen ohne GPU auf der CPU und bekommen die GPU über die
  Option `video`.

## 0.9.0

- Zwei neue Add-ons für GPUs im Home-Assistant-Host, beide experimentell:
  „HA LLM Inference (NVIDIA)“ mit CUDA 12.8 (ab Maxwell, getestet mit einer
  GTX 1060) und „HA LLM Inference (AMD/Intel)“ mit Vulkan (getestet mit einer
  Radeon RX 6900 XT, 130 Tokens pro Sekunde). Sie brauchen Home Assistant OS
  mit GPU-Unterstützung für Apps (Option `gpu`), die es bisher nur als
  Testimage gibt. Neue Option `gpu_layers` (Standard 99 = alle Schichten).
- Integration 0.7.2: Bei der Einrichtung nimmt sie das GPU-Add-on, wenn eines
  installiert ist, sonst das CPU-Add-on.
- CPU-Add-on unverändert bis auf das gemeinsame Startskript.

## 0.8.1

- Eigenes Symbol und Logo für Add-on und Integration. Die Integration bringt
  sie unter `custom_components/ha_llm_agent/brand/` mit, Home Assistant
  2026.3 und neuer zeigt sie ohne Eintrag im Brands-Repository an
  (Integration 0.7.1). Die SVG-Quellen liegen unter `brand/`.
- Modell- und Datensatzkarte auf Hugging Face verweisen auf dieses
  Repository mit Add-on und Integration.

## 0.8.0

- Automatische Modell-Updates: Mit `auto_update` (Standard: an) prüft das
  Add-on beim Start und alle 6 Stunden (`update_interval`) die Datei
  `latest.json` auf Hugging Face. Ein neues Modell lädt es im laufenden
  Betrieb, prüft die SHA-256-Summe, wechselt darauf und löscht das alte.
  Ohne Netz startet es mit dem zuletzt geladenen Modell. Eigene `model_url`
  oder `model_path` schalten die Updates ab.
- Modell v15.1 (`qwen2.5-3b-ha-v15.1-Q4_K_M.gguf`): Gewichte wie v15-soup,
  die eingebettete Chat-Vorlage nutzt ohne Systemnachricht einen deutschen
  Standard-Systemprompt statt „You are Qwen“. Mit Systemnachricht, wie von
  der Integration gesendet, ändert sich am Prompt nichts. Testergebnisse
  identisch: 299 von 300 Werkzeugaufrufen, 24 von 24 im Home-Assistant-Test.
- Auf Hugging Face liegt nur noch das aktuelle Modell.

## 0.7.1

- Neues Modell v15-soup (`qwen2.5-3b-ha-v15-soup-Q4_K_M.gguf`). Es mittelt
  die LoRA-Gewichte der drei Trainingsläufe v13, v14 und v15 („Model Soup“).
  Jeder Einzellauf machte eigene Einzelfehler, die Mittelung gleicht sie aus.
  Auf 300 Testgesprächen sind 99,7 % der Werkzeugaufrufe korrekt (v11:
  98,0 %).
- Quantisierung mit Importance Matrix aus eigenen Trainingsgesprächen. Ohne
  sie beschädigte Q4_K_M einzelne Läufe (v13: 291 von 300 unquantisiert,
  243 von 300 in Q4). Die Datei bleibt 1,9 GB groß und gleich schnell.
- „Mach den Fernseher im Schlafzimmer an“ nennt jetzt das Schlafzimmer statt
  des Media Players eines Sprachassistenten im Raum.
- „Bewässere die Balkonpflanzen“ öffnet das Ventil am Balkon, auch wenn ein
  Taster „Bewässerung Garten“ ohne Bereich daneben steht.
- „Welche Heizungen hast du verstellt?“ nennt alle verstellten Heizungen im
  Gespräch, nicht nur die letzte.
- Die Frage nach einem Raum mit teilweise nicht erreichbaren Lampen nennt
  alle nicht erreichbaren Lampen, auch die mit dem Namen des Raums.
- Integration unverändert, das Modell nutzt weiter Schema 5 (Integration
  0.7.0).

## 0.7.0

- Neues Modell v11 (`qwen2.5-3b-ha-v11-Q4_K_M.gguf`), neu trainiert ab
  Qwen2.5-3B-Instruct mit 8.000 Beispielen und 2,5 Epochen. Jeder der 64
  Fälle im Generator hat dieselbe Quote, seltene Fälle eingeschlossen.
- Schema 5: „Mach das Licht in der Küche an“ schaltet alle Lichter des
  Bereichs über `area`, „alle Lichter“ über `area: "*"`. Die Integration löst
  den Bereich selbst auf und meldet die geschalteten Lampen zurück. Bisher
  musste das Modell jede entity_id abschreiben und ließ bei vier Lampen
  gelegentlich eine aus.
- `get_state` fragt mit `domain` und `area` alle Lichter eines Bereichs ab.
- Türkontakte und Elektroautos sind im Training: „Ist die Balkontür offen?“
  fragt den Kontakt der Balkontür ab, nicht die Wohnungstür, „Wie weit kommt
  das Auto noch?“ die Reichweite statt des Ladestands.
- Außentemperatur neben Vorhersage-, Taupunkt- und Gefühlte-Temperatur-Sensoren:
  „Wie warm ist es draußen?“ liest den Außensensor, „Wie warm wird es
  morgen?“ die Vorhersage.
- Korrektur „Nein, ich meinte im Esszimmer“ schaltet alle Küchenlampen aus,
  nicht nur eine.
- Add-on: Ein Update auf das neue Standardmodell löscht die Modelldateien
  früherer Versionen in `/share/ha_llm/` (bisher blieben 1,9 GB pro Version
  liegen).
- Integration und Modell gehören zusammen: Integration 0.7.0 braucht Modell
  v11, ältere Modelle kennen `area` nicht.

## 0.6.2

- Integration: „Neu konfigurieren“ lädt den Eintrag über dessen
  Update-Listener neu. HA warnte bisher, weil der Ablauf zusätzlich selbst neu
  lud; ab HA 2026.12 wäre das ein Fehler.

## 0.6.1

- Integration: Die Server-URL lässt sich über „Neu konfigurieren“ am Eintrag
  ändern, etwa beim Wechsel vom Add-on auf llama-server auf einem GPU-Rechner.
  Die Integration prüft die neue URL über `/health`, bevor sie sie übernimmt.
  Agent und Optionen bleiben erhalten. Bisher musste der Eintrag gelöscht und
  neu angelegt werden.
- Add-on und Modell (v7) sind unverändert.

## 0.6.0

- Neues Modell v7 (`qwen2.5-3b-ha-v7-Q4_K_M.gguf`), neu trainiert ab
  Qwen2.5-3B-Instruct mit 8.000 Beispielen und 2,5 Epochen.
- Räume mit teilweise nicht erreichbaren Lampen: „Ist das Licht in der Küche
  an?“ nennt die erreichbaren Lampen mit ihrem Zustand und die übrigen als
  nicht erreichbar. Bisher antwortete das Modell „Alle Lichter sind aus“.
- „Mach das Licht in der Küche an“ schaltet bei teilweise nicht erreichbaren
  Lampen die übrigen und nennt beide Teile.
- Besteht der Nutzer mehrmals auf einem Gerät, das nicht in der Liste steht,
  lehnt das Modell jedes Mal ab, statt eine entity_id zu erfinden.
- Modus-Schalter der Heizung: „Aktiviere den Abwesenheitsmodus der Heizung“
  schaltet den Schalter „Heizung … Abwesenheitsmodus“, nicht das Thermostat.
- „das Licht im Flur“ meint immer alle Lampen im Raum. Nur der Name allein
  („Mach Küche an“) meint die Lampe, die so heißt wie der Raum.
- Bekannte Schwäche: Fragen nach Türkontakten („Ist die Balkontür im
  Wohnzimmer offen?“) treffen je nach Geräteliste manchmal den falschen
  Kontakt.

## 0.5.0

- Neues Modell v4 (`qwen2.5-3b-ha-v4-Q4_K_M.gguf`), neu trainiert ab
  Qwen2.5-3B-Instruct mit 8.000 Beispielen und 2,5 Epochen.
- „Licht in der Küche“ schaltet alle Lampen im Raum, auch wenn eine Lampe
  selbst „Küche“ heißt und daneben „Küche Links“ und „Küche Rechts“ stehen.
- Mehrere Bewässerungen: „Bewässere die Balkonpflanzen“ öffnet das Ventil auf
  dem Balkon und startet nicht die Gartenbewässerung.
- Teilerfolge nennen beide Teile: „Gartenbeleuchtung ist gerade nicht
  erreichbar. Den Rest habe ich erledigt: Flur eingeschaltet.“
- Kontrollfragen wie „Ist es jetzt an?“ beginnen mit „Ja“ oder „Nein“, passend
  zum Zustand.
- Kennt Gerätestatus wie `in_use`, `program_ended` oder `not_connected`.
- Integration: Enthält eine Aktion mehrere Geräte und ist eines davon nicht
  erreichbar, schaltet sie die übrigen und meldet nur dieses als Fehler.
  Bisher schlug die ganze Aktion fehl.

## 0.4.0

- Neues Modell v3 (`qwen2.5-3b-ha-v3-Q4_K_M.gguf`), neu trainiert ab
  Qwen2.5-3B-Instruct.
- Gespräche über mehrere Runden: „Und wieder aus“, „Welches Licht hast du
  ausgeschaltet?“, „Nein, ich meinte im Esszimmer“, „Mach das rückgängig“.
- Ist ein Gerät nicht erreichbar, meldet das Modell den Fehler, statt Erfolg
  zu behaupten. Bei mehreren Befehlen meldet es, dass nur ein Teil geklappt hat.
- Geräte, die nicht in der Liste stehen, lehnt das Modell ab, statt ein
  ähnliches Gerät zu schalten. Gibt es das Gerät in einem anderen Raum, nennt
  es diesen.
- Unterscheidet Status-Sensor und Steckdose eines Geräts („Trockner“ und
  „Trockner Power“) und kennt Markise, Mähroboter, Ventile, Gruppen und
  Buttons.
- Leere Optionen `model_url` und `model_sha256` laden das Modell, das zur
  Add-on-Version gehört. Ein Add-on-Update bringt damit auch das neue Modell.
  Ältere Standardmodelle werden nach dem Download entfernt.

## 0.3.1

- Warnung im Log und in Home Assistant, wenn die CPU kein AVX2 unterstützt.

## 0.3.0

- Erste Version mit Download von Hugging Face und SHA-256-Prüfung.
