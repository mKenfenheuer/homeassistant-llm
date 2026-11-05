# Changelog

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
