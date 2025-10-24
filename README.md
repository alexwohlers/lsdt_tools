# Me## Features

- Funktion als Ausdruck in `t` (Zeit in Sekunden) und op## Parameterübersicht

- `name` (positional, optional): Dateiname ohne Endung (Standard: "data"). Wird automatisch in measurements/ gespeichert
- `--func` (str): Funktionsausdruck in `t`, optional `n`
- `--fs` (float): Abtastrate in Hz (Standard: 100)
- `--duration` (float): Dauer in Sekunden
- `--samples` (int): Anzahl Samples (überschreibt `--duration`)
- `--amplitude` (float): Multiplikative Amplitude
- `--offset` (float): Additiver Offset
- `--noise` (float): Rausch-Std.-Abweichung (Gaussian)
- `--seed` (int): Zufalls-Seed (reproduzierbar)
- `--timestamp` (`none|relative|iso|epoch`): Zeitstempelspalte
- `--delimiter` (str): CSV-Delimiter, z. B. `,` oder `;`
- `--no-header` (Flag): Keine Kopfzeile
- `--out` (str): Ausgabedatei (überschreibt automatische Pfaderstellung aus `name`)
- `--encoding` (str): Datei-Encoding (Standard UTF-8) (`none|relative|iso|epoch`): Zeitstempelspalte
- `--delimiter` (str): CSV-Delimiter, z. B. `,` oder `;`
- `--no-header` (Flag): Keine Kopfzeile
- `--out` (str): Ausgabedatei (Standard: `measurements/data.csv`, Verzeichnis wird automatisch erstellt)
- `--encoding` (str): Datei-Encoding (Standard UTF-8)l `n` (Sample-Index)
- Abtastrate (`--fs`) und Dauer (`--duration`) oder exakte Sample-Anzahl (`--samples`)
- Amplitude, Offset, additiv gaussches Rauschen (`--noise`) und Seed
- Zeitstempel-Spalte: keine, relative Sekunden, ISO 8601 oder Epoch-Seconds
- CSV-Delimiter frei wählbar (z. B. `,` oder `;`)
- Automatische Ausgabe in das Unterverzeichnis `measurements/` (wird bei Bedarf erstellt)-Generator (CSV)

Ein kleines Python-Tool, das Werte einer angegebenen Funktion über der Zeit erzeugt und als CSV-Datei speichert. Praktisch zum Simulieren von Messdaten (z. B. Sinus, Rampen, gemischte Signale) für Tests und FFTs.

## Features

- Funktion als Ausdruck in `t` (Zeit in Sekunden) und optional `n` (Sample-Index)
- Abtastrate (`--fs`) und Dauer (`--duration`) oder exakte Sample-Anzahl (`--samples`)
- Amplitude, Offset, additiv gaussches Rauschen (`--noise`) und Seed
- Zeitstempel-Spalte: keine, relative Sekunden, ISO 8601 oder Epoch-Seconds
- CSV-Delimiter frei wählbar (z. B. `,` oder `;`)

Verfügbare Funktionen/Konstanten im Ausdruck:
`sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, exp, log, log10, sqrt, floor, ceil, abs, min, max, clamp(x,lo,hi), sign(x), pi, tau, e`.

> Sicherheit: Ausdrücke werden in einer stark eingeschränkten Umgebung ausgewertet (ohne Builtins), aber bitte nur vertrauenswürdige Ausdrücke verwenden.

## Nutzung

Voraussetzungen: Python 3.x (keine zusätzlichen Pakete nötig)

In Windows PowerShell ausführen (im Ordner dieses Repos):

```powershell
# 1 Hz Sinus, 5 s bei 100 Hz - Datei wird als measurements/sinus_1hz.csv gespeichert
python .\generate_measurements.py sinus_1hz --func "sin(2*pi*1*t)" --fs 100 --duration 5

# Mehrkomponentensignal mit Rauschen und ISO-Zeitstempel
python .\generate_measurements.py signal_mixed --func "0.5*sin(2*pi*0.2*t) + 0.2*cos(2*pi*1.5*t)" --fs 50 --duration 10 --noise 0.05 --timestamp iso

# Rechteck über sign(sin(.)) bei 2 Hz
python .\generate_measurements.py square --func "sign(sin(2*pi*2*t))" --fs 200 --duration 2

# Exakt 1000 Samples statt Dauer
python .\generate_measurements.py damped --func "exp(-0.5*t)*sin(2*pi*3*t)" --fs 500 --samples 1000

# Mit Semikolon als Trennzeichen und ohne Header
python .\generate_measurements.py fast --func "sin(2*pi*5*t)" --fs 1000 --duration 0.2 --delimiter ';' --no-header

# Ohne Name-Parameter (Standard: data.csv)
python .\generate_measurements.py --func "cos(2*pi*t)" --fs 100 --duration 1

# Mit explizitem Ausgabepfad (überschreibt automatische Benennung)
python .\generate_measurements.py --func "sin(2*pi*t)" --fs 100 --duration 1 --out custom/path.csv
```

Die erzeugten CSV-Dateien enthalten – je nach `--timestamp` – folgende Spalten:
- `none`: `index, value`
- `relative`: `index, t, value` (t in Sekunden ab 0)
- `iso`: `index, timestamp_iso, value` (UTC ISO 8601)
- `epoch`: `index, timestamp_epoch, value` (Sekunden seit Unix-Epoche)

## Beispiele für Ausdrücke

- Sinus: `sin(2*pi*f*t)`
- Kosinus: `cos(2*pi*3*t)`
- Gedämpfter Sinus: `exp(-0.5*t)*sin(2*pi*3*t)`
- Dreieck (ein einfaches, nähert sich an): `asin(sin(2*pi*1*t))`
- Rechteck: `sign(sin(2*pi*2*t))`
- Rampe: `t` oder skaliert `0.1*t`
- Clamping: `clamp(sin(2*pi*t), -0.5, 0.5)` 

Du kannst zusätzlich `--amplitude`, `--offset` und `--noise` kombinieren, z. B. `--amplitude 2 --offset 1 --noise 0.05`.

## Parameterübersicht

- `--func` (str): Funktionsausdruck in `t`, optional `n`
- `--fs` (float): Abtastrate in Hz (Standard: 100)
- `--duration` (float): Dauer in Sekunden
- `--samples` (int): Anzahl Samples (überschreibt `--duration`)
- `--amplitude` (float): Multiplikative Amplitude
- `--offset` (float): Additiver Offset
- `--noise` (float): Rausch-Std.-Abweichung (Gaussian)
- `--seed` (int): Zufalls-Seed (reproduzierbar)
- `--timestamp` (`none|relative|iso|epoch`): Zeitstempelspalte
- `--delimiter` (str): CSV-Delimiter, z. B. `,` oder `;`
- `--no-header` (Flag): Keine Kopfzeile
- `--out` (str): Ausgabedatei
- `--encoding` (str): Datei-Encoding (Standard UTF-8)

## Hinweise

- Bei `--samples` wird die Anzahl der Zeilen exakt, bei `--duration` ergibt sich `samples ≈ duration * fs` (gerundet).
- ISO-Zeitstempel sind in UTC. Relative Zeit startet immer bei `t=0`.
- Fehlerhafte Ausdrücke führen zu einer klaren Fehlermeldung mit t/n-Position.

---

## Messdaten visualisieren mit `plot_measurements.py`

Das Skript `plot_measurements.py` ermöglicht die grafische Darstellung der generierten CSV-Messdaten.

### Installation von matplotlib

```powershell
pip install matplotlib
```

### Verwendung

```powershell
# Einzelne Datei plotten (nur Dateiname, ohne Verzeichnis und Endung)
python .\plot_measurements.py sample

# Mit benutzerdefiniertem Titel und Beschriftungen
python .\plot_measurements.py sinus --title "Sinus 1 Hz" --xlabel "Zeit [s]" --ylabel "Amplitude [V]"

# Mehrere Dateien überlagert anzeigen
python .\plot_measurements.py sinus cosinus --title "Sinus vs. Kosinus"

# Als PNG-Datei speichern (statt anzeigen)
python .\plot_measurements.py data --save output_plot.png

# Mit Markern und gestrichelten Linien
python .\plot_measurements.py sample --marker o --linestyle "--"

# Ohne Grid
python .\plot_measurements.py data --no-grid
```

### Parameter für plot_measurements.py

| Parameter | Beschreibung |
|-----------|--------------|
| `files` (positional) | Dateiname(n) ohne Verzeichnis und Endung (z.B. `sinus` oder `data1 data2`) |
| `--title` | Titel des Plots |
| `--xlabel` | Beschriftung der X-Achse |
| `--ylabel` | Beschriftung der Y-Achse |
| `--grid` | Gitter anzeigen (Standard: aktiv) |
| `--no-grid` | Gitter ausblenden |
| `--marker` | Marker-Stil (`o`, `s`, `^`, `x`, `+`, `.`, etc.) |
| `--linestyle` | Linien-Stil (`-`, `--`, `-.`, `:`) |
| `--figsize` | Größe der Figur (Breite Höhe in Zoll), z.B. `--figsize 16 8` |
| `--save` | Plot als Datei speichern (z.B. `.png`, `.pdf`, `.svg`) statt anzeigen |
| `--delimiter` | CSV-Trennzeichen (Standard: `,`) |

### Funktionen

- **Automatische Format-Erkennung**: Erkennt `relative`, `iso`, `epoch` und `none` Zeitstempel automatisch
- **Mehrfach-Plots**: Mehrere Dateien können überlagert dargestellt werden
- **Zeitstempel-Formatierung**: ISO und Epoch-Zeitstempel werden automatisch formatiert
- **Export**: Speichern in PNG, PDF, SVG und andere Formate

### Workflow-Beispiel

```powershell
# 1. Messdaten generieren
python .\generate_measurements.py sinus_2hz --func "sin(2*pi*2*t)" --fs 100 --duration 5

# 2. Plotten und anzeigen
python .\plot_measurements.py sinus_2hz --title "Sinus 2 Hz Signal"

# 3. Als hochauflösendes PNG exportieren
python .\plot_measurements.py sinus_2hz --title "Sinus 2 Hz" --save sinus_plot.png
```

---

## FFT-Analyse mit `fft_measurements.py`

Das Skript `fft_measurements.py` führt eine Fast Fourier Transform (FFT) auf den Messdaten durch und speichert das Frequenzspektrum.

### Installation von numpy

```powershell
pip install numpy
```

### Verwendung

```powershell
# Einfache FFT
python .\fft_measurements.py --file measurements\sinus_2hz.csv

# Mit Hann-Fenster zur Reduktion von Spektralleckage
python .\fft_measurements.py --file measurements\data.csv --window hann

# Mit dB-Skala für Magnitude
python .\fft_measurements.py --file measurements\signal.csv --db

# Zweiseitiges Spektrum (negative + positive Frequenzen)
python .\fft_measurements.py --file measurements\data.csv --twosided

# Abtastrate manuell überschreiben
python .\fft_measurements.py --file measurements\data.csv --fs 1000

# Ausgabedatei explizit angeben
python .\fft_measurements.py --file measurements\input.csv --out fft\custom_output.csv
```

### Parameter für fft_measurements.py

| Parameter | Beschreibung |
|-----------|--------------|
| `--file` | Pfad zur Eingabe-CSV-Datei (aus measurements/) |
| `--out` | Pfad zur Ausgabe-CSV-Datei (Standard: `fft/<dateiname>.csv`) |
| `--window` | Fenster-Funktion: `none`, `hann`, `hamming`, `blackman`, `bartlett`, `kaiser` |
| `--onesided` | Nur positive Frequenzen (Standard: aktiv) |
| `--twosided` | Vollständiges Spektrum (negative + positive Frequenzen) |
| `--db` | Magnitude in dB (20*log10) statt linear |
| `--fs` | Abtastrate in Hz überschreiben (falls Autodetektion fehlschlägt) |
| `--delimiter` | CSV-Trennzeichen |

### Ausgabeformat

Die FFT-Ergebnisse werden als CSV gespeichert mit folgenden Spalten:

```csv
frequency_hz,magnitude,phase_rad
0.0,0.001,0.0
0.5,0.003,0.12
1.0,0.987,-1.57
...
```

Bei `--db` Option: `frequency_hz,magnitude_db,phase_rad`

### Fenster-Funktionen

Fenster-Funktionen reduzieren Spektralleckage (Spectral Leakage) bei nicht-periodischen Signalen:

- **none**: Kein Fenster (Rechteckfenster)
- **hann**: Gute Allzweck-Wahl, glatte Übergänge
- **hamming**: Ähnlich wie Hann, etwas bessere Frequenzauflösung
- **blackman**: Sehr gute Unterdrückung von Nebenkeulen
- **bartlett**: Dreiecksfenster, einfach
- **kaiser**: Flexibel einstellbar (Beta=8.6)

---

## Gegenüberstellung: Zeit- und Frequenzbereich mit `plot_fft_measurements.py`

Das Skript `plot_fft_measurements.py` visualisiert Messdaten und FFT-Ergebnisse nebeneinander.

### Verwendung

```powershell
# Einfache Gegenüberstellung
python .\plot_fft_measurements.py --file sinus_2hz.csv

# Mit Titel und Frequenzlimit (z.B. nur bis 50 Hz)
python .\plot_fft_measurements.py --file data.csv --title "Signal-Analyse" --fmax 50

# Mit logarithmischer Skala (dB)
python .\plot_fft_measurements.py --file signal.csv --log

# Phase als dritten Subplot anzeigen
python .\plot_fft_measurements.py --file sinus.csv --phase

# Als PNG speichern
python .\plot_fft_measurements.py --file data.csv --save analysis.png

# Größeres Plot-Format
python .\plot_fft_measurements.py --file signal.csv --figsize 20 8
```

### Parameter für plot_fft_measurements.py

| Parameter | Beschreibung |
|-----------|--------------|
| `--file` | Basis-Dateiname (z.B. `sample.csv`). Sucht in `measurements/` und `fft/` |
| `--title` | Titel des Plots |
| `--fmax` | Maximale Frequenz in Hz für FFT-Plot |
| `--log` | Logarithmische Skala (dB) für Magnitude |
| `--phase` | Phase als dritten Subplot anzeigen |
| `--figsize` | Größe der Figur (Breite Höhe), z.B. `--figsize 20 8` |
| `--save` | Speichern als Datei (z.B. `analysis.png`) |
| `--time-file` | Expliziter Pfad zur Zeitbereich-Datei |
| `--fft-file` | Expliziter Pfad zur FFT-Datei |
| `--delimiter` | CSV-Trennzeichen |

### Funktionen

- **Automatische Dateizuordnung**: Sucht automatisch passende Dateien in `measurements/` und `fft/`
- **Nebeneinander-Darstellung**: Zeitbereich links, Frequenzbereich rechts
- **Flexible Anzeige**: Optional mit Phase als drittem Subplot
- **Frequenzfilterung**: Nur relevanten Frequenzbereich anzeigen mit `--fmax`

---

## Kompletter Workflow: Von Messdaten bis zur FFT-Analyse

```powershell
# 1. Messdaten generieren (Sinus 5 Hz + 12 Hz, 100 Hz Abtastrate, 2 Sekunden)
python .\generate_measurements.py mixed_signal --func "sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)" --fs 100 --duration 2

# 2. FFT durchführen mit Hann-Fenster
python .\fft_measurements.py --file measurements\mixed_signal.csv --window hann

# 3. Gegenüberstellung plotten (nur bis 20 Hz)
python .\plot_fft_measurements.py --file mixed_signal.csv --fmax 20 --title "Zwei-Frequenz-Signal" --save mixed_analysis.png

# 4. (Optional) Nur Zeitbereich plotten
python .\plot_measurements.py mixed_signal --title "Zeitbereich"
```

Ergebnis: Das FFT-Spektrum zeigt deutliche Spitzen bei 5 Hz und 12 Hz!
