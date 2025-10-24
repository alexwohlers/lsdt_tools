#!/usr/bin/env python3
"""
FFT-Messdaten-Analyse: Führt Fast Fourier Transform auf Messdaten durch.

Liest CSV-Dateien aus dem measurements/ Verzeichnis, berechnet die FFT
und speichert die Ergebnisse (Frequenz, Magnitude, Phase) im fft/ Verzeichnis.

Beispiele:
  - Einfache FFT:
      python fft_measurements.py --file measurements/sample.csv
  - Mit Fenster-Funktion:
      python fft_measurements.py --file measurements/sinus.csv --window hann
  - Nur positive Frequenzen (einseitig):
      python fft_measurements.py --file measurements/data.csv --onesided
  - Mit dB-Skala für Magnitude:
      python fft_measurements.py --file measurements/signal.csv --db
  - Ausgabedatei explizit angeben:
      python fft_measurements.py --file measurements/input.csv --out fft/output.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    print("Fehler: numpy ist nicht installiert.")
    print("Bitte installieren mit: pip install numpy")
    sys.exit(1)


def detect_csv_format(file_path: str, delimiter: str = ",") -> str:
    """
    Erkennt das CSV-Format anhand der Spaltenüberschriften.
    Rückgabe: 'relative', 'iso', 'epoch', 'none'
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if header is None:
            return "none"
        
        header_lower = [col.lower() for col in header]
        
        if "timestamp_iso" in header_lower:
            return "iso"
        elif "timestamp_epoch" in header_lower:
            return "epoch"
        elif "t" in header_lower:
            return "relative"
        elif len(header) == 2 and "index" in header_lower and "value" in header_lower:
            return "none"
        else:
            if len(header) >= 3:
                return "relative"
            else:
                return "none"


def read_csv_signal(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, float]:
    """
    Liest CSV-Datei und extrahiert Signal und Abtastrate.
    
    Returns:
        signal: numpy array mit Messwerten
        fs: Abtastrate in Hz (geschätzt aus Zeitdifferenzen oder aus Metadaten)
    """
    csv_format = detect_csv_format(file_path, delimiter)
    
    time_values = []
    signal_values = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        # Header überspringen falls vorhanden
        if header is None or csv_format == "none":
            f.seek(0)
            if header is not None and "index" in [h.lower() for h in header]:
                next(reader)  # Header überspringen
        
        for row in reader:
            try:
                if csv_format == "none":
                    if len(row) >= 2:
                        time_values.append(float(row[0]))  # index als Zeit
                        signal_values.append(float(row[-1]))
                    elif len(row) == 1:
                        time_values.append(len(time_values))
                        signal_values.append(float(row[0]))
                
                elif csv_format == "relative":
                    if len(row) >= 3:
                        time_values.append(float(row[1]))  # t
                        signal_values.append(float(row[2]))  # value
                
                elif csv_format in ("iso", "epoch"):
                    if len(row) >= 3:
                        # Für FFT nur relative Zeit wichtig
                        signal_values.append(float(row[2]))
            
            except (ValueError, IndexError):
                continue
    
    signal = np.array(signal_values)
    
    # Abtastrate schätzen
    if len(time_values) > 1 and csv_format in ("relative", "none"):
        time_array = np.array(time_values)
        dt = np.mean(np.diff(time_array))  # Durchschnittliche Zeitdifferenz
        if dt > 0:
            fs = 1.0 / dt
        else:
            # Fallback: annehmen dass Index = Sample bei fs=1
            fs = 1.0
    else:
        # Keine Zeitinformation: fs = 1 Hz annehmen
        fs = 1.0
        print(f"Warnung: Keine Zeitinformation gefunden, nehme fs=1.0 Hz an")
    
    return signal, fs


def apply_window(signal: np.ndarray, window_type: str = "none") -> np.ndarray:
    """
    Wendet eine Fenster-Funktion auf das Signal an.
    
    Verfügbare Fenster: none, hann, hamming, blackman, bartlett, kaiser
    """
    if window_type == "none" or window_type is None:
        return signal
    
    n = len(signal)
    
    if window_type == "hann":
        window = np.hanning(n)
    elif window_type == "hamming":
        window = np.hamming(n)
    elif window_type == "blackman":
        window = np.blackman(n)
    elif window_type == "bartlett":
        window = np.bartlett(n)
    elif window_type == "kaiser":
        window = np.kaiser(n, beta=8.6)  # Beta-Wert für gute Seitenkeulen-Unterdrückung
    else:
        raise ValueError(f"Unbekanntes Fenster: {window_type}")
    
    return signal * window


def compute_fft(
    signal: np.ndarray,
    fs: float,
    window: str = "none",
    onesided: bool = True,
    db_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet die FFT eines Signals.
    
    Returns:
        frequencies: Frequenz-Array in Hz
        magnitude: Magnitude des FFT-Spektrums
        phase: Phase des FFT-Spektrums in Radiant
    """
    # Fenster anwenden
    windowed_signal = apply_window(signal, window)
    
    # FFT berechnen
    fft_result = np.fft.fft(windowed_signal)
    n = len(signal)
    
    # Frequenz-Bins
    frequencies = np.fft.fftfreq(n, d=1/fs)
    
    # Einseitig (nur positive Frequenzen) oder zweiseitig
    if onesided:
        # Nur positive Frequenzen
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        fft_result = fft_result[positive_freq_idx]
        
        # Magnitude verdoppeln (außer DC und Nyquist) für einseitige Darstellung
        magnitude = np.abs(fft_result) / n  # Normierung
        magnitude[1:-1] *= 2  # Faktor 2 für alle außer DC und Nyquist
    else:
        # Zweiseitiges Spektrum
        magnitude = np.abs(fft_result) / n
    
    # Phase berechnen
    phase = np.angle(fft_result)
    
    # dB-Skala falls gewünscht
    if db_scale:
        # Vermeidung von log(0)
        magnitude = 20 * np.log10(np.maximum(magnitude, 1e-10))
    
    return frequencies, magnitude, phase


def save_fft_results(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    output_path: str,
    delimiter: str = ",",
    db_scale: bool = False,
) -> None:
    """
    Speichert FFT-Ergebnisse als CSV.
    """
    # Ausgabeverzeichnis erstellen falls nötig
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        
        # Header
        mag_label = "magnitude_db" if db_scale else "magnitude"
        writer.writerow(["frequency_hz", mag_label, "phase_rad"])
        
        # Daten
        for freq, mag, phase in zip(frequencies, magnitude, phase):
            writer.writerow([freq, mag, phase])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Führt FFT auf Messdaten durch und speichert Ergebnisse als CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--file",
        required=True,
        help="Pfad zur Eingabe-CSV-Datei (z.B. measurements/sample.csv)",
    )
    p.add_argument(
        "--out",
        help="Pfad zur Ausgabe-CSV-Datei. Falls nicht angegeben: fft/<dateiname>.csv",
    )
    p.add_argument(
        "--window",
        choices=["none", "hann", "hamming", "blackman", "bartlett", "kaiser"],
        default="none",
        help="Fenster-Funktion zur Spektralleckage-Reduktion",
    )
    p.add_argument(
        "--onesided",
        action="store_true",
        default=True,
        help="Nur positive Frequenzen (einseitiges Spektrum)",
    )
    p.add_argument(
        "--twosided",
        action="store_true",
        help="Vollständiges Spektrum (negative + positive Frequenzen)",
    )
    p.add_argument(
        "--db",
        action="store_true",
        help="Magnitude in dB (20*log10) statt linear",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="CSV-Trennzeichen für Ein- und Ausgabe",
    )
    p.add_argument(
        "--fs",
        type=float,
        help="Abtastrate in Hz überschreiben (falls Autodetektion fehlschlägt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Eingabedatei prüfen
    input_file = args.file
    if not os.path.exists(input_file):
        # Versuchen im measurements/ Verzeichnis
        measurements_path = os.path.join("measurements", args.file)
        if os.path.exists(measurements_path):
            input_file = measurements_path
        else:
            print(f"Fehler: Datei '{args.file}' nicht gefunden.")
            sys.exit(1)
    
    # Ausgabedatei bestimmen
    if args.out:
        output_file = args.out
    else:
        # Gleicher Name, aber im fft/ Verzeichnis
        input_name = Path(input_file).stem  # Dateiname ohne Erweiterung
        output_file = os.path.join("fft", f"{input_name}.csv")
    
    # Signal einlesen
    try:
        print(f"Lese Datei: {input_file}")
        signal, fs_detected = read_csv_signal(input_file, args.delimiter)
        
        # Abtastrate überschreiben falls angegeben
        fs = args.fs if args.fs is not None else fs_detected
        
        print(f"Signal-Länge: {len(signal)} Samples")
        print(f"Abtastrate: {fs:.2f} Hz")
        print(f"Dauer: {len(signal)/fs:.3f} s")
        
        if len(signal) == 0:
            print("Fehler: Keine Daten im Signal gefunden.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # FFT berechnen
    try:
        onesided = args.onesided and not args.twosided
        
        print(f"Berechne FFT...")
        print(f"  Fenster: {args.window}")
        print(f"  Spektrum: {'einseitig' if onesided else 'zweiseitig'}")
        print(f"  Skala: {'dB' if args.db else 'linear'}")
        
        frequencies, magnitude, phase = compute_fft(
            signal=signal,
            fs=fs,
            window=args.window,
            onesided=onesided,
            db_scale=args.db,
        )
        
        print(f"FFT-Ergebnis:")
        print(f"  Frequenzauflösung: {frequencies[1]-frequencies[0]:.4f} Hz")
        print(f"  Frequenzbereich: {frequencies[0]:.2f} Hz bis {frequencies[-1]:.2f} Hz")
        print(f"  Anzahl Frequenz-Bins: {len(frequencies)}")
    
    except Exception as e:
        print(f"Fehler bei FFT-Berechnung: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Ergebnisse speichern
    try:
        save_fft_results(
            frequencies=frequencies,
            magnitude=magnitude,
            phase=phase,
            output_path=output_file,
            delimiter=args.delimiter,
            db_scale=args.db,
        )
        print(f"\nErgebnisse gespeichert: {output_file}")
    
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
