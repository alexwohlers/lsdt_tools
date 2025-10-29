#!/usr/bin/env python3
"""
Plot FFT 3D: Perspektivische Darstellung der FFT-Transformation.

Zeigt die Transformation vom Zeitbereich zum Frequenzbereich in einer
3D-perspektivischen Ansicht mit beiden Ebenen und Verbindungslinien.

Beispiele:
  - Einfache 3D-Darstellung:
      python plot_fft_3d.py sinus_1hz
  - Mit Frequenzlimit:
      python plot_fft_3d.py signal --fmax 10
  - Mit Titel:
      python plot_fft_3d.py data --title "FFT-Transformation"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
except ImportError as e:
    print(f"Fehler: Benötigte Bibliothek nicht installiert: {e}")
    print("Bitte installieren mit: pip install matplotlib numpy")
    sys.exit(1)


def detect_csv_format(file_path: str, delimiter: str = ",") -> str:
    """Erkennt das CSV-Format anhand der Spaltenüberschriften."""
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


def read_time_domain_data(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray]:
    """Liest Zeitbereich-Daten aus CSV."""
    csv_format = detect_csv_format(file_path, delimiter)
    
    x_values = []
    y_values = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if csv_format == "relative":
            for row in reader:
                try:
                    if len(row) >= 3:
                        x_values.append(float(row[1]))  # t
                        y_values.append(float(row[2]))  # value
                except (ValueError, IndexError):
                    continue
        else:
            # Fallback für andere Formate
            for idx, row in enumerate(reader):
                try:
                    if len(row) >= 2:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[-1]))
                    elif len(row) == 1:
                        x_values.append(idx * 0.01)  # Annahme: 100 Hz
                        y_values.append(float(row[0]))
                except (ValueError, IndexError):
                    continue
    
    return np.array(x_values), np.array(y_values)


def read_frequency_domain_data(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray]:
    """Liest Frequenzbereich-Daten (FFT-Ergebnisse) aus CSV."""
    frequencies = []
    magnitude = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        for row in reader:
            try:
                if len(row) >= 2:
                    frequencies.append(float(row[0]))
                    magnitude.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    
    return np.array(frequencies), np.array(magnitude)


def plot_3d_transformation(
    time_file: str,
    fft_file: str,
    title: Optional[str] = None,
    fmax: Optional[float] = None,
    tmax: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    delimiter: str = ",",
) -> None:
    """
    Plottet Zeit- und Frequenzbereich in 3D-Perspektive.
    """
    # Daten laden
    t_time, signal = read_time_domain_data(time_file, delimiter)
    
    # Zeitlimit anwenden
    if tmax is not None:
        mask = t_time <= tmax
        t_time = t_time[mask]
        signal = signal[mask]
    
    # Daten für bessere Darstellung sampeln (zu viele Punkte machen Plot unübersichtlich)
    max_time_points = 500
    
    if len(t_time) > max_time_points:
        step = len(t_time) // max_time_points
        t_time = t_time[::step]
        signal = signal[::step]
    
    # 3D Plot erstellen
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Zeitbereich: X-Achse normal (Null links, Maximum rechts)
    x_time = t_time  # Zeit normal
    y_time = np.zeros_like(t_time)  # Y=0
    z_signal = signal              # Signal normal
    ax.plot(x_time, y_time, z_signal, color='crimson', linewidth=2, label='Zeitbereich', zorder=10)

    # Frequenzbereich auf der anderen Achse unten (X=0)
    frequencies, magnitude = read_frequency_domain_data(fft_file, delimiter)
    # Frequenz-Limit anwenden
    if fmax is not None:
        mask = frequencies <= fmax
        frequencies = frequencies[mask]
        magnitude = magnitude[mask]
    x_freq = np.zeros_like(frequencies)  # X=0
    y_freq = frequencies                # Frequenz normal
    z_magnitude = magnitude             # Magnitude normal
    for y, z in zip(y_freq, z_magnitude):
        ax.plot([0, 0], [y, y], [0, z], color='steelblue', linewidth=2, alpha=0.8)

    # Zeit-Gitter
    time_min, time_max = t_time.min(), t_time.max()
    signal_min, signal_max = signal.min(), signal.max()
    signal_range = signal_max - signal_min
    for t in np.linspace(time_min, time_max, 10):
        ax.plot([t, t], [0, 0], [signal_min - 0.1*signal_range, signal_max + 0.1*signal_range], color='gray', linewidth=0.3, alpha=0.3)
    for amp in np.linspace(signal_min - 0.1*signal_range, signal_max + 0.1*signal_range, 8):
        ax.plot([time_min, time_max], [0, 0], [amp, amp], color='gray', linewidth=0.3, alpha=0.3)
    freq_min, freq_max = y_freq.min(), y_freq.max()
    mag_max = z_magnitude.max() if len(z_magnitude) > 0 else 1
    for f in np.linspace(freq_min, freq_max, 10):
        ax.plot([0, 0], [f, f], [0, mag_max * 1.1], color='gray', linewidth=0.3, alpha=0.3)
    for mag in np.linspace(0, mag_max * 1.1, 8):
        ax.plot([0, 0], [freq_min, freq_max], [mag, mag], color='gray', linewidth=0.3, alpha=0.3)

    # Achsenbeschriftungen
    ax.set_xlabel('Zeit [s]', fontsize=11, labelpad=10)
    ax.set_ylabel('Frequenz [Hz]', fontsize=11, labelpad=10)
    ax.set_zlabel('Amplitude / Magnitude', fontsize=11, labelpad=10)
    
    # Ansichtswinkel: von vorne-links oben
    ax.view_init(elev=25, azim=120)
    
    # Titel
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        filename = Path(time_file).stem
        ax.set_title(f'FFT-Transformation: {filename}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Speichern
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D-Plot gespeichert: {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plottet FFT-Transformation in 3D-Perspektive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "name",
        help="Dateiname ohne Endung (z.B. 'sample'). Sucht in measurements/ und fft/",
    )
    p.add_argument("--title", help="Titel des Plots")
    p.add_argument("--fmax", type=float, help="Maximale Frequenz in Hz für FFT-Plot")
    p.add_argument("--tmax", type=float, help="Maximale Zeit in Sekunden für Zeitbereich")
    p.add_argument("--figsize", nargs=2, type=int, default=[14, 10], help="Größe der Figur (Breite Höhe)")
    p.add_argument("--save", dest="save_path", help="Dateiname für Plot (ohne Pfad, wird in plot_fft/ gespeichert)")
    p.add_argument("--delimiter", default=",", help="CSV-Trennzeichen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Dateipfade konstruieren
    time_file = os.path.join("measurements", f"{args.name}.csv")
    fft_file = os.path.join("fft", f"{args.name}.csv")
    
    # Prüfen ob Dateien existieren
    if not os.path.exists(time_file):
        print(f"Fehler: Zeitbereich-Datei nicht gefunden: {time_file}")
        sys.exit(1)
    
    if not os.path.exists(fft_file):
        print(f"Fehler: FFT-Datei nicht gefunden: {fft_file}")
        print(f"Tipp: Führen Sie zuerst 'python fft_measurements.py {args.name}' aus")
        sys.exit(1)
    
    print(f"Zeitbereich: {time_file}")
    print(f"FFT-Daten:   {fft_file}")
    
    # Speicherpfad verarbeiten
    if args.save_path:
        if not os.path.dirname(args.save_path):
            save_path = os.path.join("plot_fft", args.save_path)
        else:
            save_path = args.save_path
    else:
        save_path = os.path.join("plot_fft", f"{args.name}_3d.png")
    
    try:
        plot_3d_transformation(
            time_file=time_file,
            fft_file=fft_file,
            title=args.title,
            fmax=args.fmax,
            tmax=args.tmax,
            figsize=tuple(args.figsize),
            save_path=save_path,
            delimiter=args.delimiter,
        )
    except Exception as exc:
        print(f"Fehler beim Plotten: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
