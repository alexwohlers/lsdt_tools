#!/usr/bin/env python3
"""
Plot FFT 3D: 3D-Darstellung von Frequenzkomponenten im Zeitbereich.

Zeigt für jede Frequenz aus dem FFT-Spektrum das entsprechende Zeitsignal
(Amplitude * cos(2πft + φ)) in 3D. Die Z-Achse zeigt die Frequenz.

Beispiele:
  - Einfache 3D-Darstellung:
      python plot_fft_3d.py sinus_1hz
  - Mit Frequenzlimit:
      python plot_fft_3d.py mixed_signal --fmax 20
  - Mit Zeitlimit:
      python plot_fft_3d.py data --tmax 1.0
  - Mit Titel:
      python plot_fft_3d.py signal --title "Frequenzkomponenten"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
except ImportError as e:
    print(f"Fehler: Benötigte Bibliothek nicht installiert: {e}")
    print("Bitte installieren mit: pip install matplotlib numpy")
    sys.exit(1)


def read_fft_data(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Liest FFT-Daten aus CSV.
    
    Returns:
        frequencies, magnitude, phase (in Radiant)
    """
    frequencies = []
    magnitude = []
    phase = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        # Header prüfen ob Phase-Spalte vorhanden
        has_phase = header and len(header) >= 3
        
        for row in reader:
            try:
                if len(row) >= 2:
                    frequencies.append(float(row[0]))
                    magnitude.append(float(row[1]))
                    if has_phase and len(row) >= 3:
                        phase.append(float(row[2]))
                    else:
                        phase.append(0.0)  # Fallback: Phase = 0
            except (ValueError, IndexError):
                continue
    
    return np.array(frequencies), np.array(magnitude), np.array(phase)


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
            for idx, row in enumerate(reader):
                try:
                    if len(row) >= 2:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[-1]))
                    elif len(row) == 1:
                        x_values.append(idx * 0.01)
                        y_values.append(float(row[0]))
                except (ValueError, IndexError):
                    continue
    
    return np.array(x_values), np.array(y_values)


def plot_frequency_components_3d(
    time_file: str,
    fft_file: str,
    title: Optional[str] = None,
    fmax: Optional[float] = None,
    tmax: Optional[float] = None,
    min_magnitude: float = 0.01,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    delimiter: str = ",",
) -> None:
    """
    Plottet Frequenzkomponenten als 3D-Signale.
    
    Für jede signifikante Frequenz wird das Signal A*cos(2πft + φ) berechnet
    und bei der entsprechenden Frequenz auf der Z-Achse geplottet.
    
    Args:
        time_file: Pfad zur Zeitbereich-CSV (für Original-Signal)
        fft_file: Pfad zur FFT-CSV (Frequenz, Amplitude, Phase)
        title: Plot-Titel
        fmax: Maximale Frequenz zum Plotten
        tmax: Maximale Zeit für Zeitachse
        min_magnitude: Minimale Amplitude für sichtbare Komponenten
        figsize: Figur-Größe
        save_path: Speicherpfad
        delimiter: CSV-Trennzeichen
    """
    # FFT-Daten laden
    frequencies, magnitude, phase = read_fft_data(fft_file, delimiter)
    
    # Frequenz-Limit anwenden
    if fmax is not None:
        mask = frequencies <= fmax
        frequencies = frequencies[mask]
        magnitude = magnitude[mask]
        phase = phase[mask]
    
    # Nur signifikante Komponenten (Magnitude > Schwellwert)
    mask = magnitude > min_magnitude
    frequencies = frequencies[mask]
    magnitude = magnitude[mask]
    phase = phase[mask]
    
    if len(frequencies) == 0:
        print("Keine signifikanten Frequenzkomponenten gefunden.")
        print(f"Tipp: Reduzieren Sie --min-magnitude (aktuell: {min_magnitude})")
        return
    
    print(f"Plotte {len(frequencies)} Frequenzkomponenten...")
    
    # Zeitbereich bestimmen
    t_time, signal_orig = read_time_domain_data(time_file, delimiter)
    if tmax is not None:
        t_max_val = min(tmax, t_time.max())
        mask_time = t_time <= t_max_val
        t_time = t_time[mask_time]
        signal_orig = signal_orig[mask_time]
    else:
        t_max_val = t_time.max()
    
    # Zeitvektor für Signalberechnung (hochauflösend)
    t = np.linspace(0, t_max_val, 10000)
    
    # 3D Plot erstellen
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Zusätzliche flache Subplots unten links (Zeitbereich) und unten rechts (Frequenzbereich)
    # Position: [links, unten, Breite, Höhe] als Anteil der Figure
    ax_time_flat = fig.add_axes([0.06, 0.04, 0.40, 0.11])  # unten links (halb so hoch)
    ax_freq_flat = fig.add_axes([0.54, 0.04, 0.40, 0.11])  # unten rechts (halb so hoch)

    # Zeitbereich: Originalsignal (flach)
    ax_time_flat.plot(t_time, signal_orig, color='b', linewidth=1.2)
    ax_time_flat.set_title('Zeitbereich', fontsize=10)
    ax_time_flat.set_xlabel('Zeit [s]', fontsize=9)
    ax_time_flat.set_ylabel('Amplitude', fontsize=9)
    ax_time_flat.tick_params(axis='both', labelsize=8)
    ax_time_flat.grid(True, alpha=0.2)

    # Frequenzbereich: Amplitudenspektrum (flach)
    # Balkenbreite aus Abtastrate und Signallänge berechnen
    n = len(signal_orig)
    dt = t_time[1] - t_time[0]  # Zeitauflösung
    fs = 1.0 / dt  # Abtastrate
    freq_resolution = fs / n  # Frequenzauflösung aus Abtastrate und Länge
    bar_width = freq_resolution * 0.8
    ax_freq_flat.bar(frequencies, magnitude, width=bar_width, color='r', edgecolor='crimson', alpha=0.7)
    ax_freq_flat.set_title('Frequenzbereich', fontsize=10)
    ax_freq_flat.set_xlabel('Frequenz [Hz]', fontsize=9)
    ax_freq_flat.set_ylabel('Magnitude', fontsize=9)
    ax_freq_flat.tick_params(axis='both', labelsize=8)
    ax_freq_flat.grid(True, alpha=0.2)
    # X-Achse unten rechts: exakt wie oben von 0 bis freq_max
    freq_max_plot = fmax if fmax is not None else frequencies.max()
    ax_freq_flat.set_xlim(0, freq_max_plot)
    
    # 1. WAND: Zeitbereich-Signal an der Stirnseite (bei Y=0, also niedrigste Frequenz)
    # Zeigt das Originalsignal als 2D-Projektion - BLAU
    freq_max_plot = fmax if fmax is not None else frequencies.max()
    x_wall_time = t_time
    y_wall_time = np.zeros_like(t_time)  # Bei Y=0 (vorne)
    z_wall_time = signal_orig
    ax.plot(x_wall_time, y_wall_time, z_wall_time, 'b-', linewidth=2.5, alpha=0.9, 
            label='Zeitbereich (Original)')
    # Fülle die Fläche unter dem Signal
    ax.plot(x_wall_time, y_wall_time, np.zeros_like(z_wall_time), 'b-', linewidth=0.5, alpha=0.3)
    for i in range(0, len(x_wall_time), max(1, len(x_wall_time)//50)):
        ax.plot([x_wall_time[i], x_wall_time[i]], [0, 0], 
                [0, z_wall_time[i]], 'b-', linewidth=0.5, alpha=0.2)
    
    # 2. WAND: Frequenzbereich-Spektrum an der Stirnseite (bei X=0, Anfang der Zeit)
    # Zeigt das Amplitudenspektrum als dünne 2D-Balken - ROT
    
    # Frequenzspektrum als dünne vertikale Linien (2D-Balken)
    for freq, mag in zip(frequencies, magnitude):
        ax.plot([0, 0], [freq, freq], [0, mag], color='r', linewidth=2.5, alpha=0.9, solid_capstyle='butt')
    
    # Label für Legende (nur einmal)
    ax.plot([0], [frequencies[0]], [magnitude[0]], 'r-', linewidth=2.5, alpha=0.9,
            label='Frequenzbereich (FFT)')

    
    # Für jede Frequenz das entsprechende Signal plotten (im 3D-Raum) - SCHWARZ
    for i, (freq, amp, ph) in enumerate(zip(frequencies, magnitude, phase)):
        # Signal berechnen: A * cos(2πft + φ)
        signal = amp * np.cos(2 * np.pi * freq * t + ph)
        
        # Bei konstanter Frequenz (Y-Achse) plotten, Amplitude auf Z-Achse
        y = np.full_like(t, freq)
        
        ax.plot(t, y, signal, color='k', linewidth=1.2, alpha=0.6)
    
    # Hauptfrequenz bestimmen (für Info-Text)
    main_freq_idx = np.argmax(magnitude)
    main_freq = frequencies[main_freq_idx]
    
    # Achsenbeschriftungen
    ax.set_xlabel('Zeit [s]', fontsize=11, labelpad=10)
    ax.set_ylabel('Frequenz [Hz]', fontsize=11, labelpad=10)
    ax.set_zlabel('Amplitude', fontsize=11, labelpad=25)
    
    # Frequenz-Achse (Y) von 0 bis fmax/max skalieren (umgekehrt)
    freq_max = fmax if fmax is not None else frequencies.max()
    ax.set_ylim(freq_max, 0)  # Umgekehrte Richtung: max -> 0
    
    # Ansichtswinkel optimieren
    ax.view_init(elev=20, azim=135)
    
    # Legende
    ax.legend(loc='upper left', fontsize=9)
    
    # Titel
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        filename = Path(time_file).stem
        ax.set_title(f'Frequenzkomponenten im Zeitbereich: {filename}', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # Layout anpassen: unten Platz für die flachen Subplots lassen
    fig.subplots_adjust(bottom=0.10, top=0.93)
    
    # Speichern
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(save_path, dpi=300)  # bbox_inches='tight' entfernt, damit Z-Label sichtbar bleibt
    print(f"3D-Plot gespeichert: {save_path}")
    print(f"  Frequenzbereich: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
    print(f"  Zeitbereich: 0 - {t_max_val:.2f} s")
    print(f"  Hauptfrequenz: {main_freq:.2f} Hz (Originalsignal hier gezeichnet)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plottet Frequenzkomponenten als 3D-Signale im Zeitbereich.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "name",
        help="Dateiname ohne Endung (z.B. 'sample'). Sucht in measurements/ und fft/",
    )
    p.add_argument("--title", help="Titel des Plots")
    p.add_argument("--fmax", type=float, help="Maximale Frequenz in Hz")
    p.add_argument("--tmax", type=float, help="Maximale Zeit in Sekunden")
    p.add_argument("--min-magnitude", type=float, default=0.01, 
                   help="Minimale Amplitude für sichtbare Komponenten")
    p.add_argument("--figsize", nargs=2, type=int, default=[14, 10], 
                   help="Größe der Figur (Breite Höhe)")
    p.add_argument("--save", dest="save_path", 
                   help="Dateiname für Plot (ohne Pfad, wird in plot_fft_3d/ gespeichert)")
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
            save_path = os.path.join("plot_fft_3d", args.save_path)
        else:
            save_path = args.save_path
    else:
        save_path = os.path.join("plot_fft_3d", f"{args.name}_3d.png")
    
    try:
        plot_frequency_components_3d(
            time_file=time_file,
            fft_file=fft_file,
            title=args.title,
            fmax=args.fmax,
            tmax=args.tmax,
            min_magnitude=args.min_magnitude,
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
