#!/usr/bin/env python3
"""
Plot FFT-Messdaten: Gegenüberstellung von Zeit- und Frequenzbereich.

Plottet Messdaten aus dem measurements/ Verzeichnis (Zeitbereich) und die
entsprechenden FFT-Ergebnisse aus dem fft/ Verzeichnis (Frequenzbereich)
nebeneinander zur Analyse.

Beispiele:
  - Einfache Gegenüberstellung:
      python plot_fft.py --file sample.csv
  - Mit Titel und Log-Skala im Frequenzbereich:
      python plot_fft.py --file sinus.csv --title "FFT-Analyse" --log
  - Nur bis 50 Hz anzeigen:
      python plot_fft.py --file data.csv --fmax 50
  - Frequenzbereich 10-100 Hz:
      python plot_fft.py --file data.csv --fmin 10 --fmax 100
  - Als PNG speichern:
      python plot_fft.py --file signal.csv --save analysis.png
  - Phase auch anzeigen (3 Subplots):
      python plot_fft.py --file sample.csv --phase
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
    import matplotlib.dates as mdates
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


def read_time_domain_data(file_path: str, delimiter: str = ",") -> Tuple[List, List, str]:
    """
    Liest Zeitbereich-Daten aus CSV.
    Returns: (x_values, y_values, format)
    """
    csv_format = detect_csv_format(file_path, delimiter)
    
    x_values = []
    y_values = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if header is None or csv_format == "none":
            f.seek(0)
            if header is not None and "index" in [h.lower() for h in header]:
                next(reader)
            
            for idx, row in enumerate(reader):
                try:
                    if len(row) >= 2:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[-1]))
                    elif len(row) == 1:
                        x_values.append(idx)
                        y_values.append(float(row[0]))
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "relative":
            for row in reader:
                try:
                    if len(row) >= 3:
                        x_values.append(float(row[1]))  # t
                        y_values.append(float(row[2]))  # value
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "iso":
            for row in reader:
                try:
                    if len(row) >= 3:
                        ts_str = row[1]
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        x_values.append(dt)
                        y_values.append(float(row[2]))
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "epoch":
            for row in reader:
                try:
                    if len(row) >= 3:
                        epoch = float(row[1])
                        dt = datetime.fromtimestamp(epoch)
                        x_values.append(dt)
                        y_values.append(float(row[2]))
                except (ValueError, IndexError):
                    continue
    
    return x_values, y_values, csv_format


def read_frequency_domain_data(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Liest Frequenzbereich-Daten (FFT-Ergebnisse) aus CSV.
    Returns: (frequencies, magnitude, phase, is_db)
    """
    frequencies = []
    magnitude = []
    phase = []
    is_db = False
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        # Prüfen ob dB-Skala
        if header and any("db" in h.lower() for h in header):
            is_db = True
        
        for row in reader:
            try:
                if len(row) >= 3:
                    frequencies.append(float(row[0]))
                    magnitude.append(float(row[1]))
                    phase.append(float(row[2]))
            except (ValueError, IndexError):
                continue
    
    return np.array(frequencies), np.array(magnitude), np.array(phase), is_db


def plot_combined(
    time_file: str,
    fft_file: str,
    title: Optional[str] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    log_scale: bool = False,
    show_phase: bool = False,
    # Kein phase_threshold mehr als Übergabeparameter
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    delimiter: str = ",",
) -> None:
    """
    Plottet Zeit- und Frequenzbereich nebeneinander (oder untereinander bei Phase).
    
    Args:
        phase_threshold: Minimale Amplitude, ab der Phase angezeigt wird (z.B. 0.1)
    """
    # Layout: wenn Phase angezeigt wird, 2x2 Grid (links Zeitbereich, rechts Amplitude und Phase untereinander)
    if show_phase:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)  # Mehr vertikaler Abstand (hspace)
        ax_time = fig.add_subplot(gs[:, 0])  # Links: beide Zeilen
        ax_freq = fig.add_subplot(gs[0, 1])  # Rechts oben: Amplitude
        ax_phase = fig.add_subplot(gs[1, 1]) # Rechts unten: Phase
    else:
        # Ohne Phase: nur 2 Plots nebeneinander
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
        ax_time, ax_freq = axes
        ax_phase = None
    
    # === Zeitbereich (links) ===
    try:
        x_time, y_time, time_format = read_time_domain_data(time_file, delimiter)
        
        has_datetime = time_format in ("iso", "epoch")
        
        ax_time.plot(x_time, y_time, linewidth=1.5, color="steelblue")
        ax_time.set_title("Zeitbereich", fontsize=12, fontweight="bold")
        
        if has_datetime:
            ax_time.set_xlabel("Zeit", fontsize=10)
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()
        else:
            ax_time.set_xlabel("Zeit [s]", fontsize=10)
        
        ax_time.set_ylabel("Amplitude", fontsize=10)
        ax_time.grid(True, alpha=0.3, linestyle="--")
        
    except Exception as e:
        print(f"Fehler beim Laden der Zeitbereich-Daten: {e}")
        ax_time.text(0.5, 0.5, f"Fehler:\n{e}", 
                     ha="center", va="center", transform=ax_time.transAxes)
    
    # === Frequenzbereich (Mitte) ===
    try:
        frequencies, magnitude, phase, is_db = read_frequency_domain_data(fft_file, delimiter)
        
        # Frequenz-Limit anwenden
        mask = np.ones(len(frequencies), dtype=bool)
        if fmin is not None:
            mask &= frequencies >= fmin
        if fmax is not None:
            mask &= frequencies <= fmax
        
        frequencies = frequencies[mask]
        magnitude = magnitude[mask]
        phase = phase[mask]
        
        # Magnitude plotten als Balkendiagramm
        # Balkenbreite aus Frequenzauflösung ableiten
        if len(frequencies) > 1:
            freq_resolution = frequencies[1] - frequencies[0]
            bar_width = freq_resolution * 0.8  # 80% der Auflösung für schmale Balken
        else:
            bar_width = 0.1
        
        ax_freq.bar(frequencies, magnitude, width=bar_width, color="crimson", edgecolor="crimson", alpha=0.7)
        ax_freq.set_title("Frequenzbereich (Magnitude)", fontsize=12, fontweight="bold")
        ax_freq.set_xlabel("Frequenz [Hz]", fontsize=10)
        
        if is_db or log_scale:
            ylabel = "Magnitude [dB]" if is_db else "Magnitude"
            if not is_db and log_scale:
                # Konvertiere zu dB wenn nicht schon in dB
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
                ax_freq.clear()
                ax_freq.bar(frequencies, magnitude_db, width=bar_width, color="crimson", edgecolor="crimson", alpha=0.7)
                ylabel = "Magnitude [dB]"
            ax_freq.set_ylabel(ylabel, fontsize=10)
        else:
            ax_freq.set_ylabel("Magnitude", fontsize=10)
        
        ax_freq.grid(True, alpha=0.3, linestyle="--")
        
        # Phase plotten (rechts) falls gewünscht
        if show_phase and ax_phase is not None:
            # Standard-Threshold: 5% des Maximums
            threshold = 0.05 * np.max(magnitude) if len(magnitude) > 0 else 0.0
            phase_to_plot = np.rad2deg(phase).copy()
            frequencies_to_plot = frequencies.copy()
            mask_above_threshold = magnitude >= threshold
            phase_to_plot = phase_to_plot[mask_above_threshold]
            frequencies_to_plot = frequencies_to_plot[mask_above_threshold]
            ax_phase.plot(frequencies_to_plot, phase_to_plot, 'o', color="crimson", markersize=4, alpha=0.7)
            ax_phase.set_title("Frequenzbereich (Phase)", fontsize=12, fontweight="bold")
            ax_phase.set_xlabel("Frequenz [Hz]", fontsize=10)
            ax_phase.set_ylabel("Phase [°]", fontsize=10)
            ax_phase.grid(True, alpha=0.3, linestyle="--")
            ax_phase.set_xlim(ax_freq.get_xlim())
            ax_phase.set_ylim(-180, 180)
            ax_phase.set_yticks([-180, -90, 0, 90, 180])
            for angle in [-180, -90, 0, 90, 180]:
                ax_phase.axhline(y=angle, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        
    except Exception as e:
        print(f"Fehler beim Laden der FFT-Daten: {e}")
        ax_freq.text(0.5, 0.5, f"Fehler:\n{e}", 
                     ha="center", va="center", transform=ax_freq.transAxes)
    
    # Gesamt-Titel
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    else:
        filename = Path(time_file).stem
        fig.suptitle(f"FFT-Analyse: {filename}", fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    
    # Speichern oder anzeigen
    # Immer speichern, nie anzeigen
    # Sicherstellen, dass Ausgabeverzeichnis existiert
    import os
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert: {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plottet Zeit- und Frequenzbereich-Daten nebeneinander.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "name",
        help="Dateiname ohne Endung (z.B. 'sample'). Sucht in measurements/ und fft/",
    )
    p.add_argument("--title", help="Titel des Plots")
    p.add_argument("--fmin", type=float, help="Minimale Frequenz in Hz für FFT-Plot")
    p.add_argument("--fmax", type=float, help="Maximale Frequenz in Hz für FFT-Plot")
    p.add_argument("--log", action="store_true", help="Logarithmische Skala (dB) für Magnitude")
    p.add_argument("--phase", action="store_true", default=True, help="Phase als dritten Subplot anzeigen (Standard: aktiv)")
    p.add_argument("--no-phase", dest="phase", action="store_false", help="Phase NICHT anzeigen")
    # Kein phase-threshold mehr als Übergabeparameter
    p.add_argument("--figsize", nargs=2, type=int, default=[16, 6], help="Größe der Figur (Breite Höhe)")
    p.add_argument("--save", dest="save_path", help="Dateiname für Plot (ohne Pfad, wird in plot_fft/ gespeichert)")
    p.add_argument("--delimiter", default=",", help="CSV-Trennzeichen")
    p.add_argument("--time-file", help="Expliziter Pfad zur Zeitbereich-Datei (überschreibt --file)")
    p.add_argument("--fft-file", help="Expliziter Pfad zur FFT-Datei (überschreibt --file)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Dateipfade bestimmen
    if args.time_file and args.fft_file:
        time_file = args.time_file
        fft_file = args.fft_file
    else:
        # Automatisch aus --file ableiten
        base_name = args.name

        # Zeitbereich-Datei: wenn der Nutzer einen relativen Pfad oder Namen angibt,
        # bauen wir measurements/<base_name> (erlaubt ist auch 'measurements/name.csv' oder absolute Pfade)
        if args.time_file:
            time_file = args.time_file
        else:
            if os.path.isabs(base_name) or base_name.startswith("measurements"):
                time_file = base_name
            else:
                time_file = os.path.join("measurements", f"{base_name}.csv")

        # FFT-Datei: standardmäßig fft/<name>.csv
        if args.fft_file:
            fft_file = args.fft_file
        else:
            file_stem = Path(base_name).stem
            fft_file = os.path.join("fft", f"{file_stem}.csv")
    
    # Prüfen ob Dateien existieren
    if not os.path.exists(time_file):
        print(f"Fehler: Zeitbereich-Datei nicht gefunden: {time_file}")
        sys.exit(1)
    
    if not os.path.exists(fft_file):
        print(f"Fehler: FFT-Datei nicht gefunden: {fft_file}")
        # Hinweis an den Nutzer: wie erzeugt man die FFT-Datei mit dem neuen CLI (name ohne Endung)
        suggested = Path(fft_file).stem
        print(f"Tipp: Führen Sie zuerst 'python fft_measurements.py {suggested}' aus")
        sys.exit(1)
    
    print(f"Zeitbereich: {time_file}")
    print(f"FFT-Daten:   {fft_file}")
    
    # Speicherpfad verarbeiten
    if args.save_path:
        # Wenn nur Dateiname, in plot_fft/ speichern
        if not os.path.dirname(args.save_path):
            save_path = os.path.join("plot_fft", args.save_path)
        else:
            save_path = args.save_path
    else:
        # Standard: plot_fft/<name>.png
        save_path = os.path.join("plot_fft", f"{args.name}.png")
    
    try:
        plot_combined(
            time_file=time_file,
            fft_file=fft_file,
            title=args.title,
            fmin=args.fmin,
            fmax=args.fmax,
            log_scale=args.log,
            show_phase=args.phase,
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
