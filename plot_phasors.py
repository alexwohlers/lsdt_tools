#!/usr/bin/env python3
"""
Phasor-Plot: Visualisiert die komplexe Addition von Sinus-Komponenten als Zeigerdiagramm.

Beispielaufruf:
  python plot_phasors.py --freqs 5 12 20 --amps 1 0.7 0.4 --phases 0 0 0 --duration 1 --fs 1000 --save phasor_demo.png

Erzeugt ein Diagramm wie im Anhang: Zeiger (blau), Summenkurve (rot), Kreise (blassblau).
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import animation
import os

def phasor_plot(freqs, amps, phases, duration=1.0, fs=1000, save_path=None, title=None):
    t = np.linspace(0, duration, int(fs*duration))
    phasors = [a * np.exp(1j * (2*np.pi*f*t + phi)) for f, a, phi in zip(freqs, amps, phases)]
    signal = np.sum(phasors, axis=0)
    
    # Figure erstellen
    fig = plt.figure(figsize=(10,5))
    
    # Gemeinsame Skala für beide Plots
    max_val = max(np.abs(signal.real).max(), np.abs(signal.imag).max(), sum(amps))
    
    # Explizite Achsen-Positionen für gleiche Höhe
    # Format: [left, bottom, width, height]
    ax_phasor = fig.add_axes([0.08, 0.15, 0.35, 0.7])  # Links, quadratisch
    ax_time = fig.add_axes([0.55, 0.15, 0.40, 0.7])    # Rechts, gleiche Höhe
    
    # Phasor-Plot links
    ax_phasor.set_aspect('equal')
    ax_phasor.grid(True, alpha=0.3)
    ax_phasor.set_xlabel(r'$\mathbb{R}$', fontsize=16)
    ax_phasor.set_ylabel(r'$\mathbb{I}$', fontsize=16)
    ax_phasor.set_xlim(-max_val*1.2, max_val*1.2)
    ax_phasor.set_ylim(-max_val*1.2, max_val*1.2)
    # Initialisiere Kreise (werden im Animate verschoben)
    circle_objs = [plt.Circle((0,0), a, color='cornflowerblue', alpha=0.2, fill=False) for a in amps]
    for c in circle_objs:
        ax_phasor.add_artist(c)
    ax_phasor.axhline(0, color='gray', lw=1)
    ax_phasor.axvline(0, color='gray', lw=1)
    arrow_objs = []
    sum_line, = ax_phasor.plot([], [], 'r-', linewidth=2)
    # Zeitbereichs-Plot rechts
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlabel('Zeit [s]', fontsize=14)
    ax_time.set_ylabel('Amplitude', fontsize=14)
    ax_time.set_xlim(t[0], t[-1])
    ax_time.set_ylim(-max_val*1.2, max_val*1.2)
    time_line, = ax_time.plot([], [], 'b-', linewidth=1.5)
    time_point, = ax_time.plot([], [], 'ro', markersize=8)
    # Animationsfunktion
    def animate(i):
        # Phasor-Plot
        for arr in arrow_objs:
            arr.remove()
        arrow_objs.clear()
        tip = 0j
        foots = []
        for idx, (f, a, phi) in enumerate(zip(freqs, amps, phases)):
            foots.append(tip)
            z = a * np.exp(1j * (2*np.pi*f*t[i] + phi))
            arr = ax_phasor.arrow(tip.real, tip.imag, z.real, z.imag, color='navy', width=0.01, head_width=0.08, length_includes_head=True)
            arrow_objs.append(arr)
            tip += z
        sum_line.set_data(signal.real[:i+1], signal.imag[:i+1])
        # Kreise an Zeigerfüßen positionieren
        for c, foot, a in zip(circle_objs, foots, amps):
            c.center = (foot.real, foot.imag)
            c.radius = a
        # Zeitbereichs-Plot
        time_line.set_data(t[:i+1], signal.real[:i+1])
        time_point.set_data([t[i]], [signal.real[i]])
        return arrow_objs + circle_objs + [sum_line, time_line, time_point]
    # Bestimme die erste Frequenz für die Umdrehung
    first_freq = freqs[0]
    t_720 = 2 / first_freq  # 2 Perioden des ersten Zeigers
    idx_720 = np.searchsorted(t, t_720)
    frames = min(idx_720 if idx_720 > 0 else len(t), len(t))
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=30, blit=False)
    # Titel setzen, falls vorhanden
    if title:
        fig.suptitle(title, fontsize=18)
    # Dateiname setzen
    if save_path:
        gif_path = os.path.join('bilder', f'{save_path}.gif')
    else:
        gif_path = os.path.join('bilder', 'phasor_animation.gif')
    ani.save(gif_path, writer='pillow', fps=30)
    print(f"Animation gespeichert: {gif_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Phasor-Plot für überlagerte Sinus-Komponenten")
    parser.add_argument('--freqs', nargs='+', type=float, required=True, help='Frequenzen in Hz')
    parser.add_argument('--amps', nargs='+', type=float, required=True, help='Amplituden')
    parser.add_argument('--phases', nargs='+', type=float, default=None, help='Phasen in Radiant')
    parser.add_argument('--duration', type=float, default=1.0, help='Dauer in Sekunden')
    parser.add_argument('--fs', type=int, default=1000, help='Abtastrate in Hz')
    parser.add_argument('--save', type=str, default=None, help='Dateistamm für PNG/GIF-Ausgabe')
    parser.add_argument('--title', type=str, default=None, help='Diagrammtitel')
    args = parser.parse_args()
    phases = [float(p) for p in args.phases] if args.phases else [0.0]*len(args.freqs)
    phasor_plot(args.freqs, args.amps, phases, args.duration, args.fs, args.save, args.title)

if __name__ == "__main__":
    main()
