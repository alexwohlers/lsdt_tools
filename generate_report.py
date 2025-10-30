#!/usr/bin/env python3
"""
Erzeugt einen PDF-Bericht aus allen Diagrammen eines Workflows.
Sammelt alle PNGs aus plot_fft_3d/, plot_fft/, plot_measurements/ und erstellt eine mehrseitige PDF.

Beispiel:
  python generate_report.py sinus_1hz --out report_sinus_1hz.pdf
"""
import os
import argparse
from pathlib import Path
from PIL import Image
from fpdf import FPDF

def collect_images(workflow_name: str) -> list:
    """Sammelt alle PNG-Dateien zu einem Workflow."""
    folders = ["plot_fft_3d", "plot_fft", "plot_measurements"]
    images = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        for img_file in folder_path.glob(f"{workflow_name}*.png"):
            images.append(img_file)
    return images

def create_pdf_report(images: list, out_path: str, title: str = None):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    # Individuelle Seitenreihenfolge: Seite 3 zuerst, dann Seite 2, dann Seite 1, Seite 4 entfällt
    order = []
    if len(images) >= 3:
        order = [2, 1, 0]
    elif len(images) == 2:
        order = [1, 0]
    elif len(images) == 1:
        order = [0]
    # Seite 4 (Index 3) wird entfernt
    max_width_mm = 190
    max_height_mm = 250
    for idx in order:
        img_path = images[idx]
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        if title:
            pdf.cell(0, 10, title, ln=True, align="C")
        pdf.cell(0, 10, str(img_path.name), ln=True, align="C")
        # Bild vor dem Einfügen skalieren (maximaler Bereich)
        try:
            with Image.open(img_path) as img:
                w_px, h_px = img.size
                # Umrechnung: 1 mm ≈ 12 px bei 300 dpi
                max_w_px = int(max_width_mm * 12)
                max_h_px = int(max_height_mm * 12)
                scale = min(max_w_px / w_px, max_h_px / h_px, 1.0)
                if scale < 1.0:
                    img = img.resize((int(w_px * scale), int(h_px * scale)), Image.LANCZOS)
                    temp_path = str(img_path) + "_tmp.png"
                    img.save(temp_path)
                    pdf.image(temp_path, x=10, y=25, w=max_width_mm)
                    os.remove(temp_path)
                else:
                    pdf.image(str(img_path), x=10, y=25, w=max_width_mm)
        except Exception as e:
            print(f"Fehler beim Einfügen von {img_path}: {e}")
    pdf.output(out_path)
    print(f"PDF-Bericht gespeichert: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="PDF-Bericht aus Workflow-Diagrammen erzeugen.")
    parser.add_argument("workflow_name", help="Workflow-Name (z.B. sinus_1hz)")
    # --out Parameter entfernt; Name wird automatisch vergeben
    parser.add_argument("--title", help="Titel für den Bericht")
    args = parser.parse_args()

    images = collect_images(args.workflow_name)
    if not images:
        print("Keine Diagramme gefunden!")
        return
    # Zielpfad im Ordner 'reports' unter gleichem Namen
    out_path = os.path.join("reports", f"{args.workflow_name}_report.pdf")
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    create_pdf_report(images, out_path, args.title)

if __name__ == "__main__":
    main()
