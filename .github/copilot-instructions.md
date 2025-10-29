# LSDT FFT Project - AI Coding Agent Instructions

## Project Overview
Signal processing toolkit for generating synthetic measurement data, performing FFT analysis, and visualizing results. Built for educational purposes with focus on simplicity and Windows PowerShell compatibility.

## Architecture & Data Flow
```
generate_measurements.py → measurements/*.csv → fft_measurements.py → fft/*.csv
                              ↓                                         ↓
                        plot_measurements.py                      plot_fft.py
                              ↓                                         ↓
                      plot_measurements/*.png                   plot_fft/*.png
```

**Directory structure convention:**
- `measurements/` - Input time-domain CSV files (auto-created)
- `fft/` - Output frequency-domain CSV files (auto-created)
- `plot_measurements/` - Time-domain plot outputs (auto-created)
- `plot_fft/` - Combined time/frequency plot outputs (auto-created)

## Key Conventions

### File Naming Pattern
All CLI tools use **positional filename arguments WITHOUT extension or directory**:
```powershell
# Correct - tool automatically adds .csv and measurements/ prefix
python generate_measurements.py sinus --func "sin(2*pi*5*t)" --fs 100 --duration 2
python plot_measurements.py sinus --save plot.png

# Avoid - old verbose style
python generate_measurements.py --out measurements/sinus.csv
python plot_measurements.py --file measurements/sinus.csv
```

### CSV Format Auto-Detection
All tools automatically detect 4 CSV formats by header inspection (see `detect_csv_format()` pattern):
- `relative`: `index,t,value` - relative time in seconds (DEFAULT)
- `iso`: `index,timestamp_iso,value` - ISO 8601 timestamps
- `epoch`: `index,timestamp_epoch,value` - Unix epoch seconds
- `none`: `index,value` - no time column

### Safe Expression Evaluation
`generate_measurements.py` uses restricted `eval()` environment (see `_build_eval_env()`):
- Variables: `t` (time in seconds), `n` (sample index)
- Math functions: `sin, cos, tan, exp, log, sqrt, sign, clamp`, etc.
- Constants: `pi, tau, e`
- **No builtins** - security by isolation

Example: `--func "sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)"`

## Essential Commands

### Complete Workflow
```powershell
# 1. Generate signal (5Hz + 12Hz mixed)
python generate_measurements.py mixed_signal --func "sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)" --fs 100 --duration 2

# 2. Compute FFT with windowing
python fft_measurements.py mixed_signal --window hann

# 3. Visualize time + frequency domain
python plot_fft.py --file mixed_signal.csv --fmax 20 --save analysis.png
```

### Dependencies
```powershell
pip install matplotlib numpy  # For FFT and plotting only
# generate_measurements.py has NO dependencies (stdlib only)
```

## Code Patterns

### Directory Auto-Creation
All output functions create parent directories automatically:
```python
import os
out_dir = os.path.dirname(out_path)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
```

### Plot Save Path Logic
When `--save` provided without directory, auto-prefix with appropriate directory:
```python
# For plot_measurements.py
if args.save_path:
    if not os.path.dirname(args.save_path):
        save_path = os.path.join("plot_measurements", args.save_path)

# For plot_fft.py
if args.save_path:
    if not os.path.dirname(args.save_path):
        save_path = os.path.join("plot_fft", args.save_path)
```

### FFT Naming Convention
Output filename matches input: `measurements/signal.csv` → `fft/signal.csv` (same stem, different directory)

## Testing Approach
No formal test suite. Validate changes by running complete workflow with known signal (e.g., single frequency sine wave - should show single FFT peak at expected frequency).

## Platform Notes
Primary target: **Windows PowerShell**. Use `;` for command chaining, not `&&`.
