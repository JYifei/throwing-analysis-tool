# Throwing Analysis Tool

This project provides a small runnable app for analysing throwing motion videos.

The goal is to make the existing analysis pipeline reproducible and easy to use,
rather than a collection of standalone scripts.

---

## What this app does

- Upload a throwing video via a web interface
- Automatically segment throws (FSM-based)
- Compare each throw against a reference model using DTW
- Generate summary scores and diagnostic plots
- Export all results as a downloadable zip file

---

## Requirements

- Python **3.12.3**
- Tested on Windows

All dependencies are pinned in `requirements.txt`.

---

## Installation

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.\.venv\Scripts\Activate.ps1     # Windows
