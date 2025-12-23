# Weekly Trend Analysis & Reporting Automation

A lightweight Python (Pandas/NumPy) project that ingests time-based CSV data, aggregates metrics by week, calculates week-over-week change, flags spikes/drops, and generates a reproducible text report.

## Features
- Parses dates and standardizes basic fields
- Weekly aggregation (overall + by department)
- Week-over-week % change for key metrics
- Flags significant increases/decreases
- Outputs a summary CSV + a human-readable report

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
