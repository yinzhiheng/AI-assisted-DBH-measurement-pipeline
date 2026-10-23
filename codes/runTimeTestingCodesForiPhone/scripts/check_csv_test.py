#!/usr/bin/env python3
import csv
from pathlib import Path

directory = Path(__file__).resolve().parents[1] / 'validation/csv-fixtures'
for mode in ('single', 'batch'):
    latest = max(directory.glob(f'*_{mode}_test-run.csv'), key=lambda p: p.stat().st_mtime)
    with latest.open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['segmentation_ms', 'image_name']
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]['image_name'] == 'rgb-comma, quote " and\nnewline.jpeg'
    assert rows[0]['segmentation_ms'] == ''
    assert rows[1]['segmentation_ms'] == '123.456789'
    assert rows[1]['image_name'] == 'rgb-example.jpeg'
    assert all(None not in row for row in rows), 'Column count mismatch'
print('PASS: single/batch CSVs have exactly two columns; quoting, blank timing and durable append verified')
