"""Render the frozen Agriculture test comparison as dependency-free SVG."""
from pathlib import Path
import csv


root = Path(__file__).resolve().parent / "results"
with (root / "Agriculture_horizon_metrics.csv").open(encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

steps = [int(row["step"]) for row in rows]
fallback = [float(row["fallback_mse"]) for row in rows]
selected = [float(row["selected_mse"]) for row in rows]
width, height = 1080, 620
left, right, top, bottom = 95, 35, 75, 80
plot_w, plot_h = width - left - right, height - top - bottom
ymax = max(fallback + selected) * 1.08


def x_of(step: int) -> float:
    return left + (step - 1) * plot_w / (len(steps) - 1)


def y_of(value: float) -> float:
    return top + plot_h * (1 - value / ymax)


def points(values: list[float]) -> str:
    return " ".join(f"{x_of(step):.2f},{y_of(value):.2f}" for step, value in zip(steps, values))


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="#ffffff"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#263238}.grid{stroke:#d9e1e6;stroke-width:1}.axis{stroke:#546e7a;stroke-width:1.5}</style>',
       f'<text x="{width/2}" y="34" text-anchor="middle" font-size="24" font-weight="700">Agriculture held-out test error by forecast horizon</text>']
for i in range(6):
    value = ymax * i / 5
    y = y_of(value)
    svg += [f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>',
            f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="14">{value:.2f}</text>']
for step in steps:
    x = x_of(step)
    svg += [f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}"/>',
            f'<text x="{x:.2f}" y="{top+plot_h+28}" text-anchor="middle" font-size="14">{step}</text>']
svg += [f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<polyline points="{points(fallback)}" fill="none" stroke="#D95F4B" stroke-width="4"/>',
        f'<polyline points="{points(selected)}" fill="none" stroke="#2474B5" stroke-width="4"/>']
for values, color in ((fallback, "#D95F4B"), (selected, "#2474B5")):
    for step, value in zip(steps, values):
        svg.append(f'<circle cx="{x_of(step):.2f}" cy="{y_of(value):.2f}" r="5" fill="{color}"/>')
svg += [f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-size="17">Forecast horizon</text>',
        f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" font-size="17" transform="rotate(-90 25 {top+plot_h/2})">MSE (train-standardized OT)</text>',
        f'<line x1="{width-280}" y1="55" x2="{width-235}" y2="55" stroke="#D95F4B" stroke-width="4"/><text x="{width-225}" y="61" font-size="15">AR-Ridge</text>',
        f'<line x1="{width-150}" y1="55" x2="{width-105}" y2="55" stroke="#2474B5" stroke-width="4"/><text x="{width-95}" y="61" font-size="15">N+S+Q+SF</text>',
        '</svg>']
(root / "Agriculture_horizon_mse.svg").write_text("\n".join(svg), encoding="utf-8")
