"""
Plot people-per-frame over time from the CSV written by part1_capture_count.py.

Usage:
  python plot_results.py                       # reads results/people_count.csv
  python plot_results.py --csv other.csv --out fig4.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/people_count.csv")
    p.add_argument("--out", default="results/people_over_time.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["people"], marker="o", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("People per frame")
    ax.set_title("People detected over time (YOLOv8n, local machine)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}  ({len(df)} points, mean {df['people'].mean():.1f}, max {df['people'].max()})")


if __name__ == "__main__":
    main()
