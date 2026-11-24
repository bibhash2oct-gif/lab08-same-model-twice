"""Look at the table before modelling it.

Prints the first rows and a summary, then saves a plot of rentals against the
hour of the day to reports/rentals_by_hour.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from bikes_data import TARGET, make_rentals

REPORTS = Path(__file__).parent / "reports"
COLOURS = {0: "#2a78d6", 1: "#eb6834"}
LABELS = {0: "weekday", 1: "weekend"}


def main() -> None:
    frame = make_rentals()
    print(frame.head())
    print(frame.describe().round(2))

    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for weekend in (0, 1):
        rows = frame[frame["is_weekend"] == weekend]
        ax.scatter(
            rows["hour"],
            rows[TARGET],
            s=8,
            alpha=0.25,
            color=COLOURS[weekend],
            label=f"{LABELS[weekend]} (one dot per station-hour)",
        )
        hourly_mean = rows.groupby("hour")[TARGET].mean()
        ax.plot(
            hourly_mean.index,
            hourly_mean.to_numpy(),
            linewidth=2,
            color=COLOURS[weekend],
            label=f"{LABELS[weekend]} average",
        )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Bikes rented in the hour")
    ax.set_title("Rentals by hour: weekday peaks at 8 am and 5-6 pm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "rentals_by_hour.png", dpi=120)
    plt.close(fig)
    print("Saved", REPORTS / "rentals_by_hour.png")


if __name__ == "__main__":
    main()
