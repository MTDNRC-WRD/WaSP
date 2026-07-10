import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_CSV = r"C:\Users\CND367\Documents\WaSP\test1\formatted_self_potential.csv"
SAVE_PNG = False
OUTPUT_PNG = r"C:\Users\CND367\Documents\WaSP\test1\formatted_self_potential_plot.png"


FIG_WIDTH = 14
FIG_HEIGHT = 8
POINT_SIZE = 45
LABEL_FONT_SIZE = 7
COLORMAP = "viridis"   # good options: viridis, plasma, turbo, cividis


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    required = ["Measurement_Number", "Longitude", "Latitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.copy()
    df["Measurement_Number"] = pd.to_numeric(df["Measurement_Number"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

    df = df.dropna(subset=["Measurement_Number", "Longitude", "Latitude"]).copy()
    df = df.sort_values("Measurement_Number").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after cleaning the input CSV.")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    scatter = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["Measurement_Number"],
        cmap=COLORMAP,
        s=POINT_SIZE,
        edgecolors="black",
        linewidths=0.4,
    )

    for _, row in df.iterrows():
        ax.annotate(
            str(int(row["Measurement_Number"])),
            (row["Longitude"], row["Latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE,
            alpha=0.85,
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Measurement Number")

    ax.set_title("Self Potential Measurement Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()