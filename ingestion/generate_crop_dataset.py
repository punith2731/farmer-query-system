import pandas as pd
import numpy as np


def generate_dataset(output_path: str = "data/mandi_prices_daily_2015_2025.csv", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Date range
    dates = pd.date_range(start="2015-01-01", end="2025-12-31")

    # Crop data (base price + MSP)
    crops_data = [
        ("Tomato", "Delhi", 2000, 0),
        ("Onion", "Delhi", 1800, 0),
        ("Potato", "Delhi", 1500, 0),
        ("Brinjal", "Delhi", 1900, 0),
        ("Cabbage", "Delhi", 1400, 0),
        ("Wheat", "Delhi", 2300, 2275),
        ("Rice", "Lucknow", 2800, 2300),
        ("Maize", "Indore", 2100, 2090),
        ("Bajra", "Jaipur", 1900, 2500),
        ("Jowar", "Nagpur", 2200, 3180),
        ("Tur Dal", "Hyderabad", 7000, 7000),
        ("Moong Dal", "Bhopal", 7800, 7755),
        ("Urad Dal", "Kanpur", 6800, 6950),
        ("Groundnut", "Ahmedabad", 6000, 5850),
        ("Soybean", "Indore", 4500, 4600),
        ("Mustard", "Jaipur", 5700, 5650),
        ("Cotton", "Nagpur", 7000, 6620),
        ("Sugarcane", "Lucknow", 3200, 3150),
        ("Tea", "Guwahati", 18000, 0),
        ("Coffee", "Chikmagalur", 14000, 0),
    ]

    rows = []

    for date in dates:
        for crop, mandi, base_price, msp in crops_data:
            # Simulate rainfall (0–10 scale)
            rainfall = np.round(rng.uniform(0, 6), 2)

            # Seasonal + random fluctuation
            seasonal = 200 * np.sin(date.dayofyear / 365 * 2 * np.pi)
            noise = int(rng.integers(-150, 150))

            price = base_price + seasonal + noise

            # Apply MSP floor
            if msp > 0:
                price = max(price, msp)

            rows.append([
                date.strftime("%Y-%m-%d"),
                crop,
                mandi,
                int(price),
                float(rainfall),
                int(msp),
            ])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["date", "crop", "mandi", "price", "rainfall_index", "msp_floor"])

    # Save dataset
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("✅ Dataset generated successfully!")
    print(f"Rows: {len(df):,}")
    print(df.head())
