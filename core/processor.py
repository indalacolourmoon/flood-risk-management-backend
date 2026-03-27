import pandas as pd


class FloodProcessor:
    def __init__(self, year1_path: str, year2_path: str):
        self.year1_path = year1_path
        self.year2_path = year2_path

    def _find_column(self, df: pd.DataFrame, possible_names: list[str]) -> str:
        """Finds a column in the dataframe by trying multiple possible case-insensitive names."""
        cols = {c.lower().strip(): c for c in df.columns}
        for name in possible_names:
            if name.lower() in cols:
                return cols[name.lower()]
        raise ValueError(f"Could not find any of the required columns: {possible_names}")

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes the dataframe columns for consistent processing."""
        try:
            # Map common variations to internal standard names
            mapping = {
                self._find_column(df, ["latitude", "lat", "y"]): "lat",
                self._find_column(df, ["longitude", "lng", "lon", "x"]): "lng",
                self._find_column(df, ["elevation", "elev", "value", "altitude"]): "elevation",
                self._find_column(df, ["system:index", "id", "index"]): "system:index",
            }
            # Rename columns
            df = df.rename(columns=mapping)
            # Ensure index is treated as string for reliable merging
            df["system:index"] = df["system:index"].astype(str)
            return df[["system:index", "lat", "lng", "elevation"]]
        except ValueError as e:
            raise ValueError(f"CSV format error: {str(e)}")

    def load_and_merge(self) -> pd.DataFrame:
        """Loads and standardizes both CSVs, then merges them on 'system:index'."""
        df1 = pd.read_csv(self.year1_path)
        df2 = pd.read_csv(self.year2_path)

        if df1.empty or df2.empty:
            raise ValueError("One or both CSV files are empty.")

        n1 = self._normalize_df(df1)
        n2 = self._normalize_df(df2)

        # Merge on 'system:index'
        merged = pd.merge(
            n1,
            n2,
            on="system:index",
            suffixes=("_y1", "_y2"),
        )

        # Optimization: Free up memory from raw dataframes
        del n1
        del n2

        if merged.empty:
            raise ValueError("No matching records found between Year 1 and Year 2 datasets. Ensure the 'system:index' columns match.")

        # Cleanup redundant columns to preserve RAM (Refinement #3)
        # We only really need one set of lat/lng since they match
        merged.drop(columns=["lat_y2", "lng_y2"], inplace=True)

        return merged

    def classify_and_compare(self, threshold: float) -> pd.DataFrame:
        """
        Applies flood classification and detects year-over-year elevation changes.
        """
        df = self.load_and_merge()

        # Classify each year
        df["status_y1"] = df["elevation_y1"].apply(
            lambda x: "Flooded" if x <= threshold else "Safe"
        )
        df["status_y2"] = df["elevation_y2"].apply(
            lambda x: "Flooded" if x <= threshold else "Safe"
        )

        # Calculate numeric delta (Spot the Difference)
        df["elevation_delta"] = df["elevation_y1"] - df["elevation_y2"]

        def get_change_status(row) -> str:
            s1, s2 = row["status_y1"], row["status_y2"]
            delta = row["elevation_delta"]
            
            # Worsened: Either it's newly flooded OR land significantly eroded (> 0.5m)
            if (s1 == "Safe" and s2 == "Flooded") or delta > 0.5:
                return "Worsened"
            # Improved: Either it's recovered OR land significantly raised (> 0.5m)
            elif (s1 == "Flooded" and s2 == "Safe") or delta < -0.5:
                return "Improved"
            else:
                return "Stable"

        df["change_analysis"] = df.apply(get_change_status, axis=1)

        # Lightweight payload — only what the map renderer needs
        clean = df[[
            "system:index",
            "lat_y1",
            "lng_y1",
            "elevation_y1",
            "elevation_y2",
            "elevation_delta",
            "status_y2",
            "change_analysis",
        ]].rename(columns={
            "lat_y1": "lat",
            "lng_y1": "lng",
        })

        return clean

    def get_summary_stats(self, processed_df: pd.DataFrame) -> dict:
        total = len(processed_df)
        if total == 0:
            return {
                "total_area_points": 0,
                "flood_risk_percentage": 0.0,
                "newly_vulnerable_points": 0,
                "improved_points": 0,
                "stable_points": 0,
                "analysis_status": "No Data",
            }

        flooded = len(processed_df[processed_df["status_y2"] == "Flooded"])
        worsened = len(processed_df[processed_df["change_analysis"] == "Worsened"])
        improved = len(processed_df[processed_df["change_analysis"] == "Improved"])
        stable = len(processed_df[processed_df["change_analysis"] == "Stable"])

        return {
            "total_area_points": total,
            "flood_risk_percentage": float(f"{(flooded / total) * 100:.2f}"),
            "newly_vulnerable_points": worsened,
            "improved_points": improved,
            "stable_points": stable,
            "analysis_status": "Success",
        }


# --- TEST BLOCK ---
if __name__ == "__main__":
    processor = FloodProcessor("data/year1.csv", "data/year2.csv")
    results = processor.classify_and_compare(19.0)
    stats = processor.get_summary_stats(results)

    print("--- Analysis Results ---")
    print(f"Total Points Processed : {stats['total_area_points']}")
    print(f"Flood Risk             : {stats['flood_risk_percentage']}%")
    print(f"Newly At-Risk Points   : {stats['newly_vulnerable_points']}")
    print(f"Improved Points        : {stats['improved_points']}")
    print("\nSample output:")
    print(results.head())
