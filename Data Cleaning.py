import pandas as pd

# Load the dataset
file_path = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Dashcam_Forensic_Analytic_Data.csv"
output_path = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Dashcam_Data_Filled.csv"

df = pd.read_csv(file_path)

# 1. Sort the data sequentially to ensure filling happens in the correct order
df = df.sort_values(by=["SourceFile", "GPSDateTime"]).reset_index(drop=True)

# 2. Group by SourceFile so we only fill gaps within the same video clip
# We apply forward-fill (ffill) to the target columns
columns_to_fill = ["GPSLatitude", "GPSLongitude", "GPSSpeed"]

df[columns_to_fill] = df.groupby("SourceFile")[columns_to_fill].ffill()

# 3. Optional: If there are still empty values at the very beginning of a video 
# (before the first GPS lock is achieved), we can backward-fill (bfill) just those slots
df[columns_to_fill] = df.groupby("SourceFile")[columns_to_fill].bfill()

# Save the filled dataset
df.to_csv(output_path, index=False)

print(f"Success! Missing values filled and saved to {output_path}")
print("\nPreview of the filled columns:")
print(df[["SourceFile", "GPSDateTime", "GPSLatitude", "GPSLongitude", "GPSSpeed"]].head(15))