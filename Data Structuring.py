import pandas as pd
import re
import os


def parse_and_clean_gps_rows(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} was not found.")
        return None


    try:
        with open(input_path, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except UnicodeError:
        with open(input_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()


    all_data_points = []
    current_source = "Unknown"
    column_row_counters = {}


    for line in lines:
        line = line.strip()
        if not line:
            continue
           
        if line.startswith("========"):
            current_source = line.replace("========", "").strip()
            column_row_counters = {}
            continue


        if ":" in line:
            parts = line.split(":", 1)
            
            # CLEANING STEP: Remove all spaces AND any surrounding quotation marks from the column name
            col_name = re.sub(r'\s+', '', parts[0]).replace('"', '')
            
            # Clean values (strip spaces and remove quotes)
            val_data = parts[1].strip().replace('"', '')

            if col_name not in column_row_counters:
                column_row_counters[col_name] = 0
            else:
                column_row_counters[col_name] += 1
                
            current_row_idx = column_row_counters[col_name]

            all_data_points.append({
                "SourceFile": current_source,
                "RowIndex": current_row_idx,
                "ColumnName": col_name,
                "Value": val_data
            })


    df_entries = pd.DataFrame(all_data_points)
   
    if df_entries.empty:
        print("No valid data entries found.")
        return None


    # Pivot to create the unique columns out of text left of the colon
    df_structured = df_entries.pivot_table(
        index=["SourceFile", "RowIndex"],
        columns="ColumnName",
        values="Value",
        aggfunc='first'
    ).reset_index()


    # Drop operational helper RowIndex
    df_structured = df_structured.drop(columns=["RowIndex"])


    # --- CLEANING STEP: Include Accelerometer so data rows are not deleted ---
    # We validate rows that contain either GPS or Accelerometer data
    data_columns = ['SourceFile', 'GPSDateTime', 'GPSLatitude', 'GPSLongitude', 'GPSSpeed', 'Accelerometer']
   
    existing_data_cols = [col for col in data_columns if col in df_structured.columns]
   
    if existing_data_cols:
        # Drop rows ONLY if they are completely empty across all target tracking columns
        df_structured = df_structured.dropna(subset=existing_data_cols, how='all')
    else:
        print("Warning: None of the target data columns were found to validate rows.")


    # --- SEQUENTIAL SORTING LAYER ---
    # Sort everything sequentially by SourceFile, then by the GPS timestamp sequence
    if 'GPSDateTime' in df_structured.columns:
        df_structured = df_structured.sort_values(by=["SourceFile", "GPSDateTime"]).reset_index(drop=True)
    else:
        df_structured = df_structured.sort_values(by=["SourceFile"]).reset_index(drop=True)


    # Save to file
    df_structured.to_csv(output_path, index=False)
    print(f"Success! Filtered row-by-column data saved to: {output_path}")
    return df_structured


# --- EXECUTION BLOCK ---
# Adjusted file paths to target your environment dynamically
input_file = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\M550_extracted_data.csv"
output_file = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\M550_Structured_GPS_Data.csv"


df_final = parse_and_clean_gps_rows(input_file, output_file)


if df_final is not None:
    print("\nPreview of the cleaned structure:")
    print(df_final.head(15))