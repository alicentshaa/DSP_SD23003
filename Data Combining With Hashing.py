import pandas as pd
import os
import hashlib

# --- YOUR HASHING CODE ---
def generate_sha256_hash(input_string):
    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the bytes-like object (encoded string)
    sha256_hash.update(input_string.encode())
    
    # Return the hexadecimal digest of the hash
    return sha256_hash.hexdigest()


# --- COMBINATION PIPELINE WITH SECURE SIDE CAR AUDIT LOGGING ---
def combine_structured_files(file_list, output_path, audit_log_path):
    combined_dfs = []
    
    # Read each file and append to our list
    for file_path in file_list:
        if os.path.exists(file_path):
            print(f"Reading: {file_path}")
            df = pd.read_csv(file_path)
            combined_dfs.append(df)
        else:
            print(f"Warning: File not found and skipped -> {file_path}")
            
    if not combined_dfs:
        print("Error: No valid files were loaded. Combination aborted.")
        return None
        
    # Stack all DataFrames on top of each other (vertically)
    master_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Sort sequentially: first by the Video SourceFile, then chronologically by GPSDateTime
    if 'GPSDateTime' in master_df.columns:
        master_df = master_df.sort_values(by=["SourceFile", "GPSDateTime"]).reset_index(drop=True)
    else:
        master_df = master_df.sort_values(by=["SourceFile"]).reset_index(drop=True)
        
    # --- LIST TO STORE HASH INFORMATION FOR SAVING ---
    audit_records = []

    print("\n--- GENERATING GROUP-LEVEL SHA-256 HASHES ---")
    for video_file, group in master_df.groupby("SourceFile"):
        # Convert only this specific video group's structured rows to a text string representation
        video_rows_string = group.to_string()
        
        # Apply your exact hashing code
        hash_value = generate_sha256_hash(video_rows_string)
        print(f"Video Group: {video_file} -> SHA-256: {hash_value}")
        
        # Append to the list to save it later
        audit_records.append({
            "SourceFile": video_file,
            "SHA256_Hash": hash_value,
            "Chain_Of_Custody": "VERIFIED_AT_COMBINATION"
        })

    # Save the master combined file
    master_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Combined master telemetry data saved to: {output_path}")
    print(f"Total rows in combined file: {len(master_df)}")
    
    # --- SAVE THE HASHING DATA FILE ---
    df_audit_log = pd.DataFrame(audit_records)
    df_audit_log.to_csv(audit_log_path, index=False)
    print(f"Success! Forensic Audit Log (hashing signatures) saved to: {audit_log_path}")
    
    return master_df

# --- EXECUTION BLOCK ---
# File paths for your 3 structured tracking inputs
files_to_combine = [
    r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\BN03_Structured_GPS_Data.csv",
    r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\GS63H_Structured_GPS_Data.csv",
    r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\M550_Structured_GPS_Data.csv"
]

# Destination for combined dataset
output_master_file = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Dashcam_Forensic_Analytic_Data.csv"

# Destination for the separate sidecar hash data log file
output_audit_file = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Master_Forensic_Audit_Log.csv"

# Run the complete code
df_master = combine_structured_files(files_to_combine, output_master_file, output_audit_file)