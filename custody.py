import os
import hashlib
import pandas as pd

# ==========================================================
# CHAIN OF CUSTODY VERIFICATION MODULE
# ==========================================================

def generate_sha256_hash(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode())
    return sha256_hash.hexdigest()


def verify_pipeline_authenticity(data_path, audit_log_path):
    """
    Verify whether uploaded dashcam dataset has been altered.

    Returns:
        True  -> Dataset authentic
        False -> Dataset altered
    """

    if not os.path.exists(data_path):
        return False

    if not os.path.exists(audit_log_path):
        return False

    master_df = pd.read_csv(data_path)
    audit_df = pd.read_csv(audit_log_path)

    reference_hashes = dict(
        zip(
            audit_df['SourceFile'],
            audit_df['SHA256_Hash']
        )
    )

    all_passed = True

    for video_file, group in master_df.groupby("SourceFile"):

        current_rows_string = group.to_string()

        current_hash = generate_sha256_hash(
            current_rows_string
        )

        original_hash = reference_hashes.get(video_file)

        if original_hash is None:
            all_passed = False

        elif current_hash != original_hash:
            all_passed = False

    return