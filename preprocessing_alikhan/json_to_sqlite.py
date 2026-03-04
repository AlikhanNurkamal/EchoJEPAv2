"""
One-time converter: combined_updated.json -> annotations.db (SQLite)

Usage:
    python json_to_sqlite.py \
        --json /home/ahmedaly/iCardio/preprocessing/json_annotation/combined_updated.json \
        --db   /home/ahmedaly/iCardio/preprocessing/json_annotation/annotations.db
"""
import json
import sqlite3
import argparse
from pathlib import Path

STUDY_FIELDS = [
    "study_designation", "age_at_visit",
    "height", "height_units", "weight", "weight_units", "bmi",
    "ejection_fraction",
    "conditions", "characterizations", "stratifications",
    "left_ventricle_diastolic_diameter", "left_ventricle_systolic_diameter",
    "left_atrium_dimensions",
    "left_ventricle", "right_ventricle", "left_atrium", "right_atrium",
    "aortic_valve", "mitral_valve", "tricuspid_valve", "pulmonic_valve",
    "pericardium", "aortic_root", "aortic_arch", "pulmonary_artery",
    "conclusions",
]


def convert(json_path: Path, db_path: Path):
    print(f"Loading {json_path} ({json_path.stat().st_size / 1e9:.2f} GB)...")
    with open(json_path) as f:
        data = json.load(f)

    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- dicoms table ---
    cur.execute("""
        CREATE TABLE dicoms (
            dicom_uuid TEXT PRIMARY KEY,
            dicom_type TEXT
        )
    """)

    dicom_rows = [
        (entry["dicom_uuid"], entry.get("type"))
        for entry in data["dicoms"]
    ]
    cur.executemany("INSERT OR IGNORE INTO dicoms VALUES (?, ?)", dicom_rows)
    print(f"  Inserted {len(dicom_rows):,} dicom rows")

    # --- studies table ---
    study_cols = ", ".join(f'"{f}" TEXT' for f in STUDY_FIELDS)
    cur.execute(f"""
        CREATE TABLE studies (
            study_uuid TEXT PRIMARY KEY,
            {study_cols}
        )
    """)

    placeholders = ", ".join("?" * (1 + len(STUDY_FIELDS)))
    study_rows = []
    for entry in data["studies"]:
        vals = [entry["study_uuid"]] + [
            json.dumps(entry.get(f)) if isinstance(entry.get(f), (list, dict)) else entry.get(f)
            for f in STUDY_FIELDS
        ]
        study_rows.append(vals)
    cur.executemany(f"INSERT OR IGNORE INTO studies VALUES ({placeholders})", study_rows)
    print(f"  Inserted {len(study_rows):,} study rows")

    conn.commit()
    conn.close()
    print(f"Done. DB written to {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    convert(Path(args.json), Path(args.db))
