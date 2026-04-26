#!/usr/bin/env python3
"""
Build iCardio RVSP train/val/holdout CSVs.
RVSP (Right Ventricular Systolic Pressure) is parsed from the
pulmonary_artery free-text field: TR component + RAP in mmHg.
Same site-based holdout split as LVEF.
Output format: "<dicom_uuid> <z_score_rvsp>" (no header, space-separated).
"""
import csv, re, sqlite3, random, math

DB      = '/home/ahmedaly/iCardio/preprocessing/json_annotation/annotations.db'
META    = '/home/ahmedaly/iCardio/preprocessing/csv_preparation/csv/intersected_dicoms_with_meta_data_final.csv'
SSY     = '/home/ahmedaly/iCardio/EchoJEPAv2/study_site_year.csv'
HOLDOUT = '/home/ahmedaly/iCardio/EchoJEPAv2/training/holdout_sites.txt'
OUT     = '/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs'
VAL_FRAC = 0.10
SEED     = 42

# ── 1. RVSP labels from DB ────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
cur  = conn.cursor()
cur.execute('SELECT study_uuid, pulmonary_artery FROM studies WHERE pulmonary_artery IS NOT NULL')
rvsp_by_study = {}
for study_uuid, text in cur.fetchall():
    flat = text.replace('\n', ' ')
    nums = re.findall(r'(\d+)\s*(?:mmHg|mm\s*Hg)', flat, re.IGNORECASE)
    if len(nums) >= 2:
        rvsp_by_study[study_uuid] = int(nums[0]) + int(nums[1])
    elif len(nums) == 1:
        rvsp_by_study[study_uuid] = int(nums[0])
conn.close()
print(f'Studies with RVSP: {len(rvsp_by_study)}')

# ── 2. Site splits ────────────────────────────────────────────────────────────
with open(HOLDOUT) as f:
    holdout_sites = {l.strip() for l in f if l.strip()}

study_to_site = {}
with open(SSY) as f:
    for row in csv.DictReader(f):
        study_to_site[row['study_icid']] = row['site_id']

holdout_studies, other_studies = set(), []
for s in rvsp_by_study:
    if study_to_site.get(s, '') in holdout_sites:
        holdout_studies.add(s)
    else:
        other_studies.append(s)

random.seed(SEED)
random.shuffle(other_studies)
n_val = max(1, int(len(other_studies) * VAL_FRAC))
val_studies   = set(other_studies[:n_val])
train_studies = set(other_studies[n_val:])
print(f'Train studies: {len(train_studies)}, val: {len(val_studies)}, holdout: {len(holdout_studies)}')

# ── 3. Map study → dicoms via metadata CSV (Standard video only) ──────────────
study_to_dicoms = {}
with open(META) as f:
    for row in csv.DictReader(f):
        s = row['study_uuid']
        if s not in rvsp_by_study:
            continue
        if row.get('type', '') != 'Standard':
            continue
        try:
            if float(row.get('n_frames', 0)) < 8:
                continue
        except ValueError:
            continue
        study_to_dicoms.setdefault(s, []).append(row['dicom_uuid'])
print(f'Studies with Standard DICOMs: {len(study_to_dicoms)}')

# ── 4. Compute z-score normalisation from train set ───────────────────────────
train_vals = [rvsp_by_study[s] for s in train_studies if s in study_to_dicoms]
mean_rvsp = sum(train_vals) / len(train_vals)
std_rvsp  = math.sqrt(sum((v - mean_rvsp)**2 for v in train_vals) / len(train_vals))
print(f'Train RVSP: mean={mean_rvsp:.2f}, std={std_rvsp:.2f} mmHg')

# ── 5. Write CSVs ─────────────────────────────────────────────────────────────
def write_split(path, studies):
    rows = []
    for s in studies:
        if s not in study_to_dicoms:
            continue
        raw = rvsp_by_study[s]
        z   = (raw - mean_rvsp) / std_rvsp
        for d in study_to_dicoms[s]:
            rows.append(f'{d} {z:.6f}\n')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f'  {path}: {len(rows)} rows')

write_split(f'{OUT}/icardio_rvsp_train.csv',   train_studies)
write_split(f'{OUT}/icardio_rvsp_val.csv',     val_studies)
write_split(f'{OUT}/icardio_rvsp_holdout.csv', holdout_studies)
print(f'\nTarget mean={mean_rvsp:.4f}, std={std_rvsp:.4f}')
print('Done.')
