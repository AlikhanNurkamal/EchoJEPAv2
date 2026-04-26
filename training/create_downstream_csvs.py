#!/usr/bin/env python3
"""
Build iCardio CSVs for three downstream tasks:
  - LVIDd  : left ventricular internal diameter at diastole (regression, cm)
  - MVRegurg: mitral valve regurgitation grade (3-class: 0=none/trace, 1=mild, 2=mod/severe)
  - Pericardial effusion (binary: 0=absent, 1=present)

Same site-based holdout split as LVEF/RVSP.
Output: "<dicom_uuid> <label>" space-separated, no header.
"""
import csv, re, sqlite3, random, math

DB      = '/home/ahmedaly/iCardio/preprocessing/json_annotation/annotations.db'
META    = '/home/ahmedaly/iCardio/preprocessing/csv_preparation/csv/intersected_dicoms_with_meta_data_final.csv'
SSY     = '/home/ahmedaly/iCardio/EchoJEPAv2/study_site_year.csv'
HOLDOUT = '/home/ahmedaly/iCardio/EchoJEPAv2/training/holdout_sites.txt'
OUT     = '/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs'
VAL_FRAC = 0.10
SEED     = 42

conn = sqlite3.connect(DB)
cur  = conn.cursor()

# ── LVIDd ─────────────────────────────────────────────────────────────────────
cur.execute('SELECT study_uuid, left_ventricle_diastolic_diameter FROM studies '
            'WHERE left_ventricle_diastolic_diameter IS NOT NULL '
            'AND left_ventricle_diastolic_diameter != ""')
lvidd_by_study = {}
for study_uuid, val in cur.fetchall():
    try:
        v = float(val)
        if 2.0 <= v <= 8.0:          # plausible range in cm
            lvidd_by_study[study_uuid] = v
    except ValueError:
        pass
print(f'Studies with LVIDd: {len(lvidd_by_study)}')

# ── MV Regurgitation ──────────────────────────────────────────────────────────
cur.execute('SELECT study_uuid, mitral_valve FROM studies '
            'WHERE mitral_valve IS NOT NULL AND mitral_valve != ""')
mvr_by_study = {}
for study_uuid, text in cur.fetchall():
    t = text.lower()
    # mod/severe first to catch "mild-to-moderate"
    if re.search(r'moderate|severe', t):
        mvr_by_study[study_uuid] = 2
    elif re.search(r'\bmild\b', t):
        mvr_by_study[study_uuid] = 1
    elif re.search(r'no significant|no regurgitation|trace|trivial|normal', t):
        mvr_by_study[study_uuid] = 0
    # else skip ambiguous
print(f'Studies with MVRegurg: {len(mvr_by_study)}  '
      f'(0={sum(1 for v in mvr_by_study.values() if v==0)}, '
      f'1={sum(1 for v in mvr_by_study.values() if v==1)}, '
      f'2={sum(1 for v in mvr_by_study.values() if v==2)})')

# ── Pericardial Effusion ───────────────────────────────────────────────────────
cur.execute('SELECT study_uuid, pericardium FROM studies '
            'WHERE pericardium IS NOT NULL AND pericardium != ""')
peri_by_study = {}
for study_uuid, text in cur.fetchall():
    t = text.lower()
    if re.search(r'no significant effusion|no effusion|without effusion', t):
        peri_by_study[study_uuid] = 0
    elif re.search(r'normal|appears normal|not well seen', t) and 'effusion' not in t:
        peri_by_study[study_uuid] = 0
    elif re.search(r'effusion', t):
        peri_by_study[study_uuid] = 1
    # skip ambiguous/unseen
print(f'Studies with Pericardial: {len(peri_by_study)}  '
      f'(0={sum(1 for v in peri_by_study.values() if v==0)}, '
      f'1={sum(1 for v in peri_by_study.values() if v==1)})')

conn.close()

# ── Site splits ───────────────────────────────────────────────────────────────
with open(HOLDOUT) as f:
    holdout_sites = {l.strip() for l in f if l.strip()}

study_to_site = {}
with open(SSY) as f:
    for row in csv.DictReader(f):
        study_to_site[row['study_icid']] = row['site_id']

# ── DICOM map ─────────────────────────────────────────────────────────────────
all_label_studies = set(lvidd_by_study) | set(mvr_by_study) | set(peri_by_study)
study_to_dicoms = {}
with open(META) as f:
    for row in csv.DictReader(f):
        s = row['study_uuid']
        if s not in all_label_studies:
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


def make_splits(label_dict):
    holdout, other = set(), []
    for s in label_dict:
        if s not in study_to_dicoms:
            continue
        if study_to_site.get(s, '') in holdout_sites:
            holdout.add(s)
        else:
            other.append(s)
    random.seed(SEED)
    random.shuffle(other)
    n_val = max(1, int(len(other) * VAL_FRAC))
    return set(other[n_val:]), set(other[:n_val]), holdout


def write_csv(path, studies, label_dict, transform=None):
    rows = []
    for s in studies:
        if s not in study_to_dicoms or s not in label_dict:
            continue
        label = label_dict[s]
        if transform:
            label = transform(label)
        for d in study_to_dicoms[s]:
            rows.append(f'{d} {label}\n')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f'  {path}: {len(rows)} rows')
    return len(rows)


# ── LVIDd CSVs ────────────────────────────────────────────────────────────────
train_s, val_s, hold_s = make_splits(lvidd_by_study)
train_vals = [lvidd_by_study[s] for s in train_s if s in lvidd_by_study]
mean_v = sum(train_vals) / len(train_vals)
std_v  = math.sqrt(sum((v - mean_v)**2 for v in train_vals) / len(train_vals))
print(f'\nLVIDd train mean={mean_v:.4f} cm, std={std_v:.4f} cm')
zscore = lambda v: f'{(v - mean_v) / std_v:.6f}'
write_csv(f'{OUT}/icardio_lvidd_train.csv',   train_s, lvidd_by_study, zscore)
write_csv(f'{OUT}/icardio_lvidd_val.csv',     val_s,   lvidd_by_study, zscore)
write_csv(f'{OUT}/icardio_lvidd_holdout.csv', hold_s,  lvidd_by_study, zscore)
print(f'  target_mean={mean_v:.4f}, target_std={std_v:.4f}')

# ── MVRegurg CSVs ─────────────────────────────────────────────────────────────
train_s, val_s, hold_s = make_splits(mvr_by_study)
print('\nMVRegurg:')
write_csv(f'{OUT}/icardio_mvregurg_train.csv',   train_s, mvr_by_study)
write_csv(f'{OUT}/icardio_mvregurg_val.csv',     val_s,   mvr_by_study)
write_csv(f'{OUT}/icardio_mvregurg_holdout.csv', hold_s,  mvr_by_study)

# ── Pericardial CSVs ──────────────────────────────────────────────────────────
train_s, val_s, hold_s = make_splits(peri_by_study)
print('\nPericardial:')
write_csv(f'{OUT}/icardio_pericardial_train.csv',   train_s, peri_by_study)
write_csv(f'{OUT}/icardio_pericardial_val.csv',     val_s,   peri_by_study)
write_csv(f'{OUT}/icardio_pericardial_holdout.csv', hold_s,  peri_by_study)

print('\nDone.')
