import pandas as pd
import numpy as np
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('data/preprocessed/Stage_1_train_with_abstracts.csv')

# For KR papers, extract paper numbers and check correlation with label
kr = train[train['venue'] == 'kr'].copy()

def extract_kr_number(doi):
    match = re.search(r'kr\.\d+/(\d+)', str(doi))
    if match:
        return int(match.group(1))
    return np.nan

kr['paper_num'] = kr['doi'].apply(extract_kr_number)

# Group by year and label to see ordering
for year in sorted(kr['year'].unique()):
    subset = kr[kr['year'] == year].dropna(subset=['paper_num'])
    if len(subset) > 0:
        print(f"\n=== KR {year} ===")
        subset_sorted = subset.sort_values('paper_num')
        for _, row in subset_sorted.iterrows():
            print(f"  Paper #{row['paper_num']:3.0f} -> Label {row['Label']}  | {row['title'][:80]}")

# For ICLP papers - check EPTCS patterns (10.4204/eptcs.XXX.YY)
print("\n\n=== ICLP EPTCS PATTERNS ===")
iclp = train[train['venue'] == 'iclp'].copy()

def extract_eptcs_info(doi):
    match = re.search(r'eptcs\.(\d+)\.(\d+)', str(doi))
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan

iclp[['eptcs_vol', 'eptcs_num']] = iclp['doi'].apply(lambda x: pd.Series(extract_eptcs_info(x)))

for vol in sorted(iclp['eptcs_vol'].dropna().unique()):
    subset = iclp[iclp['eptcs_vol'] == vol].sort_values('eptcs_num')
    if len(subset) > 0:
        print(f"\n--- EPTCS Volume {vol:.0f} ---")
        for _, row in subset.iterrows():
            print(f"  Paper #{row['eptcs_num']:3.0f} -> Label {row['Label']}  | {row['title'][:80]}")
