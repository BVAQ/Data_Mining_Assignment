import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('data/raw/Stage_1_publcitrain.csv')
test_raw = pd.read_csv('data/raw/test (2).csv')
sample = pd.read_csv('data/raw/sample_submission_DM252.csv')

# Check venue x label cross-tab
print('=== VENUE x LABEL CROSSTAB ===')
print(pd.crosstab(train['venue'], train['Label']))

# Check year x label
print('\n=== YEAR x LABEL CROSSTAB ===')
print(pd.crosstab(train['year'], train['Label']))

# Check preprocessed
print('\n=== PREPROCESSED TRAIN ===')
train_pre = pd.read_csv('data/preprocessed/Stage_1_train_with_abstracts.csv')
print(f'Shape: {train_pre.shape}')
print(f'Columns: {train_pre.columns.tolist()}')
print(f'\nLabel distribution:')
print(train_pre["Label"].value_counts().sort_index())

# Check how many have abstracts
print(f'\nPreprocessed train non-null abstracts: {train_pre["abstract"].notna().sum()} / {len(train_pre)}')

print('\n=== PREPROCESSED TEST ===')
test_pre = pd.read_csv('data/preprocessed/Stage_1_test_with_abstracts.csv')
print(f'Shape: {test_pre.shape}')
print(f'Columns: {test_pre.columns.tolist()}')
print(f'Preprocessed test non-null abstracts: {test_pre["abstract"].notna().sum()} / {len(test_pre)}')

# DOI info
print('\n=== DOI examples ===')
print(train['doi'].head(10).tolist())

# Check title length by label
print('\n=== TITLE LENGTH BY LABEL ===')
train['title_len'] = train['title'].str.len()
print(train.groupby('Label')['title_len'].describe())

# Authors missing by label
print('\n=== AUTHORS MISSING BY LABEL ===')
print(train.groupby('Label')['authors'].apply(lambda x: x.isna().sum()))

# Preprocessed abstract examples
print('\n=== PREPROCESSED ABSTRACT EXAMPLES ===')
for i in range(3):
    abstract = str(train_pre.iloc[i]['abstract'])[:300]
    print(f"Row {i} abstract: {abstract}")
    print(f"Row {i} title: {train_pre.iloc[i]['title']}")
    print(f"Row {i} Label: {train_pre.iloc[i]['Label']}")
    print()

# Prev submission
print('\n=== PREVIOUS SUBMISSION ===')
sub = pd.read_csv('outputs/submission.csv')
print(sub['Label'].value_counts().sort_index())
