import pandas as pd
import numpy as np

# Small dataset
print("=== SMALL TEST DATA ===")
ratings = pd.read_csv('data/raw/test_sample_ratings.csv')
print(f"Total: {len(ratings)} ratings")

rng = np.random.RandomState(42)
mask = rng.rand(len(ratings)) < 0.8
train = ratings[mask]
test = ratings[~mask]

print(f"Train: {len(train)} ratings ({len(train)/len(ratings)*100:.1f}%)")
print(f"Test: {len(test)} ratings ({len(test)/len(ratings)*100:.1f}%)")
print(f"Train unique users: {train['user_id'].nunique()}")
print(f"Test unique users: {test['user_id'].nunique()}")
both = len(set(train['user_id']) & set(test['user_id']))
print(f"Users in both sets: {both}")
print()

# Full dataset
print("=== FULL DATASET ===")
full = pd.read_csv('data/raw/ratings.csv')
print(f"Total: {len(full)} ratings")

mask_full = rng.rand(len(full)) < 0.8
train_full = full[mask_full]
test_full = full[~mask_full]

print(f"Train: {len(train_full)} ratings ({len(train_full)/len(full)*100:.1f}%)")
print(f"Test: {len(test_full)} ratings ({len(test_full)/len(full)*100:.1f}%)")
print(f"Train unique users: {train_full['user_id'].nunique()}")
print(f"Test unique users: {test_full['user_id'].nunique()}")
both_full = len(set(train_full['user_id']) & set(test_full['user_id']))
print(f"Users in both sets: {both_full}")
