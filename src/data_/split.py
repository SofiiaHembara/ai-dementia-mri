import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(train_df, groups=train_df['subject_id']))
    train_df, val_df = train_df.iloc[train_idx].copy(), train_df.iloc[val_idx].copy()

    return train_df, val_df, test_df