from pathlib import Path

import numpy as np
import pandas as pd


def extract_embedding(embedding, verbose=False):
    features = list()
    id2index = dict()
    index2fn = dict()
    filenames = set()
    for i, (fn, vector_embedding) in enumerate(embedding):
        fn = str(fn)
        _id = Path(fn).stem
        if _id not in id2index and fn not in filenames:
            index = len(features)
            index2fn[index] = fn
            id2index[_id] = index
            filenames.add(fn)
            features.append(vector_embedding)
        elif verbose:
            print(f"Warning: Duplicated id or filename (id={_id}, fn={fn})")
    features = np.asarray(features)
    return features, id2index, index2fn


def get_interactions_dataframe(interactions_path, display_stats=False):
    # Load interactions from CSV
    interactions_df = pd.read_csv(interactions_path)

    # Display stats
    if display_stats:
        for column in interactions_df.columns:
            print(f"Interactions - {column}: {interactions_df[column].nunique()} unique values")

    return interactions_df


def mark_evaluation_rows(interactions_df, threshold=None):
    if threshold is None:
        threshold = 1

    def _mark_evaluation_rows(group):
        # Only the last 'threshold' items are used for evaluation,
        # unless less items are available (then they're used for training)
        evaluation_series = pd.Series(False, index=group.index)
        if len(group) > threshold:
            evaluation_series.iloc[-threshold:] = True
        return evaluation_series

    # Mark evaluation rows
    interactions_df["evaluation"] = interactions_df.groupby(["user_id"])["user_id"].apply(_mark_evaluation_rows)
    # Sort transactions by timestamp
    interactions_df = interactions_df.sort_values("timestamp")
    # Reset index according to new order
    interactions_df = interactions_df.reset_index(drop=True)
    return interactions_df


def get_holdout(interactions_df):
    # Create evaluation dataframe
    holdout = []
    for user_id, group in interactions_df.groupby("user_id"):
        # Check if there's a profile for training
        profile_rows = group[~group["evaluation"]]
        predict_rows = group[group["evaluation"]]
        # Extract items
        profile = profile_rows["item_id"].values.tolist()
        if type(profile[0]) is list:
            profile = [item for p in profile for item in p]
        # Keep last interactions for evaluation
        for _, p in predict_rows.iterrows():
            timestamp = p["timestamp"]
            predict = p["item_id"]
            holdout.append([timestamp, profile, predict, user_id])
            # profile.extend(predict)  # If profile grows in evaluation
    # Store holdout in a pandas dataframe
    holdout = pd.DataFrame(
        holdout,
        columns=["timestamp", "profile", "predict", "user_id"],
    )
    holdout = holdout.sort_values(by=["timestamp"])
    holdout = holdout.reset_index(drop=True)

    # Pick interactions not used for evaluation
    new_dataset = interactions_df[~interactions_df["evaluation"]]
    # Sort transactions by timestamp
    new_dataset = new_dataset.sort_values("timestamp")
    # Reset index according to new order
    new_dataset = new_dataset.reset_index(drop=True)

    return holdout, new_dataset
