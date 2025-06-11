import pandas as pd
from sklearn.metrics import confusion_matrix

def store_scored_dataset(dataset, name, version=''):
    filename = f"Scored{name}{'' if not version else f'-{version}'}"
    dataset.to_pickle(f"{filename}.pkl")
    dataset.to_csv(f"{filename}.csv", index=False)
    dataset.to_excel(f"{filename}.xlsx", index=None, header=True)

def load_preprocessed_dataset(name):
    # Load Dataset
    data = pd.read_csv(f"../../Datasets/{name}.csv")
    print(f"Data Shape: {data.shape}")

    if data.shape[0] == 0:
        print("Nothing to Score")
        return

    # Preprocess Data
    data.drop_duplicates(subset=['Abstract'], inplace=True)
    data.dropna(subset=['Abstract'], inplace=True)

    # Drop Unnamed Columns
    # https://www.geeksforgeeks.org/how-to-drop-unnamed-column-in-pandas-dataframe/
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # Reset Index
    data.reset_index(drop=True, inplace=True)

    return data

# Cosine Similarity, KNN, Points
# output_fp: The filepath to the output dataset (the dataset that we are grading).
# target_fp: The filepath to the target dataset (the dataset that has the target scores).
# The target_fp filepath would be the name of a file in the 'Scores' folder.
# Each filepath/filename should lead to a CSV file that contains 'Title' and 'Score' as columns.
# output_threshold: This is the threshold for output scores. If a score is greater than or equal
# to a threshold it is converted into a 1 (which indicates that it contains a TMII example).
# If it's less than the threshold it is converted to a 0 (which indicates that it does not contain 
# a TMII example). To reverse this behavior (score <= threshold is 1 and score > threshold is 0)
# set reverse to True. target_threshold: This is the threshold for target scores. A score of 3 
# is the highest rating, which indicates that the paper probably contains a TMII example.
# This is the scoring that was being used, so I am going with the flow here. However, if you want 
# to count papers that had a score of 2 as a TMII example candidate, then you can use this parameter.
def measure_method_by_threshold(*, output_fp, target_fp="../../Scores/Baseline-1-BingKan.csv", output_threshold=1.0, target_threshold=3, verbose=False, reverse=False):
    target_df = pd.read_csv(target_fp)
    output_df = pd.read_csv(output_fp)

    # Raw Accuracy
    num_rows = 0
    num_correct = 0

    # Confusion Matrix
    targets = []
    outputs = []
    
    for i, row in output_df.iterrows():
        target_row = target_df.loc[target_df['Title'] == row['Title']]
        if target_row.empty:
            continue
    
        num_rows += 1

        # Finding Class from Score via Threshold
        output_score = row['Score']
        if reverse:
            output_class = 1 if output_score <= output_threshold else 0
        else:
            output_class = 1 if output_score >= output_threshold else 0
        assert output_class in [0, 1]
        
        if verbose:
            print(f"Output's Score: {output_score}")
            print(f"Output's Class: {output_class}")

        target_score = target_row['Score'].to_numpy()[0]
        target_class = 1 if target_score >= target_threshold else 0
        assert target_class in [0, 1]

        if verbose:
            print(f"Target's Score: {target_score}")
            print(f"Target's Class: {target_class}")
        
        if verbose:
            print(f"{output_class} v. {target_class}\n")
        
        if output_class == target_class:
            num_correct += 1

        targets.append(target_class)
        outputs.append(output_class)
    
    accuracy = num_correct/num_rows
    confusion = confusion_matrix(targets, outputs)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion
    }

# Zero Shot, Model
# output_fp: The filepath to the output dataset (the dataset that we are grading).
# target_fp: The filepath to the target dataset (the dataset that has the target scores).
# The target_fp filepath would be the name of a file in the 'Scores' folder.
# Each filepath/filename should lead to a CSV file that contains 'Title' and 'Score' as columns.
# Refer to the comments above for the target_threshold meaning (or the comments below).
def measure_method_by_class(*, output_fp, target_fp="../../Scores/Baseline-1-BingKan.csv", target_threshold=3, verbose=False):
    target_df = pd.read_csv(target_fp)
    output_df = pd.read_csv(output_fp)

    # Raw Accuracy
    num_rows = 0
    num_correct = 0

    # Confusion Matrix
    targets = []
    outputs = []
    
    for i, row in output_df.iterrows():
        target_row = target_df.loc[target_df['Title'] == row['Title']]
        
        if target_row.empty:
            continue
    
        num_rows += 1

        # The score is the class (0 or 1).
        output_class = row['Score']
        assert output_class in [0, 1]
            
        # The target score is converted to the class (0 or 1)
        # via the target_threshold.
        target_score = target_row['Score'].to_numpy()[0]
        target_class = 1 if target_score >= target_threshold else 0
        assert target_class in [0, 1]
        
        if verbose:
            print(f"Output Class: {output_class}")
            print(f"Target Score: {target_score}")
            print(f"Target Class: {target_class}")
            print(f"{output_class} v. {target_class}\n")
        
        if output_class == target_class:
            num_correct += 1

        targets.append(target_class)
        outputs.append(output_class)
    
    accuracy = num_correct/num_rows
    confusion = confusion_matrix(targets, outputs)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion
    }