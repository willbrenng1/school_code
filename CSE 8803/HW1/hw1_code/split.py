import pandas as pd


def split_data(
    data: pd.DataFrame, split_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ToDo: Split the dataset into train and test data using the split_ratio.
    Do not import any additional libraries for this task, doing so will cause the autograder to fail.
    For Autograder purposes only, please do not shuffle the dataset.

    Input:
        data: dataframe containing the dataset.
        split_ratio: desired ratio of the train and test splits.

    Output:
        train: train split of the data
        test: test split of the data
    """

    train = data.sample(frac=split_ratio, random_state=8803)
    test = data.drop(train.index)

    return tuple([train, test])
