import pandas as pd
from math import log
from typing import List

"""
Collection of functions to calculate the entropy, information and 
information gain of attributes in a dataset.
"""


def calculate_entropy(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the entropy for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the entropy for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated entropy (= expected information)
    """
    # TODO
    if len(dataset) == 0.0:
        return 0.0

    class_cnt = dataset[target_attribute].value_counts() # gives the number of occurrence of each unique class
    rows_cnt = len(dataset)

    entropy = 0.0
    for cnt in class_cnt:
        probability = cnt / rows_cnt
        if probability > 0.0:
            entropy -= probability * log(probability, 2)

    return entropy

def calculate_information_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information for a discrete-valued partition attribute. If set to a value, the function will calculate the information for a continuous-valued partition attribute.
    """
    # TODO
    if len(dataset) == 0.0:
        return 0.0

    weighted_entropy = 0.0
    rows_cnt = len(dataset) # total number of instances
    if split_value is None:
        unique_values = dataset[partition_attribute].unique()
        for each_unique_value in unique_values:
            subset = dataset[dataset[partition_attribute] == each_unique_value]
            if len(subset) != 0:
                probability = len(subset) / rows_cnt
                entropy_of_subset = calculate_entropy(subset, target_attribute)
                weighted_entropy += probability * entropy_of_subset
    else:
        subset_le = dataset[dataset[partition_attribute] <= split_value]
        subset_gt = dataset[dataset[partition_attribute] > split_value]

        if len(subset_le) != 0:
            probability_le = len(subset_le) / len(dataset)
            entropy_le = calculate_entropy(subset_le, target_attribute)
            weighted_entropy += probability_le * entropy_le
        if len(subset_gt) != 0:
            probability_gt = len(subset_gt) / len(dataset)
            entropy_gt = calculate_entropy(subset_gt, target_attribute)
            weighted_entropy += probability_gt * entropy_gt

    return weighted_entropy

def calculate_information_gain(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information gain for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information gain for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information gain for a discrete-valued partition attribute. If set to a value, the function will calculate the information gain for a continuous-valued partition attribute.

    Returns:
    float: The calculated information gain
    """
    # TODO
    initial_entropy = calculate_entropy(dataset, target_attribute)
    partitioned_info = calculate_information_partitioned(dataset, target_attribute, partition_attribute, split_value)

    return initial_entropy - partitioned_info
