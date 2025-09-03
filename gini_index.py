import pandas as pd
from math import log
from typing import List, Set

"""
Collection of functions to calculate the impurity and the gini index of attributes in a dataset.
"""


def calculate_impurity(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated impurity
    """
    # TODO
    if len(dataset) == 0:
        return 0.0

    class_cnt = dataset[target_attribute].value_counts()
    rows_cnt = len(dataset) # total number of instances

    sum_of_squares = 0.0
    for cnt in class_cnt:
        probability = cnt / rows_cnt
        sum_of_squares += probability ** 2

    gini_impurity = 1 - sum_of_squares
    return gini_impurity


def calculate_impurity_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).
    """
    # TODO
    if len(dataset) == 0:
        return 0.0

    weighted_impurity = 0.0
    rows_cnt = len(dataset)

    if isinstance(split, (int, float)):
        subset_le = dataset[dataset[partition_attribute] <= split]
        subset_gt = dataset[dataset[partition_attribute] > split]

        if len(subset_le) != 0:
            probability_le = len(subset_le) / rows_cnt
            impurity_le = calculate_impurity(subset_le, target_attribute)
            weighted_impurity += probability_le * impurity_le
        if len(subset_gt) != 0:
            probability_gt = len(subset_gt) / rows_cnt
            impurity_gt = calculate_impurity(subset_gt, target_attribute)
            weighted_impurity += probability_gt * impurity_gt

    if isinstance(split, Set):
        subset_in_split = dataset[dataset[partition_attribute].isin(list(split))]
        subset_not_in_split = dataset[~dataset[partition_attribute].isin(list(split))]

        if len(subset_in_split) != 0:
            probability_in = len(subset_in_split) / rows_cnt
            impurity_in = calculate_impurity(subset_in_split, target_attribute)
            weighted_impurity += probability_in * impurity_in
        if len(subset_not_in_split) != 0:
            probability_not_in = len(subset_not_in_split) / rows_cnt
            impurity_not_in = calculate_impurity(subset_not_in_split, target_attribute)
            weighted_impurity += probability_not_in * impurity_not_in

    return weighted_impurity

def calculate_gini_index(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the Gini index (= reduction of impurity) for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the Gini index for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).

    Returns:
    float: The calculated Gini index
    """
    # TODO
    initial_gini_impurity = calculate_impurity(dataset, target_attribute)
    partitioned_impurity = calculate_impurity_partitioned(dataset, target_attribute, partition_attribute, split)

    return initial_gini_impurity - partitioned_impurity