import itertools
import pandas as pd
from typing import List, Tuple, Set

import gini_index
import information_gain
from classes.decision_tree_node import DecisionTreeNode
from classes.decision_tree_leaf_node import DecisionTreeLeafNode
from classes.decision_tree_internal_node import DecisionTreeInternalNode
from classes.decision_tree_branch import DecisionTreeBranch
from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome
from classes.decision_tree_decision_outcome_above import (
    DecisionTreeDecisionOutcomeAbove,
)
from classes.decision_tree_decision_outcome_below_equal import (
    DecisionTreeDecisionOutcomeBelowEqual,
)
from classes.decision_tree_decision_outcome_equals import (
    DecisionTreeDecisionOutcomeEquals,
)
from classes.decision_tree_decision_outcome_in_list import (
    DecisionTreeDecisionOutcomeInList,
)

class DecisionTree:
    """
    A Decision Tree classifier.
    """

    def __init__(self):
        """
        Initialize the DecisionTree object.
        """

        # Function fit will later populate this variable
        self.target_attribute: str = None

        # Function fit will later produce a decision tree
        self.tree: DecisionTreeNode = None

    def fit(
        self,
        dataset: pd.DataFrame,
        target_attribute: str,
        attribute_selection_method: str,
    ):
        """
        Fit decision tree on a given dataset and target attribute, using a specified attribute selection method.

        Parameters:
        dataset (pd.DataFrame): The dataset to fit the decision tree on
        target_attribute (str): The target attribute to predict
        attribute_selection_method (str): The attribute selection method to use
        """
        # Make sure that the target_attribute is in the dataset
        if target_attribute not in dataset.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not in dataset.")

        # Make sure that the attribute_selection_method is valid
        if attribute_selection_method not in [
            "information_gain",
            "gini_index",
        ]:
            raise ValueError(
                f"Attribute selection method '{attribute_selection_method}' not valid (select either 'information_gain' or 'gini_index')."
            )

        # TODO
        self.target_attribute = target_attribute
        attribute_list = [col for col in dataset.columns if col != target_attribute]
        self.tree = self._build_tree(dataset.copy(), attribute_list, attribute_selection_method)

    def _build_tree(
        self,
        data: pd.DataFrame,
        attribute_list: List[str],
        attribute_selection_method: str,
    ) -> DecisionTreeNode:
        """
        Recursively build the decision tree.

        Parameters:
        data (pd.DataFrame): The (partial) dataset to build the decision tree with
        attribute_list (List[str]): The list of attributes to consider
        attribute_selection_method (str): The attribute selection method to use

        Returns:
        DecisionTreeNode: The root node of the decision tree
        """
        # TODO
        # Base Case 1: If the dataset is empty, this branch leads to an undefined outcome.
        if data.empty:
            raise ValueError("Empty dataset passed to _build_tree where it's not expected to be empty.")
        # Base Case 2: All instances in the current data subset belong to the same class
        if data[self.target_attribute].nunique() == 1:
            return DecisionTreeLeafNode(class_label=data[self.target_attribute].iloc[0])
        # Base Case 3: No more attributes to split on
        if not attribute_list:
            return DecisionTreeLeafNode(class_label=data[self.target_attribute].mode()[0])

        best_attribute, best_outcomes = self._find_best_split(data, attribute_list, attribute_selection_method)

        if best_attribute is None or not best_outcomes:
            return DecisionTreeLeafNode(class_label=data[self.target_attribute].mode()[0])

        branches: List[DecisionTreeBranch] = []
        for outcome in best_outcomes:
            if isinstance(outcome, DecisionTreeDecisionOutcomeEquals):
                subset_data = data[data[best_attribute] == outcome.value]
            elif isinstance(outcome, DecisionTreeDecisionOutcomeInList):
                subset_data = data[data[best_attribute].isin(list(outcome.value))]
            elif isinstance(outcome, DecisionTreeDecisionOutcomeBelowEqual):
                subset_data = data[data[best_attribute] <= outcome.value]
            elif isinstance(outcome, DecisionTreeDecisionOutcomeAbove):
                subset_data = data[data[best_attribute] > outcome.value]
            else:
                raise TypeError(f"Unknown DecisionTreeDecisionOutcome type: {type(outcome)}")

            if subset_data.empty:
                leaf_node = DecisionTreeLeafNode(class_label=data[self.target_attribute].mode()[0])
                branches.append(DecisionTreeBranch(outcome, leaf_node))
            else:
                new_attribute_list = [attr for attr in attribute_list if attr != best_attribute]
                subtree = self._build_tree(subset_data.copy(), new_attribute_list, attribute_selection_method)
                branches.append(DecisionTreeBranch(outcome, subtree))

        return DecisionTreeInternalNode(best_attribute, branches)

    def _find_best_split(
        self,
        data: pd.DataFrame,
        attribute_list: List[str],
        attribute_selection_method: str,
    ) -> Tuple[str, List[DecisionTreeDecisionOutcome]]:
        """
        Find the best split for a given dataset and attribute list. Finding the best split includes finding the best attribute to split on and also (depending on the attribute selection method) the best set of outcomes to split on this attribute.

        Parameters:
        data (pd.DataFrame): The dataset to find the best splitting attribute for
        attribute_list (List[str]): The list of attributes to consider
        attribute_selection_method (str): The attribute selection method to use

        Returns:
        str: The attribute to split on
        List[DecisionTreeDecisionOutcome]: The outcomes a split on this attribute should have
        """
        # TODO
        best_metric_value = -float('inf')
        best_attribute = None
        best_outcomes = []

        for attribute in attribute_list:
            if attribute_selection_method == "information_gain":
                current_metric_value, current_outcomes = self._calculate_information_gain(data, attribute)
            elif attribute_selection_method == "gini_index":
                current_metric_value, current_outcomes = self._calculate_gini_index(data, attribute)
            else:
                raise ValueError("Invalid attribute selection method.")

            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_attribute = attribute
                best_outcomes = current_outcomes

        return best_attribute, best_outcomes

    def _calculate_information_gain(
        self, data: pd.DataFrame, attribute: str
    ) -> Tuple[float, List[DecisionTreeDecisionOutcome]]:
        """
        Calculate the (best) information gain for a given attribute in a dataset.

        Parameters:
        data (pd.DataFrame): The dataset to calculate the information gain for
        attribute (str): The attribute to calculate the information gain for

        Returns:
        float: The calculated information gain
        List[DecisionTreeDecisionOutcome]: The outcomes the best split of this attribute has
        """
        # If self.target_attribute is not set, raise an error
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # If the attribute is not in the dataset, raise an error
        if attribute not in data.columns:
            raise ValueError(f"Attribute '{attribute}' not in dataset.")

        # TODO
        best_gain = -float('inf')
        best_outcomes: List[DecisionTreeDecisionOutcome] = []

        if data[attribute].dtype == "object":  # Categorical attribute
            current_gain = information_gain.calculate_information_gain(
                data, self.target_attribute, attribute, split_value=None
            )

            best_gain = current_gain
            unique_values = data[attribute].unique()
            best_outcomes = [DecisionTreeDecisionOutcomeEquals(value=val) for val in unique_values]

        else:  # Continuous attribute (int or float)
            unique_values = sorted(data[attribute].unique())
            split_candidates = [
                (unique_values[i] + unique_values[i + 1]) / 2
                for i in range(len(unique_values) - 1)
            ]

            for split_val in split_candidates:
                current_gain = information_gain.calculate_information_gain(
                    data, self.target_attribute, attribute, split_val
                )
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_outcomes = [
                        DecisionTreeDecisionOutcomeBelowEqual(split_val),
                        DecisionTreeDecisionOutcomeAbove(split_val),
                    ]
        return best_gain, best_outcomes

    def _calculate_gini_index(
        self, data: pd.DataFrame, attribute: str
    ) -> Tuple[float, List[DecisionTreeDecisionOutcome]]:
        """
        Calculate the (best) gini index for a given attribute in a dataset.

        Parameters:
        data (pd.DataFrame): The dataset to calculate the gini index for
        attribute (str): The attribute to calculate the gini index for

        Returns:
        float: The calculated gini index (reduction of impurity)
        List[DecisionTreeDecisionOutcome]: The outcomes the best split of this attribute has
        """
        # If self.target_attribute is not set, raise an error
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # If the attribute is not in the dataset, raise an error
        if attribute not in data.columns:
            raise ValueError(f"Attribute '{attribute}' not in dataset.")

        # TODO
        best_gini = -float('inf')
        best_outcomes: List[DecisionTreeDecisionOutcome] = []

        if data[attribute].dtype == "object":  # Categorical attribute
            unique_values = list(data[attribute].unique())

            if len(unique_values) <= 1:
                return 0.0, []

            all_values_set = set(unique_values)
            for i in range(1, len(unique_values) // 2 + 1):
                for combo in itertools.combinations(unique_values, i):
                    split_set_1 = set(combo)
                    split_set_2 = all_values_set - split_set_1

                    if not split_set_1 or not split_set_2:
                        continue

                    current_gini = gini_index.calculate_gini_index(
                        data, self.target_attribute, attribute, split_set_1
                    )

                    if current_gini > best_gini:
                        best_gini = current_gini
                        best_outcomes = [
                            DecisionTreeDecisionOutcomeInList(list(split_set_1)),
                            DecisionTreeDecisionOutcomeInList(list(split_set_2))
                        ]

        else:  # Continuous attribute (int or float)
            unique_values = sorted(data[attribute].unique())
            split_candidates = [
                (unique_values[i] + unique_values[i + 1]) / 2
                for i in range(len(unique_values) - 1)
            ]

            for split_val in split_candidates:
                current_gini = gini_index.calculate_gini_index(
                    data, self.target_attribute, attribute, split_val
                )
                if current_gini > best_gini:
                    best_gini = current_gini
                    best_outcomes = [
                        DecisionTreeDecisionOutcomeBelowEqual(split_val),
                        DecisionTreeDecisionOutcomeAbove(split_val),
                    ]
        return best_gini, best_outcomes

    def predict(self, dataset: pd.DataFrame) -> List[str | int | float]:
        """
        Predict the target attribute for a given dataset.

        Parameters:
        dataset (pd.DataFrame): The dataset to predict the target attribute for

        Returns:
        List[str | int | float]: A list of predicted class labels
        """

        # If the tree is not fitted, raise an error
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit method first.")

        # TODO
        predictions = []
        for index, row in dataset.iterrows():
            predictions.append(self._predict_tuple(row, self.tree))
        return predictions

    def _predict_tuple(
        self, tuple: pd.Series, node: DecisionTreeNode
    ) -> str | int | float:
        """
        Predict the target attribute for a given row in the dataset.
        This is a recursive function that traverses the decision tree until a leaf node is reached.

        Parameters:
        tuple (pd.Series): The row to predict the target attribute for
        node (DecisionTreeNode): The current node in the decision tree

        Returns:
        str | int | float: The predicted class label
        """
        # TODO
        if isinstance(node, DecisionTreeLeafNode):
            return node.get_label()
        elif isinstance(node, DecisionTreeInternalNode):
            attribute_to_check = node.get_label()
            value_in_tuple = tuple[attribute_to_check]

            for branch in node.get_branches():
                if branch.value_matches(value_in_tuple):
                    return self._predict_tuple(tuple, branch.get_branch_node())
            raise ValueError(
                f"No matching branch found for value '{value_in_tuple}' of attribute '{attribute_to_check}'")
        else:
            raise TypeError("Unknown node type encountered in prediction.")