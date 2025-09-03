import pandas as pd
from typing import List, Set
import math

from classes.naive_bayes_likelihoods import NaiveBayesLikelihoods
from classes.naive_bayes_prior_probabilities import NaiveBayesPriorProbabilities


class NaiveBayes:
    """
    A Naive Bayes classifier.
    """

    def __init__(self):
        # The target attribute to predict
        self.target_attribute: str = None

        # A full set of possible class labels ( = values of the target attribute)
        self.class_labels: Set[str | int | float] = None

        # The likelihoods of the classifier
        self.likelihoods: NaiveBayesLikelihoods = None

        # The prior probabilities of the classifier
        self.prior_probabilities: NaiveBayesPriorProbabilities = None

    def fit(self, dataset: pd.DataFrame, target_attribute: str):
        """
        Fit the Naive Bayes classifier to the training dataset.
        Sets the target attribute and the class labels.
        Calculates the prior probabilities, and the likelihoods.

        Parameters:
        dataset (pd.DataFrame): The training dataset
        target_attribute (str): The target attribute to predict
        """
        # Make sure that the target_attribute is in the dataset
        if target_attribute not in dataset.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not in dataset.")

        # TODO
        self.target_attribute = target_attribute
        self.class_labels = set(dataset[target_attribute].unique())
        self.prior_probabilities = self._calculate_prior_probabilities(dataset)
        self.likelihoods = self._calculate_likelihoods(dataset)

    def _calculate_prior_probabilities(
        self, dataset: pd.DataFrame
    ) -> NaiveBayesPriorProbabilities:
        """
        Calculate the prior probability for each class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesPriorProbabilities: The prior probabilities for each class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO
        prior_probs = NaiveBayesPriorProbabilities()
        total_rows = len(dataset)

        class_counts = dataset[self.target_attribute].value_counts()

        for class_label, count in class_counts.items():
            probability = count / total_rows
            prior_probs.add_prior_probability(class_label, probability)

        return prior_probs

    def _calculate_likelihoods(self, dataset: pd.DataFrame) -> NaiveBayesLikelihoods:
        """
        Calculate the likelihoods for each attribute and class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesLikelihoods: The likelihoods for each attribute and class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO
        likelihoods = NaiveBayesLikelihoods()
        feature_attributes = [col for col in dataset.columns if col != self.target_attribute]

        class_labels_to_use = self.class_labels
        if class_labels_to_use is None:
            if self.target_attribute not in dataset.columns:
                raise ValueError(
                    f"Target attribute '{self.target_attribute}' not found in the dataset provided for likelihood calculation.")
            class_labels_to_use = set(dataset[self.target_attribute].unique())

        for attribute in feature_attributes:
            for class_label in class_labels_to_use:
                subset_data = dataset[dataset[self.target_attribute] == class_label]

                if subset_data.empty:
                    if pd.api.types.is_numeric_dtype(dataset[attribute]):
                        continue
                    else:
                        all_unique_values_for_attribute_in_dataset = dataset[attribute].unique()
                        for value_candidate in all_unique_values_for_attribute_in_dataset:
                            likelihoods.add_categorical_likelihood(attribute, value_candidate, class_label, 0.0)
                        continue

                if pd.api.types.is_numeric_dtype(dataset[attribute]):
                    mean = subset_data[attribute].mean()
                    std = subset_data[attribute].std()

                    if std == 0:
                        std = 1e-6

                    likelihoods.add_continuous_likelihood(attribute, class_label, mean, std)
                else:
                    total_class_instances = len(subset_data)
                    all_unique_values_for_attribute_in_dataset = dataset[attribute].unique()

                    for value_candidate in all_unique_values_for_attribute_in_dataset:
                        count = subset_data[attribute].eq(value_candidate).sum()
                        likelihood = count / total_class_instances

                        likelihoods.add_categorical_likelihood(attribute, value_candidate, class_label, likelihood)

        return likelihoods

    def predict(self, dataset: pd.DataFrame) -> List[str | int | float]:
        """
        Predict the target attribute for a given dataset.

        Parameters:
        dataset (pd.DataFrame): The dataset to predict the target attribute for

        Returns:
        List[str | int | float]: A list of predicted class labels
        """

        # If the likelihoods or/and the prior probabilities are not calculated yet, raise an error
        if self.likelihoods is None or self.prior_probabilities is None:
            raise ValueError("Model not trained yet.")

        # TODO
        if self.class_labels is None:
            raise ValueError("Class labels not set. Model not trained correctly.")

        predictions = []
        for index, row in dataset.iterrows():
            predictions.append(self._predict_tuple(row))
        return predictions

    def _predict_tuple(self, tuple: pd.Series) -> str | int | float:
        """
        Predict the target attribute for a given row in the dataset.

        Parameters:
        tuple (pd.Series): The row in the dataset to predict the target attribute for

        Returns:
        str | int | float: The predicted class label
        """
        # TODO
        max_log_posterior_prob = -float('inf')
        predicted_class = None

        feature_attributes = [col for col in tuple.index if col != self.target_attribute]

        for class_label in self.class_labels:
            try:
                prior_prob = self.prior_probabilities.get_prior_probability(class_label)
            except ValueError:
                continue

            current_log_posterior_prob = math.log(prior_prob)

            for attribute in feature_attributes:
                value = tuple[attribute]
                try:
                    likelihood = self.likelihoods.get_likelihood(attribute, value, class_label)
                    if likelihood == 0:
                        current_log_posterior_prob = -float('inf')
                        break
                    current_log_posterior_prob += math.log(likelihood)
                except ValueError as e:
                    current_log_posterior_prob = -float('inf')
                    break

            if current_log_posterior_prob > max_log_posterior_prob:
                max_log_posterior_prob = current_log_posterior_prob
                predicted_class = class_label

        if predicted_class is None:
            if self.prior_probabilities.prior_probabilities:
                predicted_class = max(self.prior_probabilities.prior_probabilities,
                                      key=self.prior_probabilities.prior_probabilities.get)
            else:
                raise ValueError("Cannot predict: No class labels or prior probabilities available after training.")

        return predicted_class