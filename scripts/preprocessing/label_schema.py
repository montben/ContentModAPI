"""
Label schema definition for Muzzle content moderation.

Defines the 8-label multi-label classification schema and provides
utilities for converting between different dataset formats.
"""

from typing import Dict, List, Tuple
import numpy as np

# Define the 8 labels for content moderation
LABEL_SCHEMA = {
    "toxicity": 0,      # General toxic language
    "hate_speech": 1,   # Targeted hate based on identity
    "harassment": 2,    # Personal attacks and bullying
    "self_harm": 3,     # Suicide ideation and self-injury
    "violence": 4,      # Threats and violent content
    "sexual": 5,        # Sexual harassment and inappropriate content
    "profanity": 6,     # Explicit language and swearing
    "spam": 7           # Promotional content and off-topic posts
}

# Reverse mapping for label names
LABEL_NAMES = {v: k for k, v in LABEL_SCHEMA.items()}

# Label descriptions for documentation
LABEL_DESCRIPTIONS = {
    "toxicity": "General toxic, rude, or disrespectful language",
    "hate_speech": "Content targeting individuals/groups based on identity",
    "harassment": "Personal attacks, bullying, or targeted harassment",
    "self_harm": "Content related to suicide, self-injury, or eating disorders",
    "violence": "Threats of violence or graphic violent content",
    "sexual": "Sexual harassment, unwanted advances, or explicit sexual content",
    "profanity": "Explicit language, swearing, or vulgar content",
    "spam": "Promotional content, advertisements, or off-topic posts"
}


def create_label_vector(labels: Dict[str, bool]) -> np.ndarray:
    """
    Convert label dictionary to 8-bit numpy array.

    Args:
        labels: Dictionary mapping label names to boolean values

    Returns:
        8-bit numpy array representing the labels

    Example:
        >>> labels = {"toxicity": True, "hate_speech": False, "harassment": True}
        >>> create_label_vector(labels)
        array([1, 0, 1, 0, 0, 0, 0, 0])
    """
    vector = np.zeros(8, dtype=np.int8)

    for label_name, is_present in labels.items():
        if label_name in LABEL_SCHEMA:
            vector[LABEL_SCHEMA[label_name]] = int(is_present)

    return vector


def vector_to_labels(vector: np.ndarray, threshold: float = 0.5) -> Dict[str, bool]:
    """
    Convert 8-bit vector back to label dictionary.

    Args:
        vector: 8-bit numpy array or list
        threshold: Threshold for binary classification (for probability vectors)

    Returns:
        Dictionary mapping label names to boolean values
    """
    labels = {}

    for i, value in enumerate(vector):
        label_name = LABEL_NAMES[i]
        labels[label_name] = bool(value > threshold)

    return labels


def map_jigsaw_labels(jigsaw_row: Dict) -> Dict[str, bool]:
    """
    Map Jigsaw dataset labels to our 8-label schema.

    Jigsaw has: toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    return {
        "toxicity": bool(jigsaw_row.get("toxic", 0) or jigsaw_row.get("severe_toxic", 0)),
        "hate_speech": bool(jigsaw_row.get("identity_hate", 0)),
        "harassment": bool(jigsaw_row.get("insult", 0)),
        "self_harm": False,  # Not in Jigsaw dataset
        "violence": bool(jigsaw_row.get("threat", 0)),
        "sexual": bool(jigsaw_row.get("obscene", 0)),  # Approximate mapping
        "profanity": bool(jigsaw_row.get("obscene", 0)),
        "spam": False  # Not in Jigsaw dataset
    }


def map_davidson_labels(davidson_row: Dict) -> Dict[str, bool]:
    """
    Map Davidson dataset labels to our 8-label schema.

    Davidson has: hate_speech, offensive_language, neither
    """
    hate = davidson_row.get("class") == 0  # hate speech
    offensive = davidson_row.get("class") == 1  # offensive language

    return {
        "toxicity": bool(hate or offensive),
        "hate_speech": bool(hate),
        "harassment": bool(hate),  # Overlap with hate speech
        "self_harm": False,
        "violence": False,  # Would need text analysis
        "sexual": False,    # Would need text analysis
        "profanity": bool(offensive),
        "spam": False
    }


def map_hate_speech_offensive_labels(row: Dict) -> Dict[str, bool]:
    """
    Map hate_speech_offensive dataset labels to our 8-label schema.

    Dataset has: class (0=hate_speech, 1=offensive_language, 2=neither)
    """
    hate = row.get("class") == 0  # hate speech
    offensive = row.get("class") == 1  # offensive language
    neither = row.get("class") == 2  # neither

    return {
        "toxicity": bool(hate or offensive),
        "hate_speech": bool(hate),
        "harassment": bool(hate),  # Overlap with hate speech
        "self_harm": False,
        "violence": False,  # Would need text analysis
        "sexual": False,    # Would need text analysis
        "profanity": bool(offensive),
        "spam": False
    }


def map_hateval_labels(hateval_row: Dict) -> Dict[str, bool]:
    """
    Map HatEval dataset labels to our 8-label schema.

    HatEval has: HS (hate_speech), TR (target), AG (aggressiveness)
    - HS: 1 = hate speech, 0 = not hate speech
    - TR: 1 = individual target, 0 = group target
    - AG: 1 = aggressive, 0 = not aggressive
    """
    hate_speech = bool(hateval_row.get("HS", 0))
    aggressive = bool(hateval_row.get("AG", 0))
    individual_target = bool(hateval_row.get("TR", 0))

    return {
        "toxicity": bool(hate_speech or aggressive),
        "hate_speech": bool(hate_speech),
        "harassment": bool(hate_speech and individual_target),  # Hate speech targeting individuals
        "self_harm": False,
        "violence": bool(aggressive),  # Aggressive content may indicate violence
        "sexual": False,  # Would need text analysis
        "profanity": bool(aggressive),  # Aggressive language often includes profanity
        "spam": False
    }


# Dataset mapping functions registry
DATASET_MAPPERS = {
    "jigsaw": map_jigsaw_labels,
    "davidson": map_davidson_labels,
    "hate_speech_offensive": map_hate_speech_offensive_labels,
    "hateval": map_hateval_labels,
    # Add more as you integrate additional datasets
}


def get_label_statistics(labels_list: List[Dict[str, bool]]) -> Dict[str, Dict]:
    """
    Calculate statistics for label distribution.

    Args:
        labels_list: List of label dictionaries

    Returns:
        Statistics for each label including count and percentage
    """
    stats = {}
    total_samples = len(labels_list)

    for label_name in LABEL_SCHEMA.keys():
        count = sum(1 for labels in labels_list if labels.get(label_name, False))
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0

        stats[label_name] = {
            "count": count,
            "percentage": percentage,
            "description": LABEL_DESCRIPTIONS[label_name]
        }

    return stats


if __name__ == "__main__":
    # Example usage
    example_labels = {
        "toxicity": True,
        "hate_speech": False,
        "harassment": True,
        "self_harm": False,
        "violence": False,
        "sexual": False,
        "profanity": True,
        "spam": False
    }

    # Convert to vector
    vector = create_label_vector(example_labels)
    print(f"Label vector: {vector}")

    # Convert back to labels
    recovered_labels = vector_to_labels(vector)
    print(f"Recovered labels: {recovered_labels}")

    # Print schema info
    print(f"\nLabel Schema ({len(LABEL_SCHEMA)} labels):")
    for name, idx in LABEL_SCHEMA.items():
        print(f"  {idx}: {name} - {LABEL_DESCRIPTIONS[name]}")
