def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    if k <= 0:
        return [0.0, 0.0]


    top_k_recommended = recommended[:k]
    

    rel_set = set(relevant)
    hits = sum(1 for item in top_k_recommended if item in rel_set)

    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0

    return [precision, recall]