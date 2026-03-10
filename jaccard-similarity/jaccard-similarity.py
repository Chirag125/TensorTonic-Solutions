def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    union = set_a + set_b
    union = set(union)
    print(union)
    intersection = set(set_a) & set(set_b)
    if(union == 0 or (len(set_a+set_b) == 0)):
        return 0
    else:
        return len(intersection)/len(union)
    
    
    # Write code here