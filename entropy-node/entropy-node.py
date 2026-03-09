import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    length = len(y)
    x,count= np.unique(y,return_counts = True)
    p = count/length
    entropy = -np.sum(p*np.log2(p))
    return entropy
        
    return 0.0
    # Write code here
    pass