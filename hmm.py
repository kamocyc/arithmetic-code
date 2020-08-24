from typing import Dict, List

def get_stationary_distribution(mat_list: List[List[float]], emission_list: List[List[float]])-> List[float]:
    import numpy as np
    
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
        
    mat = np.matrix(mat_list)
    res = np.linalg.matrix_power(mat, 100)
    # print({"^100": res[0,:]})
    w, v = np.linalg.eig(np.transpose(mat))
    # print({"eigenvalues": w})
    p_vec = v[:,0] / np.sum(v[:,0])
    # print({"eigenvector": p_vec})
    
    emission_mat = np.transpose(np.matrix(emission_list))
    
    emission_prob = emission_mat * p_vec
    
    return sum(emission_prob.tolist(), [])

def choose_randomly(probs: Dict[str, float])-> str:
    import random
    
    p = random.random()
    acc_p = 0.0
    
    if len(probs) == 0:
        raise ValueError("probability dictionary should not empty")
    
    k = ""
    for k, i in probs.items():
        if p < acc_p + i:
            return k
        
        acc_p += i
    
    return k
