import numpy as np
from scipy.stats import bernoulli
import itertools
import random
from multiprocess import Pool
import pickle

def save_file(obj, name):
    f = open(name,"wb")
    pickle.dump(obj,f)
    f.close()

def load_file(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f) # deserialize using load()
    return obj

def generate_sample(graph, cond_probs):
    sample=np.array([0]*len(graph))
    for i in range(len(graph)):            
        parent_val= sample[graph[i]]
        index= int(parent_val.dot(1<< np.arange(parent_val.size)[::-1]))
        p_value= cond_probs[i][index]
        if(p_value ==1 or p_value == 0):
            sample[i]=p_value
        else:
            sample[i] = np.random.binomial(1,p_value)
    return sample

#paralleize this if the experiments are slow
def generate_samples(graph,cond_probs, num_samples):
    with Pool() as p:
        output = p.starmap(generate_sample , [(graph, cond_probs)]*num_samples)
    return np.array(output)

def intervene(A, cond_probs):
    new_probs = np.copy(cond_probs)
    (num_nodes, max_parent_values) = cond_probs.shape
    for (node,value) in A:
        new_probs[node] = np.zeros(max_parent_values)+value
    return new_probs

def calculate_C(graph):
    C=0
    for i in range(len(graph)):
        C+= 2**(len(graph[i]))
    return C

def is_compatible_mu(vec, A):
    for (i,val) in A:
        if vec[i] != val:
            return False
    return True


def calculate_mu_parallelized(graph, alphas, cal_A):
    mu_vals = np.zeros(len(cal_A))
    all_binary_vecs = np.array(list(itertools.product([0, 1], repeat=len(alphas))))
    filtered_vecs = all_binary_vecs[(all_binary_vecs[:,-1] == 1)]
    def calculate_mu_A(A_index):
        mu=0
        A=cal_A[A_index]
        intervened_nodes = [x[0] for x in A]
        for vec in filtered_vecs:
            if is_compatible_mu(vec,  A):
                prod=1
                for i in range(len(alphas)):
                    if i not in intervened_nodes:
                        pa_val = vec[graph[i]]
                        pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                        if vec[i]==1:
                            prod*= alphas[i][pa_index]
                        else:
                            prod*= 1-alphas[i][pa_index]
                mu+=prod
        return mu
    with Pool() as p:                
        mu_vals = p.map(calculate_mu_A, list(range(len(cal_A))) )
    return np.array(mu_vals)

def calculate_mu_from_alphas(graph, alphas, cal_A):
    mu_vals = np.zeros(len(cal_A))
    all_binary_vecs = np.array(list(itertools.product([0, 1], repeat=len(alphas))))
    filtered_vecs = all_binary_vecs[(all_binary_vecs[:,-1] == 1)]    
    for A_index in range(len(cal_A)):                
        mu=0
        A=cal_A[A_index]
        intervened_nodes = [x[0] for x in A]
        for vec in filtered_vecs:
            if is_compatible_mu(vec,  A):
                prod=1
                for i in range(len(alphas)):
                    if i not in intervened_nodes:
                        pa_val = vec[graph[i]]
                        pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                        if vec[i]==1:
                            prod*= alphas[i][pa_index]
                        else:
                            prod*= 1-alphas[i][pa_index]
                mu+=prod
        mu_vals[A_index] = mu
    return mu_vals