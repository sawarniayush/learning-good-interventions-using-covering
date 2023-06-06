from utils import *

def greedy_cover(graph):
    d = max( [len(x) for x in graph] ) #max degree
    total_alphas_to_cover =0 
    for i in range(len(graph)):
        if len(graph[i]) != 0:
            total_alphas_to_cover+= 2**len(graph[i])
    cal_I = []
    alphas_covered = 0
    mark_covered_alphas = np.ndarray((len(graph), 2**d))
    mark_covered_alphas.fill(0)
    p_of_intervening = d/(d+1)
    
    while(alphas_covered < total_alphas_to_cover ):
        I = {}
        nodes_intervened = set()
        somethingNewCovered =False
        for i in range(len(graph)):
            if len(graph[i]) > 0 and all(y in nodes_intervened for y in graph[i]):
                pa_val=np.array([ I[pa] for pa in graph[i]])
                pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                if (mark_covered_alphas[i, pa_index] ==0):
                    continue
            if(np.random.binomial(1, p_of_intervening)==1):
                I[i]= np.random.binomial(1, 0.5)
                nodes_intervened.add(i)
        for i in range(len(graph)):        
            if len(graph[i]) !=0 and i not in nodes_intervened and all(y in nodes_intervened for y in graph[i]):                
                pa_val=np.array([ I[pa] for pa in graph[i]])
                pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                if mark_covered_alphas[i, pa_index] ==0:
                    mark_covered_alphas[i, pa_index]=1
                    alphas_covered+=1
                    somethingNewCovered = True
        if (somethingNewCovered): #only add to convering set if a new alpha is covered by I
            cal_I.append(list(I.items()))
    return cal_I

def get_cover_set(graph):
    d = max( [len(x) for x in graph] ) #max degree
    total_alphas_to_cover =0 
    for i in range(len(graph)):
        if len(graph[i]) != 0:
            total_alphas_to_cover+= 2**len(graph[i])
    cal_I = []
    alphas_covered = 0
    mark_covered_alphas = np.ndarray((len(graph), 2**d))
    mark_covered_alphas.fill(0)
    p_of_intervening = d/(d+1)
    
    while(alphas_covered < total_alphas_to_cover ):
        I = {}
        nodes_intervened = set()
        somethingNewCovered =False
        for i in range(len(graph)):
            if(np.random.binomial(1, p_of_intervening)==1):
                I[i]= np.random.binomial(1, 0.5)
                nodes_intervened.add(i)
        for i in range(len(graph)):        
            if len(graph[i]) !=0 and i not in nodes_intervened and all(y in nodes_intervened for y in graph[i]):                
                pa_val=np.array([ I[pa] for pa in graph[i]])
                pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                if mark_covered_alphas[i, pa_index] ==0:
                    mark_covered_alphas[i, pa_index]=1
                    alphas_covered+=1
                    somethingNewCovered = True
        if (somethingNewCovered): #only add to convering set if a new alpha is covered by I
            cal_I.append(list(I.items()))
    return cal_I

def get_alpha_estimates(graph, cond_probs, cal_I, T):
    parent_occurence_count = np.ndarray(cond_probs.shape)
    parent_occurence_count.fill(0)
    node_val_count = np.ndarray(cond_probs.shape)
    node_val_count.fill(0)
    num_samples = int(T/len(cal_I))
    alphas = np.ndarray(cond_probs.shape)
    alphas.fill(0)
    
    for I in cal_I:
            pa_val = 0 
            cond_probs_I =intervene(I, cond_probs)
            samples= generate_samples(graph, cond_probs_I, num_samples)
            intervened_nodes= [x[0] for x in I]
            for sample in samples:
                for node in range(len(graph)):
                    pa_val = 0
                    if node in intervened_nodes:
                        continue
                    if(len(graph[node]) ==0):                  
                        parent_occurence_count[node, 0] +=1
                    else:    
                        sample_pa=sample[graph[node]]
                        pa_val=int(sample_pa.dot(1<< np.arange(sample_pa.size)[::-1]))
                        parent_occurence_count[node , pa_val] +=1
                    node_val_count[node, pa_val] +=sample[node]
    for i in range(len(graph)):
        for pa in range(2**len(graph[i])):
            if parent_occurence_count[i, pa] >0: 
                alphas[i,pa] = (node_val_count[i,pa])/(parent_occurence_count[i,pa])
            else:
                print(f"how can this happen node, i {i}, pa {pa} ")
    return alphas

def causal_bandit_covers(graph, cond_probs, cal_A, T):
    # cal_I = greedy_cover(graph)
    cal_I = get_cover_set(graph)
    print(f"Cover Size : {len(cal_I)}")
    alpha_estimates = get_alpha_estimates(graph, cond_probs, cal_I, T)
    mu_values = calculate_mu_parallelized(graph, alpha_estimates, cal_A)
    return mu_values

def causal_bandit_greedy_cover(graph, cond_probs, cal_A, T):
    cal_I = greedy_cover(graph)
    # cal_I = get_cover_set(graph)
    print(f"Cover Size : {len(cal_I)}")
    alpha_estimates = get_alpha_estimates(graph, cond_probs, cal_I, T)
    mu_values = calculate_mu_parallelized(graph, alpha_estimates, cal_A)
    return mu_values