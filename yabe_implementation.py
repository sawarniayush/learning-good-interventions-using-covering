from utils import *

def best_A_for_each_beta(betas):
    return np.argmax(betas, axis=2)

#what is beta? its a data structure similar to the conditional probability data structure with the
#the additional constraint that the values should sum to 1 for each vertex. We replicate this for all interventions 
#in cal(A) and hence we get a 3-D array
def intervene(A, cond_probs):
    new_probs = np.copy(cond_probs)
    (num_nodes, max_parent_values) = cond_probs.shape
    for (node,value) in A:
        new_probs[node] = np.zeros(max_parent_values)+value
    return new_probs

def is_compatible_old(vec,pa_index, pa, A):
    if pa!= pa_index:
        return False
    for (i,val) in A:
        if vec[i] != val:
            return False
    return True
def calculate_beta_from_alphas_old(graph, alphas, n, pa, A):
    all_binary_vecs = np.array(list(itertools.product([0, 1], repeat=len(graph))))
    beta=0
    intervened_nodes = [x[0] for x in A]
    for vec in all_binary_vecs:
        pa_val = vec[graph[n]]
        pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
        if is_compatible_old(vec, pa_index, pa, A):
            prod=1
            for i in range(len(graph)):
                if i not in intervened_nodes:
                    if vec[i]==1:
                        prod*= alphas[i][pa_index]
                    else:
                        prod*= 1-alphas[i][pa_index]
            beta+=prod
    return beta

def is_compatible(vec, ancestor_set, A):
    for (i,val) in A:
        if i in ancestor_set and vec[i] != val:
            return False
    return True

def calculate_beta_from_alphas(graph,ancestors,  alpha_estimates, n,pa , cal_A):
    ancestor_set = ancestors[n].difference(graph[n]) #calculate set difference 
    ancestor_set= list(ancestor_set)
    temp_binary_vecs = list(itertools.product([0, 1], repeat=len(ancestor_set)))
    all_binary_vecs =np.zeros((len(temp_binary_vecs),len(graph) ))
    parent_vec= [int(x) for x in bin(pa)[2:].zfill(len(graph[n]))] # generate the binary assignment of the parents
    ancestor_list= list(ancestor_set)
    for ind in range(len(temp_binary_vecs)):
        all_binary_vecs[ind][ancestor_list] = temp_binary_vecs[ind]
        all_binary_vecs[ind][graph[n]] = parent_vec
    betas=np.zeros(len(cal_A))
    for A_index in range(len(cal_A)):
        A=cal_A[A_index]
        beta=0
        intervened_nodes = [x[0] for x in A]
        for vec in all_binary_vecs:
            pa_val = vec[graph[n]]
            pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
            if is_compatible(vec,ancestors[n], A):
                prod=1
                for i in ancestor_list+graph[n]:
                    if i not in intervened_nodes:
                        pa_val = vec[graph[i]]
                        pa_index = int(pa_val.dot(1<< np.arange(pa_val.size)[::-1]))
                        if vec[i]==1:
                            prod*= alpha_estimates[i][pa_index]
                        else:
                            prod*= (1-alpha_estimates[i][pa_index])
                beta+=prod
        betas[A_index] = beta        
    return betas

def find_ancestors(graph):
    ancestors = [set() for x in range(len(graph))]
    for i in range(len(graph)):
        for pa in graph[i]:
            ancestors[i].add(pa)
            ancestors[i]=ancestors[i].union(ancestors[pa])
    return ancestors

    
#Algorithm 1 for Yabe et. al.
def estimate_betas(graph, cond_probs, cal_A, T , C):
    betas = np.ndarray(( cond_probs.shape[0], cond_probs.shape[1], len(cal_A)))
    sample_count =np.ndarray(cond_probs.shape)    
    alpha_estimates = np.ndarray(cond_probs.shape)
    sample_count.fill(0)
    alpha_estimates.fill(0) 
    betas.fill(0)
    total_samples = 0
    ancestors = find_ancestors(graph)

    for n in range(len(graph)):
        for pa in range(2**len(graph[n])):
            if(len(graph[n])==0):
                continue
            def temp_func(index):
                b= calculate_beta_from_alphas_old(graph, alpha_estimates, n, pa, cal_A[index])
                # b = calculate_beta_from_alphas(graph,ancestors[n], parent_index_in_ancestor[n], alpha_estimates, n, pa, cal_A[index])
                return b
            # betas[n,pa] = np.array([temp_func(x) for x in  list(range(len(cal_A))) ])  
            betas[n,pa] = calculate_beta_from_alphas(graph, ancestors, alpha_estimates, n,pa, cal_A )                               
            best_A = betas[n,pa,:].argmax()
            cond_probs_A = intervene(cal_A[best_A] , cond_probs)
            t_pa = 0
            t_n=0
            num_samples = int(T / C)
            total_samples +=num_samples
            samples= generate_samples(graph, cond_probs_A, num_samples)
            for sample in samples:
                sample_pa= sample[graph[n]]
                pa_val = int(sample_pa.dot(1<< np.arange(sample_pa.size)[::-1]))
                if pa_val == pa:
                    t_pa+=1
                    if sample[n]==1:
                        t_n+=1
            if t_pa >= 1: 
                alpha_estimates[n][pa] = t_n/t_pa
                sample_count[n][pa] = t_pa
    return (betas, alpha_estimates, sample_count)


#Algorithm 2 of Yabe et. al.   
def estimate_alphas(graph, cond_probs, cal_A, T, C, betas, alphas, old_sample_count):
    #equally divide T for each of the best As
    parent_occurence_count = np.ndarray(alphas.shape)
    parent_occurence_count.fill(0)
    node_val_count = np.ndarray(alphas.shape)
    node_val_count.fill(0)
    num_samples = int(T/C)
    for n in range(len(graph)):
        for pa in range(2**len(graph[n])):
            best_A = betas[n,pa,:].argmax()
            intervened_nodes = [x[0] for x in cal_A[best_A]]
            cond_probs_A = intervene(cal_A[best_A],cond_probs)
            samples= generate_samples(graph, cond_probs_A, num_samples)
            for sample in samples:
                for node in range(len(graph)):
                    if node in intervened_nodes:
                        continue
                    sample_pa=sample[graph[node]]
                    pa_val=int(sample_pa.dot(1<< np.arange(sample_pa.size)[::-1]))
                    parent_occurence_count[node , pa_val] +=1
                    if sample[node] == 1:
                        node_val_count[node, pa_val] +=1
    for i in range(len(graph)):
        for pa in range(2**len(graph[i])):
            if parent_occurence_count[i, pa] >0: 
                alphas[i,pa] = (alphas[i,pa]* old_sample_count[i,pa] + node_val_count[i,pa])/(old_sample_count[i,pa]+parent_occurence_count[i,pa])
    return alphas

def yabe_et_al(graph, cond_probs, cal_A, T):
    C=calculate_C(graph)
    print(f"C value {C}")
    (betas, alphas, sample_count)=estimate_betas(graph, cond_probs, cal_A, int(T/3), C)
    final_alphas= estimate_alphas(graph, cond_probs,cal_A, T - int(T/3), C, betas, alphas, sample_count)
    mu_values = calculate_mu_parallelized(graph, final_alphas, cal_A)
    return mu_values