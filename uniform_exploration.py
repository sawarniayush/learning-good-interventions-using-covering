from utils import *


def uniform_exploration(graph, cond_probs, cal_A, T):
    num_samples =int(T/len(cal_A))
    mu_values=[]
    for I in cal_A:
        cond_probs_I =intervene(I, cond_probs)
        samples= generate_samples(graph, cond_probs_I, num_samples)
        count_1=0
        for sample in samples:
            if sample[-1]==1:
                count_1+=1
        mu_values.append(count_1/num_samples)
    return np.array(mu_values)