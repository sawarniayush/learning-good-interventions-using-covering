from multiprocessing.pool import Pool
import sys
import time
import pickle
from yabe_implementation import *
from utils import *
from causal_bandit_covers import *
from uniform_exploration import *

# Instructions for running the algorithm
def run_experiment(model, experiment, output_file_name =None):
    graph  = model['graph']
    cal_A = model['cal_A']
    cond_probs = model['cond_probs']

    repeat = experiment['repeat']
    time_horizons = experiment['T']
    algos = experiment['algos']

    storage = {}
    storage['actual_mu_s'] = calculate_mu_parallelized(graph, cond_probs, cal_A)
    storage['best_arm'] = storage['actual_mu_s'].max()
    storage['model'] = model
    storage['experiment'] = experiment
    def temp_func(repeat_count):
        data= {
            'yabe_choice': [],
            'yabe_time':[],
            'covers_choice': [],
            'covers_time': [],
            'greedy_cover_choice': [],
            'greedy_cover_time': [],
            'uniform_choice':[],
            'uniform_time':[]
        }
        for i in range(len(time_horizons)):
            for algo in algos:
                start = time.time()
                print(f"For T= {time_horizons[i]} running {algo}")
                if algo == 'yabe':
                    data['yabe_choice'].append(yabe_et_al(graph, cond_probs, cal_A, time_horizons[i]).argmax())   
                elif algo == 'uniform':
                    data[algo+'_choice'].append(uniform_exploration(graph, cond_probs, cal_A, time_horizons[i]).argmax())
                elif algo =='covers':
                    data[algo+'_choice'].append(causal_bandit_covers(graph, cond_probs, cal_A, time_horizons[i]).argmax())
                end = time.time()
                data[algo+'_time'] = end - start
        return data
    output = []
    for i in range(repeat):
        print(f"#####Experiment Number: {i+1}#########")
        output.append(temp_func(i))
    storage['output'] = output
    save_file(storage,  output_file_name )
    return storage
               
if __name__=="__main__":
    model = load_file(sys.argv[1])
    experiment = load_file(sys.argv[2])
    run_experiment(model, experiment,sys.argv[3]+ '_'+ sys.argv[1]+'_'+sys.argv[2] )
