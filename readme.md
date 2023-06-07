# Learning Good Interventions in Causal Graphs via Covering

## Dependencies
This implementation has been tested with
- Python 3
- Numpy


## Running the algorithm
The algorithm needs a model file describing the problem instance and an experiments file. Both these files are Python dictionaries stored in pickle format. The syntax for both is given below -

1. Model file
    ```python
    #Model file is a python dictionary with the following keys
    'graph':[[],[0],[1]], #Causal DAG represented as an adjacency list with the last node as the reward node
    'cond_probs':[[0.4], [0.8, 0.4] , [0.4, 0.9]], #Conditional probability of each node taking value 1 given an assignement to its parents. The elements in the list are indexed according the the value assignment to the parent. For example, P(Node2 =1 | Parents(Node2)= '101') would be available at cond_probs[1][5]
    ''
    'cal_A': [[(1,0),(0,0)],[(0,1)]] #Given Intervention Set. Each element of the tuple (a,b) represents the intervened node (a) and the assigned value to the node (b) respectively.
    ```
2. Experiments File

    ```python
    'algos': ['yabe', 'uniform', 'covers'], # List of algorithms to be run.
    'T': [ 1000, 3000, 5000, 7000, 10000, 20000, 30000, 40000], # List of time horizons. The algorithms are run for each value of T separately. 
    'repeat': 140, # Each algorithm is run these many times. 

    ```

3. run.sh
    ```bash 
    run.sh  -m model.pkl 
            -e experiment.pkl 
            -o output_file_name #The output of the experiment would be stored in this file
    ```

