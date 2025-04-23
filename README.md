# The Chocolate Bar Problem: A Coding Challenge 

There are m chocolate bars of varying (integer) lengths and n hungry children who want 
differing amounts of chocolate (again integer amounts).  The aim is to feed all of the 
children the correct amount whilst making the fewest cuts to the chocolate bars.

## Files:

```
Technical Challenge                     —— Root folder
│                               
├── src/                        
│  |── __init__.py              
│  |──cache.py                          —— LRU Cache for past inputs
│  |──distributor.py                    —— Main solver
│  |──exceptions.py                     —— Custom error types
│  |──utils.py                          —— Default constants
|  └──tests/
│     └── test_distributor.py           —— Example tests
|── performance/
|   |── complexity_profile.png          —— Time complexity for different methods used
|   |── log-log_complexity_profile.png  —— Power law for some methods
|   └── heatmap.pdf                     —— Shows code execution time distribution
└── README.md                           —— This file
```

## How it works:

1. Validation & caching: O(n + m)
  - Check inputs (optional sanitization) and skips trivial cases.
  - Remembers past inputs in an LRU cache to skip recomputation.

2. Single item case: O(1)
  - If there is exactly one bar and one child, return 0 or 1 cut immediately.

3. Exact matches: O(n+m)
  - First, feeds any child whose demand exactly equals a whole bar (no cut needed).

4. Brute-force on small leftovers: O(kⁿ) (k is the distinct chocolate piece sizes)
  - If the remaining number of children ≤ DEFAULT_BRUTEFORCE_THRESHOLD, run a
    recursive solver on the reduced problem to guarantee the absolute minimum cuts.

5. Complement method: O(m*n) + O(m*n + nlogn)
  - Then tries to pair two children (or even three) whose combined demands fit one bar 
    exactly, so you only pay for 1 or 2 cuts instead of more. (Clever!)

6. Greedy splitting: O(m log m)
  - If no perfect fit is found, the algorithm finds the smallest bar that is big enough
    and cuts off exactly what is needed and puts the remainder back into the pool.

7. Leftover merging: O(m log m + C) (where C is the number of pieces used in the merge)
  - As a last resort, greedily merges smaller scraps to satisfy a child's demand.

## Configurable knobs:
- `enable_validation`/`enable_sanitization`: Turn input checking and sanitization on 
                                               or off at runtime.  
- `DEFAULT_CACHE_SIZE`: how many past (bars, children) inputs to remember.  
- `DEFAULT_BRUTEFORCE_THRESHOLD`: up to this many children algorithm will run a slow
                                  exact solver for better accuracy.  
- `DEFAULT_COMPLEMENT_THRESHOLD`: maximum number of children for which algorithm 
                                  attempts the 2 and 3 child complement matching.  

This approach runs in near-linear time for realistic inputs and uses only a small extra 
amount of memory.

## Usage:
Run the following command from the 'root' folder:
```
python -m src.tests.test_distributor
```

## Future Extensions

1. Exact Solver: Can also integrate exact solvers (would not be scalable).
2. Parallelise algorithm for multiple large inputs.
3. Visualization: Build a small dashboard that shows how the chocolate is distributed.
