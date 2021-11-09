from datetime import datetime
from lassort import load, localAssortF
import networkx as nx
import numpy as np
import pandas as pd

from networkx.generators.community import stochastic_block_model as sbm
from networkx.algorithms.community import modularity

N = 800
p = [[.1,.05], [.05, .1]]
n_trials = 20

def run_analysis(
    s0, 
    N=N, 
    p=p):
    sizes = [s0, N - s0]
    G = sbm(sizes, p)
    E = nx.convert_matrix.to_pandas_edgelist(G).values
    M = np.hstack([np.zeros(sizes[0]), np.ones(sizes[1])])

    assortM, assortT, Z = localAssortF(E,M,pr=np.arange(0,1,0.1))

    # average score for first group
    T0 = assortT[:sizes[0]].mean()
    # average score for second group
    T1 = assortT[sizes[0]:].mean()

    # mixing score vs. group
    r = np.corrcoef(assortT, M)[0,1]

    # modularity of partition
    A = nx.convert_matrix.to_scipy_sparse_matrix(G)

    # this is the stub count, or the edge count times 2
    m2 = A.sum()

    # intra-community edge density for each group
    e0 = A[:sizes[0], :sizes[0]].sum() / m2
    e1 = A[sizes[0]:, sizes[0]:].sum() / m2

    # degree proportion for each group
    a0 = A[:sizes[0],:].sum() / m2
    a1 = A[sizes[0]:,:].sum() / m2

    # modularity score
    Q = e0 -a0**2 + e1 - a1**2
    
    return (m2/2, e0, e1, a0, a1, Q, T0, T1)


if __name__ == "__main__":
    # find the size of the smallest group
    s0s = np.arange(10, 401, 10)
    s0s = pd.Series(np.hstack([s0s]*n_trials))
    
    results = s0s.apply(run_analysis)
    
    columns = ["m", "e0", "e1", "a0", "a1", "Q", "T0", "T1"]
    df = pd.DataFrame(
        results.to_list(),
        columns=columns
    )
    df["s0"] = s0s
    
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    df.to_csv(f"data/summary_{date_str}.csv", index=False)