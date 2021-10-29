import numpy as np


# Loads metadata and network
def load(
    networkfile, 
    metadatafile, 
    zero_index=0, 
    sep="\t",
    meta_col=0,
    header=False,
    reindex=False,
    missing_value=-1
):
    with open(networkfile) as f:
        E = np.int32([row.strip().split(sep)[:2] for row in f.readlines()])

    E -= zero_index
    if np.min(E) > 0:
        print("WARNING: minumum node index ="
              " {} (greater than 0)".format(np.min(E)))

    M = loadPartition(
        metadatafile, 
        zero_index, 
        sep, 
        meta_col,
        header,
        reindex,
        missing_value
    )

    return E, M


def loadPartition(
    partitionFile, 
    zero_index=0,
    sep="\t", 
    meta_col=0,
    header=False,
    reindex=False,
    missing_value=-1
    ):

    with open(partitionFile) as f:
        M = np.int32([row.split(sep)[meta_col] for row in f.readlines()[int(header):]])

    if reindex:
        M = reindexLabels(
            M, 
            missing_value=missing_value, 
            zero_index=zero_index)
    
        
    M -= zero_index
    if np.min(M) > 0:
        print("WARNING: minumum metadata label index ="
              " {} (greater than 0)".format(np.min(M)))

    return M


def reindexLabels(M, missing_value=-1, zero_index=0):
    if missing_value >= zero_index:
        print("WARNING: missing value may collide with zero index.")
        
    u = np.unique(M)
    non_missing = u[np.where(u!=missing_value)]
    
    # are there missing labels?
    if non_missing.max() == len(non_missing) + zero_index - 1:
        return M
    
    # relabel attributes to consecutive integers starting at zero_index
    labels = {m:(i+zero_index) for i, m in enumerate(non_missing)}
    labels[missing_value] = missing_value
    
    for i, l in enumerate(M):
        M[i] = labels[l]

    return M