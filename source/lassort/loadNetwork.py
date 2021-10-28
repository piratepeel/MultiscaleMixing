import numpy as np


import numpy as np


# Loads metadata and network
def load(
    networkfile, 
    metadatafile, 
    zero_index=0, 
    sep="\t",
    meta_col=0,
    header=False
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
        header
    )

    return E, M


def loadPartition(
    partitionFile, 
    zero_index=0,
    sep="\t", 
    meta_col=0,
    header=False
    ):

    with open(partitionFile) as f:
        M = np.int32([row.split(sep)[meta_col] for row in f.readlines()[int(header):]])

    M -= zero_index
    if np.min(M) > 0:
        print("WARNING: minumum metadata label index ="
              " {} (greater than 0)".format(np.min(M)))

    return M