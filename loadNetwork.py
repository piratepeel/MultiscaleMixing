import numpy as np


# Loads metadata and network
def load(networkfile, metadatafile, zero_index=0):
    with open(networkfile) as f:
        E = np.int32([row.strip().split()[:2] for row in f.readlines()])

    E -= zero_index
    if np.min(E) > 0:
        print("WARNING: minumum node index ="
              " {} (greater than 0)".format(np.min(E)))

    M = loadPartition(metadatafile, zero_index)

    return E, M


def loadPartition(partitionFile, zero_index=0):

    with open(partitionFile) as f:
        M = np.int32([row.split()[0] for row in f.readlines()])

    M -= zero_index
    if np.min(M) > 0:
        print("WARNING: minumum metadata label index ="
              " {} (greater than 0)".format(np.min(M)))

    return M
