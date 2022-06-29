"""
Multiscale mixing patterns in networks.

Code to calculate the multiscale assortativity from the accompanying paper:
Peel, L., Delvenne, J. C., & Lambiotte, R. (2018). 'Multiscale mixing patterns
in networks.' PNAS, 115(16), 4057-4062.
"""

import numpy as np
import scipy.sparse as sparse
import networkx as nx


def localAssortF(edgelist, node_attr, pr=np.arange(0., 1., 0.1), undir=True,
                 missingValue=-1):
    """Calculate the multiscale assortativity.

    Parameters
    ----------
    edgelist : array_like
        the network represented as an edge list,
        i.e., a E x 2 array of node pairs
    node_attr : array_like
        n length array of node attribute values
    pr : array, optional
        array of one minus restart probabilities for the random walk in
        calculating the personalised pagerank. The largest of these values
        determines the accuracy of the TotalRank vector max(pr) -> 1 is more
        accurate (default: [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    undir : bool, optional
        indicate if network is undirected (default: True)
    missingValue : int, optional
        token to indicate missing attribute values (default: -1)

    Returns
    -------
    assortM : array_like
        n x len(pr) array of local assortativities, each column corresponds to
        a value of the input restart probabilities, pr. Note if only number of
        restart probabilties is greater than one (i.e., len(pr) > 1).
    assortT : array_like
        n length array of multiscale assortativities
    Z : array_like
        N length array of per-node confidence scores

    References
    ----------
    For full details see [1]_

    .. [1] Peel, L., Delvenne, J. C., & Lambiotte, R. (2018). "Multiscale
        mixing patterns in networks.' PNAS, 115(16), 4057-4062.
    """
    # number of nodes
    n = len(node_attr)
    # number od nodes with complete attribute
    ncomp = (node_attr != missingValue).sum()
    # number of edges
    m = len(edgelist)
    # construct adjacency matrix and calculate degree sequence
    A, degree = createA(edgelist, n, m, undir)
    # construct diagonal inverse degree matrix
    D = sparse.diags(1./degree, 0, format='csc')
    # construct transition matrix (row normalised adjacency matrix)
    W = D @ A
    # number of distinct node categories
    c = len(np.unique(node_attr))
    if ncomp < n:
        c -= 1

    # calculate node weights for how "complete" the
    # metadata is around the node
    Z = np.zeros(n)
    Z[node_attr == missingValue] = 1.
    Z = (W @ Z) / degree

    # indicator array if node has attribute data (or missing)
    hasAttribute = node_attr != missingValue

    # calculate global expected values
    values = np.ones(ncomp)
    yi = (hasAttribute).nonzero()[0]
    yj = node_attr[hasAttribute]
    Y = sparse.coo_matrix((values, (yi, yj)), shape=(n, c)).tocsc()
    eij_glob = np.array(Y.T @ (A @ Y).todense())
    eij_glob /= np.sum(eij_glob)
    ab_glob = np.sum(eij_glob.sum(1)*eij_glob.sum(0))

    # initialise outputs
    assortM = np.empty((n, len(pr)))
    assortT = np.empty(n)

    WY = (W @ Y).tocsc()

    # print("start iteration")

    for i in range(n):
        pis, ti, it = calculateRWRrange(W, i, pr, n)
        if len(pr) > 1:
            for ii, pri in enumerate(pr):
                pi = pis[:, ii]

                YPI = sparse.coo_matrix((pi[hasAttribute],
                                        (node_attr[hasAttribute],
                                         np.arange(n)[hasAttribute])),
                                        shape=(c, n)).tocsr()

                trace_e = (YPI.dot(WY).toarray()).trace()
                assortM[i, ii] = trace_e

        YPI = sparse.coo_matrix((ti[hasAttribute], (node_attr[hasAttribute],
                                np.arange(n)[hasAttribute])),
                                shape=(c, n)).tocsr()
        e_gh = (YPI @ WY).toarray()
        e_gh_sum = e_gh.sum()
        Z[i] = e_gh_sum
        e_gh /= e_gh_sum
        trace_e = e_gh.trace()
        assortT[i] = trace_e

    assortT -= ab_glob
    np.divide(assortT, 1.-ab_glob, out=assortT, where=ab_glob != 0)

    if len(pr) > 1:
        assortM -= ab_glob
        np.divide(assortM, 1.-ab_glob, out=assortM, where=ab_glob != 0)

        return assortM, assortT, Z
    return None, assortT, Z


def createA(E, n, m, undir=True):
    """Create adjacency matrix and degree sequence."""
    if undir:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(list(E))
    A = nx.to_scipy_sparse_matrix(G)

    degree = np.array(A.sum(1)).flatten()

    return A, degree


def calculateRWRrange(W, i, alphas, n, maxIter=1000):
    """
    Calculate the personalised TotalRank and personalised PageRank vectors.

    Parameters
    ----------
    W : array_like
        transition matrix (row normalised adjacency matrix)
    i : int
        index of the personalisation node
    alphas : array_like
        array of (1 - restart probabilties)
    n : int
        number of nodes in the network
    maxIter : int, optional
        maximum number of interations (default: 1000)

    Returns
    -------
    pPageRank_all : array_like
        personalised PageRank for all input alpha values (only calculated if
        more than one alpha given as input, i.e., len(alphas) > 1)
    pTotalRank : array_like
        personalised TotalRank (personalised PageRank with alpha integrated
        out)
    it : int
        number of iterations

    References
    ----------
    See [2]_ and [3]_ for further details.

    .. [2] Boldi, P. (2005). "TotalRank: Ranking without damping." In Special
        interest tracks and posters of the 14th international conference on
        World Wide Web (pp. 898-899).
    .. [3] Boldi, P., Santini, M., & Vigna, S. (2007). "A deeper investigation
        of PageRank as a function of the damping factor." In Dagstuhl Seminar
        Proceedings. Schloss Dagstuhl-Leibniz-Zentrum für Informatik.
    """
    alpha0 = alphas.max()
    WT = alpha0*W.T
    diff = 1
    it = 1

    # initialise PageRank vectors
    pPageRank = np.zeros(n)
    pPageRank_all = np.zeros((n, len(alphas)))
    pPageRank[i] = 1
    pPageRank_all[i, :] = 1
    pPageRank_old = pPageRank.copy()
    pTotalRank = pPageRank.copy()

    oneminusalpha0 = 1-alpha0

    while diff > 1e-9:
        # calculate personalised PageRank via power iteration
        pPageRank = WT @ pPageRank
        pPageRank[i] += oneminusalpha0
        # calculate difference in pPageRank from previous iteration
        delta_pPageRank = pPageRank-pPageRank_old
        # Eq. [S23] Ref. [1]
        pTotalRank += (delta_pPageRank)/((it+1)*(alpha0**it))
        # only calculate personalised pageranks if more than one alpha
        if len(alphas) > 1:
            pPageRank_all += np.outer((delta_pPageRank), (alphas/alpha0)**it)

        # calculate convergence criteria
        diff = np.sum((delta_pPageRank)**2)/n
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        pPageRank_old = pPageRank.copy()

    return pPageRank_all, pTotalRank, it
  
  
  def localAssortF_numeric(A, attribute):
    """
    Calculate local assortativity of A (undirected network) with respect to the values in attribute

    Parameters
    ----------
        A : array_like (e.g. from nx.to_scipy_sparse_array(G))
            adjacency matrix
        attribute : array_like
            array of numeric values 

    Returns
    -------
    loc_ass : array_like
        array of values representing the local assortativity of each node
    """
    
    # normalize attribute
    attribute = (attribute - np.mean(attribute))/np.std(attribute)

    ## Construct transition matrix (row normalised adjacency matrix)
    # construct diagonal inverse degree matrix
    degree = A.sum(1)
    n = len(G)
    D = ss.diags(1./degree, 0, format='csc')
    W = D @ A

    ## Calculate personalized pagerank for all nodes
    pr=np.arange(0., 1., 0.1)
    per_pr = []
    for i in range(n):
        pis, ti, it = calculateRWRrange(W, i, pr, n)
        per_pr.append(ti)
    per_pr = np.array(per_pr)

    # calculate local assortativity (G is undirected, A is symmetric)
    loc_ass = (per_pr * ((A.T * attribute).T * attribute )).sum(1) / degree
    
    return loc_ass
