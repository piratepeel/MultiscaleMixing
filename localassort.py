from __future__ import division
import numpy as np
import scipy.sparse as sparse
import networkx as nx


def localAssortF(E, M, pr=np.arange(0., 1., 0.1), undir=True, missingValue=-1):

    n = len(M)
    ncomp = (M != missingValue).sum()
    m = len(E)
    A, degree = createA(E, n, m, undir)
    D = sparse.diags(1./degree, 0, format='csc')
    W = D.dot(A)
    c = len(np.unique(M))
    if ncomp < n:
        c -= 1

    # calculate node weights for how "complete" the
    # metadata is around the node
    Z = np.zeros(n)
    Z[M == missingValue] = 1.
    Z = W.dot(Z) / degree

    values = np.ones(ncomp)
    yi = (M != missingValue).nonzero()[0]
    yj = M[M != missingValue]
    Y = sparse.coo_matrix((values, (yi, yj)), shape=(n, c)).tocsc()

    assortM = np.empty((n, len(pr)))
    assortT = np.empty(n)

    eij_glob = np.array(Y.T.dot(A.dot(Y)).todense())
    eij_glob /= np.sum(eij_glob)
    ab_glob = np.sum(eij_glob.sum(1)*eij_glob.sum(0))

    WY = W.dot(Y).tocsc()

    print("start iteration")

    for i in range(n):
        pis, ti, it = calculateRWRrange(A, degree, i, pr, n)
        for ii, pri in enumerate(pr):
            pi = pis[:, ii]

            YPI = sparse.coo_matrix((pi[M != missingValue],
                                    (M[M != missingValue],
                                     np.arange(n)[M != missingValue])),
                                    shape=(c, n)).tocsr()

            trace_e = np.trace(YPI.dot(WY).toarray())
            assortM[i, ii] = trace_e

        YPI = sparse.coo_matrix((ti[M != missingValue], (M[M != missingValue],
                                np.arange(n)[M != missingValue])),
                                shape=(c, n)).tocsr()
        e_gh = YPI.dot(WY).toarray()
        Z[i] = np.sum(e_gh)
        e_gh /= np.sum(e_gh)
        trace_e = np.trace(e_gh)
        assortT[i] = trace_e

    assortM -= ab_glob
    assortM /= (1.-ab_glob + 1e-200)

    assortT -= ab_glob
    assortT /= (1.-ab_glob + 1e-200)

    return assortM, assortT, Z


# create adjacency matrix and degree sequence
def createA(E, n, m, undir=True):

    if undir:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(list(E))
        A = nx.to_scipy_sparse_matrix(G)
    else:
        A = sparse.coo_matrix((np.ones(m), (E[:, 0], E[:, 1])),
                              shape=(n, n)).tocsc()
    degree = np.array(A.sum(1)).flatten()

    return A, degree


# calculate the stationary distributions of a random walk with restart
# for different probabilties of restart (using the pagerank as a function
# approach)
def calculateRWRrange(A, degree, i, prs, n, trans=True, maxIter=1000):
    pr = prs[-1]
    D = sparse.diags(1./degree, 0, format='csc')
    W = D.dot(A)
    diff = 1
    it = 1

    F = np.zeros(n)
    Fall = np.zeros((n, len(prs)))
    F[i] = 1
    Fall[i, :] = 1
    Fold = F.copy()
    T = F.copy()

    if trans:
        W = W.T

    oneminuspr = 1-pr

    while diff > 1e-9:
        F = pr*W.dot(F)
        F[i] += oneminuspr
        Fall += np.outer((F-Fold), (prs/pr)**it)
        T += (F-Fold)/((it+1)*(pr**it))

        diff = np.sum((F-Fold)**2)
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        Fold = F.copy()

    return Fall, T, it
