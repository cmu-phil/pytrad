import numpy as np
import numpy.matlib
import itertools
from copy import deepcopy
from graph.Edge import Edge
from graph.Endpoint import Endpoint
from graph.GeneralGraph import GeneralGraph
import warnings
import sys
import scipy
import scipy.sparse
import math

def feval(parameters):
    if parameters[0] == 'local_score_CV_general':
        return local_score_CV_general(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_marginal_general':
        return local_score_marginal_general(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_CV_multi':
        return local_score_CV_multi(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'local_score_marginal_multi':
        return local_score_marginal_multi(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSum':
        if (len(parameters) == 1):
            return covSum()
        elif (len(parameters) == 2):
            return covSum(parameters[1])
        elif (len(parameters) == 3):
            return covSum(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSum(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
    elif parameters[0] == 'covNoise':
        if (len(parameters) == 1):
            return covNoise()
        elif (len(parameters) == 2):
            return covNoise(parameters[1])
        elif (len(parameters) == 3):
            return covNoise(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covNoise(parameters[1], parameters[2], parameters[3])
        else:
            return covNoise(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSEard':
        if (len(parameters) == 1):
            return covSEard()
        elif (len(parameters) == 2):
            return covSEard(parameters[1])
        elif (len(parameters) == 3):
            return covSEard(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSEard(parameters[1], parameters[2], parameters[3])
        else:
            return covSEard(parameters[1], parameters[2], parameters[3], parameters[4])
    elif parameters[0] == 'covSum':
        if (len(parameters) == 1):
            return covSum()
        elif (len(parameters) == 2):
            return covSum(parameters[1])
        elif (len(parameters) == 3):
            return covSum(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return covSum(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return covSum(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
    elif parameters[0] == 'gpr_multi_new':
        if (len(parameters) == 1):
            return gpr_multi_new()
        elif (len(parameters) == 2):
            return gpr_multi_new(parameters[1])
        elif (len(parameters) == 3):
            return gpr_multi_new(parameters[1], parameters[2])
        elif (len(parameters) == 4):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3])
        elif (len(parameters) == 5):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4])
        elif (len(parameters) == 6):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
        elif (len(parameters) == 7):
            return gpr_multi_new(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],
                                 parameters[6])
    else:
        raise Exception('Undefined function')

def kernel(x, xKern, theta):
    # KERNEL Compute the rbf kernel
    n2 = dist2(x, xKern)
    if (theta[0] == 0):
        theta[0] = 2 / np.median(n2[np.where(np.tril(n2) > 0)])
        theta_new = theta[0]
    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]
    return kx, bw_new

def Combinatorial(T0):
    # sub = Combinatorial (T0); % find all the sbusets of T0
    sub = []
    count = 0
    if (len(T0) == 0):
        sub.append(()) # a 1x0 empty matrix
    else:
        if (len(T0) == 1):
            sub.append(())
            sub.append(T0) # when T0 is a scale, it is a special case!!
        else:
            for n in range(len(T0) + 1):
                for S in list(itertools.combinations(T0, n)):
                    sub.append(S)
    return sub

def Score_G(Data, G, score_func, parameters): # calculate the score for the current G
    # here G is a DAG
    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)
        delta_score = feval([score_func, Data, i, PA, parameters])
        score = score + delta_score
    return score

def Insert_validity_test1(G, i,j, T):
    # V=Insert_validity_test1(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[:,i] != 0)[0], np.where(G.graph[i, :] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) # find the neighbours of Xj and are adjacent to Xi
    V = check_clique(G, list(np.union1d(NA, T).astype(int))) # check whether it is a clique
    return V


def check_clique(G, subnode): # check whether node subnode is a clique in G
    # here G is a CPDAG
    # the definition of clique here: a clique is defined in an undirected graph
    # when you ignore the directionality of any directed edges
    Gs = deepcopy(G.graph[np.ix_(subnode, subnode)]) # extract the subgraph
    ns = len(subnode)

    if (ns == 0):
        s = 1
    else:
        row, col = np.where(Gs == 1)
        Gs[row, col] = 6
        Gs[col, row] = 6
        if (np.all(((np.eye(ns) - np.ones((ns, ns))) * -6) == Gs)): # check whether it is a clique
            s = 1
        else:
            s = 0
    return s

def Insert_validity_test2(G, i,j, T):
    # V=Insert_validity_test(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) # find the neighbours of Xj and are adjacent to Xi

    # condition 2: every semi-directed path from Xj to Xi contains a node in union(NA,T)
    # Note: EVERY!!
    s2 = Insert_vC2_new(G, j, i, np.union1d(NA, T))
    if (s2):
        V = 1

    return V

def Insert_vC2_new(G,j,i,NAT): # validity test for condition 2 of Insert operator
    # here G is CPDAG
    # Use Depth-first-Search
    start = j
    target = i
    # stack(1)=start; % initialize the stack
    stack = [{'value':start, 'pa': {}}]
    sign = 1    # If every semi-pathway contains a node in NAT, than sign=1;
    count = 1

    while (len(stack)):
        top = stack[0]
        stack = stack[1:] # pop
        if (top['value'] == target): # if find the target, search that pathway to see whether NAT is in that pathway
            curr = top
            ss = 0
            while True:
                if (len(curr['pa'])):
                    if (curr['pa']['value'] in NAT): # contains a node in NAT
                        ss = 1
                        break
                else:
                    break
                curr = curr['pa']
            if (not ss): # do not include NAT
                sign = 0
                break
        else:
            child = np.concatenate((np.where(G.graph[:, top['value']] == 1)[0], np.where(G.graph[top['value'], :] == 6)[0]))
            sign_child = np.ones(len(child))
            # check each child, whether it has appeared before in the same pathway
            for k in range(len(child)):
                curr = top
                while True:
                    if (len(curr['pa'])):
                        if (curr['pa']['value'] == child[k]):
                            sign_child[k] = 0   # has appeared in that path before
                            break
                    else:
                        break
                    curr = curr['pa']

            for k in range(len(sign_child)):
                if (sign_child[k]):
                    stack.insert(0, {'value': child[k], 'pa': top}) # push
    return sign

def find_subset_include (s0,sub):
    # S = find_subset_include(sub(k),sub); %  find those subsets that include sub(k)
    if (len(s0) == 0 or len(sub) == 0):
        Idx = np.ones(len(sub))
    else:
        Idx = np.zeros(len(sub))
        for i in range(len(sub)):
            tmp = set(s0).intersection(set(sub[i]))
            if (len(tmp)):
                if (tmp == set(s0)):
                    Idx[i] = 1
    return Idx

def Insert_changed_score(Data,G,i,j,T,record_local_score,score_func,parameters):
    # calculate the changed score after the insert operator: i->j
    Tj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti) #  find the neighbours of Xj and are adjacent to Xi
    Paj = np.where(G.graph[j, :] == 1)[0] # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = np.union1d(NA, T).astype(int)
    tmp2 = np.union1d(tmp1, Paj)
    tmp3 = np.union1d(tmp2, [i]).astype(int)

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if (not np.setxor1d(record_local_score[j][r0][0:-1], tmp3).size):
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if (not np.setxor1d(record_local_score[j][r0][0:-1], tmp2).size): # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if ((not np.setxor1d(record_local_score[j][r0][0:-1], [-1]).size) and (not tmp2.size)):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if (s1 and s2):
            break

    if (not s1):
        score1 = feval([score_func, Data, j, tmp3, parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if(not s2):
        score2 = feval([score_func, Data, j, tmp2, parameters])
        # r = len(record_local_score[j])
        if (len(tmp2) != 0):
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, T]
    return chscore,desc,record_local_score


def Insert(G,i,j,T):
        # Insert operator
        # insert the directed edge Xi->Xj
        nodes = G.get_nodes()
        G.add_edge(Edge(nodes[i], nodes[j], -1, 1))

        for k in range(len(T)): # directing the previous undirected edge between T and Xj as T->Xj
            if G.get_edge(nodes[T[k]], nodes[j]) is not None:
                G.remove_edge(G.get_edge(nodes[T[k]], nodes[j]))
            G.add_edge(Edge(nodes[T[k]], nodes[j], -1, 1))

        return G

def Delete_validity_test(G, i,j, H):
    # V=Delete_validity_test(G, X, Y, H); % do validity test for the operator Delete; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Hj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Hi = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi) # find the neighbours of Xj and are adjacent to Xi
    s1 = check_clique(G, list(set(NA) - set(H))) # check whether it is a clique

    if (s1):
        V = 1

    return V

def Delete_changed_score(Data, G,i,j,H,record_local_score,score_func,parameters):
    # calculate the changed score after the Delete operator
    Hj = np.where(G.graph[:,j] == 6)[0] # neighbors of Xj
    Hi = np.union1d(np.where(G.graph[i,:] != 0)[0], np.where(G.graph[:, i] != 0)[0]) # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi) # find the neighbours of Xj and are adjacent to Xi
    Paj = np.union1d(np.where(G.graph[j, :] == 1)[0], [i]) # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = set(NA) - set(H)
    tmp2 = set.union(tmp1, set(Paj))
    tmp3 = tmp2 - {i}

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if (set(record_local_score[j][r0][0:-1]) == tmp3) :
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if (set(record_local_score[j][r0][0:-1]) == tmp2): # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if ((set(record_local_score[j][r0][0:-1]) == {-1}) and len(tmp2) == 0):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if (s1 and s2):
            break

    if (not s1):
        score1 = feval([score_func, Data, j, list(tmp3), parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if (not s2):
        score2 = feval([score_func, Data, j, list(tmp2), parameters])
        r = len(record_local_score[j])
        if (len(tmp2) != 0):
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, H]
    return chscore,desc,record_local_score

def Delete(G,i,j,H):
    # Delete operator
    nodes = G.get_nodes()
    if G.get_edge(nodes[i], nodes[j]) is not None:
        # delete the edge between Xi and Xj
        G.remove_edge(G.get_edge(nodes[i], nodes[j]))
    for k in range(len(H)): # directing the previous undirected edge
        if G.get_edge(nodes[j], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[j], nodes[H[k]]))
        if G.get_edge(nodes[i], nodes[H[k]]) is not None:
            G.remove_edge(G.get_edge(nodes[i], nodes[H[k]]))
        G.add_edge(Edge(nodes[j], nodes[H[k]], -1, 1))
        G.add_edge(Edge(nodes[i], nodes[H[k]], -1, 1))
    return G

def check2(G, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if (len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0):
            s = 0
            break
    return s

# function Gd = PDAG2DAG(G) % transform a PDAG to DAG
def PDAG2DAG(G):
    nodes = G.get_nodes()
    # first create a DAG that contains all the directed edges in PDAG
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()
    for edge in edges:
        if not ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or (edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW)):
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int)) # index whether the ith node has been removed. 1:removed; 0: not
    while 0 in inde:
        for i in range(Gp.num_vars):
            if (inde[i] == 0):
                sign = 0
                if (len(np.intersect1d(np.where(Gp.graph[:,i] == 1)[0], np.where(inde == 0)[0])) == 0): # Xi has no out-going edges
                    sign = sign + 1
                    Nx = np.intersect1d(np.where(Gp.graph[:,i] == 6)[0], np.where(inde == 0)[0]) # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where(Gp.graph[i, :] == 1)[0], np.where(Gp.graph[:,i]==1)[0]), np.where(inde==0)[0]) # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if (len(Nx) > 0):
                        if check2(Gp, Nx, Ax): # according to the original paper
                            sign = sign + 1
                    else:
                        sign = sign + 1
                if (sign == 2):
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.where(Gp.graph[:,i] == 6)[0]:
                        Gd.add_edge(Edge(nodes[index], nodes[i], -1, 1))
                    inde[i] = 1

    return Gd

def DAG2CPDAG(G): # transform a DAG to a CPDAG
    ## -----------------------------------------------------
    # order the edges in G
    nodes_order = list(map(lambda x : G.node_map[x], G.get_causal_ordering())) # Perform a topological sort on the nodes of G
    # nodes_order(1) is the node which has the highest order
    # nodes_order(N) is the node which has the lowest order
    edges_order= np.asmatrix([[],[]], dtype=np.int64).T
    # edges_order(1,:) is the edge which has the highest order
    # edges_order(M,:) is the edge which has the lowest order
    M = G.get_num_edges() # the number of edges in this DAG
    N = G.get_num_nodes() # the number of nodes in this DAG

    while(edges_order.shape[0] < M):
        for ny in range(N-1, -1, -1):
            j = nodes_order[ny]
            inci_all = np.where(G.graph[j, :] == 1)[0]  # all the edges that incident to j
            if (len(inci_all) != 0):
                if (len(edges_order) != 0):
                    inci = edges_order[np.where(edges_order[:, 1] == j)[0], 0] # ordered edge that incident to j
                    if (len(set(inci_all) - set(inci.T.tolist()[0])) != 0):
                        break
                else:
                    break
        for nx in range(N):
            i = nodes_order[nx]
            if(len(edges_order) != 0):
                if(len(np.intersect1d(np.where(edges_order[:,1]==j)[0], np.where(edges_order[:,0]==i)[0])) == 0 and G.graph[j,i]==1):
                    break
            else:
                if (G.graph[j, i] == 1):
                    break
        edges_order = np.r_[edges_order, np.asmatrix([i, j])]

    ## ----------------------------------------------------------------
    sign_edges = np.zeros(M) # 0 means unknown, 1 means compelled, -1 means reversible
    while (len(np.where(sign_edges == 0)[0]) != 0):
        ss = 0
        for m in range(M-1, -1, -1): # let x->y be the lowest ordered edge that is labeled "unknown"
            if sign_edges[m] == 0:
                i = edges_order[m, 0]
                j = edges_order[m, 1]
                break
        idk = np.where(edges_order[:, 1] == i)[0]
        k = edges_order[idk, 0] # w->x
        for m in range(len(k)):
            if (sign_edges[idk[m]] == 1):
                if (G.graph[j, k[m]] != 1): # if w is not a parent of y
                    id = np.where(edges_order[:, 1] == j)[0] # label every edge that incident into y with "complled"
                    sign_edges[id] = 1
                    ss = 1
                    break
                else:
                    id = np.intersect1d(np.where(edges_order[:, 0] == k[m, 0])[0], np.where(edges_order[:, 1] == j)[0]) # label w->y with "complled"
                    sign_edges[id] = 1
        if (ss):
            continue

        z = np.where(G.graph[j, :] == 1)[0]
        if (len(np.intersect1d(np.setdiff1d(z, i), np.union1d(np.union1d(np.where(G.graph[i, :] == 0)[0], np.where(G.graph[i, :] == -1)[0]), np.where(G.graph[i,:]==6)[0]))) != 0):
            id = np.intersect1d(np.where(edges_order[:,0]== i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[id] = 1  # label x->y  with "compelled"
            id1 = np.where(edges_order[:,1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = 1 # label all "unknown" edges incident into y  with "complled"
        else:
            id = np.intersect1d(np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[id] = -1  # label x->y with "reversible"

            id1 = np.where(edges_order[:,1]==j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = -1 # label all "unknown" edges incident into y with "reversible"

    # create CPDAG accoring the labelled edge
    nodes = G.get_nodes()
    Gcp = GeneralGraph(nodes)
    for m in range(M):
        if (sign_edges[m] == 1):
            Gcp.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], -1, 1))
        else:
            Gcp.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], 6, 6))

    return Gcp

def minimize(X, f, length, *varargin):
    # Minimize a differentiable multivariate function.
    #
    # Usage: X, fX, i = minimize(X, f, length, P1, P2, P3, ... )
    #
    # where the starting point is given by "X" (D by 1), and the function named in
    # the string "f", must return a function value and a vector of partial
    # derivatives of f wrt X, the "length" gives the length of the run: if it is
    # positive, it gives the maximum number of line searches, if negative its
    # absolute gives the maximum allowed number of function evaluations. You can
    # (optionally) give "length" a second component, which will indicate the
    # reduction in function value to be expected in the first line-search (defaults
    # to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
    #
    # The function returns when either its length is up, or if no further progress
    # can be made (ie, we are at a (local) minimum, or so close that due to
    # numerical problems, we cannot get any closer). NOTE: If the function
    # terminates within a few iterations, it could be an indication that the
    # function values and derivatives are not consistent (ie, there may be a bug in
    # the implementation of your "f" function). The function returns the found
    # solution "X", a vector of function values "fX" indicating the progress made
    # and "i" the number of iterations (line searches or function evaluations,
    # depending on the sign of "length") used.
    #
    # The Polack-Ribiere flavour of conjugate gradients is used to compute search
    # directions, and a line search using quadratic and cubic polynomial
    # approximations and the Wolfe-Powell stopping criteria is used together with
    # the slope ratio method for guessing initial step sizes. Additionally a bunch
    # of checks are made to make sure that exploration is taking place and that
    # extrapolation will not be unboundedly large.
    #
    # See also: checkgrad

    INT = 0.1  # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0  # extrapolate maximum 3 times the current step-size
    MAX = 20  # max 20 function evaluations per line search
    RATIO = 10  # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2  # SIG and RHO are the constants controlling the Wolfe-
    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the
    # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with RHO.

    # The code falls naturally into 3 parts, after the initial line search is
    # started in the direction of steepest descent. 1) we first enter a while loop
    # which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
    # have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
    # enter the second loop which takes p2, p3 and p4 chooses the subinterval
    # containing a (local) minimum, and interpolates it, unil an acceptable point
    # is found (Wolfe-Powell conditions). Note, that points are always maintained
    # in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
    # conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
    # was a problem in the previous line-search. Return the best value so far, if
    # two consecutive line-searches fail, or whenever we run out of function
    # evaluations or line-searches. During extrapolation, the "f" function may fail
    # either with an error or returning Nan or Inf, and minimize should handle this
    # gracefully.

    if np.size(length) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1

    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    i = 0  # zero the run length counter
    ls_failed = 0  # no previous line search has failed
    temp = [f, X]
    temp.extend(varargin)
    temp.extend([None, 2])
    f0, df0 = feval(temp)  # get function value and gradient
    fX = f0
    i = i + (1 if length < 0 else 0)  # count epochs?!
    s = -df0
    d0 = (-s.T * s)[0, 0]  # initial search direction (steepest) and slope
    x3 = red / (1 - d0)  # initial step is red/(|s|+1)

    while i < abs(length):  # while not finished
        i = i + (1 if length > 0 else 0)  # count iterations?!
        X0 = X  # make a copy of current values
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while 1:  # keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = False

            while (not success and M > 0):
                try:
                    M = M - 1
                    i = i + (1 if length < 0 else 0)  # count epochs?!
                    temp = [f, X + x3 * s]
                    temp.extend(varargin)
                    temp.extend([None, 2])
                    f3, df3 = feval(temp)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3)) or np.any(np.isinf(df3)):
                        raise Exception('')
                    success = True
                except Exception as e:  # catch any error which occured in f
                    x3 = (x2 + x3) / 2  # bisect and try again

            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            d3 = (df3.T * s)[0, 0]  # new slope
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:  # are we done extrapolating?
                break

            x1 = x2  # move point 2 to point 1
            f1 = f2
            d1 = d2
            x2 = x3  # move point 3 to point 2
            f2 = f3
            d2 = d3
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)  # make cubic extrapolation
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * (x2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))  # num. error possible, ok!
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:  # num prob | wrong sign?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 > x2 * EXT:  # new point beyond extrapolation limit?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 < x2 + INT * (x2 - x1):  # new point too close to previous point?
                x3 = x2 + INT * (x2 - x1)
        # end extrapolation

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:  # keep interpolating
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:  # choose subinterval
                x4 = x3  # move point 3 to point 4
                f4 = f3
                d4 = d3
            else:
                x2 = x3  # move point 3 to point 2
                f2 = f3
                d2 = d3

            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))  # quadratic interpolation
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)  # cubic interpolation
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A  # num. error possible, ok!

            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2  # if we had a numerical problem then bisect

            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))  # don't accept too close
            temp = [f, X + x3 * s]
            temp.extend(varargin)
            temp.extend([None, 2])
            f3, df3 = feval(temp)
            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            M = M - 1
            i = i + (1 if length < 0 else 0)  # count epochs?!
            d3 = (df3.T * s)[0, 0]  # new slope
        # end interpolation

        if (abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0):  # if line search succeeded
            X = X + x3 * s
            f0 = f3
            fX = np.vstack([fX, f0])  # update variables
            s = ((df3.T * df3)[0, 0] - df0.T * df3[0, 0]) / (df0.T * df0)[0, 0] * s - df3  # Polack-Ribiere CG direction
            df0 = df3  # swap derivatives
            d3 = d0
            d0 = (df0.T * s)[0, 0]
            if (d0 > 0):  # new slope must be negative
                s = -df0
                d0 = -(s.T * s)[0, 0]
            x3 = x3 * min(RATIO, d3 / (d0 - sys.float_info.min))  # slope ratio but max RATIO
            ls_failed = 0  # this line search did not fail
        else:
            X = X0  # restore best point so far
            f0 = F0
            df0 = dF0
            if (ls_failed or i > abs(length)):  # line search failed twice in a row
                break  # or we ran out of time, so we give up
            s = -df0  # try steepest
            d0 = -(s.T * s)[0, 0]
            x3 = 1 / (1 - d0)
            ls_failed = 1  # this line search failed
    return X, fX, i



def dist2(x, c):
    # DIST2	Calculates squared distance between two sets of points.
    #
    # Description
    # D = DIST2(X, C) takes two matrices of vectors and calculates the
    # squared Euclidean distance between them.  Both matrices must be of
    # the same column dimension.  If X has M rows and N columns, and C has
    # L rows and N columns, then the result has M rows and L columns.  The
    # I, Jth entry is the  squared distance from the Ith row of X to the
    # Jth row of C.
    #
    # See also
    # GMMACTIV, KMEANS, RBFFWD
    #

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if (dimx != dimc):
        raise Exception('Data dimension does not match dimension of centres')

    n2 = (np.matlib.ones((ncentres, 1)) * np.sum(np.multiply(x, x).T, axis=0)).T + \
         np.matlib.ones((ndata, 1)) * np.sum(np.multiply(c, c).T, axis=0) - \
         2 * (x * c.T)

    # Rounding errors occasionally cause negative entries in n2
    n2[np.where(n2 < 0)] = 0
    return n2

def pdinv(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A).T
        invU = np.eye(numData).dot(np.linalg.inv(U))
        Ainv = invU.dot(invU.T)
    except numpy.linalg.LinAlgError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
    except Exception as e:
        raise e
    return np.matlib.asmatrix(Ainv)

def eigdec(x, N, evals_only=False):
    # EIGDEC	Sorted eigendecomposition
    #
    #	Description
    #	 EVALS = EIGDEC(X, N computes the largest N eigenvalues of the
    #	matrix X in descending order.  [EVALS, EVEC] = EIGDEC(X, N) also
    #	computes the corresponding eigenvectors.
    #
    #	See also
    #	PCA, PPCA
    #

    if (N != np.round(N) or N < 1 or N > x.shape[1]):
        raise Exception('Number of PCs must be integer, >0, < dim')

    # Find the eigenvalues of the data covariance matrix
    if (evals_only):
        # Use eig function as always more efficient than eigs here
        temp_evals, _ = np.linalg.eig(x)
    else:
        # Use eig function unless fraction of eigenvalues required is tiny
        if ((N / x.shape[1]) > 0.04):
            temp_evals, temp_evec = np.linalg.eig(x)
        else:
            temp_evals, temp_evec = scipy.sparse.linalg.eigs(x, k=N, which='LM')

    # Eigenvalues nearly always returned in descending order, but just to make sure.....
    evals = np.sort(-temp_evals)
    perm = np.argsort(-temp_evals)
    evals = -evals[0:N]

    if (not evals_only):
        if (np.all(evals == temp_evals[0:N])):
            # Originals were in order
            evec = temp_evec[:, 0: N]
        else:
            # Need to reorder the eigenvectors
            evec = np.empty_like(temp_evec[:, 0: N])
            for i in range(N):
                evec[:, i] = temp_evec[:, perm[i]]

        return evals.astype(float), evec.astype(float)
    else:
        return evals.astype(float)
def gpr_multi_new(logtheta=None, covfunc=None, x=None, y=None, xstar=None, nargout=1):
    # Here we change the function gpr to gpr_multi, in which y contains a set
    # of vectors on which we do repression from x

    # gpr - Gaussian process regression, with a named covariance function. Two
    # modes are possible: training and prediction: if no test data are given, the
    # function returns minus the log likelihood and its partial derivatives with
    # respect to the hyperparameters; this mode is used to fit the hyperparameters.
    # If test data are given, then (marginal) Gaussian predictions are computed,
    # whose mean and variance are returned. Note that in cases where the covariance
    # function has noise contributions, the variance returned in S2 is for noisy
    # test targets; if you want the variance of the noise-free latent function, you
    # must substract the noise variance.
    #
    # usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y)
    #    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar)
    #
    # where:
    #
    #   logtheta is a (column) vector of log hyperparameters
    #   covfunc  is the covariance function
    #   x        is a n by D matrix of training inputs
    #   y        is a (column) vector (of size n) of targets
    #   xstar    is a nn by D matrix of test inputs
    #   nlml     is the returned value of the negative log marginal likelihood
    #   dnlml    is a (column) vector of partial derivatives of the negative
    #                 log marginal likelihood wrt each log hyperparameter
    #   mu       is a (column) vector (of size nn) of prediced means
    #   S2       is a (column) vector (of size nn) of predicted variances
    #
    # For more help on covariance functions, see "covFunctions".

    if type(covfunc) == str:
        covfunc = [covfunc]  # convert to cell if needed
    n, D = x.shape
    n, m = y.shape
    if eval(feval(covfunc)) != logtheta.shape[0]:
        raise Exception('Error: Number of parameters do not agree with covariance function')

    temp = list(covfunc.copy())
    temp.append(logtheta)
    temp.append(x)
    K = feval(temp)  # compute training set covariance matrix

    L = np.linalg.cholesky(K)  # cholesky factorization of the covariance
    alpha = solve_chol(L.T, y)

    if (
            logtheta is not None and covfunc is not None and x is not None and y is not None and xstar is None):  # if no test cases, compute the negative log marginal likelihood
        out1 = 0.5 * np.trace(y.T * alpha) + m * np.sum(np.log(np.diag(L)), axis=0) + 0.5 * m * n * np.log(
            2 * np.pi)
        if nargout == 2:  # ... and if requested, its partial derivatives
            out2 = np.matlib.zeros((logtheta.shape[0], 1))  # set the size of the derivative vector
            W = m * (np.linalg.inv(L.T) * (
                        np.linalg.inv(L) * np.matlib.eye(n))) - alpha * alpha.T  # precompute for convenience
            for i in range(len(out2) - 1, len(out2)):
                temp = list(covfunc.copy())
                temp.append(logtheta)
                temp.append(x)
                temp.append(i)
                out2[i] = np.sum(np.multiply(W, feval(temp))) / 2
    else:  # ... otherwise compute (marginal) test predictions ...
        temp = list(covfunc.copy())
        temp.append(logtheta)
        temp.append(x)
        temp.append(xstar)
        temp.append(2)  # nargout == 2
        Kss, Kstar = feval(temp)  # test covariances
        out1 = Kstar.T * alpha  # predicted means

        if nargout == 2:
            v = np.linalg.inv(L) * Kstar
            v = np.asmatrix(v)
            out2 = Kss - np.sum(np.multiply(v, v), axis=0).T

    if nargout == 1:
        return out1
    else:
        return out1, out2


K = np.matlib.empty((0, 0))

def covNoise(logtheta=None, x=None, z=None, nargout=1):
    # Independent covariance function, ie "white noise", with specified variance.
    # The covariance function is specified as:
    #
    # k(x^p,x^q) = s2 * \delta(p,q)
    #
    # where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
    # which is 1 iff p=q and zero otherwise. The hyperparameter is
    #
    # logtheta = [ log(sqrt(s2)) ]
    #
    # For more help on design of covariance functions, see "covFunctions".

    if (logtheta is None and x is None and z is None): # report number of parameters
        A = '1'

        return A

    s2 = np.exp(2 * logtheta)[0, 0] # noise variance

    if (logtheta is not None and x is not None and z is None): # compute covariance matrix
        A = s2 * np.matlib.eye(x.shape[0])
    elif (nargout == 2): # compute test set covariances
        A = s2
        B = 0   # zeros cross covariance by independence
    else: # compute derivative matrix
        A = 2 * s2 * np.matlib.eye(x.shape[0])


    if (nargout == 2):
        return A, B
    else:
        return A


def covSEard(loghyper=None, x=None, z=None, nargout=1):
    # Squared Exponential covariance function with Automatic Relevance Detemination
    # (ARD) distance measure. The covariance function is parameterized as:
    #
    # k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    #
    # where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    # D is the dimension of the input space and sf2 is the signal variance. The
    # hyperparameters are:
    #
    # loghyper = [ log(ell_1)
    #              log(ell_2)
    #               .
    #              log(ell_D)
    #              log(sqrt(sf2)) ]
    #
    # For more help on design of covariance functions, see "covFunctions".
    global K

    if (loghyper is None and x is None and z is None):
        A = '(D+1)'

        return A # report number of parameters

    n ,D = x.shape
    loghyper = loghyper.T.tolist()[0]
    ell = np.exp(loghyper[0:D]) # characteristic length scale
    sf2 = np.exp(2 * loghyper[D]) # signal variance

    if (loghyper is not None and x is not None):
        K = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T ) /2)
        A = K
    elif nargout == 2: # compute test set covariances
        A = sf2 * np.matlib.ones((z, 1))
        B = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T, np.matlib.diag( 1 /ell ) *z) / 2)
    else:
        # check for correct dimension of the previously calculated kernel matrix
        if (K.shape[0] != n or K.shape[1] != n):
            K = sf2 * np.exp(-sq_dist(np.matlib.diag(1 / ell) * x.T ) /2)

        if z <= D:  # length scale parameters
            A = np.multiply(K, sq_dist(x[:, z].T / ell[z]))
        else: # magnitude parameter
            A = 2 * K
            K = np.matlib.empty((0, 0))

    if (nargout == 2):
        return A, B
    else:
        return A

def sq_dist(a, b=None, Q=None):
    # sq_dist - a function to compute a matrix of all pairwise squared distances
    # between two sets of vectors, stored in the columns of the two matrices, a
    # (of size D by n) and b (of size D by m). If only a single argument is given
    # or the second matrix is empty, the missing matrix is taken to be identical
    # to the first.
    #
    # Special functionality: If an optional third matrix argument Q is given, it
    # must be of size n by m, and in this case a vector of the traces of the
    # product of Q' and the coordinatewise squared distances is returned.
    #
    # NOTE: The program code is written in the C language for efficiency and is
    # contained in the file sq_dist.c, and should be compiled using matlabs mex
    # facility. However, this file also contains a (less efficient) matlab
    # implementation, supplied only as a help to people unfamiliar with mex. If
    # the C code has been properly compiled and is avaiable, it automatically
    # takes precendence over the matlab code in this file.
    #
    # Usage: C = sq_dist(a, b)
    #    or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])
    #    or: c = sq_dist(a, b, Q)
    # where the b matrix may be empty.
    #
    # where a is of size D by n, b is of size D by m (or empty), C and Q are of
    # size n by m and c is of size D by 1.

    if b is None or len(b) == 0: # input arguments are taken to be identical if b is missing or empty
        b = a

    D, n = a.shape
    d, m = b.shape

    if d != D:
        raise Exception('Error: column lengths must agree.')

    if Q is None:
        C = np.matlib.zeros((n, m))
        for d in range(D):
            temp = np.tile(b[d, :], (n, 1)) - np.tile(a[d, :].T, (1, m))
            C = C + np.multiply(temp, temp)
    else:
        if (n,m) == Q.shape:
            C = np.matlib.zeros((D, 1))
            for d in range(D):
                temp = np.tile(b[d,:], (n, 1)) - np.tile(a[d,:].T, (1, m))
                temp = np.multiply(temp, temp)
                temp = np.multiply(temp, Q)
                C[d] = np.sum(temp)
        else:
            raise Exception('Third argument has wrong size.')
    return C

def solve_chol(A, B):
    # solve_chol - solve linear equations from the Cholesky factorization.
    # Solve A*X = B for X, where A is square, symmetric, positive definite. The
    # input to the function is R the Cholesky decomposition of A and the matrix B.
    # Example: X = solve_chol(chol(A),B);
    #
    # NOTE: The program code is written in the C language for efficiency and is
    # contained in the file solve_chol.c, and should be compiled using matlabs mex
    # facility. However, this file also contains a (less efficient) matlab
    # implementation, supplied only as a help to people unfamiliar with mex. If
    # the C code has been properly compiled and is avaiable, it automatically
    # takes precendence over the matlab code in this file.

    if A is None or B is None:
        raise Exception('Wrong number of arguments.')

    if (A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]):
        raise Exception('Wrong sizes of matrix arguments.')

    res = np.linalg.inv(A) * (np.linalg.inv(A.T) * B)
    return res

def covSum(covfunc, logtheta=None, x=None, z=None, nargout=1):
    # covSum - compose a covariance function as the sum of other covariance
    # functions. This function doesn't actually compute very much on its own, it
    # merely does some bookkeeping, and calls other covariance functions to do the
    # actual work.
    #
    # For more help on design of covariance functions, see "covFunctions".

    j = []
    for i in range(len(covfunc)):  # iterate over covariance functions
        f = covfunc[i]
        j.append([feval([f])])

    if (logtheta is None and x is None and z is None):  # report number of parameters
        A = j[0][0]
        for i in range(1, len(covfunc)):
            A = A + '+' + j[i][0]

        return A

    n, D = x.shape

    v = []  # v vector indicates to which covariance parameters belong
    for i in range(len(covfunc)):
        for k in range(eval(j[i][0])):
            v.append(i)
    v = np.asarray(v)

    if (logtheta is not None and x is not None and z is None):  # compute covariance matrix
        A = np.matlib.zeros((n, n))  # allocate space for covariance matrix
        for i in range(len(covfunc)):  # iteration over summand functions
            f = covfunc[i]
            temp = [f]
            t = logtheta[np.where(v == i)]
            temp.append(t[0] if len(t) == 1 else t)
            temp.append(x)
            A = A + feval(temp)

    if (
            logtheta is not None and x is not None and z is not None):  # compute derivative matrix or test set covariances
        if nargout == 2:  # compute test set cavariances
            A = np.matlib.zeros((z, 1))
            B = np.matlib.zeros((x.shape[0], z))  # allocate space
            for i in range(len(covfunc)):
                f = covfunc[i]
                temp = [f]
                t = logtheta[np.where(v == i)]
                temp.append(t[0] if len(t) == 1 else t)
                temp.append(x)
                temp.append(z)
                temp.append(2)
                AA, BB = feval(temp)  # compute test covariances and accumulate
                A = A + AA
                B = B + BB
        else:  # compute derivative matrices
            i = v[z]  # which covariance function
            j = np.sum(np.where(v[0:z] == i, 1, 0))  # which parameter in that covariance
            f = covfunc[i]
            temp = [f]
            t = logtheta[np.where(v == i)]
            temp.append(t[0] if len(t) == 1 else t)
            temp.append(x)
            temp.append(j)
            A = feval(temp)

    if (nargout == 2):
        return A, B
    else:
        return A


def local_score_CV_general(Data ,Xi ,PAi ,parameters):
    Data = np.asmatrix(Data)
    PAi = list(PAi)
    # calculate the local score
    # using negative k-fold cross-validated log likelihood as the score
    # based on a regression model in RKHS

    T = Data.shape[0]
    X = Data[:, Xi]
    var_lambda = parameters['lambda']   # regularization parameter
    k = parameters['kfold'] # k-fold cross validation
    n0 = math.floor( T /k)
    gamma = 0.01
    Thresh = 1E-5

    if (len(PAi)):
        PA = Data[:, PAi]

        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, ( T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]) # median value
        width = width * 2
        theta = 1 / (width ** 2)

        Kx, _ = kernel(X, X, (theta, 1)) # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T) # for centering of the data in feature space
        Kx = H0 * Kx * H0 # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T)/2, np.min([400, math.floor(T/2)]), evals_only=False)   # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # eig_Kx = eig_Kx[IIx]
        # eix = eix[:, IIx]
        # mx = len(IIx)

        # set the kernel for PA
        Kpa = np.matlib.ones((T, T))

        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m] ,PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 2
            theta = 1 / (width ** 2)
            Kpa = np.multiply(Kpa, kernel(PA[:, m], PA[:, m], (theta, 1))[0])

        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0 # kernel matrix for PA


        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0: T, n0: T]
                Kx_tr_te = Kx[n0: T, 0: n0]
                Kpa_te = Kpa[0:n0, 0: n0]
                Kpa_tr = Kpa[n0: T, n0: T]
                Kpa_tr_te = Kpa[n0: T, 0: n0]
                nv = n0 # sample size of validated data
            if (kk == k- 1):
                Kx_te = Kx[kk * n0:T, kk * n0: T]
                Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                    np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                       np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.matlib.eye(n1))
            tmp2 = tmp1 * Kx_tr * tmp1
            tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * var_lambda ** 2 / gamma * tmp2) * tmp1
            A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                 + 2 * n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

            B = n1 * var_lambda ** 2 / gamma * tmp2 + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 2
        theta = 1 / (width ** 2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 2)]), evals_only=False)  # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # mx = len(IIx)

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                nv = n0
            if (kk == k - 1):
                Kx_te = Kx[kk * n0: T, kk * n0: T]
                Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


def local_score_CV_multi(Data, Xi, PAi, parameters):  # calculate the local score
    # calculate the local score
    # using negative k-fold cross-validated log likelihood as the score
    # based on a regression model in RKHS
    # for variables with multi-variate dimensions
    #
    # parameters.d_label: index of each variable
    T = Data.shape[0]
    X = Data[:, parameters['dlabel'][Xi]]
    var_lambda = parameters['lambda']  # regularization parameter
    k = parameters['kfold']  # k-fold cross validation
    n0 = math.floor(T / k)
    gamma = 0.01
    Thresh = 1E-5

    if (len(PAi)):
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 3  ###
        theta = 1 / (width ** 2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # set the kernel for PA
        Kpa = np.matlib.ones((T, T))

        for m in range(len(PAi)):
            PA = Data[:, parameters['dlabel'][PAi[m]]]
            G = np.sum((np.multiply(PA, PA)), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 3  ###
            theta = 1 / (width ** 2 * PA.shape[1])
            Kpa = np.multiply(Kpa, kernel(PA, PA, (theta, 1))[0])

        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0  # kernel matrix for PA

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0: T, n0: T]
                Kx_tr_te = Kx[n0: T, 0: n0]
                Kpa_te = Kpa[0:n0, 0: n0]
                Kpa_tr = Kpa[n0: T, n0: T]
                Kpa_tr_te = Kpa[n0: T, 0: n0]
                nv = n0  # sample size of validated data
            if (kk == k - 1):
                Kx_te = Kx[kk * n0:T, kk * n0: T]
                Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                    np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                       np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.matlib.eye(n1))
            tmp2 = tmp1 * Kx_tr * tmp1
            tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * var_lambda ** 2 / gamma * tmp2) * tmp1
            A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                 - n1 * var_lambda ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                 + 2 * n1 * var_lambda ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

            B = n1 * var_lambda ** 2 / gamma * tmp2 + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
        width = width * 3  ###
        theta = 1 / (width ** 2 * X.shape[1])  #

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        CV = 0
        for kk in range(k):
            if (kk == 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                nv = n0
            if (kk == k - 1):
                Kx_te = Kx[kk * n0: T, kk * n0: T]
                Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                nv = T - kk * n0
            if (kk < k - 1 and kk > 0):
                Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                  np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                     np.arange(kk * n0, (kk + 1) * n0))]
                nv = n0

            n1 = T - nv
            A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


def local_score_marginal_general(Data, Xi, PAi, parameters):
    # calculate the local score by negative marginal likelihood
    # based on a regression model in RKHS

    T = Data.shape[0]
    X = Data[:, Xi]
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T ** 2, 1))
    width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
    width = width * 2.5  # kernel width
    theta = 1 / (width ** 2)
    H = np.matlib.eye(T) - np.matlib.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1E-5
    eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False)  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if (len(PAi)):
        PA = Data[:, PAi]

        widthPA = np.matlib.empty((PA.shape[1], 1))
        # set the kernel for PA
        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            widthPA[m] = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        widthPA = widthPA * 2.5  # kernel width

        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        logtheta0 = np.vstack([np.log(widthPA), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    else:
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        PA = np.matlib.zeros((T, 1))
        logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    score = nlml  # negative log-likelihood
    return score


def local_score_marginal_multi(Data, Xi, PAi, parameters):
    # calculate the local score by negative marginal likelihood
    # based on a regression model in RKHS
    # for variables with multi-variate dimensions
    #
    # parameters.d_label: index of each variable
    T = Data.shape[0]
    X = Data[:, parameters['dlabel'][Xi]]
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T ** 2, 1))
    widthX = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
    widthX = widthX * 2.5  # kernel width
    theta = 1 / (widthX ** 2)
    H = np.matlib.eye(T) - np.matlib.ones((T, T)) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1E-5
    eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False)  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if (len(PAi)):
        widthPA_all = np.matlib.empty((1, 0))
        # set the kernel for PA
        PA_all = np.matlib.empty((Data.shape[0], 0))
        for m in range(len(PAi)):
            PA = Data[:, parameters['dlabel'][PAi[m]]]
            PA_all = np.hstack([PA_all, PA])
            G = np.sum((np.multiply(PA, PA)), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA * PA.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            widthPA = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
            widthPA_all = np.hstack([widthPA_all, widthPA * np.matlib.ones((1, np.size(parameters['dlabel'][PAi[m]])))])
        widthPA_all = widthPA_all * 2.5  # kernel width
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        logtheta0 = np.vstack([np.log(widthPA_all.T), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA_all,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA_all,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    else:
        covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']])
        PA = np.matlib.zeros((T, 1))
        logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
        logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                              2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

        nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                         2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                         nargout=2)
    score = nlml  # negative log-likelihood
    return score