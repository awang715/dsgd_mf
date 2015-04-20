__author__ = 'Annie'

import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from operator import add

# read input data
# return a three dimension tuple
# make the min userid, movieid to be zeros
def read(files):
    token = files.split(",")
    userid = int(token[0])
    movieid= int(token[1])
    score = int(token[2])
    return userid-1,  movieid-1,  score


# initialize W and H with random numbers (0,1)
# return a list, with original input and a vector (array)
def prc(elt, factor):
    list = []
    list.append((elt, np.random.rand(factor)))
    return list

# allocate stratum
def allocate(i, j, rs, cs, mr, mc, nw, tw):     # nw is the current assinging block number, tw is total number of worker
    blockW = mc / tw + 1    # calculate block width
    shift = j - blockW * (nw - 1)   # calculate the shift size that shifts the target blocks to the diagonal
    if shift < 0:   # shift out of range
        shift = shift + mc  # shift to within the range
    nj = shift
    bi = int(((i - rs) / float(mr - rs + 1)) * tw + 1)  # filter the current stratum
    bj = int(((nj - cs) / float(mc - cs + 1)) * tw + 1)
    if bi == bj:    # the diagonal 
        return True
    else:   # filter out
        return False

# assign workers for a stratum
def alloc(i, rs, mr, tw):  # nw is the current assinging block number, tw is total number of worker
    bi = int(((i - rs) / float(mr - rs + 1)) * tw + 1)
    return bi   # return the assigned worker number

# updated function
def f(v, w, h, p_itr, ni, nj):
        L_H = list(h)   # list rdd
        L_W = list(w)
        my_dict_H = dict(L_H)   # dictionarized
        my_dict_W = dict(L_W)  
        c_itr = 0   # initialize iter variable
        for tuples in v: # iterate the stratum
            userid = tuples[0]  # user id
            movieid = tuples[1]     # movie id
            rating = tuples[2]  # rating
            W_i = my_dict_W[userid]     # get the vector in W
            H_j = my_dict_H[movieid]    # get the vector in H
            nn = dict(ni)   # dictionarized broadcasted Ni
            nnn = dict(nj)  # dictionarized broadcasted Nj
            n_i = nn[userid]    # get needed Ni, Nj
            n_j = nnn[movieid] 
            e = (100 + p_itr + c_itr) ** (-beta)    # start updating
            d_wi = (-2) * (rating - np.sum(W_i*H_j)) * H_j + (2 * l/float(n_i))*W_i     # dwi
            d_hj = (-2) * (rating - np.sum(W_i*H_j))*W_i + (2*l/float(n_j))*H_j     # dhj
            n_W_i = W_i - e * d_wi  # update wi
            n_H_j = H_j - e * d_hj  # update hj
            my_dict_W[userid] = np.asarray(n_W_i)  # update W's dictionary
            my_dict_H[movieid] = np.asarray(n_H_j)  # update H's dictionary
            c_itr += 1  # increment iter
        result = []     # initialize result list
        result.append(my_dict_W.items())    # append updated dictionary of W and H
        result.append(my_dict_H.items())
        result.append(c_itr)    # append the current iteration
        return result

# calculate the loss between V and W*H
def calcDiff(v,w,h):
        L_H = list(h)
        L_W = list(w)
        my_dict_H = dict(L_H)
        my_dict_W = dict(L_W)
        ndiff = 0.0
        for tuples in v:    # iterate through V
            userid = tuples[0]
            movieid = tuples[1]
            rv = tuples[2]
            wi = my_dict_W[userid]
            hj = my_dict_H[movieid]
            d_wh = np.sum(wi*hj)    # compute the difference
            ndiff += np.absolute(rv-d_wh)   # sum up every elt in V's difference
        return ndiff

if __name__ == "__main__":
    workerNum = int(sys.argv[2])    # get the input arguments
    factor = int(sys.argv[1])
    iteration = int(sys.argv[3])
    beta = float(sys.argv[4])
    l = float(sys.argv[5])
    input_path = sys.argv[6]
    output_w = sys.argv[7]
    output_h = sys.argv[8]  # get the output path
    sc = SparkContext("local[%d]" % workerNum, "MatrixTest")    # creating context
    textFiles = sc.textFile(input_path)     # read files
    files = textFiles.map(read)
    N_i = files.map(lambda x: (x[0], 1))    # get Ni
    N_i = N_i.reduceByKey(add)
    ni = sc.broadcast(N_i.collect())    # broadcast ni
    N_j = files.map(lambda x: (x[1], 1))    # get Ni
    N_j = N_j.reduceByKey(add)
    nj = sc.broadcast(N_j.collect())    # broadcast Nj
    rowNum = files.map(lambda x: x[0]).max()    # get max row number
    rowSNum = 0
    colNum = files.map(lambda x: x[1]).max()  # get max column number
    colSNum = 0
    W = files.map(lambda x: (x[0], 0))  # initialize W
    W = W.distinct()
    W = W.flatMap(lambda x: prc(x[0], factor))
    H = files.map(lambda x: x[1]).distinct()    # initialize H
    H = H.map(lambda x: (x, 0))
    H = H.flatMap(lambda x: prc(x[0], factor))
    r_W = dict(W.collect())     # create dictionaries of W and H
    r_H = dict(H.collect())

    prev_itr = sc.broadcast(0)  # broadcast previous iterations
    diff = 0.0  # initialize the different to check if it's converged
    for itr in range(1, iteration ): # for loop to iterate
        block_no = itr % workerNum + 1 # calculate the processing stratum's number
        V = files.filter(lambda x: allocate(x[0], x[1], rowSNum, colSNum, rowNum, colNum, block_no, workerNum) is True)     # partition v to get the current stratum
        V_worker=V.map(lambda x: (alloc(x[0], rowSNum, rowNum, workerNum),x))   # partition W
        V_W = V_worker.map(lambda (key,x): (x[0],key))
        V_W = V_W.distinct()
        Wi = V_W.join(W)
        V_H = V_worker.map(lambda (key,x): (x[1],key))   #  partition H
        V_H = V_H.distinct()
        Hj = V_H.join(H)
        V_W = Wi.map(lambda (key,x): (x[0], (key, x[1])))
        V_H = Hj.map(lambda (key,x): (x[0], (key, x[1])))
        V_W_H = V_worker.groupWith(V_W, V_H).partitionBy(workerNum)     #  joining v,w,h and start parallel work
        ndiff = 0.0
        if (itr>1):     # check if converge 
           V_W_H_check = V_W_H.map(lambda (key,x): calcDiff(x[0], x[1], x[2])).collect()
           for c in V_W_H_check:
                ndiff+=c
           if np.absolute(diff - ndiff) <= 90:
                break           # if converge then break the for loop
           else:
                diff = ndiff		   
        V_W_H_p = V_W_H.map(lambda (key,x): f(x[0], x[1], x[2], prev_itr.value, ni.value, nj.value)).collect()   # x[1][0] : V    x[1][1][1] : W     x[1][1][0] : H 
        total_itr = prev_itr.value      # update W H
        for updated in V_W_H_p:
            total_itr += updated[2]
            r_W.update(dict(updated[0]))
            r_H.update(dict(updated[1]))
        prev_itr.unpersist()
        prev_itr = sc.broadcast(total_itr)   # update number of updates
        W = sc.parallelize(r_W.items())     # update W and H
        H = sc.parallelize(r_H.items())	
    np.savetxt(output_w, np.asarray([v for (user_id, v) in sorted(r_W.items())]), delimiter=',')    # output W
    np.savetxt(output_h, np.asarray([v for (movie_id, v) in sorted(r_H.items())]).T, delimiter=',')     # output H





