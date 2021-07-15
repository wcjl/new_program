# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:13:42 2017

@author: takao
"""
import numpy as np
from PIL import Image
from pyemd import emd # https://anaconda.org/conda-forge/pyemd

eps = np.finfo(np.float64).eps

def compLikelihood(salMap, posFix):
    #return np.sum(np.log(salMap[posFix[:,0], posFix[:,1]] /  (np.sum(salMap) + eps))) / posFix.shape[0]
    return np.sum(np.log(salMap[posFix[:,0], posFix[:,1]]+eps)) / posFix.shape[0] - np.log((np.sum(salMap) + eps))

def compNSS(salMap, posFix):
    nmap = (salMap - np.mean(salMap)) / (np.std(salMap, ddof=1) + eps)
    return np.sum(nmap[posFix[:,0], posFix[:,1]]) / posFix.shape[0]

def compPer(salMap, posFix):
    return np.sum([np.sum(salMap < salMap[fix[0], fix[1]]) for fix in posFix]) / (np.size(salMap) * posFix.shape[0])

def compKl(salMap, posMap):
    fm = posMap / (np.sum(posMap) + eps)
    sm = salMap / (np.sum(salMap) + eps)
    return np.sum( fm * np.log( (fm + eps) / (sm + eps) ) )

def compEmd(salMap, posMap):
    sz = np.array(salMap.shape)
    
    # reduce image size for efficiency of caluclations
    scl = 32.0
    rSalMap = imresize(salMap, (sz/scl+0.5).astype(np.int))
    rPosMap = imresize(posMap, rSalMap.shape)
    
    nSalMap = rSalMap / (np.sum(rSalMap) + eps)
    nPosMap = rPosMap / (np.sum(rPosMap) + eps)
    
    x = np.arange(0, nSalMap.shape[1])
    y = np.arange(0, nSalMap.shape[0])
    xx, yy = np.meshgrid(x, y)
    xf = xx.flatten()
    yf = yy.flatten()
    dis = np.array([[((xf[r]-xf[c])**2+(yf[r]-yf[c])**2)**0.5 for c in range(len(xf))] for r in range(len(yf))])
    
    return emd(nSalMap.flatten(), nPosMap.flatten(), dis)

def compCc(salMap, posMap):
    return calc_cc(salMap, posMap)

def compScc(salMap, posMap):
    return calc_cc(calc_rankMap(salMap), calc_rankMap(posMap))

def compSim(salMap, posMap):
    sm = salMap / (np.sum(salMap) + eps)
    fm = posMap / (np.sum(posMap) + eps)
    return np.sum(sm * (sm < fm) + fm * (sm >= fm))

# salMap: gray image with [0..255]/[0..1]
def compAUC(salMap, posFix, negFix):
    
    # Quantize all samples
    salMap = np.uint8(salMap / np.amax(salMap) * 255)
    
    posData = salMap[posFix[:,0], posFix[:,1]]
    negData = salMap[negFix[:,0], negFix[:,1]]
    PtNum = [np.size(posData), np.size(negData)]
    
    # Determine threshold list by taking all unique values of pos & neg samples
    thListCnt = np.bincount(np.hstack([posData, negData]))
    thList = np.nonzero(thListCnt)[0]
    
    # Threshold and count
    posCnt = np.bincount(np.hstack([posData, 256]))
    negCnt = np.bincount(np.hstack([negData, 256]))
    posCumsum = np.size(posData) - np.hstack([[0], np.cumsum(posCnt)]) # more than or equal
    negCumsum = np.size(negData) - np.hstack([[0], np.cumsum(negCnt)])
    rocCurve = np.vstack([posCumsum[thList], negCumsum[thList]]).T
    rocCurve = np.vstack([rocCurve, [0,0]])
    
    return areaROC(rocCurve, PtNum)

#------------------------------------------------------------------------------#
def areaROC(rocCurve, PtNum):
    bsCnt = np.bincount((rocCurve[:,1]))
    bs = np.nonzero(bsCnt)[0]
    p_pos = [ \
        rocCurve[ np.where(rocCurve[:,1]==bsi)[0][0] ][0]/np.double(PtNum[0]) \
        for bsi in bs ]
    p_neg = bs / np.double(PtNum[1])
    x = [0] + sorted(p_neg) + [1]
    y = [0] + sorted(p_pos) + [1]
    return np.trapz(y, x)

def calc_cc(d1, d2):
    nd1 = (d1 - np.mean(d1))/(np.std(d1, ddof=1) + eps)
    nd2 = (d2 - np.mean(d2))/(np.std(d2, ddof=1) + eps)
    return np.sum(nd1 * nd2) / np.size(nd1)

def calc_rankMap(mp):
    sortIndex = np.argsort(mp.flatten())
    rankSal = np.zeros(np.size(mp))
    for n, ind in enumerate(sortIndex): rankSal[ind] = n
    return rankSal.reshape(mp.shape)

# resize np.array-type image: im=[0..255]/[0..1], sz = (height, width)
def imresize(im, sz):
    if np.amax(im) <= 1.0:
        im = im * 255
        scl = 255.0
    else:
        scl = 1
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize((sz[1], sz[0])))/scl

