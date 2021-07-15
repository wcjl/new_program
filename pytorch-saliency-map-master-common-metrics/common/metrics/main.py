# -*- coding: utf-8 -*-

import numpy as np
import os, shutil
from PIL import Image
from scipy.ndimage import filters
from datetime import datetime
import time
import random

import setPrm
import benchCore
"""
import importlib
importlib.reload(setPrm)
importlib.reload(benchCore)
import matplotlib.pyplot as plt
"""
eps = np.finfo(np.float64).eps
prm = setPrm.Prm()
if os.path.isdir(prm.outputDir) is False:
    os.makedirs(prm.outputDir)
shutil.copyfile('setPrm.py', prm.outputDir + '/setPrm.py')

def main():
    fixMean_log = []
    fixCov_log = []
    """
    curSetName = prm.fixationSets[0]
    """
    for curSetName in prm.fixationSets:
        sizeData = np.loadtxt(prm.datasetDir + "/fixations/" + curSetName + "Size.csv", delimiter=',')
        fixCell = [np.loadtxt(prm.datasetDir + "/fixations/" + curSetName + "Fix/" + str(curImgNum+1) + ".csv", delimiter=',') - 1 \
                   for curImgNum in range(sizeData.shape[0])]  # fixations files are represeted in 1-start format -> fixCell_buf: 0-start format
        fixCell = [np.array([fix for fix in fixCell[cnt] if check_cord(fix, sz)]) for cnt, sz in enumerate(sizeData)]
        normFix = [fixCell[cnt][:,0:2] / (sz-1) for cnt, sz in enumerate(sizeData)]
        
        # center bias: Maximum likelihood estimation for variance of Gaussian distribution (PRML, p. 93)
        nfix = np.vstack(normFix)
        fixMean = np.mean(nfix, axis=0)
        fixCov = np.cov(nfix, rowvar=False)
        cb_template = gauss2dm(np.int(sizeData[0, 0]), np.int(sizeData[0, 1]), fixMean, fixCov)
        cb_template = cb_template / np.max(cb_template)
        
        # store mean & cov
        fixMean_log.append(fixMean)
        fixCov_log.append(fixCov.flatten())
        
        """
        curAlgName = prm.fixationAlgs[0]
        """
        for curAlgName in prm.fixationAlgs:
            outFileName_indexLog = prm.outputDir + "/" + curSetName + "_" + curAlgName + "_indexLog.csv"
            if os.path.isfile(outFileName_indexLog):
                if prm.verbose: print("Skipping existing file: " + outFileName_indexLog)
                continue
            if prm.saveMeasuresPerImage:
                outDirPerImage = prm.outputDir + "/" + curSetName + "_" + curAlgName
                if os.path.isdir(outDirPerImage) is False: os.makedirs(outDirPerImage)
            
            tic = time.perf_counter()
            indexLog = np.zeros((prm.sigmaList.size, len(prm.fixationMeasures)))
            """
            curImgNum = 0
            """
            for curImgNum in range(sizeData.shape[0]):
                if prm.verbose: print(curAlgName + " on " + curSetName + ": " + str(curImgNum+1) + "/" + str(sizeData.shape[0]))
                
                sz = sizeData[curImgNum].astype(np.int)
                centerBias = setCenterBias(cb_template, sz)
                kSizeList = np.linalg.norm(sizeData[curImgNum]) * prm.sigmaList
                if (curAlgName == "gauss"):
                    rawSMap = centerBias.copy()
                elif (curAlgName == "uniform"):
                    rawSMap = np.ones(sz)
                else:
                    rawSMap = setRawSMap(prm, curSetName, curAlgName, curImgNum, sz)
                posFix = (normFix[curImgNum] * (sz-1) + 0.5).astype(np.int)
                
                indexLog_buf = benchMeasure(rawSMap, posFix, prm, kSizeList, curSetName, centerBias, curImgNum, normFix)
                indexLog += indexLog_buf
                if prm.saveMeasuresPerImage: np.savetxt(prm.outputDir + "/" + curSetName + "_" + curAlgName + "/" + str(curImgNum) + ".csv", indexLog_buf, delimiter=',', fmt='%f')
            # averaging over images
            indexLog /= sizeData.shape[0]
            
            toc = time.perf_counter()
            if prm.verbose: print(curAlgName + " on " + curSetName + " done in " + str(int(toc-tic)) + ' seconds!')
            np.savetxt(outFileName_indexLog, indexLog, delimiter=',', fmt='%f')
        
    # save fix mean and cov in files
    dt = datetime.now()
    outFileTime = str(dt.year) + "-" + str(dt.month) + "-" + str(dt.day) + "-" + str(dt.hour) + str(dt.minute) + str(dt.second)
    fn = prm.outputDir + "/fixMean_" + outFileTime + ".csv"
    np.savetxt(fn, fixMean_log, delimiter=',', fmt='%f')
    fn = prm.outputDir + "/fixCov_" + outFileTime + ".csv"
    np.savetxt(fn, fixCov_log, delimiter=',', fmt='%f')

def benchMeasure(rawSMap, posFix, prm, kSizeList, curSetName, 
                 centerBias=None, # Options for the measures with center bias
                 curImgNum=0, normFix=None # Options for aucs
                 ):
    smoothSMap = [filters.gaussian_filter(rawSMap, k) if k!=0 else rawSMap for k in kSizeList]
    if ("kl" in prm.fixationMeasures) or ("klwcb" in prm.fixationMeasures) \
            or ("emd" in prm.fixationMeasures) or ("emdwcb" in prm.fixationMeasures) \
            or ("cc" in prm.fixationMeasures) or ("ccwcb" in prm.fixationMeasures) \
            or ("scc" in prm.fixationMeasures) or ("sccwcb" in prm.fixationMeasures) \
            or ("sim" in prm.fixationMeasures) or ("simwcb" in prm.fixationMeasures):
        sz = np.array(rawSMap.shape)
        sigma = prm.fixMapBlurSigma[curSetName]
        posMap = setPosMap(posFix, sz, sigma)
    else:
        posMap = None
    """
    curMeasureName = prm.fixationMeasures[0]
    salMap = smoothSMap[0]
    """
    indexLog = [[calc_benchMeasure(salMap, curMeasureName, posFix, posMap, centerBias, curImgNum, normFix) \
                 for curMeasureName in prm.fixationMeasures] for salMap in smoothSMap]
    return indexLog

def calc_benchMeasure(salMap, curMeasureName, 
                      posFix=None, # Options for value-based and AUC-based metrics
                      posMap=None, # Options for distribution-based metrics
                      centerBias=None,# Options for the metrics with center bias
                      curImgNum=0, normFix=None # Options for aucs
                      ):
    sz = np.array(salMap.shape)
    if (curMeasureName == "logl"):
        return benchCore.compLikelihood(salMap, posFix)
    elif (curMeasureName == "loglwcb"):
        return benchCore.compLikelihood(salMap*centerBias, posFix)
    elif (curMeasureName == "nss"):
        return benchCore.compNSS(salMap, posFix)
    elif (curMeasureName == "nsswcb"):
        return benchCore.compNSS(salMap*centerBias, posFix)
    elif (curMeasureName == "per"):
        return benchCore.compPer(salMap, posFix)
    elif (curMeasureName == "perwcb"):
        return benchCore.compPer(salMap*centerBias, posFix)
    elif (curMeasureName == "kl"):
        return benchCore.compKl(salMap, posMap)
    elif (curMeasureName == "klwcb"):
        return benchCore.compKl(salMap*centerBias, posMap)
    elif (curMeasureName == "emd"):
        return benchCore.compEmd(salMap, posMap)
    elif (curMeasureName == "emdwcb"):
        return benchCore.compEmd(salMap*centerBias, posMap)
    elif (curMeasureName == "cc"):
        return benchCore.compCc(salMap, posMap)
    elif (curMeasureName == "ccwcb"):
        return benchCore.compCc(salMap*centerBias, posMap)
    elif (curMeasureName == "scc"):
        return benchCore.compScc(salMap, posMap)
    elif (curMeasureName == "sccwcb"):
        return benchCore.compScc(salMap*centerBias, posMap)
    elif (curMeasureName == "sim"):
        return benchCore.compSim(salMap, posMap)
    elif (curMeasureName == "simwcb"):
        return benchCore.compSim(salMap*centerBias, posMap)
    elif (curMeasureName == "aucj"):
        negFix = collect_AllNegFix(posFix, sz)
        return benchCore.compAUC(salMap, posFix, negFix)
    elif (curMeasureName == "aucjwcb"):
        negFix = collect_AllNegFix(posFix, sz)
        return benchCore.compAUC(salMap*centerBias, posFix, negFix)
    elif (curMeasureName == "aucb"):
        negFix = collect_RandomNegFix(posFix, sz)
        return benchCore.compAUC(salMap, posFix, negFix)
    elif (curMeasureName == "aucbwcb"):
        negFix = collect_RandomNegFix(posFix, sz)
        return benchCore.compAUC(salMap*centerBias, posFix, negFix)
    elif (curMeasureName == "aucs"):
        negFix = collect_shuffledNegFix(curImgNum, normFix, sz)
        return benchCore.compAUC(salMap, posFix, negFix)
    elif (curMeasureName == "aucswcb"):
        negFix = collect_shuffledNegFix(curImgNum, normFix, sz)
        return benchCore.compAUC(salMap*centerBias, posFix, negFix)
    return 0

def check_cord(cord, imgSize):
    return (cord[0] >= 0) and (cord[1] >= 0) and (cord[0] < imgSize[0]) and (cord[1] < imgSize[1])

# gaussian probability
# x: D-dimensional vector, mu: D-dimensional vector, sigma: DxD -> ret: scalar
def calcDenominator(mu, sigma):
    return np.power(2.0*np.pi, float(mu.size)/2.0) * np.power(np.linalg.det(sigma), 0.5)

def calcNumerator(x, mu, isigma):
    return -np.dot(np.dot(x-mu, isigma), x-mu) / 2.0

def calcGaussianPdf(x, mu, sigma):
    isigma = np.linalg.inv(sigma)
    return np.exp(calcNumerator(x, mu, isigma)) / calcDenominator(mu, sigma)

# x: NxD, mu: KxD, sigma: KxD^2 -> ret: NxK
def calcMultisampleGaussianPdf(x, mu, sigma):
    ret = []
    for k in range(mu.shape[0]):
        s = sigma[k, :].reshape((mu.shape[1], mu.shape[1]))
        isigma = np.linalg.inv(s)
        denominator = calcDenominator(mu[k, :], s)
        ret.append( [np.exp(calcNumerator(x[n, :], mu[k, :], isigma)) / denominator for n in range(x.shape[0])] )
    return np.array(ret)

# gaussian distribution in range 0..1.0, rows, cols: size, mean: 1 x d, cov: d x d, *ret: rows x cols
def gauss2dm(rows, cols, mean, cov):
    if mean.ndim == 1:
        mean = np.array([mean])
    d = 2
    x = np.arange(0, 1.0, 1.0/float(cols))
    y = np.arange(0, 1.0, 1.0/float(rows))
    xx, yy = np.meshgrid(x, y)
    g = np.array([yy, xx]).reshape((d, np.int(cols*rows)))
    p = calcMultisampleGaussianPdf(g.T, mean, cov.reshape((1, d*d)))
    return p.reshape((rows, cols))
    
# gaussian distribution in range 0..1, rows, cols: size, cov: d x d, *ret: rows x cols
def gauss2d(rows, cols, cov):
    return gauss2dm(rows, cols, np.ones((1, cov.shape[0]))*0.5, cov)

# resize np.array-type image: im=[0..255]/[0..1], sz = (height, width)
def imresize(im, sz):
    if np.amax(im) <= 1.0:
        im = im * 255
        scl = 255.0
    else:
        scl = 1
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize((sz[1], sz[0])))/scl

def setCenterBias(cb_template, sz):
    if (cb_template.shape[0]==sz[0]) and (cb_template.shape[1]==sz[1]):
        centerBias = cb_template
    else:
        centerBias = imresize(cb_template, sz)
    return centerBias

def setRawSMap(prm, curSetName, curAlgName, curImgNum, sz):
    if prm.algMaps_under_setName_and_algName:
        filename = prm.algMapDir + "/" + curSetName + "/" + curAlgName + "/" + str(curImgNum + 1) + ".png"
    else:
        filename = prm.algMapDir + "/" + curSetName + "/" + str(curImgNum + 1) + ".png"
    rawSMap = np.array(Image.open(filename).convert('L'), 'f')
    if (rawSMap.shape[0] != sz[0]) or (rawSMap.shape[1] != sz[1]):
        rawSMap = imresize(rawSMap, sz)
    return rawSMap

def setPosMap(posFix, sz, sigma):
    #pmsz = (100, 100)
    pmsz = sz
    fixMap = np.zeros(pmsz)
    for fix in posFix: fixMap[int(fix[0]/sz[0]*pmsz[0]+0.5), int(fix[1]/sz[1]*pmsz[1]+0.5)] += 1
    blurSize = sigma * np.linalg.norm(pmsz)
    if blurSize < eps:
        posMap = fixMap
    else:
        posMap = filters.gaussian_filter(fixMap, blurSize)
    posMap /= np.max(posMap)
    if (pmsz[0] != sz[0]) or (pmsz[1] != sz[1]): posMap = imresize(posMap, sz)
    return posMap

def collect_AllNegFix(posFix, sz):
    fixMap = np.zeros(sz)
    for fix in posFix: fixMap[int(fix[0]+0.5), int(fix[1]+0.5)] = 1
    negFix = np.array(np.where(fixMap == 0)).T
    return negFix

def collect_RandomNegFix(posFix, sz):
    allNegFix = collect_AllNegFix(posFix, sz)
    negFixInd = random.sample(range(allNegFix.shape[0]), posFix.shape[0])
    negFix = allNegFix[negFixInd, :]
    return negFix

def collect_shuffledNegFix(curImgNum, normFix, sz):
    nf = list(normFix) # リストを変更するのでnormFixに影響を与えないように深いコピーをする
    nf.pop(curImgNum)
    negFix = (np.vstack(nf) * (sz-1) + 0.5).astype(np.int)
    return negFix

if __name__ == '__main__':
    main()
