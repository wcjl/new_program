# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:05:23 2017

@author: takao
"""
import numpy as np

# Generate environment and user-specific parameters.
class Prm:
    def __init__(self):
        # Generate Environment Parameters
        self.verbose = True
        self.saveMeasuresPerImage = True
        
        # Generating fixation parameters
        #self.fixationSets = ['bruce', 'cerf', 'judd', 'imgsal', 'pascal']
        #self.fixationAlgs = ['aws', 'aim', 'sig', 'dva', 'gbvs', 'sun', 'itti']
        self.fixationSets = ['pascal']
        #self.fixationAlgs = ['densesalbi3_center_bias', 'densesalbi3_no_center_bias']
        #self.fixationAlgs = ['clique_dilation_multipath']
        self.fixationAlgs = ['clique_single']
        
        self.fixationMeasures = [
                                 "logl", "nss", "per", "kl", "emd", "cc", "scc", "sim", "aucj", "aucb", "aucs", 
#                                 "loglwcb", "nsswcb", "perwcb", "klwcb", "emdwcb", "ccwcb", "sccwcb", "simwcb", "aucjwcb", "aucbwcb", "aucswcb"
                                 ]
        #self.fixMapBlurSigma = { 'bruce' : 0.03, 'cerf' : 0.03, 'judd' : 0.03, 'imgsal' : 0.03, 'pascal' : 0.03 }
        self.fixMapBlurSigma = { 'pascal' : 0.03 }
        
        d = 0.01
        self.sigmaList = np.r_[0:0.08+d:d]
        self.sigmaLen = np.size(self.sigmaList);
        self.defaultSigma = 0.04;
        
        # Directories
        self.datasetDir = '/home/wuchenjunlin/new_program/datasets/SalObj/datasets'
        self.algMapDir = '/home/wuchenjunlin/new_program/datasets/SalObj/algmaps'
        self.outputDir = '/home/wuchenjunlin/new_program/results/cliquenqt_single'
        self.algMaps_under_setName_and_algName = True # True: algMapDir/setName/algName/***.png, False: algMapDir/setName/***.png
