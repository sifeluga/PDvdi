# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:58:20 2017

@author: z0022fat
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

class PoissonSampler:
    
    # those are never changed
    f2PI = math.pi*2.
    fMinDist = 0.634
       
    def __init__(self, M=110, N=50, AF=12.0, fPow=2.0, NN = 47, aspect = 0, tInc = 0.):
        self.M = M
        self.N = N
        self.fAF = AF
        self.fAspect = aspect
        if aspect == 0:
            self.fAspect = M/N
        self.NN = NN
        self.fPow = fPow
        self.tempInc = tInc
        
        self.M2 = round(M/2)
        self.N2 = round(N/2)
        self.density = np.zeros((M,N),dtype=np.float32)
        self.targetPoints = round(M*N / AF)
        #print(self.targetPoints)
        #print(self.fAF)
        #print(self.N2)
        
        # init varDens
        if(self.fPow > 0):
            self.variableDensity()
    
        
    def variableDensity(self):

        fNorm = 1.2 * math.sqrt(pow(self.M2,2.) + pow(self.N2*self.fAspect,2.))
        
        for j in range(-self.N2,self.N2,1):
            for i in range(-self.M2,self.M2,1):
                self.density[i+self.M2,j+self.N2] = (1. - math.sqrt(math.pow(j*self.fAspect,2.) + math.pow(i,2.))/fNorm)
                
        self.density[(self.density < 0.001)] = 0.001        
        self.density = np.power(self.density,self.fPow)
        accuDensity = math.floor(np.sum(self.density)) 
        #print(accuDensity)
        
        if accuDensity != self.targetPoints:
            scale = self.targetPoints / accuDensity
            scale *= 1.0
            self.density *= scale
            self.density[(self.density < 0.001)] = 0.001
           
        #plt.pcolormesh(self.density)
        #plt.colorbar
        #plt.show()
            
    def addPoint(self,ptN,fDens,iReg):
        
        ptNew   = np.around(ptN).astype(np.int,copy=False)
        idx     = ptNew[0] + ptNew[1]*self.M

        if self.mask[ptNew[0],ptNew[1]]:
            return -1
                    
        for j in range(max(0,ptNew[1]-iReg),min(ptNew[1]+iReg,self.N),1):
            for i in range(max(0,ptNew[0]-iReg),min(ptNew[0]+iReg,self.M),1):
                if self.mask[i,j] == True:
                    pt = self.pointArr[self.idx2arr[i + j*self.M]]
                    if pow(pt[0] - ptN[0],2.)+pow(pt[1] - ptN[1],2.) < fDens:
                    #if pow(i - ptNew[0],2.)+pow(j - ptNew[1],2.) < fDens:
                        return -1
        return idx
        
    def generate(self, seed, accu_mask=None):
           
        # set seed for deterministic results
        np.random.seed(seed)
        
        # preset storage variables
        self.idx2arr = np.zeros((self.M * self.N), dtype=np.int32)
        self.idx2arr.fill(-1)
        self.mask    = np.zeros((self.M, self.N), dtype=bool)
        self.mask.fill(False)
        self.pointArr= np.zeros((self.M * self.N, 2), dtype=np.float32)
        activeList = []
        
        # inits
        count   = 0
        pt = np.array([self.M2, self.N2],dtype=np.float32)
        
        # random jitter of inital point
        jitter = 4
        pt += np.random.uniform(-jitter/2,jitter/2, 2)
        
        # update: point matrix, mask, current idx, idx2matrix and activeList
        self.pointArr[count] = pt
        ptR             = np.around(pt).astype(np.int,copy=False)        
        idx             = ptR[0] + ptR[1]*self.M
        self.mask[ptR[0],ptR[1]]       = True
        self.idx2arr[idx]    = count
        activeList.append(idx)
        count += 1
        
        if(self.fPow == 0):
            self.fMinDist*= self.fAF
        
        # now sample points
        while(activeList):
            idxp = activeList.pop()
            curPt = self.pointArr[self.idx2arr[idxp]]
            curPtR = np.around(curPt).astype(np.int,copy=False)
            
            fCurDens = self.fMinDist 
            if(self.fPow > 0):
                fCurDens /= self.density[curPtR[0],curPtR[1]]
                
            region = int(round(fCurDens))
            
            #if count >= self.targetPoints:
            #    break
            
            for i in range(0,self.NN):
                fRad = np.random.uniform(fCurDens, fCurDens*2.)
                fAng = np.random.uniform(0., self.f2PI)              
                
                ptNew = np.array([curPt[0],curPt[1]],dtype=np.float32)
                ptNew[0] += fRad*math.cos(fAng)
                ptNew[1] += fRad*math.sin(fAng)
                ptNewR = np.around(ptNew).astype(np.int,copy=False) 
                if ptNewR[0] == curPtR[0] and ptNewR[1] == curPtR[1]:
                    continue            
                             
                if(ptNewR[0] >= 0 and ptNewR[1] >= 0 and ptNewR[0] < self.M and ptNewR[1] < self.N):
                    newCurDens = self.fMinDist/self.density[ptNewR[0],ptNewR[1]] 
                    if self.fPow == 0:
                        newCurDens = self.fMinDist
                    if self.tempInc > 0 and accu_mask is not None:
                        if accu_mask[ptNewR[0],ptNewR[1]] > self.density[ptNewR[0],ptNewR[1]] * seed + 1.01-self.tempInc:
                            continue
                    idx = self.addPoint(ptNew,newCurDens,region)
                    if idx >= 0:
                        self.mask[ptNewR[0],ptNewR[1]] = True
                        self.pointArr[count]            = ptNew
                        self.idx2arr[idx]               = count
                        activeList.append(idx)
                        count += 1                   
                        
        print("Generating finished with " + str(count) + " points.")
        #plt.pcolormesh(self.mask.transpose())
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.show()
        
        return self.mask
        #rescaled = (255.0 / self.mask.max() * (self.mask - self.mask.min())).astype(np.uint8)
        #im = Image.fromarray(rescaled)
        #im.save('test.png') 
        

if __name__ == '__main__':
    
    print('Variable density Poisson sampling with variable incoherence')
    parser = argparse.ArgumentParser(description='Executing Poisson disc sampling.')
    parser.add_argument('M',type=int)
    parser.add_argument('N',type=int)
    parser.add_argument('AF',type=float)
    parser.add_argument('-t', default=1,type=int)
    parser.add_argument('-NN', default=42,type=int)
    parser.add_argument('-p', default=2,type=float)
    parser.add_argument('-i', default=0.7,type=float)
    
    args = parser.parse_args()
    #args.t = 10
    
    print(args)
    
    accu = np.zeros((args.M, args.N), dtype=np.int32);
    
    # better call sampler now
    PS = PoissonSampler(args.M, args.N, args.AF, args.p, args.NN, 0, args.i)

    for i in range(0,args.t):
        print("Generating Poisson sampling #"+str(i))
        accu = accu + PS.generate(i+1,accu)

    plt.pcolormesh(accu.transpose())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.clim(0,4)
    plt.colorbar()
    plt.show()
    
else:
    print('I am being imported from another module')