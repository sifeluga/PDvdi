# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:58:20 2017

@author: sifeluga (Felix Lugauer)
"""

import argparse
import math
import numpy as np
from os import getcwd
# for plotting the k-space/density
import matplotlib.pyplot as plt


class PoissonSampler:
    # those are never changed
    f2PI = math.pi * 2.
    # minimal radius (packing constant) for a fully sampled k-space
    fMinDist = 0.634

    def __init__(self, M=110, N=50, AF=12.0, fPow=2.0, NN=47, aspect=0, tInc=0.):
        self.M = M
        self.N = N
        self.fAF = AF
        self.fAspect = aspect
        if aspect == 0:
            self.fAspect = M / N
        self.NN = NN
        self.fPow = fPow
        self.tempInc = tInc

        self.M2 = round(M / 2)
        self.N2 = round(N / 2)
        # need to store density matrix
        self.density = np.zeros((M, N), dtype=np.float32)
        self.targetPoints = round(M * N / AF)

        # init varDens
        if (self.fPow > 0):
            self.variableDensity()

    def variableDensity(self):
        """Precomputes a density matrix, which is used to scale the location-dependent
        radius used for generating new samples.
         """
        fNorm = 1.2 * math.sqrt(pow(self.M2, 2.) + pow(self.N2 * self.fAspect, 2.))

        # computes the euclidean distance for each potential sample location to the center
        for j in range(-self.N2, self.N2, 1):
            for i in range(-self.M2, self.M2, 1):
                self.density[i + self.M2, j + self.N2] = (
                1. - math.sqrt(math.pow(j * self.fAspect, 2.) + math.pow(i, 2.)) / fNorm)

        # avoid diving by zeros
        self.density[(self.density < 0.001)] = 0.001
        # raise scaled distance to the specified power (usually quadratic)
        self.density = np.power(self.density, self.fPow)
        accuDensity = math.floor(np.sum(self.density))

        # linearly adjust accumulated density to match desired number of samples
        if accuDensity != self.targetPoints:
            scale = self.targetPoints / accuDensity
            scale *= 1.0
            self.density *= scale
            self.density[(self.density < 0.001)] = 0.001
            # plt.pcolormesh(self.density)
            # plt.colorbar
            # plt.show()

    def addPoint(self, ptN, fDens, iReg):
        """Inserts a point in the sampling mask if that point is not yet sampled
        and suffices a location-depdent distance (variable density) to
        neighboring points. Returns the index > -1 on success."""
        ptNew = np.around(ptN).astype(np.int, copy=False)
        idx = ptNew[0] + ptNew[1] * self.M

        # point already taken
        if self.mask[ptNew[0], ptNew[1]]:
            return -1

        # check for points in close neighborhood
        for j in range(max(0, ptNew[1] - iReg), min(ptNew[1] + iReg, self.N), 1):
            for i in range(max(0, ptNew[0] - iReg), min(ptNew[0] + iReg, self.M), 1):
                if self.mask[i, j] == True:
                    pt = self.pointArr[self.idx2arr[i + j * self.M]]
                    if pow(pt[0] - ptN[0], 2.) + pow(pt[1] - ptN[1], 2.) < fDens:
                        return -1

        # success if no point was too close
        return idx

    def generate(self, seed, accu_mask=None):

        # set seed for deterministic results
        np.random.seed(seed)

        # preset storage variables
        self.idx2arr = np.zeros((self.M * self.N), dtype=np.int32)
        self.idx2arr.fill(-1)
        self.mask = np.zeros((self.M, self.N), dtype=bool)
        self.mask.fill(False)
        self.pointArr = np.zeros((self.M * self.N, 2), dtype=np.float32)
        activeList = []

        # inits
        count = 0
        pt = np.array([self.M2, self.N2], dtype=np.float32)

        # random jitter of inital point
        jitter = 4
        pt += np.random.uniform(-jitter / 2, jitter / 2, 2)

        # update: point matrix, mask, current idx, idx2matrix and activeList
        self.pointArr[count] = pt
        ptR = np.around(pt).astype(np.int, copy=False)
        idx = ptR[0] + ptR[1] * self.M
        self.mask[ptR[0], ptR[1]] = True
        self.idx2arr[idx] = count
        activeList.append(idx)
        count += 1

        # uniform density
        if (self.fPow == 0):
            self.fMinDist *= self.fAF

        # now sample points
        while (activeList):
            idxp = activeList.pop()
            curPt = self.pointArr[self.idx2arr[idxp]]
            curPtR = np.around(curPt).astype(np.int, copy=False)

            fCurDens = self.fMinDist
            if (self.fPow > 0):
                fCurDens /= self.density[curPtR[0], curPtR[1]]

            region = int(round(fCurDens))

            # if count >= self.targetPoints:
            #    break

            # try to generate NN points around an arbitrary existing point
            for i in range(0, self.NN):
                # random radius and angle
                fRad = np.random.uniform(fCurDens, fCurDens * 2.)
                fAng = np.random.uniform(0., self.f2PI)

                # generate new position
                ptNew = np.array([curPt[0], curPt[1]], dtype=np.float32)
                ptNew[0] += fRad * math.cos(fAng)
                ptNew[1] += fRad * math.sin(fAng)
                ptNewR = np.around(ptNew).astype(np.int, copy=False)
                # continue when old and new positions are the same after rounding
                if ptNewR[0] == curPtR[0] and ptNewR[1] == curPtR[1]:
                    continue

                if (ptNewR[0] >= 0 and ptNewR[1] >= 0 and ptNewR[0] < self.M and ptNewR[1] < self.N):
                    newCurDens = self.fMinDist / self.density[ptNewR[0], ptNewR[1]]
                    if self.fPow == 0:
                        newCurDens = self.fMinDist
                    if self.tempInc > 0 and accu_mask is not None:
                        if accu_mask[ptNewR[0], ptNewR[1]] > self.density[
                            ptNewR[0], ptNewR[1]] * seed + 1.01 - self.tempInc:
                            continue
                    idx = self.addPoint(ptNew, newCurDens, region)
                    if idx >= 0:
                        self.mask[ptNewR[0], ptNewR[1]] = True
                        self.pointArr[count] = ptNew
                        self.idx2arr[idx] = count
                        activeList.append(idx)
                        count += 1

        print("Generating finished with " + str(count) + " points.")

        return self.mask


def write_samples(str,mask):
    f = open(str,'w')
    for i in range(0,mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i,j] > 0:
                f.write('%d %d\n' % (i, j))
    f.close()


if __name__ == '__main__':


    usg = """PDvdiSampler.py W H AF [-N -p -i -o]
     
        Example usage: 'PDvdiSampler.py 150 110 12.5 -N 2 -o 1'
        generates two patterns of size 150x110 and 12.5-fold 
        undersampling with a graphical output
     
        W           Pattern width
        H           Pattern height
        AF          Acceleration factor (AF) > 1.0
     
        optional arguments:
        -h, --help  show this help message and exit
        -N        Number of patterns to generate jointly
        -NN      Number of neighbors (NN) ~[20;80]
        -p        Power to raise of distance from center ~[1.5;2.5]
        -i       Temporal incoherence [0;1[
        -o       Output: 0 - graphical, 1: textfile of sample locations, 2: both
     """

    parser = argparse.ArgumentParser(description='Variable density Poisson sampling'
             ' with variable incoherence. Example usage: PDvdiSampler.py 150 110 12.5 -N 2',usage=usg,add_help=False)


    parser.add_argument('W', type=int, help='Pattern width', default=110)
    parser.add_argument('H', type=int, help='Pattern height', default=50)
    parser.add_argument('AF', type=float, help='Acceleration factor (AF) > 1.0')
    parser.add_argument('-N', default=1, type=int, help='Number of patterns to generate jointly')
    parser.add_argument('-NN',default=42, type=int, help='Number of neighbors (NN) ~[20;80]')
    parser.add_argument('-p', default=2, type=float, help='Power to raise of distance from center ~[1.5;2.5]')
    parser.add_argument('-i', default=0.7, type=float, help='Temporal incoherence [0;1[')
    parser.add_argument('-o', type=int, default=0,
                        help='Output: 0 - graphical, 1: textfile of sample locations, 2: both')

    args = parser.parse_args()

    print('%d K-space(s) of size %d x %d with %.1f-fold undersampling.' % (args.N, args.W, args.H, args.AF))
    print('Power=%1.f, Neighbors=%d, Incoherence=%.1f' %(args.p, args.NN, args.i))

    # initialize accumulated mask
    accu = np.zeros((args.W, args.H), dtype=np.int32)

    # better call sampler now
    PS = PoissonSampler(args.W, args.H, args.AF, args.p, args.NN, 0, args.i)

    # plot settings
    fig = plt.figure()
    rows = 1
    cols = args.N
    if args.N > 4:
        rows = int(math.ceil(args.N/3))
        cols = 3
    #fig.add_subplot(rows,cols,1)

    # generate N k-spaces while updating accumulated mask
    for i in range(0, args.N):
        print("Generating Poisson sampling #" + str(i))
        mask = PS.generate(i + 1, accu)
        accu = accu + mask
        if args.o % 2 == 0:
            fig.add_subplot(rows,cols,i+1)
            plt.title('K-space {0}'.format(i+1))
            plt.pcolormesh(mask.transpose())
            plt.gca().set_aspect('equal', adjustable='box')
        if args.o > 0:
            write_samples('mask{0}.txt'.format(i+1),mask)
            print('File written to: %s' % (getcwd()+'\mask{0}.txt'.format(i+1)))

    if args.o % 2 == 0:
        plt.figure()
        # finally show accumulated mask
        plt.pcolormesh(accu.transpose())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Accumulated mask')
        plt.show()

else:
    print('I am being imported from another module')
