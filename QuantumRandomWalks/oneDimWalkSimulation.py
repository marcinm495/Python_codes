# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:04:42 2021

@author: Lenovo
"""
#import libraries and modules:
from __future__ import print_function
import numpy as np
import math
from numpy import linalg as LA
from functools import reduce, partialmethod
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OneDimWalk:
    
    def __init__(self, size):
        self.size = size
        
    def lRange(self):
        return -self.size
    
    def rRange(self):
        return self.size
    
    def fullRange(self):
        return 2*self.size + 1
    
    #Normalize a vector of amplitudes, s.t. sum of squares equals 1:
    def normalizeAmps(self, inpArray):
        return inpArray / LA.norm(inpArray)

    #Prepare an initial state of the entire system with amplitudes initialized for a particle
    #in the position (0,0):
    def setInitialState(self, initState, timeSteps):
        normAmps = self.normalizeAmps(initState)
        newArray = np.zeros((timeSteps, self.fullRange(), 2), dtype = np.complex128)
        for k in range(2):
            newArray[0, np.abs(self.lRange()), k] = normAmps[k]
        return newArray

    def normPhase(self, phase, m, n):
        return (1 / np.sqrt(2)) * np.exp(2 * np.pi * 1j * phase * int(m == n))
    
    def calcSingProb(self, elem):
        return np.abs(elem[0]) ** 2 + np.abs(elem[1]) ** 2

    def calcProb(self, sysMat):
        return [[ind + self.lRange(), self.calcSingProb(elem)] for ind, elem in enumerate(sysMat)]
    
    def plotProbOfMat(self, outMatrix):
        x, y = zip(*self.calcProb(outMatrix))
        plt.plot(x, y)
        plt.show()

    def showAnimation(self, calculateXY, calculateStep, step0 = 1, stepN = 50,  stepName = 'step', title = '1D Quantum Random Walk', setYlim = 1.0, interval = 500, repeat = False):
        # Initial plot:
        fig = plt.figure()
        ax = plt.axes(xlim=(-self.size, self.size), ylim=(0, setYlim))
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        plotline, = ax.plot([],[])
        # Init function:
        def initfun():
            plotline.set_data([],[])
            return plotline,
        # Update function:
        def update(step):
            x, y = calculateXY(step)
            plotline.set_data(x, y)
            plt.legend([stepName + f' = {round(calculateStep(step), 2)}'])
            return plotline,
        ani = animation.FuncAnimation(fig, update, init_func=initfun, frames=range(step0, stepN + 1), blit = False, interval = interval, repeat = repeat)
        plt.show()
        
# Simulates 1-D walk with open boundary conditions:
class OpenOneDimWalk(OneDimWalk):
    
    def __init__(self, size):
        super().__init__(size)
        
    
    def amp(self, timeSlice, k, phi, m):
        nextRow = np.take(timeSlice, k + m, axis = 0)
        return super().normPhase(phi, super().lRange() + k + m, 0) * (nextRow[0] + (m * nextRow[1]))
    
    alpha = partialmethod(amp, m = 1)
    beta = partialmethod(amp, m = -1)

    def evolveEntireMatrix(self, amp0, amp1, timeSteps, phi):
        fR = super().fullRange()
        sysMat = super().setInitialState(np.array([amp0, amp1]), timeSteps)
        def evolveSingleStep(row1, row2):
            return np.array([[self.alpha(row1, ind, phi), self.beta(row1, ind, phi)]
                              if(ind != 0 and ind != fR - 1)
                              else elem for ind, elem in enumerate(row2)])
        return np.array(reduce(evolveSingleStep, sysMat))
    
    def evolveAndPlot(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi = 0):
        if timeSteps is None:
            timeSteps = self.size
        outMatrix = self.evolveEntireMatrix(amp0, amp1, timeSteps, phi)
        super().plotProbOfMat(outMatrix)

    def evolveAndSave(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi = 0, filename = "1DQRW"):
        if timeSteps is None:
            timeSteps = self.size
        outMatrix = self.evolveEntireMatrix(amp0, amp1, timeSteps, phi)
        np.savetxt(filename, outMatrix)


    # Animation of dependance on the phase for a given fixed number of evolution steps:
    def showPhaseAnimation(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi0 = 0, phiN = np.pi, phiSteps = 50, setYlim = 1.0, interval = 500, repeat = False):
        if timeSteps is None:
            timeSteps = self.size
        def calculateXY(step):
            x, y = zip(*self.calcProb(self.evolveEntireMatrix(amp0, amp1, timeSteps, step*(phiN-phi0)/phiSteps)))
            return x, y
        def calculateStep(step):
            return step*(phiN-phi0)/phiSteps
        plot_title = 'Phase dependence of 1D Open QRW for TimeSteps'+ f' = {timeSteps}'
        super().showAnimation(calculateXY, calculateStep, stepName = 'phase', title = plot_title, stepN = phiSteps, setYlim = setYlim, interval = interval, repeat = repeat)


    # Animation of dependance on the discrete evolution timestep for a given fixed phase:
    def showTimeAnimation(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), t0 = 1, tN=None, phase = 0, setYlim=1.0, interval = 500, repeat = False):
        if tN is None:
            tN = self.size
        def calculateXY(step):
            x, y = zip(*self.calcProb(self.evolveEntireMatrix(amp0, amp1, step, phase)))
            return x, y
        def calculateStep(step):
            return step
        plot_title = 'Time evolution of 1D Open QRW for Phase'+ f' = {round(phase, 2)}'
        super().showAnimation(calculateXY, calculateStep, step0 = t0, stepN = tN, stepName = 'timestep', title = plot_title, setYlim = setYlim, interval = interval, repeat = repeat)
    

# Simulates 1-D walk with periodic boundary conditions:
class ClosedOneDimWalk(OneDimWalk):
    
    def __init__(self, size):
        super().__init__(size)
        
    
    #Shifting indices on the boundaries of the lattice:
    def moduloShift(self, num, fullRange):
        if (num >= 0 and num <= fullRange - 1):
            return num
        elif (num < 0):
            return num + fullRange - 1
        else:
            return num - fullRange + 1
    

    def amp(self, timeSlice, k, phi, fullRange, m):
        nextRow = np.take(timeSlice, self.moduloShift(k + m, fullRange), axis = 0)
        return super().normPhase(phi, super().lRange() + k + m, 0) * (nextRow[0] + (m * nextRow[1]))
    
    alpha = partialmethod(amp, m = 1)
    beta = partialmethod(amp, m = -1)

    def evolveEntireMatrix(self, amp0, amp1, timeSteps, phi):
        fR = super().fullRange()
        sysMat = super().setInitialState(np.array([amp0, amp1]), timeSteps)
        def evolveSingleStep(row1, row2):
            return np.array([[self.alpha(row1, ind, phi, fR), self.beta(row1, ind, phi, fR)]
                            for ind, elem in enumerate(row2)])
        return np.array(reduce(evolveSingleStep, sysMat))
    
    def evolveAndPlot(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi = 0):
        if timeSteps is None:
            timeSteps = self.size
        outMatrix = self.evolveEntireMatrix(amp0, amp1, timeSteps, phi)
        super().plotProbOfMat(outMatrix)

    def evolveAndSave(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi = 0, filename = "1DQRW"):
        if timeSteps is None:
            timeSteps = self.size
        outMatrix = self.evolveEntireMatrix(amp0, amp1, timeSteps, phi)
        np.savetxt(filename, outMatrix)


    # Animation of dependance on the phase for a given fixed number of evolution steps:
    def showPhaseAnimation(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), timeSteps = None, phi0 = 0, phiN = np.pi, phiSteps = 50, setYlim = 1.0, interval = 500, repeat = False):
        if timeSteps is None:
            timeSteps = self.size
        def calculateXY(step):
            x, y = zip(*self.calcProb(self.evolveEntireMatrix(amp0, amp1, timeSteps, step*(phiN-phi0)/phiSteps)))
            return x, y
        def calculateStep(step):
            return step*(phiN-phi0)/phiSteps
        plot_title = 'Phase dependence of 1D Open QRW for TimeSteps'+ f' = {timeSteps}'
        super().showAnimation(calculateXY, calculateStep, stepName = 'phase', title = plot_title, stepN = phiSteps, setYlim = setYlim, interval = interval, repeat = repeat)


    # Animation of dependance on the discrete evolution timestep for a given fixed phase:
    def showTimeAnimation(self, amp0 = np.sqrt(2) / 2, amp1 = 1j / np.sqrt(2), t0 = 1, tN=None, phase = 0, setYlim=1.0, interval = 500, repeat = False):
        if tN is None:
            tN = self.size
        def calculateXY(step):
            x, y = zip(*self.calcProb(self.evolveEntireMatrix(amp0, amp1, step, phase)))
            return x, y
        def calculateStep(step):
            return step
        plot_title = 'Time evolution of 1D Open QRW for Phase'+ f' = {round(phase, 2)}'
        super().showAnimation(calculateXY, calculateStep, step0 = t0, stepN = tN, stepName = 'timestep', title = plot_title, setYlim = setYlim, interval = interval, repeat = repeat)

def main():
    walk = ClosedOneDimWalk(50)
    #walk.evolveAndPlot(timeSteps=50, phi=0.5)
    walk.showTimeAnimation(tN=100, phase=0.1, interval=500, setYlim=0.9)
    #walk.showPhaseAnimation(timeSteps=20)


if __name__ == '__main__':
    main()
    
    
    
        
