#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:39:07 2022

@author: sebas
"""

import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu

print('This is fathon v{}'.format(fathon.__version__))

a = np.random.randn(10000)
b = np.random.randn(10000)

a = fu.toAggregated(a)
b = fu.toAggregated(b)

pymfdcca = fathon.MFDCCA(a, b)

winSizes = fu.linRangeByStep(10, 2000)
qs = np.arange(-3, 4, 0.1)
revSeg = True
polOrd = 1

n, F = pymfdcca.computeFlucVec(winSizes, qs, revSeg=revSeg, polOrd=polOrd)

list_H, list_H_intercept = pymfdcca.fitFlucVec()

plt.plot(np.log(n), np.log(F[0, :]), 'ro')
plt.plot(np.log(n), list_H_intercept[0]+list_H[0]*np.log(n), 'k-', label='h_{:.1f} = {:.2f}'.format(qs[0], list_H[0]))
half_idx = int(len(qs)/2)
plt.plot(np.log(n), np.log(F[half_idx, :]), 'co')
plt.plot(np.log(n), list_H_intercept[half_idx]+list_H[half_idx]*np.log(n),
         'k-', label='h_{:.1f} = {:.2f}'.format(qs[half_idx], list_H[half_idx]))
plt.plot(np.log(n), np.log(F[-1, :]), 'yo')
plt.plot(np.log(n), list_H_intercept[-1]+list_H[-1]*np.log(n), 'k-',
         label='h_{:.1f} = {:.2f}'.format(qs[-1], list_H[-1]))
plt.xlabel('ln(n)', fontsize=14)
plt.ylabel('ln(F(n))', fontsize=14)
plt.title('MFDCCA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.show()

plt.plot(qs, list_H, 'ro-')
plt.xlabel('q', fontsize=14)
plt.ylabel('h(q)', fontsize=14)
plt.title('h(q)', fontsize=14)
plt.show()

tau = pymfdcca.computeMassExponents()

plt.plot(qs, tau, 'ro-')
plt.xlabel('q', fontsize=14)
plt.ylabel('$\\tau$(q)', fontsize=14)
plt.title('$\\tau$(q)', fontsize=14)
plt.show()

alpha, mfSpect = pymfdcca.computeMultifractalSpectrum()

plt.plot(alpha, mfSpect, 'ro-')
plt.xlabel('$\\alpha$', fontsize=14)
plt.ylabel('f($\\alpha$)', fontsize=14)
plt.title('f($\\alpha$)', fontsize=14)
plt.show()