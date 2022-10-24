#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:22:01 2022

@author: sebas
"""

import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fathon
from fathon import fathonUtils as fu
plt.style.use(['science','notebook'])

#.Data reading
url_darwin = 'https://crudata.uea.ac.uk/cru/data/soi/soi_dar.dat'
url_tahiti = 'https://crudata.uea.ac.uk/cru/data/soi/soi_tah.dat'
url_soi = 'https://crudata.uea.ac.uk/cru/data/soi/soi_3dp.dat'
urls = [url_darwin, url_tahiti, url_soi]

p_darwin = './data/soi_dar.dat'
p_tahiti = './data/soi_tah.dat'
p_soi = './data/soi_3dp.dat'
p_bei = './data/BEI.dat'

paths = [p_darwin, p_tahiti, p_soi, p_bei]
df = []
for i in range(4):
    p = paths[i]
    if path.exists(p):
        df.append(pd.read_table(paths[i], header=None, delim_whitespace=True))
    else:
        df.append(pd.read_table(urls[i], header=None, delim_whitespace=True))
        print("None")
darwin = df[0]
tahiti = df[1]
soi = df[2]
bei = df[3]


#.Data tidying

#.Rename columns
darwin.rename(columns={0: 'Year'},inplace=True)
tahiti.rename(columns={0: 'Year'},inplace=True)
soi.rename(columns={0: 'Year', 13:'Annual'},inplace=True)
bei.rename(columns={0: 'Year'},inplace=True)

#.Write nan values
darwin[darwin==-990]=np.nan
tahiti[tahiti==-990]=np.nan
soi[soi==99.990]=np.nan
soi[soi==-99.990]=np.nan
bei.rename(columns={0: 'Year'},inplace=True)

#.Pivot data
darwin = darwin.reset_index()
da=pd.melt(darwin, id_vars='Year', value_vars=[1,2,3,4,5,6,7,8,9,10,11,12],var_name='Month',value_name='Pressure')

tahiti = tahiti.reset_index()
ta=pd.melt(tahiti, id_vars='Year', value_vars=[1,2,3,4,5,6,7,8,9,10,11,12],var_name='Month',value_name='Pressure')

soi = soi.reset_index()
so=pd.melt(soi, id_vars='Year', value_vars=[1,2,3,4,5,6,7,8,9,10,11,12,'Annual'],var_name='Month',value_name='Pressure')

bei = bei.reset_index()
be=pd.melt(bei, id_vars='Year', value_vars=[1,2,3,4,5,6,7,8,9,10,11,12],var_name='Month',value_name='Pressure')

#.Drop nan values
so.dropna()
ta.dropna()
da.dropna()
be.dropna()

#Join all data
historical = da.merge(ta, how='inner', on=['Year','Month'])  #Add tahiti to darwin measurements
historical = historical.merge(so, how='inner', on=['Year','Month'])  #Add SOI index
historical = historical.rename(columns={'Pressure_x':'Darwin','Pressure_y':'Tahiti','Pressure':'SOI'}) 
#.Divide by 10 some measurements
historical['Darwin'] = historical['Darwin'] / 10
historical['Tahiti'] = historical['Tahiti'] / 10
historical.dropna()
historical.sort_values(by=['Year','Month'],inplace=True)
#.Dataframe with dates (without days)
dates = pd.to_datetime(historical['Month'].astype(str)+'/'+historical['Year'].astype(str)).dt.date.apply(lambda x: x.strftime('%Y-%m'))

darwin = historical['Darwin']
tahity = historical['Tahiti']
soi = historical['SOI']
year = historical['Year']
bei = be['Pressure']
year_2 = be['Year']

plt.plot(year,soi)
plt.xticks(np.arange(1870,2025,25))
plt.xlabel("Time")
plt.ylabel("SOI")
plt.show()

plt.plot(year,darwin)
plt.xticks(np.arange(1870,2025,25))
plt.xlabel("Time")
plt.ylabel("Pressure (hPa)")
plt.show()

plt.plot(year,tahity)
plt.xticks(np.arange(1870,2025,25))
plt.xlabel("Time")
plt.ylabel("Pressure (hPa)")
plt.show()

#Estadistica descriptiva
from scipy import stats
stat_soi = stats.describe(soi)
stat_dar = stats.describe(darwin)
stat_tah = stats.describe(tahity)
stat_bei = stats.describe(bei)
print('soi = ',stat_soi)
print('dar = ',stat_dar)
print('tah = ',stat_tah)

print('soi = ',np.mean(soi))
print('soi = ',np.median(soi))
print('soi = ',stats.mode(soi))
from statsmodels.graphics.gofplots import qqplot
qqplot(soi, line='s')

a,b = stats.shapiro(soi)
print(a,b)
if b < 1e-3:
    print('The null can be rejected')
else:
    print('not rejected')

a = fu.toAggregated(historical['Darwin'])
b = fu.toAggregated(historical['Tahiti'])
pydcca = fathon.DCCA(a, b)
winSizes = fu.linRangeByStep(40, 100, step=5)
polOrd = 1
revseg = False
overlap = True
n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd, overlap = True, revSeg = revseg)
H, H_intercept = pydcca.fitFlucVec(logBase = 10)
plt.plot(np.log10(n), np.log10(F), 'ro')
plt.plot(np.log10(n), H_intercept+H*np.log10(n), 'k-', label='H = {:.2f}'.format(H))
plt.xlabel('log10(n)', fontsize=14)
plt.ylabel('log10(F(n))', fontsize=14)
plt.title('DCCA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.show()

winSizes = fu.linRangeByStep(3, 380, step=1)
n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd, overlap = overlap, revSeg = revseg)
H, H_intercept = pydcca.fitFlucVec(logBase = 10)
plt.plot(np.log10(n), np.log10(F), 'ro')
plt.plot(np.log10(n), H_intercept+H*np.log10(n), 'k-', label='H = {:.2f}'.format(H))
plt.xlabel('log10(n)', fontsize=14)
plt.ylabel('log10(F(n))', fontsize=14)
plt.title('DCCA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.show()

limits_list = np.array([[3,11], [35,155]], dtype=int)
list_H, list_H_intercept = pydcca.multiFitFlucVec(limits_list,logBase = 10, verbose = True)
clrs = ['k', 'b', 'm', 'c', 'y']
stls = ['-', '--', '.-']
plt.plot(np.log10(n), np.log10(F), 'ro')
for i in range(len(list_H)):
    n_rng = np.arange(limits_list[i][0], limits_list[i][1]+1, 2)
    plt.plot(np.log10(n_rng), list_H_intercept[i]+list_H[i]*np.log10(n_rng),
             clrs[i%len(clrs)]+stls[(i//len(clrs))%len(stls)], label='H = {:.2f}'.format(list_H[i]))
plt.xlabel('log10(n)', fontsize=18)
plt.ylabel('log10(F(n))', fontsize=18)
plt.legend(loc=0, fontsize=18)
plt.show()

winSizes = fu.linRangeByStep(3, 230, step=1)
n, rho = pydcca.computeRho(winSizes, polOrd= polOrd, overlap = overlap, revSeg = revseg, verbose = False)
plt.plot(n, rho, '-ro')
plt.ylim(-.5, 1)
plt.xlabel('n', fontsize=18)
plt.ylabel('$\\rho_{DCCA}$', fontsize=18)
plt.show()

#Compute confidence levels
pythresh = fathon.DCCA()
L = 300
winSizes = fu.linRangeByStep(3, 230, step=1)
nSim = 100
confLvl = 0.95
polOrd = 1
n, cInt1, cInt2 = pythresh.rhoThresholds(L, winSizes, nSim, confLvl, polOrd=polOrd, verbose=False)
plt.plot(n, rho, '-ro')
plt.plot(n, cInt1, 'g-')
plt.plot(n, cInt2, 'b-')
plt.ylim(-1, 1)
plt.xlabel('n', fontsize=18)
plt.ylabel('$\\rho_{DCCA}$', fontsize=18)
plt.savefig('./fig/fig_rho.pdf')
plt.show()

pymfdfa = fathon.MFDFA(a)
winSizes = fu.linRangeByStep(10, 380)
qs = np.arange(-10, 10)
n_c, F_c = pymfdfa.computeFlucVec(winSizes, qs, revSeg=revseg, polOrd=polOrd)
list_H_c, list_H_intercept_c = pymfdfa.fitFlucVec()
plt.plot(np.log(n_c), np.log(F_c[0, :]), 'ro')
plt.plot(np.log(n_c), list_H_intercept_c[0]+list_H_c[0]*np.log(n_c), 'k-', label='h_{:.1f} = {:.2f}'.format(qs[0], list_H_c[0]))
half_idx = int(len(qs)/2)
plt.plot(np.log(n_c), np.log(F_c[half_idx, :]), 'co')
plt.plot(np.log(n_c), list_H_intercept_c[half_idx]+list_H_c[half_idx]*np.log(n_c),
         'k-', label='h_{:.1f} = {:.2f}'.format(qs[half_idx], list_H_c[half_idx]))
plt.plot(np.log(n_c), np.log(F_c[-1, :]), 'yo')
plt.plot(np.log(n_c), list_H_intercept_c[-1]+list_H_c[-1]*np.log(n_c), 'k-',
         label='h_{:.1f} = {:.2f}'.format(qs[-1], list_H_c[-1]))
plt.xlabel('ln(n)', fontsize=14)
plt.ylabel('ln(F(n))', fontsize=14)
plt.title('MFDFA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.savefig('fig_7.pdf')
plt.show()

pymfdfa_s = fathon.MFDFA(b)
winSizes = fu.linRangeByStep(10, 380)
qs = np.arange(-10, 10)
n_s, F_s = pymfdfa_s.computeFlucVec(winSizes, qs, revSeg=revseg, polOrd=polOrd)
list_H_s, list_H_intercept_s = pymfdfa_s.fitFlucVec()
plt.plot(np.log(n_s), np.log(F_s[0, :]), 'ro')
plt.plot(np.log(n_s), list_H_intercept_s[0]+list_H_s[0]*np.log(n_s), 'k-', label='h_{:.1f} = {:.2f}'.format(qs[0], list_H_s[0]))
half_idx = int(len(qs)/2)
plt.plot(np.log(n_s), np.log(F_s[half_idx, :]), 'co')
plt.plot(np.log(n_s), list_H_intercept_s[half_idx]+list_H_s[half_idx]*np.log(n_s),
         'k-', label='h_{:.1f} = {:.2f}'.format(qs[half_idx], list_H_s[half_idx]))
plt.plot(np.log(n_s), np.log(F_s[-1, :]), 'yo')
plt.plot(np.log(n_s), list_H_intercept_s[-1]+list_H_s[-1]*np.log(n_s), 'k-',
         label='h_{:.1f} = {:.2f}'.format(qs[-1], list_H_s[-1]))
plt.xlabel('ln(n)', fontsize=14)
plt.ylabel('ln(F(n))', fontsize=14)
plt.title('MFDFA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.savefig('fig_8.pdf')
plt.show()

plt.plot(qs, list_H_c, 'ro-')
plt.plot(qs, list_H_s, 'bo-')
plt.xlabel('q', fontsize=14)
plt.ylabel('h(q)', fontsize=14)
plt.title('h(q)', fontsize=14)
plt.show()

tau_c = pymfdfa.computeMassExponents()
tau_s = pymfdfa_s.computeMassExponents()
plt.plot(qs, tau_c, 'ro-')
plt.plot(qs, tau_s, 'bo-')
plt.xlabel('q', fontsize=14)
plt.ylabel('$\\tau$(q)', fontsize=14)
plt.title('$\\tau$(q)', fontsize=14)
plt.show()

alpha_c, mfSpect_c = pymfdfa.computeMultifractalSpectrum()
alpha_s, mfSpect_s = pymfdfa_s.computeMultifractalSpectrum()
plt.plot(alpha_c[:], mfSpect_c[:], 'ro-')
plt.plot(alpha_s[:], mfSpect_s[:], 'bo-')
plt.xlabel('$\\alpha$', fontsize=14)
plt.ylabel('f($\\alpha$)', fontsize=14)
plt.title('f($\\alpha$)', fontsize=14)
plt.show()

#Multifractal Cross Correlation

pymfdcca = fathon.MFDCCA(a, b)
winSizes = fu.linRangeByStep(10, 380)
qs = np.arange(-10, 10)
n, F = pymfdcca.computeFlucVec(winSizes, qs, polOrd = polOrd, revSeg = revseg)
list_H, list_H_intercept = pymfdcca.fitFlucVec()
s = 5
plt.plot(np.log(n), np.log(F[0, :]), 'ro', markersize = s)
plt.plot(np.log(n), list_H_intercept[0]+list_H[0]*np.log(n), 'k-', label='h_{:.1f} = {:.2f}'.format(qs[0], list_H[0]))
half_idx = int(len(qs)/2)
plt.plot(np.log(n), np.log(F[half_idx, :]), 'co', markersize = s)
plt.plot(np.log(n), list_H_intercept[half_idx]+list_H[half_idx]*np.log(n),
         'k-', label='h_{:.1f} = {:.2f}'.format(qs[half_idx], list_H[half_idx]))
plt.plot(np.log(n), np.log(F[-1, :]), 'yo', markersize = s)
plt.xlabel('log(n)', fontsize=18)
plt.ylabel('log(F(n))', fontsize=18)
plt.title('MFDCCA', fontsize=18)
plt.legend(loc=0, fontsize=18)
plt.show()

plt.plot(np.log10(n), np.log10(F[0, :]), 'ro')
plt.plot(np.log10(n), list_H_intercept[0]+list_H[0]*np.log10(n), 'k-', label='h_{:.1f} = {:.2f}'.format(qs[0], list_H[0]))
half_idx = int(len(qs)/2)
plt.plot(np.log10(n), np.log10(F[half_idx, :]), 'co')
plt.plot(np.log10(n), list_H_intercept[half_idx]+list_H[half_idx]*np.log10(n),
         'k-', label='h_{:.1f} = {:.2f}'.format(qs[half_idx], list_H[half_idx]))
plt.plot(np.log10(n), np.log10(F[-1, :]), 'yo')
plt.plot(np.log10(n), list_H_intercept[-1]+list_H[-1]*np.log10(n), 'k-',
         label='h_{:.1f} = {:.2f}'.format(qs[-1], list_H[-1]))
plt.xlabel('log10(n)', fontsize=18)
plt.ylabel('log10(F(n))', fontsize=18)
plt.title('MFDCCA', fontsize=18)
plt.legend(loc=0, fontsize=18)
plt.show()

plt.plot(qs, list_H, 'bo-')
plt.xlabel('q', fontsize=18)
plt.ylabel('h(q)', fontsize=18)
plt.title('h(q)', fontsize=18)
plt.show()

tau_c = pymfdcca.computeMassExponents()
plt.plot(qs, tau_c, 'ro-')
plt.xlabel('q', fontsize=18)
plt.ylabel('$\\tau$(q)', fontsize=18)
plt.title('$\\tau$(q)', fontsize=18)
plt.show()

alpha_c, mfSpect_c = pymfdcca.computeMultifractalSpectrum()
plt.plot(alpha_c, mfSpect_c, 'ro-')
plt.xlabel('$\\alpha$', fontsize=18)
plt.ylabel('f($\\alpha$)', fontsize=18)
plt.title('f($\\alpha$)', fontsize=18)
plt.show()