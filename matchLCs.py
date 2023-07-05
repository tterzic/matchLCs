import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import sem
from tabulate import tabulate

# User input:
dataDir = '~/Data/' #Path to data directory
fileList = ['Data1.txt', 'Data2.txt'] #File names with data. 
# Data files format: 
#1st line: band	[unit] (e.g. VHE flux (>100 GeV) [cm^{-2} s^{-1 }])
#2nd line: comlumn names: MJD	Flux(E>100GeV)	FluxErr
#3rd and following lines: data in format: time (in MJD)	flux	flux uncertainty
startMJD = 58100.0	#MJD: Beginning of the time interval considered
endMJD = 58201.0	#MJD: End of the time interval considered
maxDiff = 6 #Maximal time window for quasi-simultaneous measurements in hours.
#No user input after this line


#Reading in the data.
def readData(fileName):
  counts = []
  infile = open(fileName,'r')
  axTitle = infile.readline().strip()
  line = infile.readline()
  line = infile.readline()
  while line:
    if float(line.split()[0]) >= startMJD and float(line.split()[0]) < endMJD:
      counts.append((float(line.split()[0]), float(line.split()[1]), float(line.split()[2])))
    line = infile.readline()
  infile.close()
  order = 1
  if '[' in axTitle:
    axTitle = axTitle.split('[')[0] + '[10^{' + str(order) + '} ' + axTitle.split('[')[1]
  return counts, order, axTitle


#Matching data between two lists by minimal time difference
def createMatches(flux0, flux1):
  X = flux0[:,0]
  Y = flux1[:,0]

  dist0 = np.abs(X[:, np.newaxis] - Y)
  dist0[dist0 > maxMJDdiff] = nan

  Xmin = dist0.min(axis=1)
  Xminpos = dist0.argmin(axis=1)
  Xminpos[Xmin == nan] = nan
  Ymin = dist0.min(axis=0)
  Yminpos = dist0.argmin(axis=0)
  Yminpos[Ymin == nan] = nan

  Xmatchlist = np.vstack((Xminpos[Xminpos < nan], np.squeeze(np.argwhere(Xminpos < nan)))).T
  Ymatchlist = np.vstack((np.squeeze(np.argwhere(Yminpos < nan)), Yminpos[Yminpos < nan])).T
  matchlist = np.array([x for x in set(tuple(x) for x in Xmatchlist) & set(tuple(x) for x in Ymatchlist)])
  matchlist_sort = matchlist[matchlist[:,0].argsort()]

  pairsX, pairsY = [flux1[i] for i,j in matchlist_sort], [flux0[j] for i,j in matchlist_sort]
  pairsX = np.array(pairsX)
  pairsY = np.array(pairsY)
  return (pairsX, pairsY)


#Monte Carlo simulation for estimating uncertainty on rho.
def simSpearman(X, Y):
  n = 10000 #10,000 simulated samples
  rho_sim = np.zeros(n) #Initializing array of Spearman correlation coefficients
  p_sim = np.zeros(n) #Initializing array of p-values
  N = len(X)
  for i in range(n):
    #Generating random samples using measured values with measurement uncertainties
    X_sim = np.random.normal(X[:,1], X[:,2], N)
    Y_sim = np.random.normal(Y[:,1], Y[:,2], N)
    #Calculate correlation coefficient and p-value for n samples
    rho_sim[i], p_sim[i] = spearmanr(X_sim, Y_sim)
  return (rho_sim, p_sim)



maxMJDdiff = maxDiff/24
nan = 100000
time = [str(maxDiff) + ' hrs']
tab_pairs = [time + fileList]
tab_rho = [time + fileList]
tab_p = [time + fileList]
tab_rhoMean = [time + fileList]
tab_rhoStd = [time + fileList]
tab_rhoErr = [time + fileList]
for i in range(len(fileList)):
  aux0 = [fileList[i]]
  aux1 = [fileList[i]]
  aux2 = [fileList[i]]
  aux3 = [fileList[i]]
  aux4 = [fileList[i]]
  aux5 = [fileList[i]]
  flux0 = np.array(readData(dataDir + fileList[i])[0])
  for j in range(len(fileList)):
    flux1 = np.array(readData(dataDir + fileList[j])[0])
    (pX, pY) = createMatches(flux0, flux1)
    rho,p = spearmanr(pX[:,1], pY[:,1])
    aux0.append(len(pX))
    aux1.append(rho)
    aux2.append(p)
    rho_sim,p_sim = simSpearman(pX, pY)
    aux3.append(np.mean(rho_sim))
    aux4.append(np.std(rho_sim, ddof=1))
    aux5.append(sem(rho_sim))
  tab_pairs.append(aux0)
  tab_rho.append(aux1)
  tab_p.append(aux2)
  tab_rhoMean.append(aux3)
  tab_rhoStd.append(aux4)
  tab_rhoErr.append(aux5)


print()
print('Number of pairs')
print(tabulate(tab_pairs, headers='firstrow', tablefmt='fancy_grid'))
print()
print('Spearman\'s rank correlation coefficient')
print(tabulate(tab_rho, headers='firstrow', tablefmt='fancy_grid'))
print()
print('Probability')
print(tabulate(tab_p, headers='firstrow', tablefmt='fancy_grid'))
print()
print('Mean Spearman\'s rank correlation coefficient on sims')
print(tabulate(tab_rhoMean, headers='firstrow', tablefmt='fancy_grid'))
print()
print('StDev of Spearman\'s rank correlation coefficient')
print(tabulate(tab_rhoStd, headers='firstrow', tablefmt='fancy_grid'))
print()
print('Error of Spearman\'s rank correlation coefficient')
print(tabulate(tab_rhoErr, headers='firstrow', tablefmt='fancy_grid'))
