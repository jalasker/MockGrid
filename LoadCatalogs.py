import astropy.io.fits as pf
from astropy.table import Table, join, vstack
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from MockGrid import Plotting
stats = Plotting.plotStats

def loadData(Grid, datadir = '/scratch/group/astro/desi/spectro/data/fuji/LSScats/3.1/', 
             RanFileFMT = 'ELG_{0:s}_{1:d}_clustering.ran.fits', RealFileFMT = 'ELG_{0:s}_clustering.dat.fits',
             fmt = 'fits', nRandToStack = 16, fromPickle = False, pickleFiles = None, zDataRange = [0.9, 1.2]):
    
    if fromPickle:
        raise NotImplementedError('not yet able to load data from previously pickled data files.')
        return 314
    else:
        Grid.zDataRange = zDataRange
        Grid.datadir = datadir 

        RanFile0N = RanFileFMT.format('N', 0)
        RanFile0S = RanFileFMT.format('S', 0)

        RealFileN = 'ELG_N_clustering.dat.fits'
        RealFileS = 'ELG_S_clustering.dat.fits'

        Grid.realcatN = Table.read(Grid.datadir + RealFileFMT.format('N'), format = fmt)
        Grid.realcatS = Table.read(Grid.datadir + RealFileFMT.format('S'), format = fmt)
        Grid.randcatN = Table.read(Grid.datadir + RanFile0N, format = fmt)
        Grid.randcatS = Table.read(Grid.datadir + RanFile0S, format = fmt)

        for i in range(1,nRandToStack):
            randcatNTemp = Table.read(Grid.datadir + RanFileFMT.format('N', i), format = fmt)
            randcatSTemp = Table.read(Grid.datadir + RanFileFMT.format('S', i), format = fmt)
            Grid.randcatN = vstack([Grid.randcatN, randcatNTemp])
            Grid.randcatS = vstack([Grid.randcatS, randcatSTemp])
            
        
        
        print('cleaning data')
        Grid.realcatN = Grid.realcatN[(Grid.realcatN[Grid.zkey] > Grid.zDataRange[0]) & (Grid.realcatN[Grid.zkey] < Grid.zDataRange[1])]
        Grid.realcatS = Grid.realcatS[(Grid.realcatS[Grid.zkey] > Grid.zDataRange[0]) & (Grid.realcatS[Grid.zkey] < Grid.zDataRange[1])]
        Grid.randcatN = Grid.randcatN[(Grid.randcatN[Grid.zkey] > Grid.zDataRange[0]) & (Grid.randcatN[Grid.zkey] < Grid.zDataRange[1])]
        Grid.randcatS = Grid.randcatS[(Grid.randcatS[Grid.zkey] > Grid.zDataRange[0]) & (Grid.randcatS[Grid.zkey] < Grid.zDataRange[1])]
        
        
        Grid.ndata = len(Grid.realcatN) + len(Grid.realcatS)

def loadMocks(Grid, mockdir = '/scratch/group/astro/desi/mocks/Uchuu/ELG/RhoZNorm/',
                 baseCFN = 'Cent_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}_zmin_{3:.03f}_zmax_{4:.03f}_zbox_{5:.03f}_Full.csv',
                 baseSFN = 'Sat_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}_zmin_{3:.03f}_zmax_{4:.03f}_zbox_{5:.03f}_Full.csv', fmt = 'csv',
                 plotStats = False, fromPickle = False, pickleFiles = None, 
                 applyRSD = False, axisRSD = 'z', skip_header = 10):
    Grid.applyRSD = applyRSD
    Grid.axisRSD = str(axisRSD)
    try:
        print('trying Grid.zDataRange[0], Grid.zDataRange[1]')
        zl, zu = Grid.zDataRange[0], Grid.zDataRange[1]
    except:
        print('setting Grid.zDataRange[0,1] to None')
        zl, zu = None, None
    if fromPickle:
        raise NotImplementedError('not yet able to load mocks from previously pickled data files.')
        return 314
    else:
        Grid.mockdir = mockdir
        Grid.baseCFN = baseCFN
        Grid.baseSFN = baseSFN
        for vm, sig, fs in zip(Grid.vMeans, Grid.sigmas, Grid.fsats):
            print('---')
            print(vm)
            print(sig)
            print(fs)
            print('---')
            keyVM, keySig, keyFS = int(Grid.keyscale*vm), int(Grid.keyscale*sig), int(Grid.keyscale*fs)
            if (not (zl is None)) and  (not (zu is None)):
                print('zs shouldnt be none')
                print(zl)
                print(zu)
                CentInputFN = Grid.mockdir + Grid.baseCFN.format(vm, sig, fs, zl, zu, Grid.zmock)
                SatInputFN = Grid.mockdir + Grid.baseSFN.format(vm, sig, fs, zl, zu, Grid.zmock)
            else:
                print('zs should be None')
                print(zl)
                print(zu)
                CentInputFN = Grid.mockdir + Grid.baseCFN.format(vm, sig, fs)
                SatInputFN = Grid.mockdir + Grid.baseSFN.format(vm, sig, fs)
            
            if fmt == 'csv':
                try:
                    CentCat = np.genfromtxt(CentInputFN, names = True, dtype = None, delimiter = ',', skip_header = skip_header)
                    SatCat = np.genfromtxt(SatInputFN, names = True, dtype = None, delimiter = ',', skip_header = skip_header)
                except Exception as e:
                    print(e)
                    print(CentInputFN)
                    print(SatInputFN)
                    assert(0)
            elif (fmt == 'ascii') or (fmt == 'tab'):
                CentCat = np.genfromtxt(CentInputFN, names = True, dtype = None, skip_header = skip_header)
                SatCat = np.genfromtxt(SatInputFN, names = True, dtype = None, skip_header = skip_header)
            elif fmt == 'fits':
                CentCat = Table.read(CentInputFN, format = 'fits')
                SatCat = Table.read(SatInputFN, format = 'fits')
            else:
                CentCat = np.genfromtxt(CentInputFN, names = True, dtype = None, skip_header = skip_header)
                SatCat = np.genfromtxt(SatInputFN, names = True, dtype = None, skip_header = skip_header)
            if fs > 0.0:
                FullCat = np.hstack((CentCat, SatCat))
            else:
                FullCat = CentCat
                print("SatCatShape should be zero or empty")
                print(SatCat.shape)
                print('shape of an empty array')
                print(np.array([]).shape)
                try:
                    assert(SatCat.shape == ())
                except:
                    try:
                        assert(SatCat.shape[0] == 0)
                    except:
                        assert(SatCat == np.empty(0))
            
            if Grid.applyRSD:
                poskey = Grid.axisRSD
                velkey = 'v' + Grid.axisRSD
                print(FullCat[poskey][0:10])
                print(FullCat[velkey][0:10])
                print(Grid.zmock)
                RSDCoord = FullCat[poskey].astype(float) + FullCat[velkey].astype(float)*(1+Grid.zmock)/(Grid.cosmo.H(Grid.zmock)*u.Mpc*u.s/u.km)
                ind0= RSDCoord < 0
                indBS = RSDCoord >Grid.boxsize
                
                RSDCoord[ind0] = RSDCoord[ind0] + Grid.boxsize
                RSDCoord[indBS] = RSDCoord[indBS] - Grid.boxsize
            Grid.mockDict[keyVM][keySig][keyFS] = FullCat
            if fs > 0.0:
                Grid.mockStats[keyVM][keySig][keyFS]['nsat'] = SatCat.shape[0]
            else:
                Grid.mockStats[keyVM][keySig][keyFS]['nsat'] = 0.0
            Grid.mockStats[keyVM][keySig][keyFS]['ncen'] = CentCat.shape[0]
            Grid.mockStats[keyVM][keySig][keyFS]['ntot'] = FullCat.shape[0]
        if plotStats:
            plotStats(Grid)
