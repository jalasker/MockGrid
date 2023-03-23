import astropy.cosmology as cosmo
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
from pycorr import KMeansSubsampler
import astropy.units as u
import numpy as np
from numpy import random as rand
rng = rand.default_rng()
import matplotlib.pyplot as plt

import pickle
nthreads = 32
# To activate logging
setup_logging()

def TPCFCalcData(Grid, jknsamp = 512, jknside = 512, seed = 31415,
                    edges =  (np.linspace(0., 100., 51), np.linspace(-1., 1., 201)), 
                     ells = (0,2,4), seprange = [0.1, 30.0], statType = 'smu',
                    fromPickle = False, pickleFiles = None, weightType = None):
    if fromPickle:
        raise NotImplementedError('not yet able to load data from previously pickled data 2PCFs.')
        return 314
    else:
        Grid.seprange = seprange
        Grid.jknsamp = jknsamp
        Grid.jknside = jknside
        Grid.seed = seed
        Grid.statType = statType
        Grid.edges = edges
        Grid.weightType = weightType

        Grid.data_positionsN= [Grid.realcatN['RA'].astype(float), Grid.realcatN['DEC'].astype(float), 
                          Grid.cosmo.comoving_distance(Grid.realcatN[Grid.zkey]).astype(float)]
        Grid.random_positionsN = [Grid.randcatN['RA'].astype(float), Grid.randcatN['DEC'].astype(float), 
                             Grid.cosmo.comoving_distance(Grid.randcatN[Grid.zkey]).astype(float)]
        
        if Grid.weightType == None:
            data_weights1N = np.ones(len(Grid.realcatN))
            data_weights1S = np.ones(len(Grid.realcatS))
        elif Grid.weightType.upper() == 'PIP':
            data_weights1N = Grid.realcatN['BITWEIGHTS']
            data_weights1S = Grid.realcatS['BITWEIGHTS']
        elif Grid.weightType.lower() == 'angup':
            data_weights1N = Grid.realcatN['WEIGHTS']
            data_weights1S = Grid.realcatS['WEIGHTS']
        elif (Grid.weightType.lower() == 'angup+pip') or Grid.weightType.lower() == 'pip+angup':
            import warnings
            warnings.warn('I have no reason to believe this will work, but it is in the example jupyter nb', RuntimeWarning)
            data_weights1N = Grid.realcatN['BITWEIGHTS']+ Grid.realcatN['WEIGHTS']
            data_weights1S = Grid.realcatS['BITWEIGHTS']+ Grid.realcatS['WEIGHTS']
        elif Grid.weightType.upper() == 'IIP':
            raise NotImplementedError('BRB Trying to figure out how to sum bitweights')
        elif (Grid.weightType.lower() == 'angup+iip') or Grid.weightType.lower() == 'iip+angup':
            raise NotImplementedError('BRB Trying to figure out how to sum bitweights')
            data_weights1N = Grid.realcatN['WEIGHTS'] + MagicBitweightSummer(Grid.realcatN['BITWEIGHTS'])
            data_weights1S = Grid.realcatS['WEIGHTS'] + MagicBitweightSummer(Grid.realcatS['BITWEIGHTS'])                
            
        Grid.subsamplerN = KMeansSubsampler(mode='angular', positions=Grid.data_positionsN, nsamples=Grid.jknsamp, nside=Grid.jknside, random_state=Grid.seed, position_type='rdd')
        Grid.labelsN = Grid.subsamplerN.label(Grid.data_positionsN)
        Grid.labelsRandN = Grid.subsamplerN.label(Grid.random_positionsN)

        Grid.data_positionsS= [Grid.realcatS['RA'].astype(float), Grid.realcatS['DEC'].astype(float), 
                          Grid.cosmo.comoving_distance(Grid.realcatS[Grid.zkey]).astype(float)]
        Grid.random_positionsS = [Grid.randcatS['RA'].astype(float), Grid.randcatS['DEC'].astype(float), 
                             Grid.cosmo.comoving_distance(Grid.randcatS[Grid.zkey]).astype(float)]
        Grid.subsamplerS = KMeansSubsampler(mode='angular', positions=Grid.data_positionsS, nsamples=Grid.jknsamp, nside=Grid.jknside, random_state=Grid.seed, position_type='rdd')
        Grid.labelsS = Grid.subsamplerS.label(Grid.data_positionsS)
        Grid.labelsRandS = Grid.subsamplerS.label(Grid.random_positionsS)
        
        if Grid.weightType == None:
            Grid.resultN = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsN,
                                                       randoms_positions1=Grid.random_positionsN,
                                                       data_samples1=Grid.labelsN, randoms_samples1=Grid.labelsRandN, position_type='rdd',
                                                       engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)


            Grid.resultS = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsS,
                                         randoms_positions1=Grid.random_positionsS,
                                         data_samples1=Grid.labelsS, randoms_samples1=Grid.labelsRandS, position_type='rdd',
                                         engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)
        
        else:
            Grid.resultN = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsN,
                                                       data_weights1=data_weights1, randoms_positions1=Grid.random_positionsN,
                                                       data_samples1=Grid.labelsN, randoms_samples1=Grid.labelsRandN, position_type='rdd',
                                                       engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)


            Grid.resultS = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsS,
                                         data_weights1=data_weights1, randoms_positions1=Grid.random_positionsS,
                                         data_samples1=Grid.labelsS, randoms_samples1=Grid.labelsRandS, position_type='rdd',
                                         engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)

        # Project to multipoles (monopole, quadruple, hexadecapole)
        Grid.ells = ells
        Grid.nells = len(ells)
        
        Grid.resultNS = Grid.resultN.normalize() + Grid.resultS.normalize()

        Grid.s, Grid.xiell, Grid.cov = Grid.resultNS.get_corr(ells=Grid.ells, return_sep=True, return_cov=True)


    def TPCFCalcMockGrid(Grid, fromPickle = False, pickleFiles = None, useRandoms = True,
                         makeRandomsFromMock = True, randomFN = None, skip_header = 1,
                         position_type = 'rdd'):
    if useRandoms:
        assert(makeRandomsFromMock ^ (type(randomFN) == str))
    if fromPickle:
        raise NotImplementedError('not yet able to load data from previously pickled data files.')
        return 314
    else:
        #pos = return_xyz_formatted_array(FullCat['x'],FullCat['y'],FullCat['z'])
        for vm, sig, fs in zip(Grid.vMeans, Grid.sigmas, Grid.fsats):
            keyVM, keySig, keyFS = int(Grid.keyscale*vm), int(Grid.keyscale*sig), int(Grid.keyscale*fs)
            FullCat = Grid.mockDict[keyVM][keySig][keyFS] 
            if position_type.lower().startswith('rd'):
                FullCatRA = FullCat['ra']
                FullCatDec = FullCat['dec']
            if position_type.lower() == 'rdd':
                FullCatCD = Grid.cosmo.comoving_distance(FullCat['zobs']).astype(float)
                pos = [FullCatRA, FullCatDec, FullCatCD]
            elif position_type.lower() == 'rd':
                pos = [FullCatRA, FullCatDec]
            elif position_type.lower() == 'xyz':
                FullCatX = FullCat['x']
                FullCatY = FullCat['y']
                FullCatZ = FullCat['z']
                pos = [FullCatX, FullCatY, FullCatZ]
            if useRandoms:
                if makeRandomsFromMock:
                    if position_type.lower() == 'rdd':
                        randPosMock = [rng.uniform(0.0, 360, FullCatRA.shape), np.degrees(np.arcsin(rng.uniform(-1, 1, FullCatRA.shape))), MockCDs]
                    elif position_type.lower() == 'rd':
                        randPosMock = [rng.uniform(0.0, 360, FullCatRA.shape), np.degrees(np.arcsin(rng.uniform(-1, 1, FullCatRA.shape)))]
                    else:
                        randPosMock = [rng.uniform(0.0, Grid.boxsize, FullCatX.shape),
                                       rng.uniform(0.0, Grid.boxsize, FullCatY.shape), 
                                       rng.uniform(0.0, Grid.boxsize, FullCatZ.shape)]
                else:
                    if (randFile.split('.')[-1] == 'h5') or (randFile.split('.')[-1] == 'hdf5') or (randFile.split('.')[-1] == 'h5py'):
                        randCat = h5py.File(randFile, 'r')
                    elif (randFile.split('.')[-1] == 'fits') or (randFile.split('.')[-1] == 'fit'):
                        randCat = pf.open(randFile)[1].data
                    elif (randFile.split('.')[-1] == 'csv'):
                        randCat = np.genfromtxt(randFile, delimiter = ',', skip_header = skip_header, names = True, dtype = None)
                    else:
                        randCat = np.genfromtxt(randFile, skip_header = skip_header, names = True, dtype = None)
                        
                    if position_type.lower().startswith('rd'):
                        randRA = randCat['ra'][:]
                        randDec = randCat['dec'][:]
                    if position_type.lower() == 'rdd':
                        randCD = Grid.cosmo.comoving_distance(randCat['zobs'][:]).astype(float)
                        randPosMock = [randRA, randDec, randCD]
                    elif position_type.lower() == 'rd':
                        randPosMock = [randRA, randDec]
                    elif position_type.lower() == 'xyz':
                        randX = randCat['x'][:]
                        randY = randCat['y'][:]
                        randZ = randCat['z'][:]
                        randPosMock = [randX, randY, randZ]
                    
            if useRandoms:
                res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,los = 'z',
                                         engine='corrfunc', boxsize=Grid.boxsize, nthreads=Grid.nthreads,
                                         randoms_positions1 = randPosMock, position_type = position_type,
                                         compute_sepsavg=False)
            else:
                res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,los = 'z',
                                         engine='corrfunc', boxsize=Grid.boxsize, nthreads=Grid.nthreads,
                                         position_type = position_type, compute_sepsavg=False)
            sTemp, xiellTemp = res.get_corr(ells=Grid.ells, return_sep=True, return_cov=False)
            Grid.TPCFDict[keyVM][keySig][keyFS] = (sTemp, xiellTemp)
