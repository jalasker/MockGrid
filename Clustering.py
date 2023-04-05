import astropy.cosmology as cosmo
from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, utils, setup_logging
from pycorr import KMeansSubsampler
import astropy.units as u
import numpy as np
from numpy import random as rand
rng = rand.default_rng()
import matplotlib.pyplot as plt
import scipy.interpolate as interpol
import pickle
nthreads = 32
# To activate logging
setup_logging()

def _format_bitweights(bitweights):
    # From xirunpc.py in desihub/LSS Repo
    if bitweights.ndim == 2: return list(bitweights.T)
    return [bitweights]

def wFKP(z, nz, tracer = 'ELG'):
    PDict = {'ELG': 4000, 'LRG': 10000, 'QSO': 6000, 'BGS' : 7000}
    P0 = PDict[tracer]
    return 1.0/(1.0 + P0*nz)

def TPCFCalcData(Grid, jknsamp = 512, jknside = 512, seed = 31415,
                    rmin = -1, rmax = 2, nrbins = 31, nmubins = 51, 
                     ells = (0,2,4), statType = 'smu',
                    fromTemp = False, saveTemp = True, weightType = None):

        Grid.jknsamp = jknsamp
        Grid.jknside = jknside
        Grid.seed = seed
        Grid.statType = statType
        edges =  (np.logspace(rmin, rmax, nrbins), np.linspace(-1, 1., nmubins))
        Grid.rmin = rmin; Grid.rmax = rmax; Grid.nrbins = nrbins; Grid.nmubins = nmubins
        Grid.edges = edges
        Grid.weightType = weightType
        if fromTemp:
            if not (Grid.weightType is None):
                Grid.resultN = np.load(Grid.tempstorage + '/DESI_Fuji_NGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}_{6}weights_TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], Grid.weightType), allow_pickle = True, fix_imports = True)
                Grid.resultS = np.load(Grid.tempstorage + '/DESI_Fuji_SGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}_{6}weights_TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], Grid.weightType), allow_pickle = True, fix_imports = True)
            else:
                Grid.resultN = np.load(Grid.tempstorage + '/DESI_Fuji_NGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], Grid.weightType), allow_pickle = True, fix_imports = True)
                Grid.resultS = np.load(Grid.tempstorage + '/DESI_Fuji_SGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], Grid.weightType), allow_pickle = True, fix_imports = True)
        else:

            Grid.data_positionsN= [Grid.realcatN['RA'].astype(float), Grid.realcatN['DEC'].astype(float), 
                              Grid.cosmo.comoving_distance(Grid.realcatN[Grid.zkey]).astype(float)]
            Grid.random_positionsN = [Grid.randcatN['RA'].astype(float), Grid.randcatN['DEC'].astype(float), 
                                 Grid.cosmo.comoving_distance(Grid.randcatN[Grid.zkey]).astype(float)]
            

            if Grid.weightType == None:
                data_weights1N = np.ones(len(Grid.realcatN))
                data_weights1S = np.ones(len(Grid.realcatS))
            else:
                data_weights1N = np.ones(len(Grid.realcatN))
                data_weights1S = np.ones(len(Grid.realcatS))
                randoms_weights1N = np.ones(len(Grid.randcatN))
                randoms_weights1S = np.ones(len(Grid.randcatS))

                Grid.weightList = sorted(Grid.weightType.split('+'))
                for wType in Grid.weightList:
                    if wType.lower() == 'pip':
                        assert(not 'iip' in Grid.weightList)
                        data_weights1N = data_weights1N + _format_bitweights(Grid.realcatN['BITWEIGHTS'])
                        data_weights1S = data_weights1S +_format_bitweights(Grid.realcatS['BITWEIGHTS'])
                    elif wType.lower() == 'pip2':
                        data_weights1N =  _format_bitweights(Grid.realcatN['BITWEIGHTS'])
                        data_weights1S = _format_bitweights(Grid.realcatS['BITWEIGHTS'])
                        break
                    elif wType.lower() == 'pip3':
                        data_weights1N =  Grid.realcatN['BITWEIGHTS']
                        data_weights1S = Grid.realcatS['BITWEIGHTS']
                        break
                    elif wType.lower() == 'angup':
                        data_weights1N *= Grid.realcatN['WEIGHT']
                        data_weights1S *= Grid.realcatS['WEIGHT']
                        randoms_weights1N *= Grid.randcatN['WEIGHT']
                        randoms_weights1S *= Grid.randcatS['WEIGHT']
                    elif wType.lower() == 'iip':
                        assert(not 'pip' in Grid.weightList)
                        data_weights1N *= 1.0/Grid.realcatN['PROB_OBS']
                        data_weights1S *= 1.0/Grid.realcatS['PROB_OBS']

                    elif wType.lower() == 'fkp':
                        data_weights1N *= Grid.realcatN['WEIGHT_FKP']
                        data_weights1S *= Grid.realcatS['WEIGHT_FKP']
                        randoms_weights1N *= Grid.randcatN['WEIGHT_FKP']
                        randoms_weights1S *= Grid.randcatS['WEIGHT_FKP']   
                    elif wType.lower() == 'zfail':
                        data_weights1N *= Grid.realcatN['WEIGHT_ZFAIL']
                        data_weights1S *= Grid.realcatS['WEIGHT_ZFAIL']
                        randoms_weights1N *= Grid.randcatN['WEIGHT_ZFAIL']
                        randoms_weights1S *= Grid.randcatS['WEIGHT_ZFAIL']            
                
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
                                             randoms_positions1=Grid.random_positionsS,data_samples1=Grid.labelsS,
                                             randoms_samples1=Grid.labelsRandS, position_type='rdd',
                                             engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)
                '''
                Grid.dataTPCFDictN['noweight'] = Grid.resultN
                Grid.dataTPCFDictS['noweight'] = Grid.resultS
                '''
            else:
                Grid.resultN = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsN,
                                                           data_weights1=data_weights1N, randoms_positions1=Grid.random_positionsN,
                                                           randoms_weights= randoms_weights1N,data_samples1=Grid.labelsN,
                                                            randoms_samples1=Grid.labelsRandN, position_type='rdd',
                                                           engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)


                Grid.resultS = TwoPointCorrelationFunction(Grid.statType, Grid.edges, data_positions1=Grid.data_positionsS,
                                             data_weights1=data_weights1S, randoms_positions1=Grid.random_positionsS,
                                             randoms_weights= randoms_weights1S,data_samples1=Grid.labelsS,
                                             randoms_samples1=Grid.labelsRandS, position_type='rdd',
                                             engine='corrfunc', compute_sepsavg=False, nthreads=Grid.nthreads)
                '''
                Grid.dataTPCFDictN[weightList.join('+')] = Grid.resultN
                Grid.dataTPCFDictS[weightList.join('+')] = Grid.resultS
                '''
        if saveTemp and (not fromTemp):
            if not (Grid.weightType is None):
                Grid.resultN.save_txt(Grid.tempstorage + '/DESI_Fuji_NGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}_{6}weights_TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], '+'.join(Grid.weightList)), return_std = True)
                Grid.resultS.save_txt(Grid.tempstorage + '/DESI_Fuji_SGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}_{6}weights_TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1], '+'.join(Grid.weightList)), return_std = True)
            else:
                Grid.resultN.save_txt(Grid.tempstorage + '/DESI_Fuji_NGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1]), return_std = True)
                Grid.resultS.save_txt(Grid.tempstorage + '/DESI_Fuji_SGC_SMu_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}_zmin_{4:.02f}_zmax_{5:.02f}TPCF.txt'.format(rmin, rmax,nrbins, nmubins, Grid.zDataRange[0], Grid.zDataRange[1]), return_std = True)
        # Project to multipoles (monopole, quadruple, hexadecapole)
        Grid.ells = ells
        Grid.nells = len(ells)
        
        Grid.resultNS = Grid.resultN.normalize() + Grid.resultS.normalize()
        '''
        if Grid.weightType == None:
            Grid.dataTPCFDictNS['noweight'] = Grid.resultNS
        else:
            Grid.dataTPCFDictNS[weightList.join('+')] = Grid.resultNS
        '''
        Grid.s, Grid.xiell, Grid.cov = Grid.resultNS.get_corr(ells=Grid.ells, return_sep=True, return_cov=True)


def TPCFCalcMockGrid(Grid, fromTemp = False, saveTemp = True, useRandoms = True,
                     makeRandomsFromMock = True, randomFN = None, skip_header = 1,
                     position_type = 'rdd', skipUnloadedCats = True,
                     baseDataDir = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3.1/'):



    #posMock = [FullCatRA, FullCatDec, MockCDs]
    #randPosMock = [rng.uniform(0.0, 360, FullCatRA.shape), np.degrees(np.arcsin(rng.uniform(-1, 1, FullCatRA.shape))), MockCDs]
    #resultMock = TwoPointCorrelationFunction(args.statType, edges, data_positions1=posMock,
    #                            randoms_positions1=randPosMock, position_type='rdd',
    #                            engine='corrfunc', compute_sepsavg=False, nthreads=nthreads)
    #resultMock.save_txt(args.plotDirBase + '/UchuuMockTPCF' + suffix + '.txt', return_std = False)

    #sMock, xiellMock = resultMock.get_corr(ells=(0), return_sep=True, return_cov=False)

    if useRandoms:
        assert(makeRandomsFromMock ^ (type(randomFN) == str))
    
    #pos = return_xyz_formatted_array(FullCat['x'],FullCat['y'],FullCat['z'])
    for vm, sig, fs in zip(Grid.vMeans, Grid.sigmas, Grid.fsats):
        if fromTemp:
            res = np.load(Grid.tempstorage + '/UchuuMock_SMu_vm_{4:.01f}_sig_{5:.01f}_fsat_{6:.03f}_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}TPCF.txt'.format(Grid.rmin, Grid.rmax,Grid.nrbins, Grid.nmubins, vm, sig, fs), allow_pickle = True)
        else:
            keyVM, keySig, keyFS = int(Grid.keyscale*vm), int(Grid.keyscale*sig), int(Grid.keyscale*fs)
            try:
                FullCat = Grid.mockDict[keyVM][keySig][keyFS] 
            except:
                print('for params vm = {0:.02f}; sigma = {1:.02f}; fsat = {2:.02f}; cat not loaded. '.format(vm, sig, fs))
                if skipUnloadedCats:
                    Grid.TPCFDict[keyVM][keySig][keyFS] = (None, None)
                    continue
            if position_type.lower().startswith('rd'):
                FullCatRA = FullCat['ra']
                FullCatDec = FullCat['dec']
            if position_type.lower() == 'rdd':
                FullCatCD = Grid.cosmo.comoving_distance(FullCat['zobs']).astype(float)

                pos = [ FullCatRA, FullCatDec, FullCatCD ]
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
                        randPosMock = [rng.uniform(0.0, 360, FullCatRA.shape), np.degrees(np.arcsin(rng.uniform(-1, 1, FullCatRA.shape))), FullCatCD]
                    elif position_type.lower() == 'rd':
                        assert(0)
                        randPosMock = [rng.uniform(0.0, 360, FullCatRA.shape), np.degrees(np.arcsin(rng.uniform(-1, 1, FullCatRA.shape)))]
                    else:
                        assert(0)
                        randPosMock = [rng.uniform(0.0, Grid.boxsize, FullCatX.shape),
                                       rng.uniform(0.0, Grid.boxsize, FullCatY.shape), 
                                       rng.uniform(0.0, Grid.boxsize, FullCatZ.shape)]
                else:
                    assert(0)
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
                        assert(0)
                        randPosMock = [randRA, randDec]
                    elif position_type.lower() == 'xyz':
                        assert(0)
                        randX = randCat['x'][:]
                        randY = randCat['y'][:]
                        randZ = randCat['z'][:]
                        randPosMock = [randX, randY, randZ]
            if not 'fkp' in Grid.weightType.lower():
                if useRandoms:

                    res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,
                                             engine='corrfunc',  nthreads=Grid.nthreads,
                                             randoms_positions1 = randPosMock, position_type = position_type,
                                             compute_sepsavg=False)
                else:
                    assert(0)
                    res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,los = 'z',
                                             engine='corrfunc', boxsize=Grid.boxsize, nthreads=Grid.nthreads,
                                             position_type = position_type, compute_sepsavg=False)
            else:

                ELGnzN = np.genfromtxt(baseDataDir + 'ELG_N_nz.txt', names = True, dtype = None, skip_header = 1)
                ELGnzS = np.genfromtxt(baseDataDir + 'ELG_S_nz.txt', names = True, dtype = None, skip_header = 1)
                ELGNz = ELGnzN['Nbin'] + ELGnzS['Nbin']
                ELGVz = ELGnzN['Vol_bin'] + ELGnzS['Vol_bin']
                ELGZl = ELGnzN['zlow']
                ELGZm = ELGnzN['zmid']
                ELGZh = ELGnzN['zhigh']
                ELGnz = ELGNz/ELGVz
                nzspl = interpol.UnivariateSpline(ELGZm, ELGnz, s = 0)
                #Check later if zobs is right or if there should be a variable/argument version
                nz = nzspl(FullCat['zobs'])
                weightsMock = wFKP(FullCat['zobs'], nz, tracer = 'ELG')
                if useRandoms:

                    res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,
                                             engine='corrfunc',  nthreads=Grid.nthreads, data_weights1=weightsMock,
                                             randoms_weights1=weightsMock,randoms_positions1 = randPosMock, 
                                             position_type = position_type, compute_sepsavg=False)
                else:
                    assert(0)
                    res = TwoPointCorrelationFunction('smu', Grid.edges, data_positions1=pos,los = 'z',
                                             engine='corrfunc', boxsize=Grid.boxsize, nthreads=Grid.nthreads,
                                             position_type = position_type, compute_sepsavg=False)
            if saveTemp:
                res.save_txt(Grid.tempstorage + '/UchuuMock_SMu_vm_{4:.01f}_sig_{5:.01f}_fsat_{6:.03f}_rmin_{0:.01f}_rmax_{1:.01f}_nrbins_{2:d}_nmubins_{3:d}TPCF.txt'.format(Grid.rmin, Grid.rmax,Grid.nrbins, Grid.nmubins, vm, sig, fs), return_std = False)
        sTemp, xiellTemp = res.get_corr(ells=Grid.ells, return_sep=True, return_cov=False)
        Grid.TPCFDict[keyVM][keySig][keyFS] = (sTemp, xiellTemp)
