import astropy.io.fits as pf
from astropy.table import Table, join, vstack
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt 
import MockGrid.Clustering as clst



def PlotTPCFMockGrid(Grid,colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', 
                  '#999999', '#e41a1c', '#dede00'], fmts = ['o', 's', 'p', '*', 'h', 'v', '+', 'x', 'D', 'd', '2', '3', '4'],
                    outfileBase = None, overplotData = True, ells2plot = None, seprange = [5, 30]):
    #may want to change this so that there are separate plots in sigma bins rather than using line style/format
    try:
        s, xiell = Grid.TPCFDict[Grid.vMeansOnly[0]*Grid.keyscale][Grid.sigmasOnly[0]*Grid.keyscale][Grid.fsatsOnly[0]*Grid.keyscale]

    except Exception as e:
        print('exception')
        print(e)
        #print('TPCFs dont yet exist. Calculating them.')
        #clst.TPCFCalcMockGrid(Grid)
    #Make separate fig/axarr for each sigma value... too crowded with all together
    sigFigs = []
    sigAxes = []
    for sig in Grid.sigmasOnly:
        if not (ells2plot is None):
            nells2plot = len(ells2plot)
        else:
            nells2plot = Grid.nells
        if nells2plot > 1:
            fig, axarr = plt.subplots(len(np.unique(Grid.vMeansOnly)),nells2plot, figsize = (6*nells2plot,5*len(Grid.vMeansOnly)))
            sigFigs.append(fig)
            sigAxes.append(axarr)
        else:
            fig, axarr = plt.subplots(len(np.unique(Grid.vMeansOnly)), 1, figsize = (6,5*len(Grid.vMeansOnly)))
            sigFigs.append(fig)
            sigAxes.append(axarr)
    for indM, vm in enumerate(Grid.vMeansOnly):
        keyVM= int(Grid.keyscale*vm)
        for indS, sig in enumerate(Grid.sigmasOnly):
            keySig = int(Grid.keyscale*sig)
            fig = sigFigs[indS]
            axarr = sigAxes[indS]
            for indF, fs in enumerate(Grid.fsatsOnly):
                keyFS = int(Grid.keyscale*fs)
                if not (ells2plot is None):
                    for ell in ells2plot:
                        if int(ell) == 0: ill = 0
                        elif int(ell) == 2: ill = 1
                        elif int(ell) == 4: ill = 2
                        else: raise ValueError('only valid ells are (0,2,4) and the provided value was {0}'.format(ell))
                        if nells2plot > 1:
                            ax2plot = axarr[indM][ill]
                            plotOneTPCF(Grid, vm = vm, sig = sig, fs = fs, axes = ax2plot, overplotData = (indF == 0), ells2plot = ells2plot, seprange = seprange, fmt = fmts[indS], c = colors[indF], ill = ill, ell = ell)
                            #axarr[indM][ill].plot(s, s * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
                        else:
                            ax2plot = axarr[indM]
                            plotOneTPCF(Grid, vm = vm, sig = sig, fs = fs, axes = ax2plot, overplotData = (indF == 0), ells2plot = ells2plot, seprange = seprange, fmt = fmts[indS], c = colors[indF], ill = ill, ell = ell)
                            #axarr[indM].plot(s, s * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
                else:
                    for ill, ell in enumerate(Grid.ells):
                        ax2plot = axarr[indM][ill]
                        plotOneTPCF(Grid, vm = vm, sig = sig, fs = fs, axes = ax2plot, overplotData = (indF == 0), ells2plot = ells2plot, seprange = seprange, fmt = fmts[indS], c = colors[indF], ill = ill, ell = ell)
                        #axarr[indM][ill].plot(s, s * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
    for fig, axarr, sig in zip(sigFigs, sigAxes, Grid.sigmasOnly):
        fig.suptitle('sigma = {0:.03f}'.format(sig), fontsize = 20)
        
        fig.tight_layout()
        if not (outfileBase is None):
            if not (Grid.weightType is None): 
                fig.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_{4}Weights_{3}Multipoles.png'.format(sig, seprange[0], seprange[1], nells2plot, '+'.join(Grid.weightList)))
            else:
                fig.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_NoWeights_{3}Multipoles.png'.format(sig, seprange[0], seprange[1], nells2plot))
            #plt.show()
        else:
            plt.show()

        plt.close(fig)
def plotOneTPCF(Grid, vm = 100.0, sig = 30.0, fs = 0.18, axes = None, overplotData = True, ells2plot = None, seprange = [5, 30], fmt = '-', c = 'b', ill = 0, ell = 0 ):

    if axes is None:
        axes = plt.subplots(1,1)
        #plt.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw)

    try:
        keyVM= int(Grid.keyscale*vm)
        keySig = int(Grid.keyscale*sig)
        keyFS = int(Grid.keyscale*fs)
        s, xiell = Grid.TPCFDict[keyVM][keySig][keyFS]
    except:
        print('for params vm = {0:.02f}; sigma = {1:.02f}; fsat = {2:.02f}; TPCF doesnt exist. '.format(vm, sig, fs))
        return 0

    if (s is None) or (xiell is None):
        print('for params vm = {0:.02f}; sigma = {1:.02f}; fsat = {2:.02f}; TPCF is NONE. '.format(vm, sig, fs))
        return 1

    if not (ells2plot is None):
        nells2plot = len(ells2plot)
        for ell in ells2plot:
            if int(ell) == 0: ill = 0
            elif int(ell) == 2: ill = 1
            elif int(ell) == 4: ill = 2
            else: raise ValueError('only valid ells are (0,2,4) and the provided value was {0}'.format(ell))
            if nells2plot > 1:
                axes.plot(s, s * xiell[ill], marker = fmt ,c = c, label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
            else:
                axes.plot(s, s * xiell[ill], marker = fmt ,c = c, label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
    else:
        for ill, ell in enumerate(Grid.ells):
            axes.plot(s, s * xiell[ill], marker = fmt ,c = c, label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))

    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel(r'$log_{\rm 10}(s)$')
    axes.set_ylabel(r'$log_{\rm 10}(s*\xi)$')
    axes.axvline(x=seprange[0], c = 'Gray')
    axes.axvline(x=seprange[1], c = 'Gray')
    if overplotData:
        std = np.array_split(np.diag(Grid.cov)**0.5, Grid.nells)


        #axarr[x][y].errorbar(Grid.s, Grid.s * Grid.xiell[y] , c = 'k', yerr = Grid.s*std[y], label='data')
        axes.errorbar(Grid.s, Grid.s * Grid.xiell[ill] , c = 'k', yerr = Grid.s*std[ill], label='data')
    if ill == 0:
        axes.legend(fontsize = 9, loc = 'lower left')

'''
def plotParGrid(Grid, parvar = 'Vpeak', parbins = np.linspace(0.0, 2000.0, 2001), outFileBase = None):
    axarrGrid = []
    figGrid = []
    for vmtemp in Grid.vMeansOnly:
        figTot, axarrTot = plt.subplots(np.unique(Grid.sigmasOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], 
                                        figsize = (11*np.unique(Grid.sigmasOnly).shape[0], 10*np.unique(Grid.fsatsOnly).shape[0]))
        axarrGrid.append(axarrTot)
        figGrid.append(figTot)
    
    print(len(axarrGrid))
    print(axarrTot.shape)
    for indVM, vm in enumerate(Grid.vMeansOnly):
        keyVM = Grid.keyscale*vm
        axGrid = axarrGrid[indVM]
        print(type(axGrid))
        print(axGrid.shape)
        print(axGrid.dtype)
        fig = figGrid[indVM]
        fig.suptitle('vmean = {0:.01f}'.format(vm))
        for indSig, sig in enumerate(Grid.sigmasOnly):
            keySig = Grid.keyscale*sig
            for indFS, fs in enumerate(Grid.fsatsOnly):
                keyFS = Grid.keyscale*fs
                print(type(axGrid[indSig]))
                if (np.unique(Grid.sigmasOnly).shape[0] == 1):
                    ax = axGrid[indFS]
                elif (np.unique(Grid.fsatsOnly).shape[0] == 1):
                    ax = axGrid[indSig]

                FullMock = Grid.mockDict[keyVM][keySig][keyFS]
                print(FullMock.keys())
                CentMock = FullMock[FullMock['pid'] == -1]
                SatMock = FullMock[FullMock['pid'] != -1]
                print('satmockshape')
                print(SatMock.shape)
                histParFull, _ = np.histogram(FullMock[parvar], bins = parbins)
                histParCent, _ = np.histogram(CentMock[parvar], bins = parbins)
                histParSat, _ = np.histogram(SatMock[parvar], bins = parbins)
                
                ax.step(parbins[:-1], histParFull, c = 'k', label = 'All Halos', where = 'pre')
                ax.step(parbins[:-1], histParCent, c = 'b', label = 'Central Halos', where = 'pre')
                ax.step(parbins[:-1], histParSat, c = 'Orange', label = 'Satellite Halos', where = 'pre')
                
                if parvar == 'Vpeak':
                    ax.axvline(x = vm, c = 'k', label = 'vmean = {0:.01f}'.format(vm))
                    ax.axvline(x = vm + sig, c = 'Grey', label = 'sigma = {0:.01f}'.format(sig))
                    ax.axvline(x = vm - sig, c = 'Grey')
                ax.set_xlabel(parvar)
                ax.set_ylabel('N Gal/halo')
                ax.legend()
                

    if outFileBase is None:
        figTot.show()
    else:
        figTot.savefig(outfileBase + '_{0}Distro.png'.format(parvar))
'''
def plotParGrid(Grid, parvar = 'Vpeak', parbins = np.linspace(0.0, 2000.0, 2001), outfileBase = None):
    axarrGrid = []
    figGrid = []
    for vmtemp in Grid.vMeansOnly:
        figTot, axarrTot = plt.subplots(np.unique(Grid.sigmasOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], 
                                        figsize = (70*np.unique(Grid.sigmasOnly).shape[0], 5*np.unique(Grid.fsatsOnly).shape[0]))
        axarrGrid.append(axarrTot)
        figGrid.append(figTot)
        
    for indVM, vm in enumerate(Grid.vMeansOnly):
        keyVM = Grid.keyscale*vm
        axGrid = axarrGrid[indVM]
        fig = figGrid[indVM]
        fig.suptitle('vmean = {0:.01f}'.format(vm), fontsize = 30)
        for indSig, sig in enumerate(Grid.sigmasOnly):
            keySig = Grid.keyscale*sig
            for indFS, fs in enumerate(Grid.fsatsOnly):
                keyFS = Grid.keyscale*fs
                if np.unique(Grid.sigmasOnly).shape[0] > 1:
                    ax = axGrid[indSig][indFS]
                else:
                    ax = axGrid[indFS]
                FullMock = Grid.mockDict[keyVM][keySig][keyFS]
                CentMock = FullMock[FullMock['pid'] == -1]
                SatMock = FullMock[FullMock['pid'] != -1]
                print('satmockshape')
                print(SatMock.shape)
                histParFull, _ = np.histogram(FullMock[parvar], bins = parbins)
                histParCent, _ = np.histogram(CentMock[parvar], bins = parbins)
                histParSat, _ = np.histogram(SatMock[parvar], bins = parbins)
                
                ax.step(parbins[:-1], histParFull, c = 'k', label = 'All Halos', where = 'pre')
                ax.step(parbins[:-1], histParCent, c = 'b', label = 'Central Halos', where = 'pre')
                ax.step(parbins[:-1], histParSat, c = 'Orange', label = 'Satellite Halos', where = 'pre')
                
                if parvar == 'Vpeak':
                    ax.axvline(x = vm, c = 'k', label = 'vmean = {0:.01f}'.format(vm))
                    ax.axvline(x = vm + sig, c = 'Grey', label = 'sigma = {0:.01f}'.format(sig))
                    ax.axvline(x = vm - sig, c = 'Grey')
                ax.set_xlabel(parvar, fontsize = 24)
                ax.set_ylabel('N Gal/halo', fontsize = 24)
                ax.legend( fontsize = 16)
                

    if outfileBase is None:
        figTot.show()
    else:
        figTot.savefig(outfileBase + '_{0}Distro.png'.format(parvar))

def plotStats(Grid, outfileBase = None, dFracDown = 0.9, dFracUp = 1.1, nToNorm =14790000, datasized = False):

    figTot, axarrTot = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotTot = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figSat, axarrSat = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotSat = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figCen, axarrCen = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotCen = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    '''
    figTot, axarrTot = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], figsize = (11*np.unique(Grid.fsatsOnly).shape[0], 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotTot = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figSat, axarrSat = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], figsize = (11*np.unique(Grid.fsatsOnly).shape[0], 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotSat = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figCen, axarrCen = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], figsize = (11*np.unique(Grid.fsatsOnly).shape[0], 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotCen = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    '''
    if datasized:
        vminTot = Grid.ndata*dFracDown
        vmaxTot = Grid.ndata*dFracUp
    else:
        vminTot = nToNorm*dFracDown
        vmaxTot = nToNorm*dFracUp
    for indVM, vm in enumerate(Grid.vMeansOnly):
        keyVM = Grid.keyscale*vm
        axTot = axarrTot[indVM]
        axSat = axarrSat[indVM]
        axCen = axarrCen[indVM]
        currArrTot = arr2PlotTot[indVM]
        currArrSat = arr2PlotSat[indVM]
        currArrCen = arr2PlotCen[indVM]
        for indSig, sig in enumerate(Grid.sigmasOnly):
            keySig = Grid.keyscale*sig
            for indFS, fs in enumerate(Grid.fsatsOnly):
                keyFS = Grid.keyscale*fs
                currArrTot[indSig][indFS] = int(Grid.mockStats[keyVM][keySig][keyFS]['ntot'])
                currArrSat[indSig][indFS] = int(Grid.mockStats[keyVM][keySig][keyFS]['nsat'])
                currArrCen[indSig][indFS] = int(Grid.mockStats[keyVM][keySig][keyFS]['ncen'])

        imTot = axTot.imshow(currArrTot, vmin = vminTot, vmax = vmaxTot)
        imSat = axSat.imshow(currArrSat)
        imCen = axCen.imshow(currArrCen)
        

        # Show all ticks and label them with the respective list entries
        axTot.set_xticks(np.arange(len(Grid.fsatsOnly)), labels=Grid.fsatsOnly.round(decimals=3))
        axTot.set_yticks(np.arange(len(Grid.sigmasOnly)), labels=Grid.sigmasOnly.round(decimals=3))

        axSat.set_xticks(np.arange(len(Grid.fsatsOnly)), labels=Grid.fsatsOnly.round(decimals=3))
        axSat.set_yticks(np.arange(len(Grid.sigmasOnly)), labels=Grid.sigmasOnly.round(decimals=3))

        axCen.set_xticks(np.arange(len(Grid.fsatsOnly)), labels=Grid.fsatsOnly.round(decimals=3))
        axCen.set_yticks(np.arange(len(Grid.sigmasOnly)), labels=Grid.sigmasOnly.round(decimals=3))

        # Rotate the tick labels and set their alignment.
        plt.setp(axTot.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(axSat.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(axCen.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(Grid.sigmasOnly)):
            for j in range(len(Grid.fsatsOnly)):
                textTot = axTot.text(j, i, '{0:d}'.format(int(currArrTot[i][j])),
                               ha="center", va="center", color="w", fontsize = 14)
                textSat = axSat.text(j, i, '{0:d}'.format(int(currArrSat[i][j])),
                               ha="center", va="center", color="w", fontsize = 14)
                textCen = axCen.text(j, i, '{0:d}'.format(int(currArrCen[i][j])),
                               ha="center", va="center", color="w", fontsize = 14)
                
                #textTot = axTot.text(j, i, '{0:.02f}'.format(arr2PlotTot[i, j]),
                #               ha="center", va="center", color="w")
                #textSat = axSat.text(j, i, '{0:.02f}'.format(arr2PlotSat[i, j]),
                #               ha="center", va="center", color="w")
                #textCen = axCen.text(j, i, '{0:.02f}'.format(arr2PlotCen[i, j]),
                #               ha="center", va="center", color="w")

        axTot.set_title("NGal in mock (vmean = {0:.03f})".format(vm), fontsize = 16)
        axTot.set_xlabel('sat. frac.')
        axTot.set_ylabel('sigma')

        axSat.set_title("NSat in mock (vmean = {0:.03f})".format(vm), fontsize = 16)
        axSat.set_xlabel('sat. frac.')
        axSat.set_ylabel('sigma')

        axCen.set_title("NCent in mock (vmean = {0:.03f})".format(vm), fontsize = 16)
        axCen.set_xlabel('sat. frac.')
        axCen.set_ylabel('sigma')
        plt.colorbar(imTot,ax=axTot)
        plt.colorbar(imCen,ax=axCen)
        plt.colorbar(imSat,ax=axSat)
    if outfileBase is None:
        figTot.show()
        figCen.show()
        figSat.show()
    else:
        figTot.savefig(outfileBase + '_NTot.png')
        figSat.savefig(outfileBase + '_NSat.png')
        figCen.savefig(outfileBase + '_NCen.png')
def plotChi2Heatmap(Grid, seprange = [10.0, 40.0], outfileBase = None, fullcov = True, monopoleOnly = True):
    figTot, axarrTot = plt.subplots(np.unique(Grid.sigmasOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotTot = [ np.empty((Grid.vMeansOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.sigmasOnly]
    print(arr2PlotTot[0].shape)
    if monopoleOnly:
        #sN, xiellN, covN = Grid.resultN.get_corr(ells=(0), return_sep=True, return_cov=True)
        #sS, xiellS, covS = Grid.resultS.get_corr(ells=(0), return_sep=True, return_cov=True)

        #covDataChi2 = covN + covS
        #xiellDataChi2 = xiellN + xiellS
        resultNS = Grid.resultN.normalize() + Grid.resultS.normalize()

        sDataChi2, xiellDataChi2, covDataChi2 = Grid.resultNS.get_corr(ells=(0), return_sep=True, return_cov=True)
    else:
        resultNS = Grid.resultN.normalize() + Grid.resultS.normalize()

        sDataChi2, xiellDataChi2, covDataChi2 = Grid.resultNS.get_corr(ells=Grid.ells, return_sep=True, return_cov=True)
    
    for indSig, sig in enumerate(Grid.sigmasOnly):
        keySig = Grid.keyscale*sig
        if np.unique(Grid.sigmasOnly).shape[0] > 1:
            axTot = axarrTot[indSig]
        else:
            axTot = axarrTot
        currArrTot = arr2PlotTot[indSig]
        for indVM, vm in enumerate(Grid.vMeansOnly):
            keyVM = Grid.keyscale*vm
            for indFS, fs in enumerate(Grid.fsatsOnly):
                #printCond = (np.abs(vm - 200.0) < 0.01 ) & (np.abs(fs - 0.09) < 0.001 ) & (np.abs(sig - 30.0) < 0.01 )
                printCond = True
                keyFS = Grid.keyscale*fs
                sMock, xiellMock = Grid.TPCFDict[keyVM][keySig][keyFS]
                if monopoleOnly:
                    dxi = xiellDataChi2 - xiellMock[0]
                    sepCond = (sMock > seprange[0]) & (sMock < seprange[1])
                    if printCond:
                        print('vm = {0}; fs = {1}; sig = {2}'.format(vm, fs, sig))
                        print('preInv dxi, cov.shape, cov[0].shape')
                        print(dxi.shape)
                        print(Grid.cov.shape)
                        print(Grid.cov[0].shape)
                        print('preInv cov, cov[0]')
                        print(Grid.cov)
                        print(Grid.cov[0])
                        print('xiellDataChi2')
                        print(xiellDataChi2)
                        print('xiellDataChi2[sepCond]')
                        print(xiellDataChi2[sepCond])
                        print('----')
                        print('xiellMock[0]')
                        print(xiellMock[0])
                        print('xiellMock[0][sepCond]')
                        print(xiellMock[0][sepCond])

                    covInv = np.linalg.inv(covDataChi2)
                    if printCond:
                        print('postInv covInv.shape, covInv[0].shape')
                        print(covInv.shape)
                        print(covInv[0].shape)
                else:
                    dxi = Grid.xiell - xiellMock
                    sepCond = (sMock > seprange[0]) & (sMock < seprange[1])
                    covInv = numpy.linalg.inv(Grid.cov)
                Grid.ndof = np.sum(sepCond) - 3
                if fullcov:
                    assert(0)
                    chi2s = dxi.T*(covInv)*dxi
                    print('chi2s.shape full cov')
                    print(chi2s.shape)
                    chi2 = np.sum(chi2s[sepCond]/Grid.ndof)
                else:
                    std = np.sqrt(np.diag(covDataChi2))
                    if printCond:
                        print('vm = {0}; fs = {1}; sig = {2}'.format(vm, fs, sig))
                        print('dxi')
                        print(dxi)
                        print('dxi**2')
                        print(dxi**2)
                        print('std')
                        print(std)

                        print('dxi[sepCond]')
                        print(dxi[sepCond])
                        print('dxi[sepCond]**2')
                        print(dxi[sepCond]**2)
                        print('std[sepCond]')
                        print(std[sepCond])
                    chi2s = dxi**2/std
                    if printCond:
                        print('vm = {0}; fs = {1}; sig = {2}'.format(vm, fs, sig))
                        print('chi2s.shape diag')
                        print(chi2s.shape)
                        print('chi2s values diag')
                        print(chi2s)
                        print('chi2s[sepCond] values diag')
                        print(chi2s[sepCond])
                    chi2 = np.sum(chi2s.T[sepCond])
                if printCond:
                    print('ndof')
                    print(Grid.ndof)
                currArrTot[indVM][indFS] = chi2/Grid.ndof
        print('currArrTot.shape')
        print(currArrTot.shape)
        print('indSig')
        imTot = axTot.imshow(currArrTot)
        plt.colorbar(imTot,ax=axTot)

        
        # Show all ticks and label them with the respective list entries
        axTot.set_xticks(np.arange(len(Grid.fsatsOnly)), labels=Grid.fsatsOnly.round(decimals=3))
        axTot.set_yticks(np.arange(len(Grid.vMeansOnly)), labels=Grid.vMeansOnly.round(decimals=3))


        # Rotate the tick labels and set their alignment.
        plt.setp(axTot.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        

        # Loop over data dimensions and create text annotations.
        for i in range(len(Grid.vMeansOnly)):
            for j in range(len(Grid.fsatsOnly)):
                try:
                    textTot = axTot.text(j, i, '{0:.03f}'.format(currArrTot[i][j]),
                               ha="center", va="center", color="w", fontsize = 8)
                except:
                    print('i,j')
                    print("{0:d} {1:d}".format(i,j))
                    print('chi2')
                    print(currArrTot[i][j])
                    textTot = axTot.text(j, i, '{0}'.format('NaN'),
                               ha="center", va="center", color="w", fontsize = 8)
                    
               
                
                

        axTot.set_title("Chi2 Grid (sigma = {0:.03f}\n seprange = {1:.02f} - {2:.02f}\n weights = {3})".format(sig, seprange[0], seprange[1], '+'.join(Grid.weightList)), fontsize = 20)
        axTot.set_xlabel('sat. frac.')
        axTot.set_ylabel('vmean')
    if outfileBase is None:
        figTot.show()
    else:
        if monopoleOnly:
            if not (Grid.weightType is None): 
                figTot.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_{3}Weights_MonopoleOnlyChi2Heatmaps.png'.format(sig, seprange[0], seprange[1], '+'.join(Grid.weightList)))
            else:
                figTot.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_NoWeights_MonopoleOnlyChi2Heatmaps.png'.format(sig, seprange[0], seprange[1]))
        else:
            if not (Grid.weightType is None): 
                figTot.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_{3}Weights_MultipleMultipolesChi2Heatmaps.png'.format(sig, seprange[0], seprange[1], '+'.join(Grid.weightList)))
            else:
                figTot.savefig(outfileBase + '_sigma{0:.03f}_sepRange_{1:.02f}_{2:.02f}_NoWeights_MultipleMultipolesChi2Heatmaps.png'.format(sig, seprange[0], seprange[1],))

def plotOneHOD(Grid, mvar, vm = None, sig = None, fs = None, 
                hostIDKey = 'id', parentIDKey  = 'pid', allHaloFileName = None):
    if allHaloFileName is None:
        allHaloFileName = '/scratch/group/astro/desi/mocks/Uchuu/ELG/hdf/'
    if (vm is None) or (sig is None) or (fs is None):
        raise ValueError('This function requires specifying a vmean, sigma, fsat combo for which to plot an HOD')
    keyVM = Grid.keyscale*vm
    keySig = Grid.keyscale*sig
    keyFS = Grid.keyscale*fs
    
    ThisMock = Grid.mockDict[keyVM][keySig][keyFS]
    
    TheseSats = ThisMock[ThisMock['pid'] != ThisMock['id']]
    TheseCents = ThisMock[ThisMock['pid'] == -1]
    del ThisMock
    
    CentMass = TheseCents['Mvir']
    
    try:
        allHalos = Grid.allHalos
    except:
        Grid.allHalos = h5py.File( allHaloFileName, 'r')