import astropy.io.fits as pf
from astropy.table import Table, join, vstack
import numpy as np
from matplotlib.colors import Normalize





def PlotTPCFMockGrid(self,colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', 
                  '#999999', '#e41a1c', '#dede00'], fmts = ['o', 's', 'p', '*', 'h', 'v', '+', 'x', 'D', 'd', '2', '3', '4'],
                    outfileBase = None, overplotData = True, ells2plot = None):
    #may want to change this so that there are separate plots in sigma bins rather than using line style/format
    try:
        s, xiell = Grid.TPCFDict[Grid.vMeansOnly[0]*Grid.keyscale][Grid.sigmasOnly[0]*Grid.keyscale][Grid.fsatsOnly[0]*Grid.keyscale]

    except Exception as e:
        print(e)
        print('TPCFs dont yet exist. Calculating them.')
        Grid.TPCFCalcMockGrid()
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
            fig, axarr = plt.subplots(len(np.unique(Grid.vMeansOnly)),1, figsize = (6,5*len(Grid.vMeansOnly)))
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
                s, xiell = Grid.TPCFDict[keyVM][keySig][keyFS]
                if not (ells2plot is None):
                    for ell in ells2plot:
                        if int(ell) == 0: ill = 0
                        elif int(ell) == 2: ill = 1
                        elif int(ell) == 4: ill = 2
                        else: raise ValueError('only valid ells are (0,2,4) and the provided value was {0}'.format(ell))
                        if nells2plot > 1:
                            axarr[indM][ill].plot(s, s**2 * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
                        else:
                            axarr[indM].plot(s, s**2 * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
                else:
                    for ill, ell in enumerate(Grid.ells):
                        axarr[indM][ill].plot(s, s**2 * xiell[ill], marker = fmts[indS],c = colors[indF], label = 'vm={0:.02f};fs={1:.02f}'.format(vm, fs))
    for fig, axarr, sig in zip(sigFigs, sigAxes, Grid.sigmasOnly):
        print('sigma = {0:.03f}'.format(sig))
        fig.suptitle('sigma = {0:.03f}'.format(sig))
        for x in range(len(np.unique(Grid.vMeansOnly))):
            if nells2plot > 1:
                for y in range(nells2plot):
                    axarr[x][y].set_xscale('log')
                    axarr[x][y].set_yscale('log')
                    axarr[x][y].set_xlabel(r'$log_{\rm 10}(s^2)$')
                    axarr[x][y].set_ylabel(r'$log_{\rm 10}(s^2*\xi)$')
                    axarr[x][y].axvline(x=Grid.seprange[0], c = 'Gray')
                    axarr[x][y].axvline(x=Grid.seprange[1], c = 'Gray')
                    if overplotData:
                        std = np.array_split(np.diag(Grid.cov)**0.5, Grid.nells)

                        #yerr = np.diagonal(np.sqrt(np.diag(covN + covS)))
                        #print('yerr')
                        #print(yerr)
                        #axarr[x][y].errorbar(Grid.sN, (Grid.sN**2 * Grid.xiellN[y] + Grid.sS**2 * Grid.xiellS[y])/2.0, c = 'k', yerr = Grid.sN*std[y], label='data')
                        axarr[x][y].errorbar(Grid.s, Grid.s**2 * Grid.xiell[y] , c = 'k', yerr = Grid.s*std[y], label='data')
                    if y == 0:
                        axarr[x][y].legend(fontsize = 9, loc = 'lower left')
            else:
                ell = ells2plot[0]
                if int(ell) == 0: ill = 0
                elif int(ell) == 2: ill = 1
                elif int(ell) == 4: ill = 2
                axarr[x].set_xscale('log')
                axarr[x].set_yscale('log')
                axarr[x].set_xlabel(r'$log_{\rm 10}(s)$')
                axarr[x].set_ylabel(r'$log_{\rm 10}(s^2*\xi)$')
                axarr[x].axvline(x=Grid.seprange[0], c = 'Gray')
                axarr[x].axvline(x=Grid.seprange[1], c = 'Gray')
                if overplotData:
                    std = np.array_split(np.diag(Grid.cov)**0.5, Grid.nells)

                    #yerr = np.diagonal(np.sqrt(np.diag(covN + covS)))
                    #print('yerr')
                    #print(yerr)
                    #axarr[x].errorbar(Grid.sN, (Grid.sN**2 * Grid.xiellN[ill] + Grid.sS**2 * Grid.xiellS[ill])/2.0, c = 'k', yerr = Grid.sN**2*std[ill], label='data')
                    axarr[x].errorbar(Grid.s, Grid.s**2 * Grid.xiell[ill] , c = 'k', yerr = Grid.s**2*std[ill], label='data')
                    axarr[x].legend(fontsize = 9, loc = 'lower left')

        fig.tight_layout()
    if not (outfileBase is None):
        fig.savefig(outfileBase + '_sigma{0:.03f}.png'.format(sig))
        plt.show()
    else:
        plt.show()

    plt.close(fig)
def plotParGrid(self, parvar = 'Vpeak', parbins = np.linspace(0.0, 2000.0, 2001), outFileBase = None):
    axarrGrid = []
    figGrid = []
    for vmtemp in Grid.vMeansOnly:
        figTot, axarrTot = plt.subplots(np.unique(Grid.sigmasOnly).shape[0], np.unique(Grid.fsatsOnly).shape[0], 
                                        figsize = (11*np.unique(Grid.sigmasOnly).shape[0], 10*np.unique(Grid.fsatsOnly).shape[0]))
        axarrGrid.append(axarrTot)
        figGrid.append(figTot)
        
    for indVM, vm in enumerate(Grid.vMeansOnly):
        keyVM = Grid.keyscale*vm
        axGrid = axarrGrid[indVM]
        fig = figGrid[indVM]
        fig.suptitle('vmean = {0:.01f}'.format(vm))
        for indSig, sig in enumerate(Grid.sigmasOnly):
            keySig = Grid.keyscale*sig
            for indFS, fs in enumerate(Grid.fsatsOnly):
                keyFS = Grid.keyscale*fs
                ax = axGrid[indSig][indFS]
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
                ax.set_xlabel(parvar)
                ax.set_ylabel('N Gal/halo')
                ax.legend()
                

    if outFileBase is None:
        figTot.show()
    else:
        figTot.savefig(outfileBase + '_{0}Distro.png'.format(parvar))
        
def plotStats(self, outfileBase = None, dFracDown = 0.9, dFracUp = 1.1):
    figTot, axarrTot = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotTot = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figSat, axarrSat = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotSat = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    figCen, axarrCen = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotCen = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    vminTot = Grid.ndata*dFracDown
    vmaxTot = Grid.ndata*dFracUp
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
                               ha="center", va="center", color="w", fontsize = 8)
                textSat = axSat.text(j, i, '{0:d}'.format(int(currArrSat[i][j])),
                               ha="center", va="center", color="w", fontsize = 8)
                textCen = axCen.text(j, i, '{0:d}'.format(int(currArrCen[i][j])),
                               ha="center", va="center", color="w", fontsize = 8)
                
                #textTot = axTot.text(j, i, '{0:.02f}'.format(arr2PlotTot[i, j]),
                #               ha="center", va="center", color="w")
                #textSat = axSat.text(j, i, '{0:.02f}'.format(arr2PlotSat[i, j]),
                #               ha="center", va="center", color="w")
                #textCen = axCen.text(j, i, '{0:.02f}'.format(arr2PlotCen[i, j]),
                #               ha="center", va="center", color="w")

        axTot.set_title("NGal in mock (vmean = {0:.03f})".format(vm))
        axTot.set_xlabel('sat. frac.')
        axTot.set_ylabel('sigma')

        axSat.set_title("NSat in mock (vmean = {0:.03f})".format(vm))
        axSat.set_xlabel('sat. frac.')
        axSat.set_ylabel('sigma')

        axCen.set_title("NCent in mock (vmean = {0:.03f})".format(vm))
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
def plotChi2Heatmap(self, seprange = [10.0, 40.0], outfileBase = None, fullcov = True, monopoleOnly = True):
    figTot, axarrTot = plt.subplots(np.unique(Grid.vMeansOnly).shape[0], 1, figsize = (11, 10*np.unique(Grid.vMeansOnly).shape[0]))
    arr2PlotTot = [ np.empty((Grid.sigmasOnly.shape[0], Grid.fsatsOnly.shape[0])) for vmtemp in Grid.vMeansOnly]
    if monopoleOnly:
        sN, xiellN, covN = Grid.resultN.get_corr(ells=(0), return_sep=True, return_cov=True)
        sS, xiellS, covS = Grid.resultS.get_corr(ells=(0), return_sep=True, return_cov=True)

        covDataChi2 = covN + covS
        xiellDataChi2 = xiellN + xiellS
    
    for indVM, vm in enumerate(Grid.vMeansOnly):
        keyVM = Grid.keyscale*vm
        axTot = axarrTot[indVM]
        currArrTot = arr2PlotTot[indVM]
        for indSig, sig in enumerate(Grid.sigmasOnly):
            keySig = Grid.keyscale*sig
            for indFS, fs in enumerate(Grid.fsatsOnly):
                keyFS = Grid.keyscale*fs
                sMock, xiellMock = Grid.TPCFDict[keyVM][keySig][keyFS]
                if monopoleOnly:
                    dxi = xiellDataChi2 - xiellMock[0]
                    sepCond = (sMock > seprange[0]) & (sMock < seprange[1])
                    print('preInv dxi, cov.shape, cov[0].shape')
                    print(dxi.shape)
                    print(Grid.cov.shape)
                    print(Grid.cov[0].shape)
                    covInv = numpy.linalg.inv(covDataChi2)
                    print('postInv')
                    print(covInv.shape)
                    print(covInv[0].shape)
                else:
                    dxi = Grid.xiell - xiellMock
                    sepCond = (sMock > seprange[0]) & (sMock < seprange[1])
                    covInv = numpy.linalg.inv(Grid.cov)
                Grid.ndof = np.sum(sepCond) - 3
                if fullcov:
                    
                    chi2s = dxi.T*(covInv)*dxi
                    print('chi2s.shape full cov')
                    print(chi2s.shape)
                    chi2 = np.sum(chi2s[sepCond]/Grid.ndof)
                else:
                    std = np.sqrt(np.diag(covDataChi2))
                    chi2s = dxi**2/std
                    print('chi2s.shape diag')
                    print(chi2s.shape)
                    chi2 = np.sum(chi2s.T[sepCond])
                currArrTot[indSig][indFS] = chi2/Grid.ndof

        imTot = axTot.imshow(currArrTot)
        plt.colorbar(imTot,ax=axTot)

        
        # Show all ticks and label them with the respective list entries
        axTot.set_xticks(np.arange(len(Grid.fsatsOnly)), labels=Grid.fsatsOnly.round(decimals=3))
        axTot.set_yticks(np.arange(len(Grid.sigmasOnly)), labels=Grid.sigmasOnly.round(decimals=3))


        # Rotate the tick labels and set their alignment.
        plt.setp(axTot.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        

        # Loop over data dimensions and create text annotations.
        for i in range(len(Grid.sigmasOnly)):
            for j in range(len(Grid.fsatsOnly)):
                try:
                    textTot = axTot.text(j, i, '{0:d}'.format(int(currArrTot[i][j])),
                               ha="center", va="center", color="w", fontsize = 8)
                except:
                    print('i,j')
                    print("{0:d} {1:d}".format(i,j))
                    print('chi2')
                    print(currArrTot[i][j])
                    textTot = axTot.text(j, i, '{0}'.format('NaN'),
                               ha="center", va="center", color="w", fontsize = 8)
                    
               
                
                

        axTot.set_title("Chi2 Grid (vmean = {0:.03f})".format(vm))
        axTot.set_xlabel('sat. frac.')
        axTot.set_ylabel('sigma')
    if outfileBase is None:
        figTot.show()
    else:
        figTot.savefig(outfileBase + '_Chi2Heatmaps.png')

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