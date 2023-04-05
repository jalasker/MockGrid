import astropy.io.fits as pf
import numpy as np
import os
import astropy.cosmology as cosmo
Planck15 = cosmo.Planck15
class MockGrid:
    def __init__(self, vmean = None, mmin = None, mstep = None, mmax = None, 
                 sigma = None, sigmin = None, sigstep = None, sigmax = None,
                 fsat = None, fsmin = None, fsstep = None, fsmax = None, vcut = None,
                 keyscale = 10000, ranseed = None, nthreads = None, boxsize = 2000.0,
                cosmology = Planck15, zkey = 'Z', zmock = 1.03, tempstorage = 'scratch', 
                gridName = None):
        self.cosmo = cosmology

        if tempstorage == 'scratch':
            if 'PSCRATCH' in os.environ:
                self.tempstorage = os.environ['PSCRATCH'] + '/MockGridTempStorage/'
            elif 'CSCRATCH' in os.environ:
                self.tempstorage = os.environ['CSCRATCH'] + '/MockGridTempStorage/'
            else:
                self.tempstorage = os.environ['SCRATCH'] + '/MockGridTempStorage/'


        if not (gridName is None):
            self.tempstorage = self.tempstorage + '/' + gridName + '/'
            self.gridName = gridName
        else:
            self.gridName = ''
            
        if not (os.path.isdir(self.tempstorage)):

            os.makedirs(self.tempstorage)


        # ensure that you are either passing a single value 
        # for gaussian parameters or initializing a grid
        # but not both
        assert((vmean is None) or ((mmin is None) and (mstep is None) and (mmax is None)))
        assert((sigma is None) or ((sigmin is None) and (sigstep is None) and (sigmax is None)))
        assert((fsat is None) or ((fsmin is None) and (fsstep is None) and (fsmax is None)))
        
        self.vGrid = not (mmin is None)
        self.sigGrid = not (sigmin is None)
        self.fsGrid = not (fsmin is None)
        self.ranseed = ranseed
        self.boxsize = boxsize
        self.zkey = zkey
        self.zmock = zmock
        self.vcut = vcut
        
        if nthreads is None:
            nthreads = 1
        self.nthreads = nthreads
        os.environ['NUMEXPR_MAX_THREADS'] = str(self.nthreads)
        if not (self.vcut is None):

            if not (self.ranseed is None):
                self.suffix = '_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}_Vcut{3:.01f}_ranseed_{4:d}'
            else:
                self.suffix = '_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}_Vcut{3:.01f}'
        else:
            if not (self.ranseed is None):
                self.suffix = '_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}_ranseed_{3:d}'
            else:
                self.suffix = '_vm_{0:.02f}_sig_{1:.02f}_fs_{2:.03f}'

        
        self.nGrids = int(self.vGrid) + int(self.sigGrid) + int(self.fsGrid)
        
        #self.nestOrder = []
        #self.nestOrderNames = []
        
        if self.vGrid:
            self.vMeansOnly = np.arange(mmin, mmax+mstep, mstep)
            #self.nestOrder.append(self.vMeansTemp)
            #self.nestOrderNames.append('vmean')
        else:
            self.vMeansOnly = np.array([vmean])
        if self.sigGrid:
            self.sigmasOnly = np.arange(sigmin, sigmax + sigstep, sigstep)
            #self.nestOrder.append(self.sigmasTemp)
            #self.nestOrderNames.append('sigma')
        else:
            self.sigmasOnly = np.array([sigma])
        if self.fsGrid:
            self.fsatsOnly = np.arange(fsmin, fsmax + fsstep, fsstep)
            #self.nestOrder.append(self.fsatsTemp)
            #self.nestOrderNames.append('fsat')
        else:
            self.fsatsOnly = np.array([fsat])
        
        self.vMeans, self.sigmas, self.fsats = np.meshgrid(self.vMeansOnly, self.sigmasOnly, self.fsatsOnly)
        
        self.vMeans = self.vMeans.flatten()
        self.sigmas = self.sigmas.flatten()
        self.fsats = self.fsats.flatten()
        
        self.mockDict = {}
        self.mockStats = {}
        self.TPCFDict = {}
        self.chi2Dict = {}

        self.dataTPCFDictN = {}
        self.dataTPCFDictS = {}
        self.dataTPCFDictNS = {}

        
        self.keyscale = keyscale
        #need to loop through params, creating subdicts if there is another layer of nesting below
        for vm in self.vMeansOnly:
            keyVM = int(self.keyscale*vm)
            self.mockDict[keyVM] = {}
            self.mockStats[keyVM] = {}
            self.TPCFDict[keyVM] = {}
            self.chi2Dict[keyVM] = {}
            for sigma in self.sigmasOnly:
                keySig = int(self.keyscale*sigma)
                self.mockDict[keyVM][keySig] = {}
                self.mockStats[keyVM][keySig] = {}
                self.TPCFDict[keyVM][keySig] = {}
                self.chi2Dict[keyVM][keySig] = {}
                for fsat in self.fsatsOnly:
                    keyFS = int(self.keyscale*fsat)
                    self.mockStats[keyVM][keySig][keyFS] = {}
                    self.mockDict[keyVM][keySig][keyFS] = {}
                    #do we add keys for different weight types here or upon generation? 
                    #if upon generation, will need to re-alphabetically order the weight strings after 
                    #splitting on +
                    self.TPCFDict[keyVM][keySig][keyFS] = {}
                    self.chi2Dict[keyVM][keySig][keyFS] = {}
