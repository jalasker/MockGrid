import astropy.io.fits as pf
from astropy.table import Table, join, vstack
import numpy as np
import os
import pickle


def saveObj(Grid, filedir = './Pickles/', filenamebase = 'MockGridTemp'):
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    vns = Grid.__dict__.keys()
    vnfile = open(filedir + filenamebase + '_varnames'  + '.pickle', 'w')
    for vn in vns:
        vnfile.write('{0}\n'.format(vn))
        with open(filedir + filenamebase + '_' + vn + '.pickle', 'wb') as handle:
            pickle.dump(Grid.__dict__[vn], handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadObj(Grid, filedir = './Pickles/', filenamebase = 'MockGridTemp'):

    vnfile = open(filedir + filenamebase + '_varnames'  + '.pickle', 'r')
    for vn in vnfile.readlines():
        vnstrip = vn.rstrip()
        print(repr(vnstrip))
        with open(filedir + filenamebase + '_' + vnstrip + '.pickle', 'rb') as handle:
                Grid.__dict__[vnstrip] = pickle.load(handle)

def loadElementsFromPickle(Grid, filedir = './Pickles/', filenamebase = 'MockGridTemp',
                           listOfElements = ['ells'], overwriteElements = False):
    if not os.path.exists(filedir):
        print('Pickle File directory not found')
        return 314
    vnFile = open(filedir + filenamebase + '_varnames'  + '.pickle', 'r')
    vnsSaved = vnFile.readlines()
    vnsLoaded = Grid.__dict__.keys()
    for e in listOfElements:
        if not (e in vnsSaved):
            print('variable {0} wasnt saved. Something bad should happen now.'.format(e))
        if e in vnsLoaded:
            if overwriteElements:
                print('overwriting {0}'.format(e))
                with open(filedir + filenamebase + '_' + e + '.pickle', 'rb') as handle:
                    Grid.__dict__[e] = pickle.load(handle)
            else:
                print('not overwriting {0}'.format(e))
        else:
            with open(filedir + filenamebase + '_' + e + '.pickle', 'rb') as handle:
                #pickle.dump(Grid.__dict__[vn], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('loading {0}'.format(e))
                Grid.__dict__[e] = pickle.load(handle)
            

def saveElementsAsPickle(Grid, filedir = './Pickles/', filenamebase = 'MockGridTemp', 
                        listOfElements = ['ells'], overwriteElements = False):
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(filedir + filenamebase + '_varnames' + '.pickle'):
        print('making new set of files')
        vnsSaved = open(filedir + filenamebase + '_varnames'  + '.pickle', 'w')
        vnSavedList = []
    else:
        vnsSaved = open(filedir + filenamebase + '_varnames' + '.pickle', 'r+').readlines()
        vnSavedList = [str(vn.rstrip()) for vn in vnsSaved]
    #for vn in vnSavedList:
    #    vnsSaved.write('{0}\n'.format(vn.rstrip()))
    vnsLoaded = Grid.__dict__.keys()
    for e in listOfElements:
        pfname = filedir + filenamebase + '_' + e + '.pickle'
        if os.path.isfile(pfname):
            if overwriteElements:
                print('overwriting {0}'.format(e))
                with open(filedir + filenamebase + '_' + e + '.pickle', 'wb') as handle:
                    pickle.dump(Grid.__dict__[e], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('pickle file for element {0} already exists.'.format(e))
                return 3141
            vnsSaved.write('{0}\n'.format(e))
            vnSavedList.append(e)
            
        else:
            print('variable {0} not loaded so unable to save it'.format(e))
