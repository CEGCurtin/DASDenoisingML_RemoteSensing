import numpy as np
from struct import unpack

import logging

logging.basicConfig(level=logging.INFO)

def fLoadSilixaTDMS(fname,scan_size=2000):
    '''Function loads Silixa TDMS file
    v.0.2, Roman Pevzner
    
    scan_size - number of byts to scan for aquisition parameters
    
    D       - data
    mlength - measure length (m)
    dx      - chanel spacing (m)
    dt      - sampling interval (s)
    
    '''
    D,dx,mlength,dt=np.zeros((0,0),dtype='int16'),0,0,0
    SPAT_RES = b'SpatialResolution[m]'
    MES_LENGTH = b'MeasureLength[m]'
    SAMPL_FREQ = b'SamplingFrequency[Hz]'
    GPS_TIME = b'GPSTimeStamp'

    with open(fname,'rb') as f:
        f.seek(20,0)
        #unpack()
        hh=unpack('i',f.read(4))
        header_size=hh[0] + 28
        f.seek(0,0)
        sbuf = f.read(scan_size)
        f.seek(0,2)
        nbytes = f.tell()
        
        ss = sbuf.find(SAMPL_FREQ)+len(SAMPL_FREQ)+4
        col = sbuf[ss:ss+8]
        ss = sbuf.find(SPAT_RES)+len(SPAT_RES)+4
        col = col+sbuf[ss:ss+8]
        ss = sbuf.find(GPS_TIME)+len(GPS_TIME)+4
        col = col+sbuf[ss:ss+16]
        ss = sbuf.find(MES_LENGTH)+len(MES_LENGTH)+4
        col = col+sbuf[ss:ss+4]
        sfreq,dx,gt1,gt2,mlength = unpack('ddQqI',col) #(float64,float64,uint32,uint64,int64)
        dt = 1.0/sfreq
        
        ntr = int(mlength/dx)
        nsmp = int((nbytes - header_size)/(2*ntr))
        f.seek(header_size,0)
        #for n in range(nsmp):
        #    D[:,n] = np.fromfile(f,dtype='int16',count=ntr)
        D = np.fromfile(f,dtype='int16',count=int(nbytes/2))

    
    D = D.reshape(nsmp,ntr)
    np.transpose(D)
    return D,dx,mlength,dt