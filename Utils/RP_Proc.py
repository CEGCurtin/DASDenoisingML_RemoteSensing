import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft,ifft

#%% Pre-processing Routines
def fRingdownRemoval(R):
    mR = np.median(R,1)
    Rf = np.transpose(np.transpose(R) - np.transpose(mR))
    return Rf

#%% Signal to Noise Ratio Utils

def fCalcFrameSNR(D,percent,maxlag_np):

    ntr = np.size(D[0,:])
    nsmp = np.size(D[:,0])

    g_max = np.zeros(ntr-1,dtype='float')
    ccf_lags = signal.correlation_lags(nsmp,nsmp)
    gv=(np.abs(ccf_lags)<maxlag_np)
    for n in range(ntr-1):
        ccf=signal.correlate((D[:,n]-D[:,n].mean())/(D[:,n].std()*nsmp),(D[:,n+1]-D[:,n+1].mean())/D[:,n+1].std(),'full')
        g_max[n] = np.max(ccf[(gv)])

    SNR_vec = np.sqrt(g_max/(1-g_max))

    return stats.trim_mean(SNR_vec,percent/100)

def fCalcSNR2DFrame(D,window):
    nsmp,ntr = np.shape(D)
    SNR2D = np.zeros((nsmp,ntr-1))

    w2 = window // 2
    for n in range(nsmp):
        s1 = max(n - w2, 0)
        s2 = min(n + w2, nsmp-1)
        nn = s2-s1+1
        L = nn*2-1
        Dd = D[s1:s2+1,:]
        Dd = Dd - np.mean(Dd,axis=0)
        D0 = Dd[:,0:ntr-1]
        D1 = Dd[:,1:ntr]

        R = np.real(ifft(fft(D0,L,axis=0)*fft(np.flipud(D1.conj()),L,axis=0),axis=0))
        R/=np.std(D0,axis=0)*np.std(D1,axis=0)*nn
        g_max_v = np.max(R,axis=0)
        #print(np.shape(R))
        SNR2D[n,:] = np.sqrt(g_max_v/(1-g_max_v))

    return SNR2D

#%% Ormsby Filter
def fOrmsbASDesign(dt, nnp, fOrmsb):
    df = (1000.0 / (nnp * dt))  # for ms
    Fr = np.arange(0, (nnp - 1) * df + df, df)
    flt = np.zeros(nnp)

    flt[np.logical_and(Fr >= fOrmsb[1], Fr <= fOrmsb[2])] = 1
    nf = np.sum(np.logical_and(Fr >= fOrmsb[0], Fr < fOrmsb[1]))
    o = (np.cos(np.arange(0, nf) / nf * np.pi - np.pi) + 1) / 2
    flt[np.logical_and(Fr >= fOrmsb[0], Fr < fOrmsb[1])] = o

    nf = np.sum(np.logical_and(Fr > fOrmsb[2], Fr <= fOrmsb[3]))
    o = (np.cos(np.arange(0, nf) / nf * np.pi - np.pi) + 1) / 2
    flt[np.logical_and(Fr > fOrmsb[2], Fr <= fOrmsb[3])] = np.flip(o)

    return Fr, flt

def fOrmsbyBPFilter(R, dt, f1, f2, f3, f4, filt_mode):
    nnp, ntr = R.shape

    _, amp_fl_spec = fOrmsbASDesign(dt, nnp, [f1, f2, f3, f4])

    amp_fl_spec = np.array(amp_fl_spec, dtype=np.float32)
    amp_fl_spec = amp_fl_spec.reshape(1, nnp)

    if filt_mode < 0:#to make a notch filter set filt_mode to anything less than 0
        amp_fl_spec = 1 - amp_fl_spec

    S = np.fft.fft(np.array(R, dtype=np.float32), axis=0)
    S[nnp // 2:nnp, :] = 0

    for k in range(ntr):
        S[:, k] = S[:, k] * amp_fl_spec

    Rd = np.fft.ifft(S, axis=0)
    Rd = 2*np.real(Rd)


    return Rd
