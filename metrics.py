import numpy as np
import onnxruntime
import librosa

def fwsegsnr(clean, enhanced, sr, eps,
             n_fft, hop, win, window):
    X = np.abs(librosa.stft(clean, n_fft=n_fft, hop_length=hop, win_length=win, window=window))
    Xh = np.abs(librosa.stft(enhanced, n_fft=n_fft, hop_length=hop, win_length=win, window=window))

    F, T = X.shape
    fwseg = 0

    for t in range(T):
        w = X[:, t] ** 2
        if np.sum(w) == 0:
            continue
        w /= (np.sum(w) + eps)
        
        num = X[:, t] ** 2
        den = (X[:, t] - Xh[:, t]) ** 2 + eps
        snr_f = np.log10(num / (den + eps))

        fwseg += np.sum(w * snr_f)

    fwseg = 10 * fwseg / (T + eps)
    return fwseg

def cepstral_distance(clean, enhanced, sr, n_mfcc=13):
    clean = clean / np.max(np.abs(clean))
    enhanced = enhanced / np.max(np.abs(enhanced))
    C = librosa.feature.mfcc(y=clean, sr=sr, n_mfcc=n_mfcc)
    Ch = librosa.feature.mfcc(y=enhanced, sr=sr, n_mfcc=n_mfcc)
    T = min(C.shape[1], Ch.shape[1])
    C = C[:, :T]
    Ch = Ch[:, :T]
    diff = (C - Ch) ** 2
    per_frame = np.sqrt(np.sum(diff, axis=0))
    CD = np.mean(per_frame)
    return CD

def DNSMOS(y, sr):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    target_len = 144160
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    
    input_audio = np.expand_dims(y.astype(np.float32), axis=0)  
    session = onnxruntime.InferenceSession("models/sig_bak_ovr.onnx")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_audio})
    # Overall, Speech Mos, Background Mos
    return outputs[0]
    
    