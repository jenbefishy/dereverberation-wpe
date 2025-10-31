import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests

def download_dataset():
    url = "https://drive.usercontent.google.com/download?id=1RarjxOgWkaDV8EjH_eLX169y89PVa3sg&confirm=t"
    output = "nonblind_test.clean.zip"
    
    if not Path(output).exists():
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        
        with open(output, 'wb') as file, tqdm(
            desc=output,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        
        print("Файл успешно скачан:", output)
    else:
        print("Файл уже найден")



def read_normalize(path):
    y, sr = sf.read(path)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    y = y / np.max(np.abs(y))
    return y, sr


def plot_audio_spectrogram(
    y, sr,
    n_fft, hop_length, win_length, window,
    fmin=50, fmax_speech=8000, top_db=80
):
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    num_channels = y.shape[0]
    plt.figure(figsize=(10, 3 * num_channels))

    fmax = min(fmax_speech, sr / 2)

    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)

        D = np.abs(librosa.stft(y[i], n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window))
        D_db = librosa.amplitude_to_db(D, ref=np.max, top_db=top_db)

        librosa.display.specshow(
            D_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='hz', cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Канал {i+1}")

        plt.ylim(fmin, fmax)

        yticks = [100, 500, 1000, 2000, 4000, int(fmax)]
        yticks = [f for f in yticks if fmin <= f <= fmax]
        ylabels = [f"{int(f/1000)}k" if f >= 1000 else str(int(f)) for f in yticks]
        plt.yticks(yticks, ylabels, fontsize=9)
        plt.ylabel("Частота (Гц)", fontsize=11)
        plt.xlabel("Время (с)", fontsize=11)

        plt.tight_layout(pad=1.5)

    plt.show()

