import librosa
import numpy as np


def extract_features(audio_path: str) -> np.ndarray:
    """
    Extrai features de áudio compatíveis com o CSV do GTZAN.
    Reproduz exatamente as mesmas features usadas no treino do modelo.

    Args:
        audio_path: caminho para o arquivo .wav

    Returns:
        np.ndarray com 57 features na ordem correta
    """
    # Carrega o áudio com taxa de amostragem padrão do GTZAN (22050 Hz)
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma_stft.mean()
    chroma_stft_var = chroma_stft.var()

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    rms_var = rms.var()

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    spectral_centroid_var = spectral_centroid.var()

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = spectral_bandwidth.mean()
    spectral_bandwidth_var = spectral_bandwidth.var()

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = zero_crossing_rate.mean()
    zero_crossing_rate_var = zero_crossing_rate.var()

    # Harmony e Perceptr
    harmony, perceptr = librosa.effects.hpss(y)
    harmony_mean = harmony.mean()
    harmony_var = harmony.var()
    perceptr_mean = perceptr.mean()
    perceptr_var = perceptr.var()

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # MFCCs (20 coeficientes, média e variância de cada)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_features = []
    for i in range(20):
        mfcc_features.append(mfcc[i].mean())
        mfcc_features.append(mfcc[i].var())

    # Montar vetor na ordem exata do CSV
    features = np.array([
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var,
        perceptr_mean, perceptr_var,
        tempo,
        *mfcc_features
    ])

# Trava de paridade — garante que o vetor tem exatamente 57 features.
    # Usa raise em vez de assert porque assert é removido quando o Python
    # roda em modo otimizado (-O), e essa garantia não pode sumir.
    expected = len(get_feature_names())
    if len(features) != expected:
        raise ValueError(
            f"Número de features incorreto: esperado {expected}, obtido {len(features)}"
        )

    return features   

def get_feature_names() -> list:
    """
    Retorna os nomes das features na ordem correta.
    Útil para validação e debug.
    """
    names = [
        'chroma_stft_mean', 'chroma_stft_var',
        'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var',
        'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var',
        'perceptr_mean', 'perceptr_var',
        'tempo'
    ]
    for i in range(1, 21):
        names.append(f'mfcc{i}_mean')
        names.append(f'mfcc{i}_var')

    return names