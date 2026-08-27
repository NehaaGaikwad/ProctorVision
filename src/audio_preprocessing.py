import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_FILE = "segments/speech_001.wav"

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

print("Loading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

print(f"Sample Rate: {sr} Hz")
print(f"Duration: {len(audio) / sr:.2f} seconds")
print(f"Samples: {len(audio)}")

audio = audio / (np.max(np.abs(audio)) + 1e-9)

print("Audio normalized.")

mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

log_mel = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)

print(f"Mel Spectrogram Shape: {log_mel.shape}")

os.makedirs("output", exist_ok=True)

output_file = "output/speech_001_spectrogram.png"

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    log_mel,
    sr=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")
plt.tight_layout()

plt.savefig(
    output_file,
    dpi=150
)

plt.close()

print(f"Spectrogram saved: {output_file}")