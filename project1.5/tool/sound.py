import librosa
y, sr = librosa.load("dataset/accident/2.wav", sr=None)
print(f"샘플링 레이트: {sr}Hz, 길이: {len(y)/sr:.2f}초")