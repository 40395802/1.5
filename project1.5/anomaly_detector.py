import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa
import os
import logging
import joblib
from dataclasses import dataclass
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import signal
from threading import Thread, Lock
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import warnings
from librosa.util.exceptions import ParameterError
from matplotlib.animation import FuncAnimation


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Matplotlib 설정
mpl_use('Agg')
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'

# 설정 클래스
@dataclass
class AudioConfig:
    sample_rate: int
    duration: int
    channels: int
    yamnet_sample_rate: int
    buffer_size: int

@dataclass
class ModelConfig:
    contamination: str
    n_estimators: int
    threshold_percentile: int
    grid_search: bool
    update_interval: float
    use_autoencoder: bool
    autoencoder_epochs: int
    feature_dim: int
    bandpass_filter: bool
    augmentation: bool

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    dataset_path: str
    model_path: str

# 오디오 버퍼 클래스
class RingBuffer:
    def __init__(self, max_size: int, frame_size: int):
        self.max_size = max_size
        self.frame_size = frame_size
        self.buffer = np.zeros((max_size, frame_size), dtype=np.float32)
        self.index = 0
        self.lock = Lock()

    def put(self, data: np.ndarray):
        flattened = data.squeeze()
        with self.lock:
            if len(flattened) == self.frame_size:
                self.buffer[self.index % self.max_size] = flattened
                self.index += 1

    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.index == 0:
                return None
            idx = (self.index - 1) % self.max_size
            return self.buffer[idx]

# 코어 로직 클래스
class AccidentSoundAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self._init_gpu()
        self._load_yamnet()
        self._init_models()
        self.audio_buffer = RingBuffer(
            max_size=config.audio.buffer_size,
            frame_size=int(config.audio.sample_rate * config.audio.duration)
        )
        self.scaler = StandardScaler()
        self.logger = self._configure_logger()

    def _process_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        target_len = int(sr * self.config.audio.duration)
        chunks = []
        # 너무 짧은 경우: 패딩 또는 반복
        if len(audio) < target_len:
            if len(audio) > 0:  # 비어있지 않은 경우만 처리
                padding_needed = target_len - len(audio)
                if padding_needed <= len(audio):  # 짧은 경우 패딩
                    audio = np.pad(audio, (0, padding_needed), mode='constant')
                else:  # 매우 짧은 경우 반복
                    repeat_count = int(np.ceil(target_len / len(audio)))
                    audio = np.tile(audio, repeat_count)[:target_len]
            chunks.append(audio)
        # 긴 경우: 잘라서 청크로 분할
        else:
            num_chunks = int(np.floor(len(audio) / target_len))
            for i in range(num_chunks):
                start = i * target_len
                end = start + target_len
                chunks.append(audio[start:end])
            # 남은 부분 처리 (짧은 샘플 포함)
            remaining = audio[num_chunks * target_len:]
            if len(remaining) > 0:
                if len(remaining) < target_len:
                    remaining = np.pad(remaining, (0, target_len - len(remaining)), mode='constant')
                chunks.append(remaining)
        return chunks

    def _init_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                self.logger.info("GPU 메모리 동적 할당 활성화")
            except RuntimeError as e:
                self.logger.warning(f"GPU 설정 실패: {e}")

    def _load_yamnet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

    def _init_models(self):
        self.ocsvm = SGDOneClassSVM(
            nu=0.1,
            shuffle=True,
            tol=1e-4,
            max_iter=1000
        )
        self.autoencoder = self._build_autoencoder()
        self.threshold = None

    def _build_autoencoder(self):
        input_dim = self.config.model.feature_dim
        encoder_input = tf.keras.layers.Input(shape=(input_dim,))
        # Encoder
        x = tf.keras.layers.Dense(64, activation='relu')(encoder_input)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        # Decoder
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
        return tf.keras.Model(inputs=encoder_input, outputs=decoder_output)

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        audio = audio / (rms + 1e-6)
        return np.sign(audio) * np.log1p(np.abs(audio))

    def _apply_bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.config.model.bandpass_filter:
            nyq = 0.5 * sr
            low = 300 / nyq
            high = 4000 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.lfilter(b, a, audio)
        return audio

    def _augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.config.model.augmentation:
            # Pitch Shift
            if np.random.rand() > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-2, 2))
            # Time Stretch
            if np.random.rand() > 0.5:
                rate = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)
            # 노이즈 추가
            if np.random.rand() > 0.5:
                audio = audio + np.random.normal(0, 0.005, audio.shape)
            # 시간 이동
            if np.random.rand() > 0.5:
                shift = np.random.randint(-sr // 2, sr // 2)
                audio = np.roll(audio, shift)
            # 목표 길이에 맞게 패딩 또는 자르기
            target_len = self.config.audio.yamnet_sample_rate * self.config.audio.duration
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
        return audio

    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Mel-spectrogram 추출
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        # MFCC 및 기타 특징 추출
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(S=S)
        # 각 특징의 평균값 계산
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(log_S, axis=1)
        ])
        # feature_dim에 맞게 크기 조정
        target_dim = self.config.model.feature_dim
        if len(features) > target_dim:
            features = features[:target_dim]  # 초과하는 경우 잘라냄
        elif len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)))  # 부족한 경우 패딩
        return features

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            audio = self._normalize_audio(audio)
            audio = self._apply_bandpass_filter(audio, sr)
            if sr != self.config.audio.yamnet_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=self.config.audio.yamnet_sample_rate,
                    res_type='kaiser_best'
                )
            target_len = self.config.audio.yamnet_sample_rate * self.config.audio.duration
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            return audio
        except Exception as e:
            self.logger.error(f"전처리 실패: {str(e)}")
            raise

    def train(self, new_samples_only=False):
        X = []
        file_list = [f for f in os.listdir(self.config.dataset_path) 
                     if f.lower().endswith(('.wav', '.mp3'))]
        if not file_list:
            raise ValueError("학습 데이터 없음")
        if new_samples_only and os.path.exists(self.config.model_path):
            self.load_model()
            self.logger.info("기존 모델 로드 완료. 추가 학습 시작...")
        else:
            self.logger.info("새로운 모델 학습 시작...")
        for fname in file_list:
            path = os.path.join(self.config.dataset_path, fname)
            try:
                audio, sr = librosa.load(path, sr=self.config.audio.yamnet_sample_rate)
                processed_chunks = self._process_audio(audio, sr)
                for chunk in processed_chunks:
                    chunk = self._augment_audio(chunk, sr)
                    features = self._extract_features(chunk, sr)
                    X.append(features)
            except Exception as e:
                self.logger.error(f"파일 처리 실패: {fname} - {str(e)}")
        if len(X) == 0:
            raise ValueError("학습할 데이터가 없습니다.")
        # 데이터 표준화
        X = self.scaler.fit_transform(X)
        # Autoencoder 학습
        if self.config.model.use_autoencoder:
            self.autoencoder.compile(optimizer='adam', loss='mse')
            self.autoencoder.fit(
                X, X,
                epochs=self.config.model.autoencoder_epochs,
                batch_size=32,
                verbose=0
            )
        # OCSVM 학습
        self.ocsvm.fit(X)
        self._set_threshold(X)
        self._save_model()

    def _set_threshold(self, X: np.ndarray):
        scores = self.ocsvm.score_samples(X)
        self.threshold = np.percentile(
            scores, 
            100 - self.config.model.threshold_percentile
        )
        self.logger.info(f"임계값 설정 완료: {self.threshold:.2f}")

    def _save_model(self):
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        joblib.dump({
            'model': self.ocsvm,
            'autoencoder': self.autoencoder.get_weights() if self.config.model.use_autoencoder else None,
            'threshold': self.threshold,
            'scaler': self.scaler
        }, self.config.model_path)
        self.logger.info(f"모델 저장 완료: {self.config.model_path}")

    def load_model(self):
        data = joblib.load(self.config.model_path)
        self.ocsvm = data['model']
        self.threshold = data['threshold']
        self.scaler = data['scaler']
        if self.config.model.use_autoencoder and data['autoencoder'] is not None:
            self.autoencoder.set_weights(data['autoencoder'])
        self.logger.info("모델 로드 완료")

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        try:
            processed_audio = self.preprocess_audio(audio, sr)
            features = self._extract_features(processed_audio, sr)
            features_scaled = self.scaler.transform([features])
            # OCSVM 점수
            ocsvm_score = self.ocsvm.score_samples(features_scaled)[0]
            # Autoencoder 점수
            ae_score = 0.0
            if self.config.model.use_autoencoder:
                reconstructed = self.autoencoder.predict(features_scaled, verbose=0)
                ae_score = np.mean(np.square(features_scaled - reconstructed))
            # 앙상블 점수 계산
            combined_score = ocsvm_score - (ae_score * 0.3 if self.config.model.use_autoencoder else 0)
            is_accident = combined_score > self.threshold
            return {
                'is_accident': is_accident,
                'score': combined_score,
                'waveform': processed_audio,
                'mel_spectrogram': librosa.power_to_db(
                    librosa.feature.melspectrogram(y=processed_audio, sr=sr),
                    ref=np.max
                )
            }
        except Exception as e:
            self.logger.error(f"예측 실패: {str(e)}")
            return {'is_accident': False, 'score': 0}

# 모니터링 클래스
class AudioMonitor:
    def __init__(self, detector: AccidentSoundAnomalyDetector, config: Config):
        self.detector = detector
        self.config = config.audio
        self.model_config = config.model
        self.is_running = False
        self.stream = None
        self.last_status = None
        self.last_update_time = 0
        self.active_figures = []
        self.max_figures = 5

        # 실시간 그래프 초기화
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(2, 1, figsize=(12, 6))
        self.line_wave, = self.ax_wave.plot([], [])
        self.spec_img = self.ax_spec.imshow(np.zeros((128, 128)), aspect='auto', origin='lower', cmap='inferno')
        self.ax_wave.set_title("Waveform")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_spec.set_title("Mel Spectrogram")
        self.ax_spec.set_xlabel("Time")
        self.ax_spec.set_ylabel("Frequency")
        plt.colorbar(self.spec_img, ax=self.ax_spec, format='%+2.0f dB')
        plt.tight_layout()

    def _callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"오디오 스트림 오류: {status}")
        self.detector.audio_buffer.put(indata.copy().squeeze())

    def start(self):
        import time
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=self._callback,
            blocksize=int(self.config.sample_rate * self.config.duration)
        )
        self.stream.start()
        logging.info("모니터링 시작... (Ctrl+C 종료)")

        # Matplotlib 애니메이션 설정
        ani = FuncAnimation(self.fig, self._update_plot, interval=500, blit=False)
        plt.show()

    def _update_plot(self, frame):
        audio = self.detector.audio_buffer.get()
        if audio is None:
            # 오디오 데이터가 없는 경우 기존 그래프를 그대로 유지
            return self.line_wave, self.spec_img

        result = self.detector.predict(audio, self.config.sample_rate)
        waveform = result['waveform']
        mel_spec = result['mel_spectrogram']
        is_accident = result['is_accident']

        # Waveform 업데이트
        time_axis = np.linspace(0, len(waveform) / self.config.sample_rate, len(waveform))
        self.line_wave.set_data(time_axis, waveform)
        self.ax_wave.relim()  # 축 범위 재설정
        self.ax_wave.autoscale_view()

        # Mel Spectrogram 업데이트
        self.spec_img.set_data(mel_spec)
        self.spec_img.set_clim(vmin=np.min(mel_spec), vmax=np.max(mel_spec))

        # 콘솔에 상태 출력
        status_text = '🚨 사고' if is_accident else '✅ 정상'
        timestamp = datetime.now().strftime('%H:%M:%S')
        log = f"[{timestamp}] {status_text} | 신뢰도: {result['score']:.2f}"
        print(log)

        
        return self.line_wave, self.spec_img

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("\n")
        logging.info("모니터링 중지")
        plt.close('all')