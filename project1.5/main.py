import logging
import sys
import yaml
import os
from anomaly_detector import Config, AudioConfig, ModelConfig, AccidentSoundAnomalyDetector, AudioMonitor
import warnings

def load_config() -> Config:
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")
    
    with open(config_path, encoding='utf-8') as f:  # UTF-8 인코딩 명시
        cfg = yaml.safe_load(f)
    
    return Config(
        audio=AudioConfig(**cfg['audio']),
        model=ModelConfig(**cfg['model']),
        dataset_path=cfg['dataset_path'],
        model_path=cfg['model_path']
    )

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logging.info("사용자에 의해 프로그램 종료됨")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        logging.error("처리되지 않은 예외", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("debug.log", encoding="utf-8")]
    )
    
    cfg = load_config()
    detector = AccidentSoundAnomalyDetector(cfg)
    
    if not os.path.exists(cfg.model_path):
        logging.info("초기 모델 학습 시작...")
        try:
            detector.train()
        except ValueError as e:
            logging.error(f"모델 학습 실패: {str(e)}")
            sys.exit(1)
    else:
        logging.info("기존 모델 로드 및 추가 학습 시작...")
        try:
            detector.load_model()
            detector.train(new_samples_only=True)
        except Exception as e:
            logging.error(f"모델 로드 또는 추가 학습 실패: {str(e)}")
            sys.exit(1)
    
    monitor = AudioMonitor(detector, cfg)
    try:
        monitor.start()
    except KeyboardInterrupt:
        logging.info("사용자에 의해 모니터링 중지됨")
        monitor.stop()
    finally:
        logging.info("프로그램 정상 종료")

if __name__ == "__main__":
    main()