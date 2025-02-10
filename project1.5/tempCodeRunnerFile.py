import logging
import sys
import yaml
import os
from anomaly_detector import Config, AudioConfig, ModelConfig, AccidentSoundAnomalyDetector, AudioMonitor
import warnings

def load_config() -> Config:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    return Config(
        audio=AudioConfig(**cfg['audio']),
        model=ModelConfig(**cfg['model']),
        dataset_path=cfg['dataset_path'],
        model_path=cfg['model_path']
    )

def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("debug.log", encoding="utf-8")]
    )
    
    cfg = load_config()
    detector = AccidentSoundAnomalyDetector(cfg)
    
    if not os.path.exists(cfg.model_path):
        logging.info("Initial model training...")
        detector.train()
    else:
        detector.load_model()
        detector.train(new_samples_only=True)
    
    monitor = AudioMonitor(detector, cfg)
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == "__main__":
    main()