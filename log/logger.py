import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_file: str = "claude_api.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler (rotates at 5MB, keeps 3 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
