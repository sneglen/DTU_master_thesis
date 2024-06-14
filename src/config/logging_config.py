import logging
import sys
from pathlib import Path

logger_name = None

# Custom level: NOTICE
NOTICE_LEVEL_NUM = 25 # between INFO=20 and WARNING=30 
logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")

def notice(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        self._log(NOTICE_LEVEL_NUM, message, args, **kws)

logging.Logger.notice = notice


def setup_logging(log_path):

  BASE_DIR = Path(__file__).resolve().parent
  LOGS_DIR = Path(BASE_DIR, log_path)
  LOGS_DIR.mkdir(parents=True, exist_ok=True)

  logging_config = {
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
          "minimal": {"format": "%(message)s"},
          "detailed": {
              "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
          },
      },
      "handlers": {
          "console": {
              "class": "logging.StreamHandler",
              "stream": sys.stdout,
              "formatter": "minimal",
              "level": NOTICE_LEVEL_NUM,
          },
          "info": {
              "class": "logging.handlers.RotatingFileHandler",
              "filename": Path(LOGS_DIR, "info.log"),
              "maxBytes": 10485760,  # 1 MB
              "backupCount": 10,
              "formatter": "detailed",
              "level": logging.INFO,
          },
          "error": {
              "class": "logging.handlers.RotatingFileHandler",
              "filename": Path(LOGS_DIR, "error.log"),
              "maxBytes": 10485760,  # 1 MB
              "backupCount": 10,
              "formatter": "detailed",
              "level": logging.ERROR,
          },
          "debug": {
              "class": "logging.handlers.RotatingFileHandler",
              "filename": Path(LOGS_DIR, "debug.log"),  
              "maxBytes": 10485760,  # 1 MB
              "backupCount": 10,
              "formatter": "detailed",
              "level": logging.DEBUG, 
          },        
      },
      "root": {
          "handlers": ["console", "info", "error"],
          "level": logging.INFO,
          "propagate": True
      },
      "loggers": {
          "production_logger": {  
              "handlers": ["console", "info", "error"], 
              "level": logging.INFO,
              "propagate": False  
          },    
          "development_logger": {  
              "handlers": ["console", "info", "error", "debug"], 
              "level": logging.DEBUG,
              "propagate": False  
          },    
      },
  }

  logging.config.dictConfig(logging_config)


def initiate_logging(cfg):
  logger_name = cfg['logger']['logger_name']
  setup_logging(cfg['logger']['log_path'])
  logger = logging.getLogger(logger_name)
  logger.notice("<SESSION STARTED>\n")

  return logger