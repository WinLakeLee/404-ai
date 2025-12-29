import logging
import os
from datetime import datetime

class DailyLogger:
    def __init__(self, log_dir="logs", log_prefix="result"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_prefix = log_prefix
        self.logger = None
        self._setup_logger()

    def _get_log_path(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{self.log_prefix}_{today}.log")

    def _setup_logger(self):
        log_path = self._get_log_path()
        self.logger = logging.getLogger(f"DailyLogger_{self.log_prefix}")
        self.logger.setLevel(logging.INFO)
        # 중복 핸들러 방지
        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, msg, level="info"):
        self._setup_logger()  # 날짜 변경 시 핸들러 갱신
        if level == "info":
            self.logger.info(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "error":
            self.logger.error(msg)
        else:
            self.logger.debug(msg)

    def save_result(self, result):
        """
        result(dict 또는 str)을 로그 파일에 저장
        """
        import json
        self._setup_logger()
        if isinstance(result, dict):
            self.logger.info("RESULT: " + json.dumps(result, ensure_ascii=False))
        else:
            self.logger.info(f"RESULT: {result}")

# 사용 예시:
# from _logger_ import DailyLogger
# logger = DailyLogger()
# logger.log("메시지")
# logger.save_result(result_dict)
