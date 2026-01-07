import logging
import os
from datetime import datetime


class DailyLogger:
    def __init__(self, log_dir="logs", log_prefix="result"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_prefix = log_prefix
        # 레벨별 매핑: (date_str, logger)
        self.loggers = {}
        # 핸들러는 첫 로그 시점까지 생성하지 않음 (날짜별 폴더 정확성 보장)
        # 로그 보존일수: 기본 30일, 환경변수 LOG_RETENTION_DAYS로 재정의
        try:
            self.retention_days = int(os.environ.get("LOG_RETENTION_DAYS", "30"))
        except Exception:
            self.retention_days = 30
        # 아카이브 보존 개월수: 기본 6개월, 환경변수 LOG_ARCHIVE_RETENTION_MONTHS로 재정의
        try:
            self.archive_retention_months = int(
                os.environ.get("LOG_ARCHIVE_RETENTION_MONTHS", "6")
            )
        except Exception:
            self.archive_retention_months = 6

    def _get_log_dir(self):
        today = datetime.now().strftime("%Y-%m-%d")
        d = os.path.join(self.log_dir, today)
        os.makedirs(d, exist_ok=True)
        # prune old date-folders on access (archive then remove)
        try:
            self._prune_old_dirs()
        except Exception:
            pass
        return d

        def _prune_old_dirs(self):
            """보존일보다 오래된 날짜 폴더를 아카이브하고 오래된 아카이브를 정리합니다.

            동작:
            - `self.retention_days`보다 오래된 날짜 형식 폴더는 `logs/archived/`에
                `<YYYY-MM-DD>.tar.gz` 형태로 압축하여 저장(이미 존재하면 건너뜀)한 뒤 원본 폴더를 삭제합니다.
            - `archive_retention_months`보다 오래된 아카이브 파일은 삭제합니다.
            """

        import shutil
        import tarfile

        now = datetime.now()

        # ensure archive dir exists
        archive_root = os.path.join(self.log_dir, "archived")
        os.makedirs(archive_root, exist_ok=True)

        # 1) Archive old date folders
        for name in os.listdir(self.log_dir):
            path = os.path.join(self.log_dir, name)
            if not os.path.isdir(path):
                continue
            if name == "archived":
                continue
            # expect folder names like YYYY-MM-DD
            try:
                dt = datetime.strptime(name, "%Y-%m-%d")
            except Exception:
                continue
            age_days = (now - dt).days
            if age_days > self.retention_days:
                archive_name = f"{name}.tar.gz"
                archive_path = os.path.join(archive_root, archive_name)
                # if archive exists, just remove folder
                if not os.path.exists(archive_path):
                    try:
                        with tarfile.open(archive_path, "w:gz") as tar:
                            # add folder contents under top-level folder name
                            tar.add(path, arcname=name)
                    except Exception:
                        # failed to archive: best-effort, skip deletion
                        continue
                # remove original folder after successful archive (or if archive already existed)
                try:
                    shutil.rmtree(path)
                except Exception:
                    # ignore failures
                    pass

        # 2) Prune old archives (by months)
        try:
            for fname in os.listdir(archive_root):
                if not fname.endswith(".tar.gz"):
                    continue
                fpath = os.path.join(archive_root, fname)
                # try parse date from filename <YYYY-MM-DD>.tar.gz
                base = fname[:-7]
                try:
                    dt = datetime.strptime(base, "%Y-%m-%d")
                except Exception:
                    # fallback to file mtime
                    try:
                        mtime = os.path.getmtime(fpath)
                        dt = datetime.fromtimestamp(mtime)
                    except Exception:
                        continue
                age_days = (now - dt).days
                # approximate months as 30 days per month
                if age_days > (self.archive_retention_months * 30):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except Exception:
            pass

    def _setup_logger(self, level: str = "info"):
        """
        특정 레벨용 로거를 준비(또는 반환)합니다. 로그는 날짜별 폴더와
        레벨별 파일에 기록됩니다.
        """
        level = (level or "info").lower()
        today = datetime.now().strftime("%Y-%m-%d")

        # If we already have a logger for this level and it's for today, reuse it
        meta = self.loggers.get(level)
        if meta is not None:
            date_str, existing_logger = meta
            if date_str == today:
                return existing_logger
            # date changed: remove handlers from old logger to avoid duplicate writes
            try:
                for h in list(existing_logger.handlers):
                    try:
                        existing_logger.removeHandler(h)
                        h.close()
                    except Exception:
                        pass
            except Exception:
                pass

        log_dir = self._get_log_dir()
        filename = f"{self.log_prefix}_{level}.log"
        log_path = os.path.join(log_dir, filename)

        logger = logging.getLogger(f"DailyLogger_{self.log_prefix}_{level}_{today}")
        logger.setLevel(logging.DEBUG)

        # Add new file handler for today's log
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # store meta
        self.loggers[level] = (today, logger)
        return logger

    def log(self, msg, level="info"):
        # 요청된 레벨용 로거가 존재하는지 확인(날짜 변경 처리)
        lvl = (level or "info").lower()
        logger = self._setup_logger(lvl)
        if lvl == "info":
            logger.info(msg)
        elif lvl == "warning":
            logger.warning(msg)
        elif lvl == "error":
            logger.error(msg)
        else:
            logger.debug(msg)

    def save_result(self, result):
        """
        result(dict 또는 str)을 로그 파일에 저장합니다.
        """
        import json

        # Save results to info-level log for the current date
        logger = self._setup_logger("info")
        if isinstance(result, dict):
            logger.info("RESULT: " + json.dumps(result, ensure_ascii=False))
        else:
            logger.info(f"RESULT: {result}")


# 사용 예시:
# from _daily_logger import DailyLogger
# logger = DailyLogger()
# logger.log("메시지")
# logger.save_result(result_dict)
