import logging
import sys

def setup_logging():
    """设置统一的 logging 格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(processName)s %(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete.")