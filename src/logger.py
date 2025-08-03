import logging
import os
import sys
from datetime import datetime
import re

# ======================================================
# 📌 STEP 1: Create a unique log file name using timestamp
# ======================================================
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# ======================================================
# 📌 STEP 2: Define folder path where logs will be stored
# ======================================================
logs_path = os.path.join(os.getcwd(), "logs")

# ======================================================
# 📌 STEP 3: Create logs directory if it doesn’t exist
# ======================================================
os.makedirs(logs_path, exist_ok=True)

# ======================================================
# 📌 STEP 4: Full path to the log file inside logs folder
# ======================================================
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# ======================================================
# 📌 STEP 5: Configure file logging (UTF-8 safe with emojis)
# ======================================================
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
file_formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# ======================================================
# 📌 STEP 6: Console logging (REMOVE emojis for Windows console)
# ======================================================
class EmojiStripperFormatter(logging.Formatter):
    """Custom formatter to remove emojis for console output"""
    def format(self, record):
        # Remove all non-ASCII characters (emojis, symbols)
        record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg))
        return super().format(record)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = EmojiStripperFormatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# ======================================================
# 📌 STEP 7: Root logger setup
# ======================================================
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# ======================================================
# 📌 STEP 8: Quick test to ensure logging works
# ======================================================
if __name__ == "__main__":
    logging.info("✅ Logging system initialized successfully! 🚀")
    logging.info("This message will keep emojis in log file but show plain text in console.")