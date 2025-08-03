import logging
import os
from datetime import datetime

# Step 1: Create a log file name using current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Step 2: Define folder path where logs will be stored
logs_path = os.path.join(os.getcwd(), "logs")

# Step 3: Create the logs directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Step 4: Full path to the log file inside the logs folder
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Step 5: Configure logging to write to that file with proper format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Step 6: Example usage to confirm logging works
if __name__ == "__main__":
    logging.info("✅ Logging has started successfully.")


# Logging Setup – Simple Explanation

# Step 1: We import the tools needed:
#         - logging: to save messages about what our code is doing
#         - os: to work with folders and file paths
#         - datetime: to add current date and time into the log file name

# Step 2: We create a unique log file name using the current time
#         This keeps each run separate and easy to track

# Step 3: We create a folder called "logs" (if it doesn't exist)
#         This folder will store our log files safely

# Step 4: We set the full path where the log file will be saved
#         The file will go inside the "logs" folder

# Step 5: We configure how logging should work
#         - Where to save logs
#         - What each log entry should look like (time, line, name, level, message)
#         - What kind of messages to save (INFO and above)

# This setup helps us track problems or actions in our app step by step