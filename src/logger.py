import logging
import os
from datetime import datetime

# Create a log file name with current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Set the path for logs directory and log file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the logs directory if it does not exist
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    # Example usage of logging
    logging.info("Logging has started.") 


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