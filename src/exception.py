import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
# This script helps to find errors easily when your code crashes.

# We first import 'sys' to get information like which file and which line caused the error.
# We also use a logging setup to keep records of error messages.

# There's a function that builds a detailed message:
# - It checks where the error happened (file name and line number).
# - It shows what the actual error message is.
# - This helps you understand exactly what went wrong.

# Then there's a class called CustomException:
# - It creates a special error with all the important details.
# - When the error is printed, you get a full message with location and reason.
# - This makes fixing bugs faster and easier.  