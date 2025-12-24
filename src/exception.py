import sys
from src.logging import logging  # your custom logger

def error_message_detail(error, error_detail):
    """
    Returns a detailed error message with file name, line number and actual error
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail):
        """
        Custom exception for the project

        :param error: actual error object
        :param error_detail: sys module to get traceback
        """
        self.error_message = error_message_detail(error, error_detail=error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message



