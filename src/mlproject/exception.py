import sys

class CustomExceptionHandler(Exception):
    def __init__(self):
        pass
    
    @staticmethod
    def get_exception_info(error):
        exception_type = type(error)
        exception_name = str(error)
        return exception_type, exception_name
    
    @staticmethod
    def error_details(error, error_details):
        exception_type, exception_name = CustomExceptionHandler.get_exception_info(error)
        filename = error_details.exc_info()[2].tb_frame.f_code.co_filename
        lineno = error_details.exc_info()[2].tb_lineno
        
        print(f"Exception: {exception_name} of type: {exception_type} occurred in Python script: '{filename}' at line: {lineno}")
