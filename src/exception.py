import sys
import logging
sys # provides various functions and variables that are used to manipulate different parts of the python runtime environement 
    # any exception that is getting controlled sys will happen that information     

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename   
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error))
    return error_message
    

class CustomException(Exception):
    def __innit__(self,error_message, error_detail:sys):
        super().__innit__(error_message)
        self.error_message = error_message = error_message_detail(error_message, error_detail = error_detail)
    
    def __str__(self):
        return self.error_message

    if __name__ == "__main__":
        try:
            a=1/0
        except Exception as e:
            logging.info('Divide by Zero ')
            raise CustomException(e, sys)
