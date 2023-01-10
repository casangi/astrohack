import time

from datetime import datetime

def debug(message:str):
    """ Prints formatted message with logging levels for debug

    Args:
        message (str): Message to print
    """

    DEBUG = "\033[01;32m"
    RESET = "\033[01;00;39m"

    print("[" + DEBUG + str(datetime.fromtimestamp(time.time())) + " Debug" + RESET + "]: " + message)
        
def info(message:str):
    """ Prints formatted message with logging levels for info

    Args:
        message (str): Message to print
    """

    INFO = "\033[01;34m"
    RESET = "\033[01;00;39m"

    print("[" + INFO + str(datetime.fromtimestamp(time.time())) + " System Info" + RESET + "]: " + message)
        
def warning(message:str):
    """ Prints formatted message with logging levels for warning

    Args:
        message (str): Message to print
    """

    WARNING = "\033[01;33m"
    RESET = "\033[01;00;39m"

    print("[" + WARNING + str(datetime.fromtimestamp(time.time())) + " Warning" + RESET + "]: " + message)
        
def error(message:str):
    """ Prints formatted message with logging levels for error

    Args:
        message (str): Message to print
    """

    ERROR = "\033[05;31m"       
    RESET = "\033[01;00;39m"
        
    print("[" + ERROR + str(datetime.fromtimestamp(time.time())) + " Error" + RESET + "]: " + message)
        
def critical(message:str):
    """ Prints formatted message with logging levels for critical

    Args:
        message (str): Message to print
    """

    CRITICAL = "\033[05;31m" 
    RESET = "\033[01;00;39m"

    print("[" + CRITICAL + str(datetime.fromtimestamp(time.time())) + " Critical" + RESET + "]: " + message)