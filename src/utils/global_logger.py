import json
import os
import logging
import datetime
from termcolor import colored

LOG_DIR = 'logs'
LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
logger = logging.getLogger('pytorch')
logger.setLevel(LOG_LEVEL)
logger.handlers = []
logger.propagate = 0

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

fileHandeler = logging.FileHandler(os.path.sep.join([
    LOG_DIR if os.path.isdir(LOG_DIR) and os.access(LOG_DIR, os.W_OK) else '/logs',
    datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
]))
fileHandeler.setLevel(LOG_LEVEL)

consoleHandeler = logging.StreamHandler()
consoleHandeler.setLevel(LOG_LEVEL)
fileFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fmt = colored('%(asctime)s', 'blue') +' - '+ colored('%(name)s', 'magenta', attrs=['bold']) + ' - '+ \
              colored('%(levelname)s:', 'green') +' - '+ colored('%(message)s', 'white')
consoleFormatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
fileHandeler.setFormatter(fileFormatter)
consoleHandeler.setFormatter(consoleFormatter)

logger.addHandler(fileHandeler)
logger.addHandler(consoleHandeler)



