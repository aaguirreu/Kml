import time

__version__ = '1.0'
__author__ = 'Álvaro Aguirre'
__date__ = time.strftime("%Y-%m-%dT%Hh%Mm%Ss")

time_log = []
step_log = []

#from sec2vec_module import sec2vec
from .logging import log_step, save_time_log, log_results
from .mlize import run_models, classification
from .vectorize import evaluate_all_vectorizations
from .run import kmers, extract_taxonomy