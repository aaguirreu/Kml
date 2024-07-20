import time

__version__ = '1.0'
__author__ = 'Álvaro Aguirre'
__date__ = time.strftime("%Y-%m-%dT%Hh%Mm%Ss")

time_log = []
step_log = []

#from sec2vec_module import sec2vec
from .vectorize import calculate_tfidf
from .kmer_logging import log_step, save_time_log, log_results
from .compare import compare_kmers
from .run import kmers