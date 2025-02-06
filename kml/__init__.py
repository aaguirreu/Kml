import time

__version__ = '1.0'
__author__ = 'Álvaro Aguirre'
__date__ = time.strftime("%Y-%m-%dT%Hh%Mm%Ss")

step_log = []
results = []
all_results = []

from .logging import log_step, log_results, log_memory_usage
from .mlize import run_models, extract_taxonomy, extract_species_from_filename
from .vectorize import vectorization_methods
from .results import results_to_df, plot_grouped_bar, plot_roc_curve_by_model, plot_correct_incorrect_bar
from .run import run_all, evaluate_all_vectorizations, all_results