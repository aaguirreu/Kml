import time
import os
import csv
import psutil
from . import step_log, __author__, __date__

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    # Log memory usage in MB
    log_step(f"Memory usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def log_step(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}')
    step_log.append(message)
    with open('logs/logs.log', 'a') as log_file:
        log_file.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}\n')

def log_results(output, files, best_k, k_range):
    # If 'files' is a string, wrap it in a list to avoid iterating over each character.
    if isinstance(files, str):
        files = [files]
    
    log_output_filepath = os.path.join(output, f'performance.log')
    with open(log_output_filepath, 'w') as log_file:
        log_file.write(f"""#####################################
# Title: Kmer                       #
# Author: {__author__}            #
# Date: {__date__}        #
# Outfile: {__date__}.log #
#####################################

###########
# Results #
###########

- Frequency:

""")
        # Write information for each file
        for filepath in files:
            log_file.write(f"file: {filepath}\n")
            try:
                file_size = os.path.getsize(filepath)
            except Exception as e:
                file_size = f"Error getting file size: {e}"
            log_file.write(f"size: {file_size} bytes\n")
            if k_range.start == k_range.stop - 1:
                log_file.write(f"kmer {k_range.start}\n")
            else:
                log_file.write(f"kmer {k_range.start}-{k_range.stop - 1}\n")
            log_file.write(f"The best performance using this file was through {best_k}-mer analysis.\n\n")

    log_step(f"Results log saved to {log_output_filepath}")
