import time
import os
import csv
from kmer import step_log, time_log, __author__, __date__

def log_step(message):
    print(message)
    step_log.append(message)
    with open('logs/logs.log', 'a') as log_file:
        log_file.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}\n')

def save_time_log(filepath):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a') as time_file:
        writer = csv.writer(time_file, delimiter='\t')
        if not file_exists:
            headers = ['Genome', 'k', 'vector_secs']
            if len(time_log[0]) > 3:
                headers.append('sec2vec_secs')
            writer.writerow(headers)
        writer.writerows(time_log)
    log_step(f"Time log saved to {filepath}")

def log_results(files, best_performance_file, best_k, k_range):
    log_output_filepath = os.path.join('logs', f'{__date__}.log')
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
        log_file.write(f"The best performance was achieved with file {best_performance_file} using {best_k}-mer analysis.\n\n")
        for filepath in files:
            log_file.write(f"file: {filepath}\n")
            log_file.write(f"size: {os.path.getsize(filepath)} bytes\n")
            if k_range.start == k_range.stop - 1:
                log_file.write(f"kmer {k_range.start}\n")
            else:
                log_file.write(f"kmer {k_range.start}-{k_range.stop - 1}\n")
            log_file.write(f"The best performance using this file was through {best_k}-mer analysis.\n\n")

    log_step(f"Results log saved to {log_output_filepath}")
