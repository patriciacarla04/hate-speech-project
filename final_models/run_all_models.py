import subprocess


# Define the command for each model training
commands = [
    ['python', 'xlmr_multilingual.py', '--train_data', 'multi_duplicate_all.csv', '--test_data_en', 'EN_test.csv', '--test_data_it', 'IT_test.csv', '--test_data_slo','SLO_test.csv', '--output_dir', 'xlmr_duplicate_all', '--results_dir', 'results_duplicate_all', '--batch_size', '32', '--max_len','128','--epochs','1'],
    ['python', 'xlmr_multilingual.py', '--train_data', 'multi_duplicate_disagreement.csv', '--test_data_en', 'EN_test.csv', '--test_data_it', 'IT_test.csv', '--test_data_slo','SLO_test.csv', '--output_dir', 'xlmr_duplicate_disagreement', '--results_dir', 'results_duplicate_disagreement', '--batch_size', '32', '--max_len','128','--epochs','1'],
    ['python', 'xlmr_multilingual.py', '--train_data', 'multi_remove_disagreement.csv', '--test_data_en', 'EN_test.csv', '--test_data_it', 'IT_test.csv', '--test_data_slo','SLO_test.csv', '--output_dir', 'xlmr_remove_disagreement', '--results_dir', 'results_remove_disagreement', '--batch_size', '32', '--max_len','128','--epochs','1'],
    ['python', 'xlmr_disagreement_class.py', '--train_data', 'multi_disagreement_class.csv', '--test_data', 'multi_disagreement_class_test_test.csv', '--output_dir', 'xlmr_disagreement_class', '--results_dir', 'results_disagreement_class', '--batch_size', '32', '--max_len','128','--epochs','1']
]

# List to hold the processes
processes = []

# Start each training process
for command in commands:
    p = subprocess.Popen(command)
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.wait()