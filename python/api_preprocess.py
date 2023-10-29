'''
This script contains an example of a preprocessed data Petal Metrics API call.
You will need a valid developer API key to access.

Usage: python api_preprocess.py -k $API_KEY
'''
import argparse
import random

import api


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--api_key', type=str, required=True,
                    help='API key for the Petal Metrics API')
args = parser.parse_args()

random_eeg_data = [
    [random.randint(0,100) / 100 for i in range(150)]
    for num in range(4)
]
calculations = api.request_metrics(
    api_key=args.api_key,
    eeg_data=random_eeg_data,
    metrics=['preprocessed_data'],
)
print(calculations['data'])
