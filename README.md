# ED-sKWS: Early-Decision Spiking Neural Networks for Rapid,and Energy-Efficient Keyword Spotting
This is the official implementation of ED-sKWS(https://arxiv.org/abs/2406.12726). This resportory is built from SpArch (https://github.com/idiap/sparch), please see the SpArch for installation.


## Installation

    git clone https://github.com/idiap/sparch.git
    cd sparch
    pip install -r requirements.txt
    python setup.py install

## Run experiments

All experiments on the speech command recognition datasets can be run from the `run_exp.py` script. The experiment configuration can be specified using parser arguments. Run the command `python run_exp.py -h` to get the descriptions of all possible options. For instance, if you want to run a new SNN experiment with adLIF neurons on the SC dataset,

    python run_exp.py --model_type adLIF --dataset_name sc \
        --data_folder <PATH-TO-DATASET-FOLDER> --new_exp_folder <OUTPUT-PATH>

You can also continue training from a checkpoint

    python run_exp.py --use_pretrained_model 1 --load_exp_folder <OUTPUT-PATH> \
        --dataset_name sc --data_folder <PATH-TO-DATASET-FOLDER> \
        --start_epoch <LAST-EPOCH-OF-PREVIOUS-TRAINING>

## Run experiments on SC-100 dataset

The script for SC-100 dataset(coming soon) can be run from 'run_exp_100words.py', and run as the other speech recognition dataset.

    python run_exp_100words.py --model_type adLIF \
        --data_folder <PATH-TO-DATASET-FOLDER> --new_exp_folder <OUTPUT-PATH>

You can also continue training from a checkpoint

    python run_exp_100words.py --use_pretrained_model 1 --load_exp_folder <OUTPUT-PATH> \
        --dataset_name sc --data_folder <PATH-TO-DATASET-FOLDER> \
        --start_epoch <LAST-EPOCH-OF-PREVIOUS-TRAINING>
