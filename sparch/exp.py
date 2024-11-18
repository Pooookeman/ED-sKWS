#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""
import errno
import logging
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


from sparch.dataloaders.nonspiking_datasets import load_hd_or_sc
from sparch.dataloaders.spiking_datasets import load_shd_or_ssc
from sparch.dataloaders.words100_dataset import load_100words
from sparch.models.anns import ANN
from sparch.models.snns import SNN
from sparch.parsers.model_config import print_model_options
from sparch.parsers.training_config import print_training_options

logger = logging.getLogger(__name__)


def ES_entropy_acc(output, y):
    num_classes = torch.tensor(output.shape[-1])

    pi_logits = F.softmax(output, dim=-1)
    entropy = -(1/torch.log(num_classes))  * (pi_logits * torch.log(pi_logits)).sum(dim=-1)

    timesteps = torch.argmin(entropy, dim=0)

    selected_features = torch.empty((output.shape[1], output.shape[2])).cuda()
    for b in range(output.shape[1]):
        selected_features[b] = output[timesteps[b], b]
    pred = torch.argmax(selected_features, dim=1)

    acc = np.mean((y == pred).detach().cpu().numpy())  
    return acc, timesteps.float().mean()

def ES_alpha_acc(output, y, alpha=0.9):
    p = F.softmax(output, dim=-1)
    B, D = p.shape[1], p.shape[2]

    stop_timestep = torch.full((B,), p.shape[0]-1).cuda()
    saved_features = p[-1, ...]

    saved_mask = torch.zeros(B, dtype=torch.bool).cuda()

    for t in range(p.shape[0]):
        logits_t = p[t]

        condition = (logits_t > alpha).any(dim=1) & ~saved_mask

        stop_timestep[condition] = t
        saved_features[condition] = logits_t[condition]

        saved_mask |= condition

        if saved_mask.all():
            break
    
    pred = torch.argmax(saved_features, dim=1)
    acc = np.mean((y == pred).detach().cpu().numpy()) 
    return acc, stop_timestep
        

def TemporalLoss(output, y, criterion):
    Loss_es = 0
    Loss_es += criterion(torch.mean(output, dim=0), y)
    return Loss_es

def TETLoss(output, y, criterion, lamb=1e-3):
    T = output.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(output[t, ...], y)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(output).fill_(1.0)
        Loss_mmd = MMDLoss(output, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total

def V2Loss(output, y, criterion):
    T = output.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(torch.mean(output[:(t+1), ...], dim=0), y)
    Loss_es = Loss_es / T # L_TET
    return Loss_es

def V2Loss_window(output, y, criterion):
    window = 20
    T = output.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(torch.mean(output[max(0,t-window):(t+1), ...], dim=0), y)
    Loss_es = Loss_es / T # L_TET
    return Loss_es

def calculate_acc(output, y, alpha_list = [0.9, 0.95, 0.99]):
    acc_per_time = []
    acc_acum_time = []
    alpha_results_list = []
    for t in range(output.size(0)):
        pred = torch.argmax(output[t], dim=1)
        acc = np.mean((y == pred).detach().cpu().numpy())  
        acc_per_time.append(acc)

        mean_out = output[:(t+1)].mean(0)
        pred = torch.argmax(mean_out, dim=1)
        acc = np.mean((y == pred).detach().cpu().numpy())  
        acc_acum_time.append(acc)
    entropy_results = ES_entropy_acc(output, y)
    for alpha in alpha_list:
        alpha_results_list.append(ES_alpha_acc(output, y, alpha))
    return acc_per_time, acc_acum_time, entropy_results, alpha_results_list

class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):

        # New model config
        self.model_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias
        self.bidirectional = args.bidirectional

        # Training config
        self.use_pretrained_model = args.use_pretrained_model
        self.only_do_testing = args.only_do_testing
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.log_tofile = args.log_tofile
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.use_regularizers = args.use_regularizers
        self.reg_factor = args.reg_factor
        self.reg_fmin = args.reg_fmin
        self.reg_fmax = args.reg_fmax
        self.use_augm = args.use_augm
        self.exp_name = args.exp_name

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()
        print_model_options(args)
        print_training_options(args)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if not self.only_do_testing:

            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                self.train_one_epoch(e)
                best_epoch, best_acc = self.valid_one_epoch(e, best_epoch, best_acc)

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                self.net = torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                )
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        if self.dataset_name in ["sc", "ssc"]:
            self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info(
                "\nThis dataset uses the same split for validation and testing.\n"
            )

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_bdir" if self.bidirectional else "_udir"
            outname += "_reg" if self.use_regularizers else "_noreg"
            outname += "_lr" + str(self.lr)
            exp_folder = os.path.join("exp", self.exp_name, outname.replace(".", "_"))
        # For a new model check that out path does not exist
        # if not self.use_pretrained_model and os.path.exists(exp_folder):
        #     raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        # For the spiking datasets
        if self.dataset_name in ["shd", "ssc"]:

            self.nb_inputs = 700
            self.nb_outputs = 20 if self.dataset_name == "shd" else 35

            self.train_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=True,
            )
            self.valid_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=False,
            )
            if self.dataset_name == "ssc":
                self.test_loader = load_shd_or_ssc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    nb_steps=100,
                    shuffle=False,
                )
            if self.use_augm:
                logging.warning(
                    "\nWarning: Data augmentation not implemented for SHD and SSC.\n"
                )

        # For the non-spiking datasets
        elif self.dataset_name in ["hd", "sc"]:

            self.nb_inputs = 40
            self.nb_outputs = 20 if self.dataset_name == "hd" else 35

            self.train_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=True,
            )
            self.valid_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=False,
            )
            if self.dataset_name == "sc":
                self.test_loader = load_hd_or_sc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    use_augm=self.use_augm,
                    shuffle=False,
                )
            if self.use_augm:
                logging.info("\nData augmentation is used\n")
        else:
            raise ValueError(f"Invalid dataset name {self.dataset_name}")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        input_shape = (self.batch_size, None, self.nb_inputs)
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=self.device)
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif self.model_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:

            self.net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

            self.net = ANN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                ann_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")

        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        self.net.train()

        alpha_list = [0.9, 0.95, 0.99]
        losses, accs1, accs2 = [], [], []
        entropy_acc= []
        entropy_ave_timestep = 0
        alpha_acc_list = [0 for _ in range(len(alpha_list))]
        alpha_ave_timestep_list = [0 for _ in range(len(alpha_list))]
        epoch_spike_rate = 0

        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output, firing_rates, temporal_firing_rate = self.net(x)

            # Compute loss
            # loss_val = self.loss_fn(output.mean(dim=0), y)
            # loss_val = self.loss_fn(output[-1,:,:], y) #cumulative softmax
            loss_val = TETLoss(output, y, self.loss_fn)
            # loss_val = V2Loss(output, y, self.loss_fn)
            # loss_val = V2Loss_window(output, y, self.loss_fn)
            losses.append(loss_val.item())

            # Spike activity
            if self.net.is_snn:
                epoch_spike_rate += torch.mean(firing_rates)

                if self.use_regularizers:
                    reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                    loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

            # Compute accuracy with labels
            acc_per_time, acc_acum_time, entropy_results, alpha_results_list=calculate_acc(output, y, alpha_list)
            
            accs1.append(acc_per_time)
            accs2.append(acc_acum_time)
            entropy_acc.append(entropy_results[0])
            for idx in range(len(alpha_results_list)):
                alpha_acc_list[idx] += alpha_results_list[idx][0]
                alpha_ave_timestep_list[idx] += alpha_results_list[idx][1]
            entropy_ave_timestep += entropy_results[1]

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs2)
        logging.info(f"Epoch {e}: train acc2={train_acc}")

        train_acc = np.mean(accs1)
        logging.info(f"Epoch {e}: train acc1={train_acc}")
        
        train_acc = np.mean(entropy_acc)
        logging.info(f"Epoch {e}: train entropy acc={train_acc} average stop time {entropy_ave_timestep/step}")

        for idx in range(len(alpha_list)):
            logging.info(f"Epoch {e}: train alpha acc={alpha_acc_list[idx]/step} average stop time {alpha_ave_timestep_list[idx]/step} for alpha {alpha_list[idx]}")

        # Train spike activity of whole epoch
        if self.net.is_snn:
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: train mean act rate={epoch_spike_rate}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: train elapsed time={elapsed}")

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            alpha_list = [0.9, 0.95, 0.99]
            losses, accs1, accs2 = [], [], []
            epoch_spike_rate = 0
            early_stop_spike_rate = 0
            entropy_acc = []
            entropy_ave_timestep = 0
            alpha_acc_list = [0 for _ in range(len(alpha_list))]
            alpha_ave_timestep_list = [0 for _ in range(len(alpha_list))]

            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(self.valid_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates, temporal_firing_rate = self.net(x)

                # Compute loss
                # loss_val = self.loss_fn(output[-1,:,:], y)
                loss_val = TETLoss(output, y, self.loss_fn)
                # loss_val = V2Loss(output, y, self.loss_fn)
                # loss_val = V2Loss_window(output, y, self.loss_fn)
                losses.append(loss_val.item())

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)
                # Compute accuracy with labels
                acc_per_time, acc_acum_time, entropy_results, alpha_results_list=calculate_acc(output, y, alpha_list)
            
                accs1.append(acc_per_time)
                accs2.append(acc_acum_time)
                entropy_acc.append(entropy_results[0])
                for idx in range(len(alpha_results_list)):
                    alpha_acc_list[idx] += alpha_results_list[idx][0]
                    alpha_ave_timestep_list[idx] += alpha_results_list[idx][1].float().mean()
                entropy_ave_timestep += entropy_results[1]


            # Train loss of whole epoch
            train_loss = np.mean(losses)
            logging.info(f"Epoch {e}: train loss={train_loss}")

            # Validation accuracy of whole epoch
            valid_acc = np.mean(accs2)
            logging.info(f"Epoch {e}: valid acc2={valid_acc}")
            acc2_path = os.path.join(self.log_dir, "valid_acc2_"+str(e)+".txt")
            np.savetxt(acc2_path, np.mean(accs2,0))
            
            valid_acc = np.mean(accs1)
            acc_last_step = np.mean(accs1, axis=0)[-1]
            logging.info(f"Epoch {e}: valid acc1={valid_acc}")
            acc1_path = os.path.join(self.log_dir, "valid_acc1_"+str(e)+".txt")
            np.savetxt(acc1_path, np.mean(accs1,0))

            valid_acc = np.mean(entropy_acc)
            logging.info(f"Epoch {e}: valid entropy acc={valid_acc} average stop time {entropy_ave_timestep/step}")

            for idx in range(len(alpha_list)):
                logging.info(f"Epoch {e}: train alpha acc={alpha_acc_list[idx]/step} average stop time {alpha_ave_timestep_list[idx]/step} for alpha {alpha_list[idx]}")

            # Update learning rate
            self.scheduler.step(acc_last_step)

            # Update best epoch and accuracy
            if acc_last_step > best_acc:
                best_acc = acc_last_step
                best_epoch = e

                # Save best model
                if self.save_best:
                    torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                    logging.info(f"\nBest model saved with valid acc={acc_last_step}")

            logging.info("\n-----------------------------\n")

            return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            alpha_list = [0.9, 0.95, 0.99]
            losses, accs1, accs2 = [], [], []
            epoch_spike_rate = 0
            early_stop_spike_rate = 0
            entropy_acc = []
            entropy_ave_timestep = 0
            alpha_acc_list = [0 for _ in range(len(alpha_list))]
            alpha_ave_timestep_list = [0 for _ in range(len(alpha_list))]


            logging.info("\n------ Begin Testing ------\n")

            # Loop over batches from test set
            for step, (x, raw_audio, y) in enumerate(test_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates, temporal_firing_rate  = self.net(x)

                # Compute loss
                # loss_val = self.loss_fn(output[-1,:,:], y)
                loss_val = TETLoss(output, y, self.loss_fn)
                # loss_val = V2Loss(output, y, self.loss_fn)
                # loss_val = V2Loss_window(output, y, self.loss_fn)
                losses.append(loss_val.item())
                
                # Compute accuracy with labels
                acc_per_time, acc_acum_time, entropy_results, alpha_results_list=calculate_acc(output, y, alpha_list)
            
                accs1.append(acc_per_time)
                accs2.append(acc_acum_time)
                entropy_acc.append(entropy_results[0])
                for idx in range(len(alpha_results_list)):
                    alpha_acc_list[idx] += alpha_results_list[idx][0]
                    alpha_ave_timestep_list[idx] += alpha_results_list[idx][1].float().mean()
                entropy_ave_timestep += entropy_results[1]
                
                ES_timestep = alpha_results_list[idx][1]
                
                for idx, t in enumerate(ES_timestep): 
                    early_stop_spike_rate += temporal_firing_rate[idx, :t].sum()/torch.numel(temporal_firing_rate)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)

            # Test loss
            test_loss = np.mean(losses)
            logging.info(f"Test loss={test_loss}")

            # Test accuracy
            test_acc = np.mean(accs2)
            logging.info(f"Test acc2={test_acc}")
            acc2_path = os.path.join(self.log_dir, "test_acc2.txt")
            np.savetxt(acc2_path, np.mean(accs2,0))

            test_acc = np.mean(accs1)
            logging.info(f"Test acc1={test_acc}")
            acc1_path = os.path.join(self.log_dir, "test_acc1.txt")
            np.savetxt(acc1_path, np.mean(accs1,0))

            
            test_acc = np.mean(entropy_acc)
            logging.info(f" Test entropy acc={test_acc} average stop time {entropy_ave_timestep/step}")

            for idx in range(len(alpha_list)):
                logging.info(f"Test alpha acc={alpha_acc_list[idx]/step} average stop time {alpha_ave_timestep_list[idx]/step} for alpha {alpha_list[idx]}")


            # Test spike activity
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Test mean act rate={epoch_spike_rate}")

            logging.info(f"Test mean early stop act rate={early_stop_spike_rate/step}")

            logging.info("\n-----------------------------\n")
