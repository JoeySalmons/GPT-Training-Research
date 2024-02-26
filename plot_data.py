"""
5/22/23
Plot the data from the training process
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import datetime
import yaml


def get_date_from_string(string):
    # string is of the form "20230522_094831"
    # return a datetime object
    return datetime.datetime.strptime(string, "%Y%m%d_%H%M%S")


def get_most_recent_folder(folder):
    # folder is the path to the folder containing the wandb log folders
    # return the path to the most recent folder

    # get all the folders
    folders = os.listdir(folder)

    # filter out the ones that aren't folders
    folders = [f for f in folders if os.path.isdir(os.path.join(folder, f))]

    # filter out the ones that don't start with "run-"
    folders = [f for f in folders if f.startswith("run-")]

    # get the dates
    dates = [get_date_from_string(re.findall(r"\d{8}_\d{6}", f)[0]) for f in folders]

    # find the most recent date
    most_recent_date = max(dates)

    # get the index of the most recent date
    index = dates.index(most_recent_date)

    # return the path to the most recent folder and the folder name
    return os.path.join(folder, folders[index]), folders[index]


def read_config_file(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data


def read_output_file(file_path):
    with open(file_path, 'r') as file:
        output_data = file.read()
    return output_data


def store_information(folder):
    most_recent_folder, name = get_most_recent_folder(folder)
    # Now go into the "files" folder
    most_recent_folder = os.path.join(most_recent_folder, 'files')
    config_file_path = os.path.join(most_recent_folder, 'config.yaml')
    output_file_path = os.path.join(most_recent_folder, 'output.log')

    config_data = read_config_file(config_file_path)
    output_data = read_output_file(output_file_path)

    # Process the output data
    # Split the output data by lines
    output_data = output_data.split('\n')
    # Only keep the lines that start with "iter"
    output_data = [line for line in output_data if line.startswith("iter")]
    # Split each line by spaces
    output_data = [line.split() for line in output_data]

    # Extract iter, loss, time, and mfu for each element in output_data
    output_info = []
    for line in output_data:
        info = {
            'iter': int(line[1].strip(':')),
            'loss': float(line[3].rstrip(',')),
            'time': float(line[5].rstrip('ms,')),
            'mfu': float(line[7].rstrip('%'))
        }
        output_info.append(info)

    # Store the information in a dictionary

    data = {
        'config': config_data,
        'output': output_info
    }
    return data


def plot_stats(data, config_dict, cutoff, folder_name, wandb_log_folder, log_dict=None, window_size=20, runtime=0):
    # extract the data from data
    iters = [info['iter'] for info in data['output']]
    losses = [info['loss'] for info in data['output']]
    time = [info['time'] for info in data['output']]
    mfu = [info['mfu'] for info in data['output']]
    # set first mfu to equal second mfu
    mfu[0] = mfu[1]
    batch_size = data['config']['batch_size']['value']
    block_size = data['config']['block_size']['value']
    learning_rate = data['config']['learning_rate']['value']
    num_samples = data['config']['num_samples']['value']
    # 3 sig figs min_lr
    min_lr = round(data['config']['min_lr']['value'], 2 - int(np.floor(np.log10(abs(data['config']['min_lr']['value'])))))
    max_iters = data['config']['max_iters']['value']
    # 3 sig figs runtime in minutes
    runtime = round(runtime / 60, 2 - int(np.floor(np.log10(abs(runtime / 60)))))

    # extract run info from config_dict
    sampling_temp = config_dict['temperature']

    # extract the data from log_dict
    # dict_keys(['run_name', 'iter', 'loss', 'lr', 'mfu', 'acc_iter', 'accuracy', 'frac_skipped', 'frac_off_by_one', 'median_distance', 'mean_distance', 'variance_distance'])
    if log_dict is not None:
        run_name = log_dict['run_name']  # for plot title
        # this first set should be identical to the data from data['output'] except for learning rate
        iters = log_dict['iter']
        losses = log_dict['loss']
        learning_rates = log_dict['lr']
        mfu = log_dict['mfu']
        # this second set is taken at different intervals (acc_iter) and contains different information (from testing the model)
        acc_iters = log_dict['acc_iter']
        accuracies = log_dict['accuracy']
        frac_skipped = log_dict['frac_skipped']
        frac_off_by_one = log_dict['frac_off_by_one']
        median_distance = log_dict['median_distance']
        mean_distance = log_dict['mean_distance']
        variance_distance = log_dict['variance_distance']
        # accuracy, etc. are logged for each acc_iter, so we can plot them directly

    # Plot the losses
    # if the window size is greater than the number of iterations, set the window size to the number of iterations
    if window_size > len(iters):
        window_size = len(iters)
    # if we have a lot of iters, increase the window size
    if len(iters) > 1000:
        window_size = len(iters) // 50
    avg_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(18, 12))

    # Plot average losses and raw losses
    plt.subplot(2, 1, 1)
    plt.plot(iters[window_size - 1:], avg_losses, label='Moving Average')
    plt.plot(iters, losses, alpha=0.5, label='Raw Data')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # limit y axis to the range of the data up to 3 if any loss is less than 3, otherwise just use the min and max
    # max_loss = 3
    # min_loss = min(losses)
    # if min_loss > 3:
    #     max_loss = max(losses)
    # plt.ylim(bottom=min_loss, top=max_loss)
    plt.legend()
    plt.grid(True, which='both')

    # Add text information
    plt.text(0.45, 0.95, f"Batch Size: {batch_size}", transform=plt.gca().transAxes)
    plt.text(0.45, 0.9, f"Block Size: {block_size}", transform=plt.gca().transAxes)
    plt.text(0.45, 0.85, f"Initial LR: {learning_rate}", transform=plt.gca().transAxes)
    plt.text(0.45, 0.8, f"Min LR: {min_lr}", transform=plt.gca().transAxes)
    plt.text(0.45, 0.75, f"Max Iters: {max_iters}", transform=plt.gca().transAxes)
    plt.text(0.45, 0.7, f"Cutoff: {cutoff} iter", transform=plt.gca().transAxes)

    plt.text(0.65, 0.95, f"Runtime: {runtime} min", transform=plt.gca().transAxes)
    plt.text(0.65, 0.9, f"Window Size: {window_size} (moving avg)", transform=plt.gca().transAxes)
    plt.text(0.65, 0.85, f"Last Avg Loss: {round(avg_losses[-1], 4)}", transform=plt.gca().transAxes)
    plt.text(0.65, 0.8, f"Saved to: {folder_name}", transform=plt.gca().transAxes)
    plt.text(0.65, 0.75, f"Samples per eval: {num_samples}", transform=plt.gca().transAxes)
    plt.text(0.65, 0.7, f"Sampling Temp: {sampling_temp}", transform=plt.gca().transAxes)

    if log_dict is not None:
        plt.title(f"Loss vs Iteration ({run_name})")
        # Plot accuracies
        plt.subplot(2, 2, 3)
        plt.plot(acc_iters, accuracies, label='Accuracy')
        plt.plot(acc_iters, frac_skipped, label='Fraction Skipped')
        plt.plot(acc_iters, frac_off_by_one, label='Fraction Off By One')
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        # y axis is percentage, from 0 to 100, so limit it to that
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, which='both')

        # Plot the rest of the stats
        plt.subplot(2, 2, 4)
        plt.plot(acc_iters, median_distance, label='Median Distance')
        plt.plot(acc_iters, mean_distance, label='Mean Distance')
        plt.plot(acc_iters, variance_distance, label='Variance Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Stat')
        plt.legend()
        plt.grid(True, which='both')
    else:
        plt.title(f"Loss vs Iteration")
        # Plot time
        plt.subplot(2, 2, 3)
        plt.plot(iters, time, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Time (ms)')
        plt.title('Time')

        # Plot mfu
        plt.subplot(2, 2, 4)
        plt.plot(iters, mfu, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('MFU (%)')
        plt.title('MFU')

    plt.tight_layout()

    # name file with folder_name
    file_name = folder_name + '_stats.png'
    # save to plots folder
    plt.savefig('plots/' + file_name, dpi=300)
    print(f"Saved plot to {os.getcwd()}/plots")
    # save to wandb\folder_name
    plt.savefig(os.path.join(wandb_log_folder, folder_name, file_name), dpi=300)
    print(f"Saved plot to {os.path.join(wandb_log_folder, folder_name)}")
    # plt.show()


def save_plots(log_stats_dict, config_dict):
    # log_stats_dict is a dictionary with:
    # run_name, iter, loss, lr, mfu, acc_iter, accuracy, frac_skipped, frac_off_by_one, median_distance, mean_distance, variance_distance
    # where the values for iter, loss, lr, and mfu are logged at a different frequency than the rest

    # load from wandb log folder
    # wandb_log_folder = "D:\\Documents\\WSL\\Arch\\llm-training\\wandb"
    # wandb_log_folder = "/mnt/d/Documents/WSL/Arch/llm-training/wandb"
    # or just get current working directory + wandb
    wandb_log_folder = os.getcwd() + "/wandb"

    # Usage
    data = store_information(wandb_log_folder)

    # also read a JSON file in order to get the runtime (the file is wandb-summary.JSON, the runtime is in seconds)
    most_recent_folder, folder_name = get_most_recent_folder(wandb_log_folder)
    print(most_recent_folder)
    # read the file
    wandb_summary_file = os.path.join(most_recent_folder, 'files/wandb-summary.json')
    with open(wandb_summary_file, 'r') as file:
        summary_data = file.read()
    # convert to a dictionary
    summary_data = eval(summary_data)
    # get the runtime
    runtime = summary_data['_runtime']

    cutoff = 000
    plot_stats(data, config_dict, cutoff, folder_name, wandb_log_folder, log_dict=log_stats_dict, runtime=runtime)
