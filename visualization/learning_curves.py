import argparse
import os
import glob
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import seaborn as sns


def find_logfile(log_dir):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        print('No logfile specified and %s path did not exist.' % log_dir)
        return None

    try:
        logfile = max(glob.iglob(os.path.join(log_dir, '*.log')), key=os.path.getctime)
    except ValueError:
        logfile = None
    if logfile is None:
        print('Unable to find *.log file in %s directory. Please specify a weight file manually.\n' % log_dir)
        return None
    return logfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', nargs='?', help='The log file to use.')
    parser.add_argument('-c', '--combined', default=False, action='store_true', help='Show combined loss.')
    args = parser.parse_args()

    show_combined = args.combined

    logfile = args.logfile
    if logfile is None:
        logfile = find_logfile('log')
    if logfile is None:
        parser.print_usage()
        exit(1)

    re_iteration_loss = re.compile(
        '^I(?P<month>\d{2})(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<seconds>\d{2}\.\d+)\s*?.+ Iteration (?P<iteration>\d+), loss = (?P<loss>[+-]?\d+(\.\d+)?(e[+-]\d+)?)$',
        re.MULTILINE)

    re_loss_output = re.compile(
        '^I(?P<month>\d{2})(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<seconds>\d{2}\.\d+)\s*?.+ Train net output #(?P<num>\d+): (?P<name>.+?) = .+? \(.+? = (?P<loss>[+-]?\d+(\.\d+)?(e[+-]\d+)?) loss\)$',
        re.MULTILINE)

    re_loss_test = re.compile(
        '^I(?P<month>\d{2})(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<seconds>\d{2}\.\d+)\s*?.+ Test net output #(?P<num>\d+): (?P<name>.+?) = .+? \(.+? = (?P<loss>[+-]?\d+(\.\d+)?(e[+-]\d+)?) loss\)$',
        re.MULTILINE)

    re_learning_rate = re.compile(
        '^I(?P<month>\d{2})(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<seconds>\d{2}\.\d+)\s*?.+ Iteration (?P<iteration>\d+), lr = (?P<lr>[+-]?\d+(\.\d+)?(e[+-]\d+)?)$',
        re.MULTILINE)

    loss_value = []
    loss_iteration = []
    loss_time = []

    losses = {}
    validation_losses = {}

    lr_value = []
    lr_iteration = []

    last_iteration = 0
    assert logfile is not None
    with open(logfile, 'r') as f:
        for line in f:
            match = re_iteration_loss.search(line)
            if match is not None:
                month, day, hour, minute, second = match.group('month', 'day', 'hour', 'minute', 'seconds')
                time = datetime.time(int(hour), int(minute), int(float(second)), int((float(second) - int(float(second)))*1000000))

                iteration, loss = match.group('iteration', 'loss')

                loss_value.append(float(loss))
                loss_iteration.append(float(iteration))
                loss_time.append(time)

                last_iteration = float(iteration)

            match = re_loss_test.search(line)
            if match is not None:
                month, day, hour, minute, second = match.group('month', 'day', 'hour', 'minute', 'seconds')
                time = datetime.time(int(hour), int(minute), int(float(second)),
                                     int((float(second) - int(float(second))) * 1000000))

                num, name, loss = match.group('num', 'name', 'loss')

                if name not in validation_losses:
                    validation_losses[name] = dict(iteration=[], time=[], value=[])

                validation_losses[name]['iteration'].append(last_iteration)
                validation_losses[name]['time'].append(time)
                validation_losses[name]['value'].append(loss)

            match = re_loss_output.search(line)
            if match is not None:
                month, day, hour, minute, second = match.group('month', 'day', 'hour', 'minute', 'seconds')
                time = datetime.time(int(hour), int(minute), int(float(second)),
                                     int((float(second) - int(float(second))) * 1000000))

                num, name, loss = match.group('num', 'name', 'loss')

                if name not in losses:
                    losses[name] = dict(iteration=[], time=[], value=[])

                losses[name]['iteration'].append(last_iteration)
                losses[name]['time'].append(time)
                losses[name]['value'].append(loss)

            match = re_learning_rate.search(line)
            if match is not None:
                iteration, lr = match.group('iteration', 'lr')

                lr_value.append(float(lr))
                lr_iteration.append(float(iteration))

                last_iteration = float(iteration)

    if len(lr_iteration) > 1:
        lr_value.append(lr_value[len(lr_value)-1])
        lr_iteration.append(last_iteration)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    ax1.set_title('Training error')
    ax2.set_title('Validation error')
    ax3.set_title('Learning rate')
    ax3.set_ylim([0.1 * np.min(lr_value), 10 * np.max(lr_value)])
    ax3.set_xlabel('iteration')

    for name in losses:
        ax1.plot(losses[name]['iteration'], losses[name]['value'], linestyle='-', marker='.', label=name)
    if show_combined:
        ax1.plot(loss_iteration, loss_value, linestyle=':', marker='.', label='combined loss')

    for name in validation_losses:
        ax2.plot(validation_losses[name]['iteration'], validation_losses[name]['value'], linestyle='-', marker='.', label=name)

    ax3.semilogy(lr_iteration, lr_value, label='learning rate')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
