# -*- coding: utf-8 -*-
"""
This script can cut Blackrock files to a certain length

This work is based on python-neo blackrockio and blackrockrawio by:
  * Chris Rodgers
  * Michael Denker
  * Lyuba Zehl
  * Samuel Garcia

and was created by Björn Müller.

This script can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3

It can be used from a terminal by executing "python /path/to/cut_blackrock_files.py {arguments}"
There are two positional arguments that are used together, these are number of samples and sampling rate.
The file is then cut to a number of samples that is given by its own sampling rate and the specified rate.
So e.g. with 100 1000, a file with sampling rate 1000 gets cut to 100 samples, while a file with sampling rate 30000
gets cut to 3000 samples, because then the time span covered by both files is equally long.
They must be specified first if they should be used and are overridden by -b <int>.

All possible options are:
    -b <length in MiB>          : Specifies sizes of largest file
    -a                          : Load all nsX files
    -f </path/to/files>         : Path to input files
    --nsx {number of nsX}       : Specify which nsX to load, multiple ints possible
    --nev                       : Load nev file
    --nsx_path </path/to/file>  : Override path for nsX files
    --nev_path </path/to/file>  : Override path for nev file
    --same_size                 : Create output files that all have the same size

An example for usage is:
python ./cut_blackrock_files.py 1000 30000 -f ./l101210-001 -a --nev_path ./l101210-001-02

An example for cutting to specified length is:
python ./cut_blackrock_files.py -b 10 --same_size -f ./l101210-001 -a

"""

import argparse
import os

import numpy as np


def cut_nsx_variant_a(filenames, nsx_nb, nb_samples=None, sampling_rate=None, length_bytes=None, same_size=False):

    """Loads the values needed to cut nsX files of version 2.1 and calls the function that cuts the files"""

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    dt0 = [
        ('file_id', 'S8'),
        ('label', 'S16'),
        ('period', 'uint32'),
        ('channel_count', 'uint32')]

    nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

    offset_dt0 = np.dtype(dt0).itemsize
    shape = nsx_basic_header['channel_count']
    file_sampling_rate = int(30000 / nsx_basic_header['period'])

    offset_header = offset_dt0 + np.dtype('uint32').itemsize * shape
    if same_size and length_bytes is not None:
        nb_samples = (length_bytes - offset_header) / (2 * shape)
        file_sampling_rate = 1
        sampling_rate = 1
    length = (int((nb_samples * file_sampling_rate) / sampling_rate) + 1) * shape * 2

    cut_file(filename, offset_header + length)


def cut_nsx_variant_b(filenames, nsx_nb, nb_samples=None, sampling_rate=None, length_bytes=None, same_size=False):

    """Loads the values needed to cut nsX files of version 2.2 or 2.3 and calls the function that cuts the files"""

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    bytes_in_headers = np.memmap(filename, dtype='uint32', offset=10, shape=1)[0]
    file_sampling_rate = 30000 / np.memmap(filename, dtype='uint32', offset=286, shape=1)[0]
    channel_count = np.memmap(filename, dtype='int32', offset=310, shape=1)[0]

    offset_header = bytes_in_headers + 9
    shape = channel_count
    if same_size and length_bytes is not None:
        nb_samples = (length_bytes - offset_header) / (2 * shape)
        file_sampling_rate = 1
        sampling_rate = 1

    nb_samples = int((nb_samples * file_sampling_rate) / sampling_rate)

    length = (nb_samples + 1) * shape * 2

    cut_file_nsx_variant_b(filename, offset_header + length, offset_header,
                           nb_samples, channel_count)


def cut_nev(filenames, nb_samples=None, sampling_rate=None, length_bytes=None, same_size=False):

    """Loads the values needed to cut nev files and calls the function that cuts the files"""

    filename = '.'.join([filenames['nev'], 'nev'])

    header = np.memmap(filename, dtype='uint32', shape=3, offset=12)

    offset_header = header[0]
    data_size = header[1]
    file_sampling_rate = header[2]

    if length_bytes is not None and same_size:
        nb_samples = int((length_bytes - offset_header) / data_size)
        i = nb_samples
    else:
        dt0 = [
            ('timestamp', 'uint32'),
            ('packet_id', 'uint16'),
            ('value', 'S{0}'.format(data_size - 6))]

        raw_data = np.memmap(filename, mode='r', offset=offset_header, dtype=dt0)

        i = 0
        try:
            while raw_data[i]['timestamp'] < int((nb_samples * file_sampling_rate) / sampling_rate):
                i += 1  # Because size of all data packets (including Packet ID 0) is the same
        except IndexError:
            raise ValueError("More samples specified than included in file")

    cut_file(filename, i * data_size + offset_header)


def cut_file(filename, total_length):

    """Cuts files down to specified length"""

    if total_length > os.path.getsize(filename):
        raise ValueError('More samples specified than included in file')

    part_to_write = np.memmap(filename, shape=total_length, dtype='uint8')  # uint8 so size is always one byte

    part_to_write.tofile("".join([filename, "_cut"]))

    return part_to_write


def cut_file_nsx_variant_b(filename, total_length, offset_header, nb_samples, channel_nb):

    """Cuts files of version 2.2 or 2.3, also needs to edit number of samples following the header"""

    if total_length > os.path.getsize(filename):
        raise ValueError('More samples specified than included in file')

    file_nb_samples = np.memmap(filename, shape=1, offset=offset_header - 4, dtype='uint32')[0]

    while file_nb_samples <= nb_samples:
        nb_samples -= file_nb_samples
        offset_header += 2 * channel_nb * file_nb_samples + 9  # 9 bytes is length of data packet header
        total_length += 9
        file_nb_samples = np.memmap(filename, shape=1, offset=offset_header-4, dtype='uint32')[0]

    if total_length > os.path.getsize(filename):
        raise ValueError('More samples specified than included in file')

    part_to_write = np.memmap(filename, shape=total_length, dtype='uint8')  # uint8 so size is always one byte

    print(total_length)
    print(filename)
    writer = np.memmap("".join([filename, "_cut"]), shape=total_length, dtype='uint8', mode='w+')
    writer[:] = part_to_write[:]

    nb_samples_to_write = nb_samples + 1
    for i in range(4):
        writer[offset_header + i - 4] = np.uint8(nb_samples_to_write % 256)
        nb_samples_to_write /= 256

    return part_to_write


def extract_nsx_file_spec(filenames, nsx_nb):

    """Returns the file version of the nsX files"""

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    dt0 = [
        ('file_id', 'S8'),
        ('ver_major', 'uint8'),
        ('ver_minor', 'uint8')]

    nsx_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]

    if nsx_file_id['file_id'].decode() == 'NEURALSG':
        spec = '2.1'
    elif nsx_file_id['file_id'].decode() == 'NEURALCD':
        spec = '{0}.{1}'.format(nsx_file_id['ver_major'], nsx_file_id['ver_minor'])
    else:
        raise IOError('Unsupported NSX file type.')

    return spec


def get_nsx_samples(filenames, nsx_nb, spec, length_bytes):

    """Reads the number of samples required for specified length in bytes from nsX files"""

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    if spec == '2.1':
        channel_count = np.memmap(filename, dtype='int32', offset=28, shape=1)[0]
        bytes_in_headers = 11 + channel_count * 4
        sampling_rate = int(30000 / np.memmap(filename, dtype='int32', offset=24, shape=1)[0])
    elif spec in ['2.2', '2.3']:
        bytes_in_headers = np.memmap(filename, dtype='uint32', offset=10, shape=1)[0]
        channel_count = np.memmap(filename, dtype='int32', offset=310, shape=1)[0]
        sampling_rate = np.memmap(filename, dtype='uint32', offset=290, shape=1)[0]

    return int((length_bytes - bytes_in_headers) / (2 * channel_count)), sampling_rate


def get_nev_samples(filenames, length_bytes):

    """Reads the number of samples required for specified length in bytes from nev files"""

    filename = '.'.join([filenames['nev'], 'nev'])
    data_size = np.memmap(filename, dtype='uint32', shape=1, offset=16)[0]
    offset_header = np.memmap(filename, dtype='uint32', shape=1, offset=12)[0]
    nb_samples = int((length_bytes - offset_header) / data_size)
    print(length_bytes)
    print("DS:", data_size)
    print(nb_samples)
    print(offset_header + nb_samples * data_size)
    time_stamp = np.memmap(filename, dtype='uint32', shape=1, offset=offset_header + nb_samples * data_size)[0]
    print(time_stamp)
    return int(time_stamp*1000 / np.memmap(filename, dtype='uint32', shape=1, offset=20)[0]), 1000


# Creating argparse parser with desired arguments
parser = argparse.ArgumentParser()

parser.add_argument("nb_samples", type=int, nargs='?', default=None)
parser.add_argument("sampling_rate", type=int, nargs='?', default=None)
parser.add_argument("-b", "--bytes", type=int, default=None)
parser.add_argument("-a", "--all", action='store_true')
parser.add_argument("-f", "--filenames", type=str, default=None)
parser.add_argument("--nsx", type=str, nargs='*', default=[])
parser.add_argument("--nev", action='store_true')
parser.add_argument("--nsx_path", type=str, default=None)
parser.add_argument("--nev_path", type=str, default=None)
parser.add_argument("--same_size", action='store_true')

args = parser.parse_args()

# Check if arguments are correctly specified where argparse can't check itself
if not args.nev and not args.nsx and not args.all:
    parser.error("You need to specify what you want to cut")

if (args.nb_samples is None or args.sampling_rate is None) and args.bytes is None:
    parser.error("You need to specify the length of the output, either input the number of samples with the "
                 "corresponding sampling rate or use -b <length in Mebibytes>")

# Read the values into local variables
nb_samples = args.nb_samples
sampling_rate = args.sampling_rate
load_nev = args.nev

if args.nsx and args.nsx[0] == 'all':
    nsx_nb = 'all'
else:
    nsx_nb = []
    for nb in args.nsx:
        nsx_nb.append(int(nb))

if args.all:
    nsx_nb = 'all'
    load_nev = True

if args.bytes is not None:
    length_bytes = args.bytes * 1024 * 1024
else:
    length_bytes = None

filenames = {}
if args.filenames is not None:
    filenames['nsx'] = os.path.abspath(args.filenames)
    filenames['nev'] = os.path.abspath(args.filenames)
if args.nsx_path is not None:
    filenames['nsx'] = os.path.abspath(args.nsx_path)
if args.nev_path is not None:
    filenames['nev'] = os.path.abspath(args.nev_path)

# Check if filenames are specified, needs to be done down here because local dictionary filenames must be set first
if not filenames:
    parser.error("No filenames specified, please use -f or --nsx_path and/or --nev_path")
elif nsx_nb and 'nsx' not in filenames:
    parser.error("You need to specify a path for the nsX you want to cut, please use -f or --nsx_path")
elif load_nev and not ('nev' in filenames):
    parser.error("You need to specify a path for the nev you want to cut, please use -f or --nev_path")

# Finding available nsX
if nsx_nb == 'all':
    nsx_nb = []
    for i in range(1, 7):
        filename = ".".join([filenames['nsx'], 'ns%i' % i])
        if os.path.exists(filename):
            nsx_nb.append(i)

# Check if nsX or nev has higher file size
nsx_is_largest = False

for i in nsx_nb:
    if load_nev and os.path.getsize(".".join([filenames['nev'], 'nev'])) < os.path.getsize(
            '.'.join([filenames['nsx'], 'ns%i' % i])):
        nsx_is_largest = True
        break
    elif not load_nev:
        nsx_is_largest = True

# Choose wanted number of samples from largest file so the single files don't become larger than length_bytes
if length_bytes is not None and not args.same_size:
    if nsx_is_largest:
        nb_samples, sampling_rate = get_nsx_samples(filenames, max(nsx_nb),
                                                    extract_nsx_file_spec(filenames, max(nsx_nb)), length_bytes)
        print(nb_samples)
    else:
        nb_samples, sampling_rate = get_nev_samples(filenames, length_bytes)
        print(nb_samples)

# Load nsX with different routine depending on file version
for i in nsx_nb:
    spec = extract_nsx_file_spec(filenames, i)
    if spec == '2.1':
        cut_nsx_variant_a(filenames, i, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
                          same_size=args.same_size)
    elif spec in ['2.2', '2.3']:
        cut_nsx_variant_b(filenames, i, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
                          same_size=args.same_size)
# Load nev if wanted
if load_nev:
    cut_nev(filenames, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
            same_size=args.same_size)

# TODO: same times from each file (???)
