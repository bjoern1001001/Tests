# -*- coding: utf-8 -*-
"""
Script for cutting files in the Blackrock raw format.

This work is based on python-neo blackrockio and blackrockrawio by:
  * Chris Rodgers
  * Michael Denker
  * Lyuba Zehl
  * Samuel Garcia

This script can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3
"""

import argparse
import os

import numpy as np


def cut_nsx_variant_a(filenames, nsx_nb, nb_samples=None, sampling_rate=None, length_bytes=None, split=False):

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    dt0 = [
        ('file_id', 'S8'),
        ('label', 'S16'),
        ('period', 'uint32'),
        ('channel_count', 'uint32')]

    nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

    offset_dt0 = np.dtype(dt0).itemsize
    shape = nsx_basic_header['channel_count']
    file_sampling_rate = int(30000/nsx_basic_header['period'])

    offset_header = offset_dt0 + np.dtype('uint32').itemsize * shape
    if split and length_bytes is not None:
        nb_samples = (length_bytes - offset_header) / (2 * shape)
        file_sampling_rate = 1
        sampling_rate = 1
    length = (int((nb_samples*file_sampling_rate)/sampling_rate)+1)*shape*2

    cut_file(filename, offset_header + length)


def cut_nsx_variant_b(filenames, nsx_nb, nb_samples=None, sampling_rate=None, length_bytes=None, split=False):

        filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

        bytes_in_headers = np.memmap(filename, dtype='uint32', offset=10, shape=1)[0]
        file_sampling_rate = 30000/np.memmap(filename, dtype='uint32', offset=286, shape=1)[0]
        channel_count = np.memmap(filename, dtype='int32', offset=310, shape=1)[0]

        offset_header = bytes_in_headers + 9
        shape = channel_count
        if split and length_bytes is not None:
            nb_samples = (length_bytes - offset_header)/(2*shape)
            file_sampling_rate = 1
            sampling_rate = 1

        nb_samples = int((nb_samples*file_sampling_rate)/sampling_rate)
        length = (nb_samples+1)*shape*2

        cut_file_nsx_variant_b(filename, offset_header + length, offset_header,
                               nb_samples, channel_count)


def cut_nev(filenames, nb_samples=None, sampling_rate=None, length_bytes=None, split=False):

    filename = '.'.join([filenames['nev'], 'nev'])

    header = np.memmap(filename, dtype='uint32', shape=3, offset=12)

    offset_header = header[0]
    data_size = header[1]
    file_sampling_rate = header[2]

    if length_bytes is not None and split:
        nb_samples = int((length_bytes-offset_header)/data_size)
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
                i += 1                               # Because size of all data packets (including Packet ID 0) is the same
        except IndexError:
            raise ValueError("More samples specified than included in file")

    cut_file(filename, i * data_size + offset_header)


def cut_file(filename, total_length):

    if total_length > os.path.getsize(filename):
        raise ValueError('More samples specified than included in file')

    part_to_write = np.memmap(filename, shape=total_length, dtype='uint8')     # uint8 so size is always one byte

    part_to_write.tofile("".join([filename, "_cut"]))

    return part_to_write


def cut_file_nsx_variant_b(filename, total_length, offset_header, nb_samples, channel_nb):

    if total_length > os.path.getsize(filename):
        raise ValueError('More samples specified than included in file')

    file_nb_samples = np.memmap(filename, shape=1, offset=offset_header - 4, dtype='uint32')[0]

    if file_nb_samples <= nb_samples:
        nb_samples -= file_nb_samples
        offset_header += 2 * channel_nb * file_nb_samples + 9  # 9 bytes is length of data packet header

    part_to_write = np.memmap(filename, shape=total_length, dtype='uint8')      # uint8 so size is always one byte

    writer = np.memmap("".join([filename, "_cut"]), shape=total_length, dtype='uint8', mode='w+')
    writer[:] = part_to_write[:]

    nb_samples_to_write = nb_samples + 1
    for i in range(4):
        writer[offset_header+i-4] = np.uint8(nb_samples_to_write % 256)
        nb_samples_to_write /= 256

    return part_to_write


def extract_nsx_file_spec(filenames, nsx_nb):

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

    filename = '.'.join([filenames['nsx'], 'ns%i' % nsx_nb])

    if spec == '2.1':
        channel_count = np.memmap(filename, dtype='int32', offset=28, shape=1)[0]
        bytes_in_headers = 11 + channel_count*4
        sampling_rate = int(30000/np.memmap(filename, dtype='int32', offset=24, shape=1)[0])
    elif spec in ['2.2', '2.3']:
        bytes_in_headers = np.memmap(filename, dtype='uint32', offset=10, shape=1)[0]
        channel_count = np.memmap(filename, dtype='int32', offset=310, shape=1)[0]
        sampling_rate = np.memmap(filename, dtype='uint32', offset=290, shape=1)[0]

    return int((length_bytes-bytes_in_headers)/(2*channel_count)), sampling_rate


def get_nev_samples(filenames, length_bytes):
    filename = '.'.join([filenames['nev'], 'nev'])
    data_size = np.memmap(filename, dtype='uint32', shape=1, offset=16)[0]
    offset_header = np.memmap(filename, dtype='uint32', shape=1, offset=12)[0]
    nb_samples = int((length_bytes-offset_header)/data_size)
    time_stamp = np.memmap(filename, dtype='uint32', shape=1, offset=offset_header+nb_samples*data_size)[0]
    return int(time_stamp/np.memmap(filename, dtype='uint32', shape=1, offset=20)[0])*1000, 1000


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

nb_samples = args.nb_samples
sampling_rate = args.sampling_rate
if args.bytes is not None:
    length_bytes = args.bytes * 1024 * 1024
else:
    length_bytes = None
filenames = {}
if args.filenames is not None:
    filenames['nsx'] = args.filenames
    filenames['nev'] = args.filenames
if args.nsx_path is not None:
    filenames['nsx'] = args.nsx_path
if args.nev_path is not None:
    filenames['nev'] = args.nev_path
if not filenames:
    parser.error("No filenames specified, please use -f or --nsx_path and/or --nev_path")
elif args.nsx and 'nsx' not in filenames:
    parser.error("You need to specify a path for the nsX you want to cut, please use -f or --nsx_path")
elif args.nev and 'nev' not in filenames:
    parser.error("You need to specify a path for the nev you want to cut, please use -f or --nev_path")

if not args.nev and not args.nsx and not args.all:
    parser.error("You need to specify what you want to cut")

if (nb_samples is None or sampling_rate is None) and length_bytes is None:
    parser.error("You need to specify the length of the output, either input the number of samples with the "
                 "corresponding sampling rate or use -b <length in Megabytes>")

if args.all:
    args.nev = True
    nsx_nb = 'all'

if args.nsx[0] != 'all':
    nsx_nb = []
    for nb in args.nsx:
        nsx_nb.append(int(nb))
else:
    nsx_nb = args.nsx[0]

if nsx_nb == 'all':
    nsx_nb = []
    for i in range(1, 7):
        filename = ".".join([filenames['nsx'], 'ns%i' % i])
        if os.path.exists(filename):
            nsx_nb.append(i)


nsx_is_largest = False

for i in nsx_nb:
    if args.nev and os.path.getsize(".".join([filenames['nev'], 'nev'])) < os.path.getsize(
            '.'.join([filenames['nsx'], 'ns%i' % i])):
        nsx_is_largest = True
        break


if length_bytes is not None and not args.same_size:
    if nsx_is_largest:
        nb_samples, sampling_rate = get_nsx_samples(filenames, max(nsx_nb),
                                                    extract_nsx_file_spec(filenames, max(nsx_nb)), length_bytes)
    else:
        nb_samples, sampling_rate = get_nev_samples(filenames, length_bytes)

for i in nsx_nb:
    spec = extract_nsx_file_spec(filenames, i)
    if spec == '2.1':
        cut_nsx_variant_a(filenames, i, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
                          split=args.same_size)
    elif spec in ['2.2', '2.3']:
        cut_nsx_variant_b(filenames, i, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
                          split=args.same_size)
if args.nev:
    cut_nev(filenames, nb_samples=nb_samples, sampling_rate=sampling_rate, length_bytes=length_bytes,
            split=args.same_size)

# TODO: same times from each file (???)
