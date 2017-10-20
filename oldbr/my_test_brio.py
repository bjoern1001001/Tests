from __future__ import absolute_import

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from numpy.testing import assert_equal

import numpy as np
import quantities as pq
import math
from neo.io.blackrockio import BlackrockIO
from neo.io.blackrockio_v4 import BlackrockIO as old_brio
import matplotlib.pyplot as plt
from neo.rawio import BlackrockRawIO

from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path

#################
# from __future__ import absolute_import

import os
import sys
import re
import warnings

import unittest

import numpy as np
import quantities as pq

from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import *

#from neo.io.neuralynxio import NewNeuralynxIO
#from neo.io.neuralynxio import NeuralynxIO as OldNeuralynxIO
from neo import AnalogSignal

import time

# check scipy
try:
    from distutils import version
    import scipy.io
    import scipy.version
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err

# dummy class used only for automatic downloading of test data
#  class DownloadDataIO(BaseTestIO, unittest.TestCase):
#      ioclass = BlackrockIO
#      files_to_download = [
#          'FileSpec2.3001.nev',
#          'FileSpec2.3001.ns5',
#          'FileSpec2.3001.ccf',
#          'FileSpec2.3001.mat']

dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataNikos2/i140703-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilou/l101210-001'
dirname = '/home/arbeit/Downloads/files_for_testing_neo/DataNikos/n130715-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataEnya/e170131-002'   # File missing event data
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana/s131214-002'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataTanya/t130910-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataTanya2/a110914-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana_second/s140203-003'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana_third/s131209-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/blackrock/FileSpec2.3001'
oldbrio_reader = None
newbrio_reader = None


# old_block = None
# new_block = None
def old_brio_load():
    # ioclass = old_brio
    # files_to_test = ['FileSpec2.3001']
    oldbrio_reader = old_brio(dirname, verbose=True)
    old_block = oldbrio_reader.read_block(
        # n_starts=[None], n_stops=None,
        channels='all',  # {1, 2, 3, 4, 5, 6, 7, 8, 129, 130},
        nsx_to_load=2,
        units='all',
        load_events=True,
        load_waveforms=True,
        scaling='voltage', n_starts=-50*pq.s, n_stops=100000*pq.s)
        # scaling='voltage', n_starts=0.02*pq.s, n_stops=0.04*pq.s)
    # output(old_block)
    print 'Loading old IO done'
    return old_block


def new_brio_load():
    newbrio_reader = BlackrockIO(dirname, nsx_to_load=2)
    # new_block = newbrio_reader.read_block()
    # print(newbrio_reader)
    # newbrio_reader.parse_header()
    new_block = newbrio_reader.read_block(load_waveforms=True)#, time_slices=[(0.001366667*pq.s, 2.0*pq.s)])  # signal_group_mode="group-by-same-units")#load_waveforms=True)
    # output(new_block)
    print 'Loading new IO done'
    return new_block


def output(block):
    for seg in block.segments:
        print('seg', seg.index)
        for epoch in seg.epochs:
            print("FOUND EPOCH")
        for anasig in seg.analogsignals:
            print(' AnalogSignal', anasig.name, anasig.shape, anasig.t_start, anasig.sampling_rate)
            print('ChannelIndex', anasig.annotations['channel_id'])
            # print(anasig.channel_index.channel_id) => SHOULD WORK; BUT DOESN'T!!!!!!!!!!!
        for st in seg.spiketrains:
            if st is not None:
                print(' SpikeTrain', st.name, st.shape, st.waveforms.shape, st.times[:].rescale(pq.s)[:5])
                      #st.waveforms[:].rescale(pq.s)[:5])
        for ev in seg.events:
            print(' Event', ev.name, ev.times.shape)
    print ('*' * 10)


# def count_channels(block):
#     num_channels = 0
#     for seg in block.segments:
#         for anasig in seg.analogsignals:
#             a = block.annotations

def compare_neo_content(bl1, bl2):
    print('*' * 5, 'Comparison of blocks', '*' * 5)
    object_types_to_test = [Segment, ChannelIndex, Unit, AnalogSignal,
                            SpikeTrain, Event, Epoch]
    # object_types_to_test = [SpikeTrain]
    for objtype in object_types_to_test:
        print('Testing {}'.format(objtype))
        children1 = bl1.list_children_by_class(objtype)
        children2 = bl2.list_children_by_class(objtype)

        if len(children1) != len(children2):
            warnings.warn('Number of {} is different in both blocks ({} != {'
                          '}). Skipping comparison'.format(objtype,
                                                           len(children1),
                                                           len(children2)))
            continue

        for child1, child2 in zip(children1, children2):
            compare_annotations(child1.annotations, child2.annotations, objtype)
            # compare_attributes(child1, child2)


def compare_failing_classes(old_block,
                            new_block):  # more precise comparison of the classes that fail compare_neo_content, here AnaSig, Event and ChannelIndex
    print('*')


def compare_annotations(anno1, anno2, objecttype):
    if len(anno1) != len(anno2):
        warnings.warn('Different numbers of annotations! {} != {'
                      '}\nSkipping further comparison of this '
                      'annotation list.'.format(
            anno1.keys(), anno2.keys()))
        print('In:', objecttype)
        # time.sleep()
        return
    assert anno1.keys() == anno2.keys()
    for key in anno1.keys():
        anno1[key] = anno2[key]


def print_annotations_id(block,
                         objtype):  # because comparison will always say false, so need to check content => for IDs
    children1 = block.list_children_by_class(objtype)
    for child1 in children1:
        try:  # => if 'Unit_id' in child1.annotations   [.keys] => raus!!!
            print('Unit_ID:', child1.annotations['unit_id'])
        except:
            pass
        try:
            print('Channel_ID:', child1.annotations['channel_id'])
        except:
            pass
        try:
            print('ID:', child1.annotations['id'])
        except:
            pass
    print('*' * 10)


def print_annotations_all(
        block,
        objtype):
    objects = block.list_children_by_class(objtype)
    print ('Object Type: ', objtype)
    for obj in objects:
        for key in obj.annotations.keys():
            print('Key: ', key)
            print('Value: ', obj.annotations[key])
        print('*' * 20)

def print_annotations_of_object(obj):
    for key in obj.annotations.keys():
        print('Key: ', key)
        print('Value: ', obj.annotations[key])
    print('*' * 20)

def print_attributes_of_object(object):
    attribs = object._all_attrs
    # print(attribs)
    for attrib in attribs:
        if attrib[0] is not 'signal':
            print(attrib[0], ': ', object.__getattribute__(attrib[0]))
        else:
            print(attrib[0], ': ', object[:])
        print('*' * 10)
    print('*' * 20)


def print_attributes_of_all_objects(block, objtype):
    objects = block.list_children_by_class(objtype)
    index = 0
    print('Type: ', objtype)
    for object in objects:
        print('                                       *****Number: ', index)
        # print_attributes_of_object(object)
        index = index + 1
        # print('ANASIG FROM CHANIND: ', object.analogsignals[0][object.index[0]])
        print_attributes_of_object(object)


def child_objects(block, objtype):
    return block.list_children_by_class(objtype)


def chanind_anasig_relation(block):  # GOOD
    chaninds = block.list_children_by_class(ChannelIndex)
    for chanind in chaninds:
        print(chanind.name)
        anasigs = chanind.analogsignals
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.channel_index.name)
        units = chanind.units
        for unit in units:
            print(unit.name)
        print('*' * 10)


def chanind_unit_relation(block):  # GOOD
    chaninds = block.list_children_by_class(ChannelIndex)
    for chanind in chaninds:
        print(chanind)
        anasigs = chanind.units
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.channel_index)
        units = chanind.units

        print('*' * 10)


def unit_st_relation(block):
    chaninds = block.list_children_by_class(Unit)
    for chanind in chaninds:
        print(chanind.name)
        anasigs = chanind.spiketrains
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.unit)
        print('*' * 10)


def st_unit_relation(block):
    chaninds = block.list_children_by_class(SpikeTrain)
    for chanind in chaninds:
        print(chanind.name)
        anasig = chanind.unit
        try:
            print(anasig.name, 'ChannelIndex: ', anasig.spiketrains)
        except:
            print("No Unit linked to this SpikeTrain")
        print('*' * 10)


def segment_anasig_relation(block):
    chaninds = block.list_children_by_class(Segment)
    for chanind in chaninds:
        print(chanind)
        anasigs = chanind.analogsignals
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.segment)
        print('*'*10)

def segment_st_relation(block):
    chaninds = block.list_children_by_class(Segment)
    for chanind in chaninds:
        print(chanind)
        anasigs = chanind.spiketrains
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.segment)
        print('*'*10)

def segment_epoch_relation(block):
    chaninds = block.list_children_by_class(Segment)
    for chanind in chaninds:
        print(chanind)
        anasigs = chanind.epochs
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.segment)
        print('*'*10)

def segment_event_relation(block):
    chaninds = block.list_children_by_class(Segment)
    for chanind in chaninds:
        print(chanind)
        anasigs = chanind.events
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.segment)
        print('*'*10)

def block_chanind_relation(block):
    print(block)
    anasigs = block.channel_indexes
    for anasig in anasigs:
        print(anasig.name, 'ChannelIndex: ', anasig.block)
    print('*'*10)

def block_segment_relation(block):
    print(block)
    anasigs = block.segments
    for anasig in anasigs:
        print(anasig.name, 'ChannelIndex: ', anasig.block)
    print('*'*10)


def compare_array_content(rescale_factor, array1, array2):
    array1 = (array1 / rescale_factor).magnitude
    array2 = array2.magnitude
    print(rescale_factor, '****************************')
    if np.allclose(array1, array2, atol=0.0001, rtol=0.0001):
        print('Good')
    else:
        print('Failed')
    #  if (array1 == array2).all:
    #      print('Good')
    #  else:
    #      print('Failed')


def compare_object_content(old_block, new_block, objtype):
    objolds = old_block.list_children_by_class(objtype)
    objnews = new_block.list_children_by_class(objtype)
    # print(objolds, objnews)
    for objold, objnew in zip(objolds, objnews):
        oldarray = objold[:]  # specific for SpikeTrain and Event
        newarray = objnew[:]  # specific for AnaSig
        # print(oldarray)
        # print(newarray)
        rescale_factor = oldarray.flat[0] / newarray.flat[0]
        index = 1
        while math.isnan(rescale_factor):
            rescale_factor = oldarray.flat[index] / newarray.flat[index]
            index += 1
        compare_array_content(rescale_factor, oldarray, newarray)


def plot(old_block, new_block):
    ar = np.zeros_like(np.ndarray(30000))
    st = old_block.list_children_by_class(SpikeTrain)[0]
    anasig = old_block.list_children_by_class(AnalogSignal)[0]
    array = anasig[:].magnitude
    print(array)
    for t in st[:].magnitude:
        print (t)
        ar[int(t-5)] = 10
    plt.plot(ar, 'x')
    plt.plot(array)
    # plt.figure(2)
    newar = np.zeros_like(np.ndarray(30000))
    newst = new_block.list_children_by_class(SpikeTrain)[0]
    newanasig = new_block.list_children_by_class(AnalogSignal)[0]
    newarray = anasig[:].magnitude
    print(newarray)
    for nt in newst[:].magnitude * 30000:
        print (nt)
        newar[int(nt-5)] = 10
    plt.plot(newar, 'x')
    plt.plot(newarray)
    plt.show()


def plotnew(old_block):
    ar = np.zeros_like(np.ndarray(31000))
    st = old_block.list_children_by_class(SpikeTrain)[0]
    anasig = old_block.list_children_by_class(AnalogSignal)[0]
    array = anasig[:].magnitude
    print(array)
    for t in st[:].magnitude * 30000:
        print (t)
        ar[int(t-5)] = 10
    plt.plot(ar, 'x')
    plt.plot(array)
    plt.show()


def run_test():
    startold = time.time()
    # old_block = old_brio_load()
    finishold = time.time()
    print('This took ', finishold-startold, ' seconds')
    #output(old_block)
    #raise ValueError
    startnew = time.time()
    new_block = new_brio_load()
    finishnew = time.time()
    print('This took ', finishnew - startnew, ' seconds')
    output(new_block)
    #plot(old_block, new_block)
    #compare_neo_content(old_block, new_block)
    objtypes = [Segment, ChannelIndex, Unit, AnalogSignal,
                            SpikeTrain, Event, Epoch]
    #plot(old_block, new_block)
    # for objtype in objtypes:
    #     print('*'*100)
    #     print('OLD IO')
    #     print_annotations_all(old_block, objtype)
    #     print('*' * 100)
    #     print('NEW IO')
    #     print_annotations_all(new_block, objtype)
    # print("OLD IO")
    # print_annotations_of_object(old_block)
    # print("NEW IO")
    # print_annotations_of_object(new_block)
        # print("OLD BLOCK ANNOTATIONS")
        # print_annotations_of_object(old_block)
        # print("NEW BLOCK ANNOTATIONS")
        ## print_annotations_of_object(new_block)
        # chan_ind = child_objects(old_block, ChannelIndex)
        # print('NEW Event Annotations')
        # print_annotations_all(old_block, Unit)
        # print('NEW Epoch Attributes')        # NEED TO DO THIS FOR AAAAALLLLLL OBJECT TYPES!!!!!!!!!!!!!! Unit SpikeTrain Event Epoch
        # print_attributes_of_all_objects(new_block, SpikeTrain)
    #print_attributes_of_object(new_block.list_children_by_class(AnalogSignal)[96])
    #print_attributes_of_all_objects(old_block, AnalogSignal)
    #compare_array_content(1, new_block.list_children_by_class(AnalogSignal)[96][:], old_block.list_children_by_class(AnalogSignal)[0][:])
        # chanind_anasig_relation(new_block)
        # chanind_unit_relation(new_block)
        # unit_st_relation(new_block)
        # st_unit_relation(new_block)
        # segment_anasig_relation(new_block)
        # segment_st_relation(new_block)
        # segment_event_relation(new_block)
        # segment_epoch_relation(new_block)
        # block_chanind_relation(new_block)
        # block_segment_relation(new_block)
    #compare_object_content(old_block, new_block, AnalogSignal)
        #print_annotations_id(new_block, Unit)
        #print_annotations_id()
        #print_attributes_of_object(new_block)
        #print_attributes_of_object(old_block)
        #print_annotations_of_object(new_block)
        #print_annotations_of_object(old_block)
        #anasig = new_block.list_children_by_class(AnalogSignal)[2]
        #print anasig.shape
    #testEvent = Event(times=[1, 2, 3, 4, 5], labels=['a', 'b', 'c', 'd', 'e'], units=pq.mV, name='Test')
    #print(testEvent)


run_test()
