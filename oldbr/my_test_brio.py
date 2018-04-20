from __future__ import absolute_import
import math
from neo.io.blackrockio import BlackrockIO
from neo.io.blackrockio_v4 import BlackrockIO as old_brio
#import matplotlib.pyplot as plt
from neo.rawio import BlackrockRawIO
import warnings
import numpy as np
from neo.core import *
from neo import AnalogSignal
import time
import quantities as pq

# check scipy
try:
    from distutils import version
    import scipy.io
    import scipy.version
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err



dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataNikos2/i140703-001a'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataNikos2Public/Testfiles_10MB/i140703-001'       #2.3
# dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilou/Testfiles_10000_samples/l101210-001a'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilou/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilouPublic/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilouPublic/Testfiles_1MB/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilouPublic/Testfiles_5MB/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilouPublic/Testfiles_10MB/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataLilouPublic/Testfiles_20MB/l101210-001'        #2.1
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataNikos/n130715-001'                             #2.3
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataEnya/e170131-002'   # File missing event data  #2.3
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana/s131214-002'  # Do not load ns6!!!!       #2.2
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataTanya/t130910-001'                             #2.1
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataTanya2/a110914-001'                            #2.1
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana_second/s140203-003'                       #2.3
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/DataSana_third/s131209-001'                         #2.2
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/blackrock/FileSpec2.3001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/neo/blackrock/blackrock_2_1/l101210-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180116-land-m-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180116-land-v-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180125-land-m-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180125-land-v-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180216-draw-m-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180216-draw-v-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180206-draw-m-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/y180206-draw-v-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestFiles1MiB/y180216-draw-m-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestFiles1MiB/y180216-draw-v-001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-pause-m-006'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-pause-v-006'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-pauseSolo-m-008'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-reset-m-005'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-reset-v-005'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-resetSolo-m-007'
dirname = '/home/arbeit/Downloads/files_for_testing_neo/V4A/TestPause/y180301-reset-m-005_cut'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/Synchro_tests/synctest_180306_draw_v_001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/Synchro_tests/synctest_180306_draw_m_001'
dirname1 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/Synchro_tests/synctest_180306_land_v_002'
dirname2 = '/home/arbeit/Downloads/files_for_testing_neo/V4A/Synchro_tests/synctest_180306_land_m_002'
oldbrio_reader = None
newbrio_reader = None
oldmlio_reader = None


def old_mlio_load():        # Own routine for loading matlab files
    pass
    #return old_block


def old_brio_load():
    oldbrio_reader = old_brio(dirname, verbose=True)#,
                             #nev_override='-'.join([dirname, '03']))
    old_block = oldbrio_reader.read_block(
        #n_starts=[None, 1.1*pq.s], n_stops=[None, 1.4*pq.s],
        channels='all',  # '{1, 3, 5, 7, 95},
        nsx_to_load=2,
        units='all',
        load_events=True,
        load_waveforms=True,
        scaling='voltage', lazy=False)
        # scaling='voltage', n_starts=0.02*pq.s, n_stops=0.04*pq.s)
    # output(old_block)
    print('Loading old IO done')
    return old_block


def new_brio_load():
    newbrio_reader = BlackrockIO(dirname, nsx_to_load=2)#, nev_override='-')  # channels_to_load={1, 3, 5, 7, 95})#, nev_override='-'.join([dirname, '03']))
    newbrio_reader2 = BlackrockIO(dirname2, nsx_to_load=2)#, nev_override='-')  # channels_to_load={1, 3, 5, 7, 95})#, nev_override='-'.join([dirname, '03']))
    # new_block = newbrio_reader.read_block()
    # print(newbrio_reader)
    # newbrio_reader.parse_header()
    new_block = newbrio_reader.read_block(load_waveforms=False, signal_group_mode="split-all")#, nsx_to_load='all')#, time_slices=[(float('-inf'), float('inf')), (1*pq.s, 2*pq.s), (3*pq.s, 4 * pq.s)])#(1 *pq.s, 2 * pq.s)])
    new_block2 = newbrio_reader2.read_block(load_waveforms=False, signal_group_mode="split-all")#, nsx_to_load='all')#, time_slices=[(float('-inf'), float('inf')), (1*pq.s, 2*pq.s), (3*pq.s, 4 * pq.s)])#(1 *pq.s, 2 * pq.s)])
                                            #, signal_group_mode="group-by-same-units") #, time_slices=[(1.0, 40.0)])#, time_slices=[(0.001366667*pq.s, 2.0*pq.s)])  # signal_group_mode="group-by-same-units")#load_waveforms=True)
    # output(new_block)
    print ('Loading new IO done')
    return new_block, new_block2


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
                print(' SpikeTrain', st.name, st.shape, st.waveforms.shape, st.times[:][:5])
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

def print_annotations_chidx(block):
    objects = block.list_children_by_class(AnalogSignal)
    print ('Object Type: ', AnalogSignal)
    for obj in objects:
        print(obj.name)
        for key in obj.channel_index.annotations.keys():
            print('Key: ', key)
            print('Value: ', obj.channel_index.annotations[key])
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
        # print(object)
        print_attributes_of_object(object)


def child_objects(block, objtype):
    return block.list_children_by_class(objtype)


def chanind_anasig_relation(block):  # GOOD
    chaninds = block.list_children_by_class(ChannelIndex)
    for chanind in chaninds:
        print(chanind.name, chanind)
        anasigs = chanind.analogsignals
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.channel_index, anasig[:])
            print(id(anasig))
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
        print(chanind.name, chanind)
        anasigs = chanind.spiketrains
        for anasig in anasigs:
            print(anasig.name, 'ChannelIndex: ', anasig.unit)
        print('*' * 10)


def st_unit_relation(block):
    chaninds = block.list_children_by_class(SpikeTrain)
    for chanind in chaninds:
        print(chanind.name, chanind)
        anasig = chanind.unit
        try:
            print(anasig.name, 'Unit: ', anasig.spiketrains)
            for st in anasig.spiketrains:
                print(st.unit)
            print()
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

def compare_array_content_event(array1, array2):
    if (array1 == array2).all:
        print('Good')
    else:
        print('Failed')

def compare_object_content(old_block, new_block, objtype, attr):
    objolds = old_block.list_children_by_class(objtype)
    objnews = new_block.list_children_by_class(objtype)
    #print(objolds, objnews)
    for objold, objnew in zip(objolds, objnews):
        oldarray = getattr(objold, attr)  # specific for SpikeTrain and Event
        newarray = getattr(objnew, attr)  # specific for AnaSig
        # print(oldarray)
        # print(newarray)
        if len(oldarray.flat) == 0 and len(newarray.flat) == 0:
            print("Good")
            return
        #print oldarray
        #print newarray
        index = 0
        while newarray.flat[index] == 0:
            index += 1
            #rescale_factor = oldarray.flat[index] / newarray.flat[index]
        rescale_factor = oldarray.flat[index] / newarray.flat[index]
        if(rescale_factor!=1):
            warnings.warn(''.join(["Rescale factor is ", str(rescale_factor)]))
        compare_array_content(rescale_factor, oldarray, newarray)

def compare_object_content_anasig(old_block, new_block):
    objolds = old_block.list_children_by_class(AnalogSignal)
    objnews = new_block.list_children_by_class(AnalogSignal)
    #print(objolds, objnews)
    for objold, objnew in zip(objolds, objnews):
        oldarray = objold[:]
        newarray = objnew[:]  # specific for AnaSig
        if len(oldarray.flat) == 0 and len(newarray.flat) == 0:
            print("Good")
            return
        index = 0
        while newarray.flat[index] == 0:
            index += 1
        rescale_factor = oldarray.flat[index] / newarray.flat[index]
        if(rescale_factor!=1):
            warnings.warn(''.join(["Rescale factor is ", str(rescale_factor)]))
        compare_array_content(rescale_factor, oldarray, newarray)

def compare_object_content_event(old_block, new_block):
    objolds = old_block.list_children_by_class(Event)
    objnews = new_block.list_children_by_class(Event)
    for objold in objolds:
        if(len(objold)==0):
            objolds.remove(objold)
    i = 0
    while i < len(objnews):
        if (len(objnews[i]) == 0):
            objnews.pop(i)
            i-=1
        i += 1

    if len(objolds)==0 and len(objnews)==0:
        print("Good, because empty!")
        return
    for objold, objnew in zip(objolds, objnews):
        oldarray = objold.times  # specific for SpikeTrain and Event
        newarray = objnew.times  # specific for AnaSig
        index = 0
        while newarray.flat[index] == 0:                #!!!!!!TANYA2 FAILS HERE!!!
            index += 1
        rescale_factor = oldarray.flat[index] / newarray.flat[index]
        if(rescale_factor!=1):
            warnings.warn(''.join(["Rescale factor is ", str(rescale_factor)]))
        compare_array_content(rescale_factor, oldarray, newarray)
    for objold, objnew in zip(objolds, objnews):
        oldarray = getattr(objold, 'labels')  # specific for SpikeTrain and Event
        newarray = getattr(objnew, 'labels')  # specific for AnaSig
        compare_array_content_event(oldarray, newarray)


def compare_all_objects(old_block, new_block):
    compare_object_content(old_block, new_block, SpikeTrain, 'times')
    compare_object_content(old_block, new_block, SpikeTrain, 'waveforms')
    compare_object_content_anasig(old_block, new_block)
    compare_object_content_event(old_block, new_block)

def plot(old_block, new_block):
    ar = np.zeros_like(np.ndarray(30000))
    st = old_block.list_children_by_class(SpikeTrain)[0]
    anasig = old_block.list_children_by_class(AnalogSignal)[0]
    array = anasig[:].magnitude
    print(array)
    for t in st[:].magnitude:
        print (t)
        ar[int(t-6000)] = 10
    plt.plot(ar, 'x')
    plt.plot(array)
    # plt.figure(2)
    newar = np.zeros_like(np.ndarray(30000))
    newst = new_block.list_children_by_class(SpikeTrain)[0]
    newanasig = new_block.list_children_by_class(AnalogSignal)[0]
    newarray = newanasig[:].magnitude
    print(newarray)
    for nt in newst[:].magnitude * 30000:
        print (nt)
        newar[int(nt-6000)] = 10
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


def test_merge():
    waveforms1 = np.array([[[0., 1.],
                            [0.1, 1.1]],
                           [[2., 3.],
                            [2.1, 3.1]],
                           [[4., 5.],
                            [4.1, 5.1]]]) * pq.mV
    data1 = np.array([3, 4, 5])
    data2quant = np.array([4.5, 8, 9]) * pq.s
    data1quant = data1 * pq.s
    train1 = SpikeTrain(data1quant, waveforms=waveforms1,
                        name='n', arb='arbb', t_stop=10.0 * pq.s, array_annotations={'a': np.array([9, 8, 7])})
    train2 = SpikeTrain(data2quant, waveforms=waveforms1,
                        name='n', arb='arbb', t_stop=10.0 * pq.s, array_annotations={'a': np.array([19, 18, 17])})
    train3 = train2.merge(train1)
    print(train3.array_annotations)

    data = range(10)
    rate = 1000 * pq.Hz
    signal = AnalogSignal(data, sampling_rate=rate, units="mV", array_annotations={'b': np.array(['a'])})

    data = [i + 1 for i in range(10)]
    rate = 1000 * pq.Hz
    signal2 = AnalogSignal(data, sampling_rate=rate, units="mV", array_annotations={'b': np.array(['cc'])})

    data = [i + 2 for i in range(10)]
    rate = 1000 * pq.Hz
    signal3 = AnalogSignal(data, sampling_rate=rate, units="mV", array_annotations={'b': np.array(['ddd'])})

    signal4 = signal3.merge(signal2)

    print(signal4)
    print(signal4.array_annotations)

    signal5 = signal4.merge(signal)

    print(signal5)
    print(signal5.array_annotations)

    evt = Event([1.1, 1.5, 1.7] * pq.ms,
                labels=np.array(['test event 1',
                                 'test event 2',
                                 'test event 3'], dtype='S'),
                array_annotations={'a': np.array(['1.1', '1.5', '1.7'])})
    evt2 = Event([1.2, 1.6, 1.8] * pq.ms,
                 labels=np.array(['test event 4',
                                  'test event 5',
                                  'test event 6'], dtype='S'),
                 array_annotations={'a': np.array(['1.2', '1.6', '1.8'])})

    evt3 = evt2.merge(evt)
    print(evt3)
    print(evt3.array_annotations)

    evt5 = Event([1.23, 1.67, 1.89] * pq.ms,
                 labels=np.array(['test event 41',
                                  'test event 51',
                                  'test event 61'], dtype='S'),
                 array_annotations={'a': np.array(['1.23', '1.67', '1.89'])})

    print(evt3.merge(evt5))
    print(evt3.merge(evt5).array_annotations)

    epctarg = Epoch([1.1, 1.4] * pq.ms,
                    durations=[20, 40] * pq.ns,
                    labels=np.array(['test epoch 1 1',
                                     'test epoch 1 2',
                                     'test epoch 1 3',
                                     'test epoch 2 1',
                                     'test epoch 2 2',
                                     'test epoch 2 3'], dtype='S'),
                    array_annotations={'a': np.array(['1.1', '1.4'])})
    epctarg1 = Epoch([1.12, 1.42] * pq.ms,
                     durations=[21, 41] * pq.ns,
                     labels=np.array(['test epoch 1 1',
                                      'test epoch 1 2',
                                      'test epoch 1 3',
                                      'test epoch 2 1',
                                      'test epoch 2 2',
                                      'test epoch 2 3'], dtype='S'),
                     array_annotations={'a': np.array(['1.12', '1.42'])})
    epctarg2 = Epoch([1.123, 1.423] * pq.ms,
                     durations=[22, 42] * pq.ns,
                     labels=np.array(['test epoch 1 1',
                                      'test epoch 1 2',
                                      'test epoch 1 3',
                                      'test epoch 2 1',
                                      'test epoch 2 2',
                                      'test epoch 2 3'], dtype='S'),
                     array_annotations={'a': np.array(['1.123', '1.423'])})

    epc = epctarg.merge(epctarg2)

    print(epc)
    print(epc.array_annotations)
    epc1 = epc.merge(epctarg1)

    print(epc1)
    print(epc1.array_annotations)

    sig = IrregularlySampledSignal([1.1, 1.5, 1.7],
                                   signal=[20., 40., 60.], time_units=pq.s,
                                   units=pq.V, array_annotations={'a': np.array(['a'])})
    sig1 = IrregularlySampledSignal([1.1, 1.5, 1.7],
                                    signal=[20.4, 40.4, 60.4], time_units=pq.s,
                                    units=pq.V, array_annotations={'a': np.array(['d'])})
    sig2 = IrregularlySampledSignal([1.1, 1.5, 1.7],
                                    signal=[20.5, 40.5, 60.5], time_units=pq.s,
                                    units=pq.V, array_annotations={'a': np.array(['g'])})

    sig3 = sig.merge(sig1)

    print(sig3)
    print(sig3.array_annotations)

    print(sig3.merge(sig2))
    print(sig3.merge(sig2).array_annotations)


def npify(a):
    a = np.ndarray(a)


def annotations_at_index(d, index):

    index_annotations = {}

    for ann in d:
        print("HERE")
        index_annotations[ann] = d[ann][index]

    return index_annotations

def getitem_example():
    epctarg = Epoch([1.1, 1.4] * pq.ms,
                    durations=[20, 40] * pq.ns,
                    labels=np.array(['test epoch 1 1',
                                     'test epoch 1 2',
                                     'test epoch 1 3',
                                     'test epoch 2 1',
                                     'test epoch 2 2',
                                     'test epoch 2 3'], dtype='S'),
                    array_annotations={'a': np.array(['1.1', '1.4'])})
    epctarg1 = Epoch([1.12, 1.42] * pq.ms,
                     durations=[21, 41] * pq.ns,
                     labels=np.array(['test epoch 1 1',
                                      'test epoch 1 2',
                                      'test epoch 1 3',
                                      'test epoch 2 1',
                                      'test epoch 2 2',
                                      'test epoch 2 3'], dtype='S'),
                     array_annotations={'a': np.array(['1.12', '1.42'])})
    epctarg2 = Epoch([1.123, 1.423] * pq.ms,
                     durations=[22, 42] * pq.ns,
                     labels=np.array(['test epoch 1 1',
                                      'test epoch 1 2',
                                      'test epoch 1 3',
                                      'test epoch 2 1',
                                      'test epoch 2 2',
                                      'test epoch 2 3'], dtype='S'),
                     array_annotations={'a': np.array(['1.123', '1.423'])})

    epc = epctarg.merge(epctarg2)

    print(type(epc[1]))
    print(type(epc[1:]))
    epcc = Epoch(times=epc[1])

    print(epcc.times)
    #print(epc.array_annotations)

    ###epc1 = epc.merge(epctarg1)

    #print(epc1)
    #print(epc1.array_annotations)

    waveforms1 = np.array([[[0., 1.],
                            [0.1, 1.1]],
                           [[2., 3.],
                            [2.1, 3.1]],
                           [[4., 5.],
                            [4.1, 5.1]]]) * pq.mV
    data1 = np.array([3, 4, 5])
    data2quant = np.array([4.5, 8, 9]) * pq.s
    data1quant = data1 * pq.s
    train1 = SpikeTrain(data1quant, waveforms=waveforms1,
                        name='n', arb='arbb', t_stop=10.0 * pq.s)

    # Currently does not happen, because I'm changing things in SpikeTrain
    print(type(train1[0]))  # Returns int/float Quantity, which seems "defective".
    # This happens because numpy returns only a scalar, not a single element array for  a scalar index
    print(train1[0:1])    # Returns SpikeTrain because numpy returns array (or subclass, here: SpikeTrain)
                                # for slices indices
    # => Things like train[0].annotations or train[0].t_stop cannot be requested
    # This gave problems when sorting etc., where those properties are set
    # (but create an error because of setting nonexistent attributes)]

    q = pq.Quantity([1,2,3], units=pq.s)
    test1 = q[0:1]  # This returns an array quantity because numpy returns arrays when using slices
    test2 = q[0]    # This returns a (broken) scalar quantity, which causes problems when treated like a SpikeTrain etc.
    # test2 cannot be used like a SpikeTrain and not as times=test2 as well,
    # because it is not a list/array/whatever

    print(test1)
    print(test2)


def test_rescale():
    epctarg = Epoch([1.1, 1.4] * pq.ms,     # TODO: Shouldn't this result in an error?
                    durations=[20, 40, 50] * pq.ns,
                    labels=np.array(['test epoch 1 1',
                                     'test epoch 1 2',
                                     'test epoch 1 3',
                                     'test epoch 2 1',
                                     'test epoch 2 2',
                                     'test epoch 2 3'], dtype='S'),
                    array_annotations={'a': np.array(['1.1', '1.4'])})

    print_attributes_of_object(epctarg)
    print_attributes_of_object(epctarg.rescale(pq.s))

    evt = Event([1.1, 1.5, 1.7] * pq.ms,
                labels=np.array(['test event 1',
                                 'test event 2',
                                 'test event 3'], dtype='S'),
                array_annotations={'a': np.array(['1.1', '1.5', '1.7'])})

    print_attributes_of_object(evt)
    print_attributes_of_object(evt.rescale(pq.s))
    print(evt.rescale(pq.s).units)
    print(type(epctarg.times))

    data = range(10)
    rate = 1000 * pq.Hz
    signal = AnalogSignal(data, sampling_rate=rate, units="mV", array_annotations={'b': np.array(['a'])})

    print_attributes_of_object(signal)
    print_attributes_of_object(signal.rescale(pq.V))
    print_annotations_of_object(signal.rescale(pq.V))

    waveforms1 = np.array([[[0., 1.],
                            [0.1, 1.1]],
                           [[2., 3.],
                            [2.1, 3.1]],
                           [[4., 5.],
                            [4.1, 5.1]]]) * pq.mV
    data1 = np.array([3, 4, 5])
    data2quant = np.array([4.5, 8, 9]) * pq.s
    data1quant = data1 * pq.s
    train1 = SpikeTrain(data1quant, waveforms=waveforms1,
                        name='n', arb='arbb', t_stop=10.0 * pq.s, array_annotations={'a': np.array([9, 8, 7])})
    print_attributes_of_object(train1)
    print_attributes_of_object(train1.rescale(pq.ms))


def run_test():
    np.set_printoptions(threshold=6)
    #reader = BlackrockRawIO(dirname, nsx_to_load=5)
    #reader.parse_header()
    #print(reader.get_analogsignal_chunk())
    startold = time.time()
    #old_block = old_brio_load()
    finishold = time.time()
    print('This took ', finishold-startold, ' seconds')
    #output(old_block)
    #raise ValueError
    startnew = time.time()
    new_block, new_block2 = new_brio_load()
    finishnew = time.time()
    print('This took ', finishnew - startnew, ' seconds')
    #time.sleep(100)
    #output(new_block)
    #print_annotations_all(old_block, Unit)
    #print_annotations_all(new_block, Unit)
    #plot(old_block, new_block)
    #compare_neo_content(old_block, new_block)
    #raise NotImplementedError
    objtypes = [ChannelIndex, Unit, AnalogSignal,
                            SpikeTrain, Event, Epoch, Segment]
        # plot(old_block, new_block)
    #print("OLD")
    #for a in old_block.list_children_by_class(AnalogSignal):
     #   print('*********')
     #   print(a[:])
   # print("NEW")
    #for a in new_block.list_children_by_class(AnalogSignal):
    #    print('*********')
    #   print(a[:])
    # objtypes = [AnalogSignal]
    # print(new_block.channel_indexes[0].analogsignals)
    """for objtype in objtypes:
         print('*'*100)
         print('OLD IO')
         print_attributes_of_all_objects(old_block, objtype)
         # print('*' * 100)
         # print('NEW IO')
         # print_attributes_of_all_objects(new_block, objtype)
    for objtype in objtypes:
         print('*'*100)
         print('OLD IO')
         # print_annotations_all(old_block, objtype)
         # print('*' * 100)
         # print('NEW IO')
         # print_annotations_all(new_block, objtype)
    print("OLD IO")
    print_annotations_of_object(old_block)
    print_attributes_of_object(old_block)"""
    #objtypes = [ChannelIndex]
    for objtype in objtypes:
        print('*' * 100)
        print('NEW IO')
        #print_attributes_of_all_objects(new_block, objtype)
        # print('*' * 100)
        # print('NEW IO')
        print_attributes_of_all_objects(new_block2, objtype)
    print_attributes_of_object(new_block)
    """
    for objtype in objtypes:
        print('*' * 100)
        print('NEW IO')
        print_annotations_all(new_block, objtype)
    print("NEW IO")
    print_annotations_of_object(new_block)
    print_attributes_of_object(new_block)
    print(finishnew-startnew)"""
    """
    #st_unit_relation(new_block)
        # print("OLD BLOCK ANNOTATIONS")
        # print_annotations_of_object(old_block)
        # print("NEW BLOCK ANNOTATIONS")
        ## print_annotations_of_object(new_block)
        # chan_ind = child_objects(old_block, ChannelIndex)
        # print('NEW Event Annotations')
        # print_annotations_all(old_block, Unit)
        # print('NEW Epoch Attributes')        # NEED TO DO THIS FOR AAAAALLLLLL OBJECT TYPES!!!!!!!!!!!!!! Unit SpikeTrain Event Epoch
    #print_annotations_all(new_block, ChannelIndex)
    #print_attributes_of_all_objects(new_block, AnalogSignal)
    #print_annotations_chidx(new_block)
    #chanind_unit_relation(new_block)
    #print_attributes_of_object(new_block.list_children_by_class(AnalogSignal)[96])
    #for i in objtypes:
    #    print_attributes_of_all_objects(new_block, i)
    #compare_array_content(1, new_block.list_children_by_class(AnalogSignal)[96][:], old_block.list_children_by_class(AnalogSignal)[0][:])
    chanind_anasig_relation(new_block)
    chanind_unit_relation(new_block)
    unit_st_relation(new_block)
    st_unit_relation(new_block)
    segment_anasig_relation(new_block)
    segment_st_relation(new_block)
    segment_event_relation(new_block)
    segment_epoch_relation(new_block)
    block_chanind_relation(new_block)
    block_segment_relation(new_block)"""
    """chanind_anasig_relation(old_block)
    chanind_unit_relation(old_block)
    unit_st_relation(old_block)
    st_unit_relation(old_block)
    segment_anasig_relation(old_block)
    segment_st_relation(old_block)
    segment_event_relation(old_block)
    segment_epoch_relation(old_block)
    block_chanind_relation(old_block)
    block_segment_relation(old_block)"""
    # for seg in new_block.list_children_by_class(Segment):
    #     print(seg.t_stop)
    #     print(len(seg.spiketrains))
    #     for sptr in seg.spiketrains:
    #         print(sptr)

    # sig = new_block.list_children_by_class(AnalogSignal)[-1:]
    # for elem in sig:
    #     for i, a in enumerate(elem):
    #         if a > 3000*pq.mV:
    #             # b = i + 3
    #             print(i)
        # print(elem.name)
        # print(max(elem[10698:21117]))
        # print(elem[10117:10617])
        # print('******************************************************************')
    # print('**************************************************************')
    # tr1 = tr.sort()
    # for elem in tr:
    #     print(elem)


    #print_attributes_of_all_objects(new_block, AnalogSignal)

    # for elem in new_block.list_children_by_class(SpikeTrain)[1]:
    #     print(elem)

    for seg in new_block.list_children_by_class(Segment):
    #     print(seg.analogsignals[0].t_start)
    #     print(seg.analogsignals[0].t_stop)
        #for elem in seg.analogsignals[0]:
        #    print(elem)
        print("seg", seg.t_start)
        print("seg", seg.t_stop)
        print(seg.events[:])
        for ev in seg.events:
            print(ev)

        # for anasig in seg.analogsignals:
        #     print('sig')
        #     for item in anasig:
        #         if item.magnitude != 38.75:
        #             print(item)
        # try:
        #     print("sptr", seg.spiketrains[0].t_start)
        #     print("sptr", seg.spiketrains[0].t_stop)
        #     # for st in seg.spiketrains:
        #     #     print(len(st))
        # except:
        #     print("No Spiketrain")
    #print(new_block.list_children_by_class(Segment)[3].spiketrains[:])




run_test()
