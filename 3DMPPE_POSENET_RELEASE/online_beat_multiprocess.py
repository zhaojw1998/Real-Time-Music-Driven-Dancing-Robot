# -*- coding:utf-8 -*-
from multiprocessing import Process, Queue, Value
import os
import time
import random
import numpy as np
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.models import BEATS_LSTM
from madmom.processors import IOProcessor, process_online
from madmom.io import write_beats

def generate(v):
    while(True):
        for i in range(10):
            v.put(i)

def read(q, v):
    print('Process to read beats: %s' % os.getpid())
    i=0
    while(True):
        #print(q.qsize())
        if not q.empty():
            value = q.get()
            print('get beat', i)
            i+=1
        if np.random.rand() > 0.99995:
            print('select new motion:', v.get())
        #time.sleep(1)   

def write(q):
    kwargs = dict(
    fps = 100,
    correct = True,
    infile = None,
    outfile = None,
    max_bpm = 170,
    min_bpm = 60,
    #nn_files = [BEATS_LSTM[0]],
    transition_lambda = 100,
    num_frames = 1,
    online = True,
    verbose = 1
    )   
    def beat_callback(beats, output=None):
        if len(beats) > 0:
            # Do something with the beat (for now, just print the array to stdout)
            q.put(beats[0])
            print(beats)
    print('Process to write betas: %s' % os.getpid())
    in_processor = RNNBeatProcessor(**kwargs)
    beat_processor = DBNBeatTrackingProcessor(**kwargs)
    out_processor = [beat_processor, beat_callback]
    processor = IOProcessor(in_processor, out_processor)
    process_online(processor, **kwargs)

if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    v = Queue(maxsize=1)
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q, v))
    pg = Process(target=generate, args=(v,))

    pw.start()
    pr.start()
    pg.start()

    pw.join()
    pr.join()
    pg.join()

