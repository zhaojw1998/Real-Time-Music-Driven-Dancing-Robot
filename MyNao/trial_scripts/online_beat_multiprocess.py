# -*- coding:utf-8 -*-
from multiprocessing import Process, Queue
import os
import time
import random
import numpy as np
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.models import BEATS_LSTM
from madmom.processors import IOProcessor, process_online
from madmom.io import write_beats
from online_beat_extract import write
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

