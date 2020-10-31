import numpy as np
import os
import time
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.models import BEATS_LSTM
from madmom.processors import IOProcessor, process_online
from madmom.io import write_beats

def beat_extractor(queue_beat):
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
    verbose = 0
    )   
    def beat_callback(beats, output=None):
        if len(beats) > 0:
            # Do something with the beat (for now, just print the array to stdout)
            if not queue_beat.empty():
                _ = queue_beat.get()
            queue_beat.put(beats[0])
            print(beats)
    #print('Process to write betas: %s' % os.getpid())
    in_processor = RNNBeatProcessor(**kwargs)
    beat_processor = DBNBeatTrackingProcessor(**kwargs)
    out_processor = [beat_processor, beat_callback]
    processor = IOProcessor(in_processor, out_processor)
    process_online(processor, **kwargs)

def beat_simulator(queue_beat):
    t1=time.time()
    while True:
        time.sleep(60/120)
        t=time.time()-t1
        if not queue_beat.empty():
            _ = queue_beat.get()
        queue_beat.put(t)