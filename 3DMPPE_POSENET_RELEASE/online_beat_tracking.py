from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.models import BEATS_LSTM
from madmom.processors import IOProcessor, process_online
from madmom.io import write_beats
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
        print(beats)

in_processor = RNNBeatProcessor(**kwargs)
beat_processor = DBNBeatTrackingProcessor(**kwargs)
out_processor = [beat_processor, beat_callback]
processor = IOProcessor(in_processor, out_processor)
process_online(processor, **kwargs)