

import random
from music21 import *

# get random sample of riffs from training data
# Get a sample of k unique elements from the list
k = 10
selected_riff_sample = random.sample(riffs, k)

# task: pull one with closest starting note to end
note_to_match = previous_melody[-1]
proximities = []
for i in selected_riff_sample:
    proximities[i] = abs((note_to_match.pitch.midi) - (selected_riff_sample[i].pitch.midi))

lowest_proximity = min(proximities)
next_riff = selected_riff_sample[lowest_proximity]

# append to score
