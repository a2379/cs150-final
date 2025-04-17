"""CS 150 final project - Automata component"""

from music21 import *

# About the data representation

"""
Arrays are generally shaped (I, N, 8) = bool, where:
- I is the I-th instrument
- N refers to the absolute position in terms of 16th nodes
- 8 refers to the 7 notes of a scale (plus an octave)
- bool is the on/off state of the note (similar to 1-hot encoding)

For example, (0, 2, 4) = True means the instrument at index 0 plays a 16th note
of the second note of the scale on the 2nd beat. Multiple True values with an
increasing middle index means the note is held for multiple 16th notes.
"""

# External functions

def grid_to_stream(grid: list, bpm: int):
    s = stream.Stream()
    return s

# Music21 stuff

def scale_idx(idx):
    # Minor scale
    if idx in [0, 1, 3, 4, 6, 7]:
        return 2 * idx
    else:
        return 2 * idx - 1

def idx_to_pitch(idx):
    return 4 * 12 + scale_idx(idx)

def song_array_to_stream(arr: np.ndarray) -> stream.Stream:
    s = stream.Stream()
    last_pitch = {x: False for x in range(8)}
    for inst in arr:
        p = stream.Part()
        for pos in inst:
            for n in pos:
                if n:
                    if last_pitch[n]:
                        last_pitch[n].duration.quarterLength += 0.5
                    else:
                        last_pitch[n] = note.Note(idx_to_pitch(n),
                                                  quarterLength=0.5)
                else:
                    if last_pitch[n]:
                        l = [x for x in last_pitch.values() if x]
                        p.append(chord.Chord(l))
                        last_pitch[n] = False
                    else:
                        pass
        l = [x for x in last_pitch.values() if x]
        p.append(chord.Chord(l))
        s.append(p)
    return s

# Algorithmic stuff

def make_lut(neighbor_range, num):
    # Size of sequence (center cell + neighbors)
    s = neighbor_range[0] + neighbor_range[1] + 1

    # Binary representation of wolfram number (unpadded)
    b = bin(num)[2:]

    # All permutations of sequences
    l = ["".join(seq) for seq in itertools.product("01", repeat=s)]

    # Binary representation of wolfram number (padded)
    bs = b.rjust(len(l), "1")

    # Build lookup table
    lut = {}
    for i, item in enumerate(reversed(l)):
        lut[item] = bs[i]
    return lut

def arr_to_string(arr):
    out = ""
    for obj in arr:
        if obj:
            out += "1"
        else:
            out += "0"
    return out

def rule_num_to_func(neighbor_range: tuple, num: int):
    """Range is pair (left, right). num is Wolfram automata number."""
    left, right = neighbor_range
    lut = make_lut(neighbor_range, num)

    # In: single 1x8 sample
    # Out: next 1x8 sample
    def func(arr: np.ndarray):
        out = np.full((8), False)
        for i, val in enumerate(arr):
            # If range is out of bounds, assume zero.
            subarr = np.take(arr, range(i - left, i + right + 1), mode="wrap")
            out[i] = bool(int(lut[arr_to_string(subarr)]))
        return out

    return func

def get_song_length(sec: int = 10):
    return ((ui_state.bpm // 60) * sec) // 4

def grid_to_song_array(grid: np.ndarray, bpm: int) -> np.ndarray:
    f = rule_num_to_func(ui_state.neighbor_range, ui_state.rule_num)
    l = get_song_length()
    state = np.array(ui_state.grid[0])
    out = np.full((1, l, 8), False)
    for i in range(l):
        tmp = f(state)
        out[0][i] = tmp
        state = tmp
    return out
