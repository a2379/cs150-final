"""CS 150 final project - Automata component"""

from music21 import *
import neural_network.nn as nn
import rhythm.rhythm as rhythm

threshold = 0.5

# External functions


def grid_to_stream(grid1: list, grid2: list, bpm: int):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(bpm))
    p = stream.Part()
    p.append(clef.TrebleClef())
    p.append(key.KeySignature(0))  # C major
    # p.append(key.KeySignature(-3))  # C minor
    p.append(instrument.Guitar())
    n_measures = 16
    notes_per_measure = 16
    nn_input = []
    for i in range(n_measures * notes_per_measure):
        next_state = transition_function(grid1, grid2)
        # if i % 4 == 0:
        nn_input.append(state_to_nn_input(next_state))
        p.append(state_to_music21_dynamics(next_state))
        p.append(state_to_music21_note(next_state))
        grid1 = grid2
        grid2 = next_state
    s.append(p)

    # All Python lists
    harmony, bass = nn.generate_harmony(nn_input, "jazz")

    harmony_part = stream.Part()
    harmony_part.insert(0, instrument.Piano())
    bass_part = stream.Part()
    bass_part.insert(0, instrument.Piano())
    bass_part.append(clef.BassClef())

    rhythm_gen = rhythm.RhythmGenerator()
    # final_melody = rhythm_gen.arrange_piece(nn_input)
    final_harmony = rhythm_gen.arrange_piece(harmony)
    # final_bass = rhythm_gen.arrange_piece(bass)
    print(final_harmony)

    # (melody, harmony, bass)
    convert_to_music21((harmony_part, bass_part), final_harmony, bass)

    s.append(harmony_part)
    s.append(bass_part)
    return s


# Music21
def convert_to_music21(sections, h, b):
    for measure in h:
        for item in measure:
            hNote = chord.Chord(item[0]) if item[0] else note.Rest()
            hNote.quarterLength = item[1] / 4
            sections[0].append(hNote)

        # bNote = chord.Chord(b[i]) if b[i] else note.Rest()
        # bNote.quarterLength = 0.25
        # sections[1].append(bNote)


def state_to_nn_input(s):
    l = [int_to_note_str(i) for i, x in enumerate(s) if x > threshold]
    list_len = len(l)
    if list_len == 1:
        return l[0]
    elif list_len > 3:
        return tuple(l[:3])
    else:
        return tuple(l)


def int_to_note_str(x):
    if x < 44:
        oct = "4"
    else:
        oct = "5"
    note_lut = {
        0: "C",
        1: "D",
        2: "D#",
        # Skip fourth (F)
        3: "G",
        4: "G#",
        # Skip seventh
        5: "C",
    }
    note_lut.update({x + 5: y for x, y in note_lut.items()})
    return note_lut[x] + oct


def state_to_music21_dynamics(s: list):
    notes_active = [x for x in s if x > threshold]
    avg = sum(notes_active) / max(len(notes_active), 1.0)
    return dynamics.Dynamic(avg)


def state_to_music21_note(s: list):
    notes = [
        note.Note(index_to_pitch(i), quarterLength=0.25)
        for i, x in enumerate(s)
        if x > threshold
    ]
    if notes:
        return chord.Chord(notes, quarterLength=0.25)
    else:
        return note.Rest(quarterLength=0.25)


def scale_index(idx):
    # Minor scale (commented out)
    """lut = {
        0: 0,
        1: 2, # Second is 2 semitones
        2: 3, # Minor third is 3 semitones
        3: 5,
        4: 7,
        5: 8, # Flat sixth
        6: 10,
        7: 12, # Octave
    }"""
    # Hira Joshi pentatonic minor
    lut = {
        0: 0,
        1: 2,
        2: 3,
        # Skip fourth
        3: 7,
        4: 8,
        # Skip seventh
        5: 12,
    }
    lut2 = {x + 5: y + 12 for x, y in lut.items()}  # Second octave dict
    lut.update(lut2)  # Merge them
    return lut[idx]


def index_to_pitch(idx):
    base_octave = 4
    notes_per_octave = 12
    return base_octave * notes_per_octave + scale_index(idx)


# Automata


def transition_function(s1, s2):
    """About the rule:

    - 2-octave range across an 8-note minor scale (list length is 16)
    - Wolfram rule type is 31 (considering next 2 notes on left/right)
    - Pseudo-infinite boundaries using an extra 2 padding cells on each end
    - Padding cells are assumed to be at threshold
    - If t-1 center cell is above threshold, subtract t-1 neighbors
    - If t-1 center cell is below threshold, add t-1 neighbors
    - Further neighbors have smaller influence
    - Reverse this relation for t-2, but with smaller weights overall
    """
    scale_size = 10  # 10 for pentatonic, 16 for regular
    l = [0.0] * scale_size
    padding = [0.0, 0.0]
    l = padding + l + padding
    s1 = padding + s1 + padding
    s2 = padding + s2 + padding
    scale = 0.5
    inner_weight_t1 = 0.5 * scale
    outer_weight_t1 = 0.3 * inner_weight_t1
    inner_weight_t2 = 0.2 * inner_weight_t1
    outer_weight_t2 = 0.1 * inner_weight_t2
    stop_index = len(l) - 2
    for i in range(2, stop_index):
        center = s1[i]
        if center > threshold:
            l[i] = (
                center
                - threshold
                - (inner_weight_t1 * s1[i - 1])
                - (outer_weight_t1 * s1[i - 2])
                - (inner_weight_t1 * s1[i + 1])
                - (outer_weight_t1 * s1[i + 2])
                - (inner_weight_t2 * s2[i - 1])
                - (outer_weight_t2 * s2[i - 2])
                - (inner_weight_t2 * s2[i + 1])
                - (outer_weight_t2 * s2[i + 2])
            )
        else:
            l[i] = (
                center
                + (inner_weight_t1 * s1[i - 1])
                + (outer_weight_t1 * s1[i - 2])
                + (inner_weight_t1 * s1[i + 1])
                + (outer_weight_t1 * s1[i + 2])
                + (inner_weight_t2 * s2[i - 1])
                + (outer_weight_t2 * s2[i - 2])
                + (inner_weight_t2 * s2[i + 1])
                + (outer_weight_t2 * s2[i + 2])
            )
    # Remove padding and normalize
    l = [normalize(x) for x in l[2:-2]]
    return l


def normalize(x):
    return max(min(x, 1.0), 0.0)
