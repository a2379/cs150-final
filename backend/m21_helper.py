"""
Utilities for converting structured rhythm/harmony/melody data into a music21 stream.
Used for rendering musical scores or MIDI playback based on generated neural net output.
"""

from music21 import *

# Maps fractional quarter note lengths to music21 note types
dots_to_type = {
    0.125: "32nd",
    0.25: "16th",
    0.5: "eighth",
    1.0: "quarter",
    2.0: "half",
    4.0: "whole",
}


# -------------------------- Conversion Function ------------------------------
def convert_to_music21(m, h, b, bpm):
    """
    Converts melody, harmony, and bass data into a full music21 stream.
    """

    s = stream.Stream()
    s.append(tempo.MetronomeMark(bpm))
    m_part = stream.Part()
    m_part.insert(0, instrument.Guitar())
    m_part.append(key.KeySignature(0))  # C major
    h_part = stream.Part()
    h_part.insert(0, instrument.Piano())
    b_part = stream.Part()
    b_part.insert(0, instrument.Piano())
    b_part.append(clef.BassClef())

    for i in range(len(m)):
        for j in range(len(m[i])):
            mNote = chord.Chord(m[i][j][0]) if m[i][j][0] else note.Rest()
            dots = m[i][j][1] / 4
            mNote.quarterLength = dots
            mNote.type = dots_to_type[dots]
            m_part.append(mNote)

        for j in range(len(h[i])):
            hNote = chord.Chord(h[i][j][0]) if h[i][j][0] else note.Rest()
            dots = h[i][j][1] / 4
            hNote.quarterLength = dots
            hNote.type = dots_to_type[dots]
            h_part.append(hNote)

        for j in range(len(b[i])):
            bNote = chord.Chord(b[i][j][0]) if b[i][j][0] else note.Rest()
            dots = b[i][j][1] / 4
            bNote.quarterLength = dots
            bNote.type = dots_to_type[dots]
            b_part.append(bNote)

    s.insert(0, m_part)
    s.insert(0, h_part)
    s.insert(0, b_part)
    return s
