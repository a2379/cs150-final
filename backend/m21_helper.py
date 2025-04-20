from music21 import *


def convert_to_music21(m, h, b, bpm):
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
            mNote.quarterLength = m[i][j][1] / 4
            m_part.append(mNote)

        for j in range(len(h[i])):
            hNote = chord.Chord(h[i][j][0]) if h[i][j][0] else note.Rest()
            hNote.quarterLength = h[i][j][1] / 4
            h_part.append(hNote)

        for j in range(len(b[i])):
            bNote = chord.Chord(b[i][j][0]) if b[i][j][0] else note.Rest()
            bNote.quarterLength = b[i][j][1] / 4
            b_part.append(bNote)

    s.append(m_part)
    s.append(h_part)
    s.append(b_part)
    return s
