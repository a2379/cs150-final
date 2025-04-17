import music21 as m21
from random import choice

NOTES = ["C", "D", "E", "F", "G", "A", "B", "C"]
LOW_BOUND = m21.note.Note("C4").pitch.midi
HIGH_BOUND = m21.note.Note("A5").pitch.midi

"""
Monte Carlo Rules: Riffs 
    1. riff start and end notes must be C4
    2. riffs must be at least 4 quarter durations long
    3. riffs must have at least 4 unique pitches

    3. intervals can't be 4th/7th/tritone
    4. intervals must be within 1 octave higher
    5. notes must be within C4 ~ A5, inclusive
"""
def extract_riffs(notes):
    allowed_intervals = [0, 1, 2, 4, 5]
    melody = []

    melody.append("C4")

    for _ in range(len(notes)):
        while True:
            candidatePitch = choice(NOTES)
            candidateOctave = choice([4, 5])
            candidateNote = f"{candidatePitch}{candidateOctave}"

            candidateNoteObj = m21.note.Note(candidateNote)

            if not (LOW_BOUND <= candidateNoteObj.pitch.midi <= HIGH_BOUND):
                continue

            prev_note = melody[-1]
            intvl = m21.interval.Interval(m21.note.Note(prev_note), candidateNoteObj)

            if abs(intvl.semitones) in allowed_intervals:
                melody.append(candidateNote)
                break

    melody.append("C4")
    return melody

def process_corpus_data():
    chorales = m21.corpus.chorales.Iterator()
    for chorale in chorales:
        notes = 

def if __name__ == "__main__":
    sequences = process_corpus_data()
    extract_riffs([''])
