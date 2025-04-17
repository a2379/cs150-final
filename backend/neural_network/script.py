import os
import importlib
import music21 as m21
import nn
import time


def convert_to_music21(sections, m, h, b):
    for i in range(len(m)):
        mNote = m21.chord.Chord(m[i])
        mNote.quarterLength = 1.0
        sections[0].append(mNote)

        hNote = m21.chord.Chord(h[i]) if h[i] != "Rest" else m21.note.Rest()
        hNote.quarterLength = 1.0
        sections[1].append(hNote)

        bNote = m21.chord.Chord(b[i]) if b[i] != "Rest" else m21.note.Rest()
        bNote.quarterLength = 1.0
        sections[2].append(bNote)


def generate_harmony(melody_pitches, genre):
    path = f"./{genre}/pretrained_metadata.py"

    if os.path.isfile(path):
        metadata = importlib.import_module(f"{genre}.pretrained_metadata")
        model = nn.load_model(metadata)
        print(genre)
        return nn.generate_harmony(model, melody_pitches)
    else:
        print(f"'{genre}' is an invalid genre.")


def main():
    genre = "rock"

    # start_time = time.time()
    # model = nn.train_network(f"./{genre}/*.midi")
    # nn.save_model(model, f"{genre}")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time / 60:.4f} minutes")
    # return

    melody = m21.stream.Part()
    melody.insert(0, m21.instrument.Piano())
    harmony = m21.stream.Part()
    harmony.insert(0, m21.instrument.Piano())
    bass = m21.stream.Part()
    bass.insert(0, m21.instrument.Piano())
    bass.append(m21.clef.BassClef())

    melody_pitches = [
        ("C4", "E4", "G4"),
        ("C4", "E4", "G4"),
        ("C4", "E4", "G4"),
        ("C4", "E4", "G4"),
        ("E4", "G4"),
        "E4",
        "G4",
        "E4",
        "C4",
        ("E4", "G4"),
        ("E4", "G4"),
        ("E4", "G4"),
        ("E4", "G4"),
        ("C4", "G4"),
        ("C4", "G4"),
        ("C4", "G4"),
        ("C4", "G4"),
        "C4",
        "E4",
        "G4",
        "E4",
        "C4",
    ]

    # harmony_pitches, bass_pitches = nn.generate_harmony(model, melody_pitches)
    harmony_pitches, bass_pitches = generate_harmony(melody_pitches, genre)
    print(melody_pitches)
    print(harmony_pitches)
    print(bass_pitches)

    sections = (melody, harmony, bass)

    convert_to_music21(sections, melody_pitches, harmony_pitches, bass_pitches)

    s = m21.stream.Stream()
    for section in sections:
        s.insert(0, section)

    s.show()


if __name__ == "__main__":
    main()
