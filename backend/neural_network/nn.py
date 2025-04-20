import os
from importlib import import_module
from itertools import combinations
import torch
import neural_network.model as m

# chorales = m21.corpus.chorales.Iterator()
genres = ["jazz", "gospel", "rock"]


# Generates harmonies given a melody sequence
def generate_harmony(melody, genre, pretrained):
    path = f"./neural_network/{genre}/pretrained_metadata.py"

    if genre in genres and os.path.isfile(path):
        metadata = import_module(f"neural_network.{genre}.pretrained_metadata")
        if pretrained:
            model = m.load_model(metadata)
        else:
            print(f"No -pretrained flag: loading {genre} midis...")
            midi_path = f"./neural_network/{genre}/*.midi"
            model = m.train_network(midi_path, metadata)
            m.save_model(model, genre)
        return predict_notes(model, melody)
    else:
        print(f"'{genre}' is an invalid genre.")


def squeeze_note_from_chord_permutations(chord):
    note = m.encoded_notes["C4"]
    if chord and isinstance(chord, tuple):
        if chord in m.encoded_notes:
            note = m.encoded_notes[chord]
        else:
            for i in range(len(chord), 0, -1):
                for combo in combinations(chord, i):
                    if combo in m.encoded_notes:
                        note = m.encoded_notes[combo]
                        break

    # Is single note, not chord
    elif chord in m.encoded_notes:
        note = m.encoded_notes[chord]

    return note


def predict_notes(model, melody):
    tensor_input = []
    for chord in melody:
        note = squeeze_note_from_chord_permutations(chord)
        tensor_input.append(note)

    melodyTensor = torch.tensor(
        tensor_input,
        dtype=torch.long,
    ).unsqueeze(0)
    model.eval()

    # Predict harmony notes from sequence of melody notes
    with torch.no_grad():
        harmony_output, bass_output = model(melodyTensor)
        harmony_output = harmony_output.squeeze(0)
        harmony_predicted = torch.argmax(harmony_output, dim=1)
        bass_output = bass_output.squeeze(0)
        bass_predicted = torch.argmax(bass_output, dim=1)

        # Invert k,v pairs of m.encoded_notes to derive final pitches
        encoded_inverse = {i: note for note, i in m.encoded_notes.items()}
        return [encoded_inverse[i.item()] for i in harmony_predicted], [
            encoded_inverse[i.item()] for i in bass_predicted
        ]
