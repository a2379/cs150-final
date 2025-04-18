import os
import importlib
import glob
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import music21 as m21

# Load built-in corpus
chorales = m21.corpus.chorales.Iterator()

# Notes are encoded in tuples of three possible and unique notes
encodedNotes = {}


# Simple LSTM-based model to predict harmony from melody
class HarmonyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super(HarmonyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.harmony_fc = nn.Linear(hidden_dim, vocab_size)
        self.bass_fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        h_output = self.harmony_fc(x)
        b_output = self.bass_fc(x)
        return h_output, b_output


# Save recently trained model to pretrained path
def save_model(model, genre):
    torch.save(model.state_dict(), f"./{genre}/pretrained")
    with open(f"./{genre}/pretrained_metadata.py", "w") as file:
        file.write(f"CORPUS_ENCODED_SIZE={len(encodedNotes)}\n")
        file.write(f"NOTE_TO_IDX={encodedNotes}\n")
        file.write(f"PRETRAINED_PATH='./{genre}/pretrained'\n")


# Load pretrained model and model metadata
def load_model(metadata):
    model = HarmonyModel(metadata.CORPUS_ENCODED_SIZE)
    if os.path.exists(metadata.PRETRAINED_PATH):
        model.load_state_dict(torch.load(metadata.PRETRAINED_PATH))
        model.eval()

        global encodedNotes
        encodedNotes = metadata.NOTE_TO_IDX
    else:
        print("No pretrained model found. Training new model.")
        # model = train_network()
    return model


def extract_pitches(obj):
    if not obj:
        return ()
    if obj.isNote:
        return obj.nameWithOctave
    elif obj.isChord:
        return tuple(p.nameWithOctave for p in obj.pitches)


# Extracts melody-harmony pairs from the music21 corpus with harmony as chords
def process_corpus_data(midiPath):
    sequences = []

    for fname in glob.glob(midiPath):
        try:
            stream = m21.converter.parse(fname)
        except m21.midi.MidiException as _:
            continue

        mNotes = []
        hNotes = []
        bNotes = []

        # Store parts by name
        parts = {"Melody": None, "Harmony": None, "Bass": None}

        # Tranpose Note objects from music21 corpus
        try:
            for part in stream.parts:
                # Melody
                if any(s in part.partName for s in ["Percuss", "Drum"]):
                    continue
                elif any(s in part.partName for s in ["Bass", "Brass"]):
                    parts["Bass"] = part.flatten().notes
                elif any(s in part.partName for s in ["Piano", "Melody"]):
                    parts["Melody"] = part.flatten().notes
                else:
                    parts["Harmony"] = part.flatten().notes
        except:
            continue

        print(fname)

        # Tranpose melody-harmony from Note objects
        if parts["Melody"]:
            for note in parts["Melody"]:
                melodyNote = extract_pitches(note)
                if not melodyNote:
                    continue

                harmNote = ()
                if parts["Harmony"]:
                    matching_notes = parts["Harmony"].getElementsByOffset(
                        note.offset, mustBeginInSpan=False
                    )
                    if not matching_notes:
                        continue
                    harmNote = extract_pitches(matching_notes[0])

                bassNote = ()
                if parts["Bass"]:
                    matching_notes = parts["Bass"].getElementsByOffset(
                        note.offset, mustBeginInSpan=False
                    )
                    if not matching_notes:
                        continue
                    bassNote = extract_pitches(matching_notes[0])

                mNotes.append(melodyNote)
                hNotes.append(harmNote)
                bNotes.append(bassNote)

        if mNotes and hNotes and bNotes and len(mNotes) == len(hNotes) == len(bNotes):
            sequences.append((mNotes, hNotes, bNotes))

    return sequences


# Encodes an index to a tuple of three unique notes as tensors for training
def encode_sequences(sequences):
    global encodedNotes
    uniqueNotes = set(n for seq in sequences for notes in seq for n in notes)

    encodedNotes = {note: i for i, note in enumerate(uniqueNotes)}

    encodedData = [
        (
            torch.tensor([encodedNotes[n] for n in melody], dtype=torch.long),
            torch.tensor([encodedNotes[n] for n in harmony], dtype=torch.long),
            torch.tensor([encodedNotes[n] for n in bass], dtype=torch.long),
        )
        for melody, harmony, bass in sequences
    ]

    return encodedData


# Train a new PyTorch model with hyperparams and optimizers per the tutorials
def train_network(midiPath):
    # Generate encoded training data from music21 corpus
    sequences = process_corpus_data(midiPath)
    encodedData = encode_sequences(sequences)

    # Hyperparameters per the PyTorch tutorials
    vocab_size = len(encodedNotes)
    model = HarmonyModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop by # of epochs
    epoch = 0
    overfit_count = 0
    anchor_loss = float("inf")
    while overfit_count < 3:
        # for epoch in range(epochs):
        epoch += 1
        CELoss = 0
        for melody, harmony, bass in encodedData:
            optimizer.zero_grad()
            h_output, b_output = model(melody.unsqueeze(0))
            hloss = criterion(h_output.squeeze(0), harmony)
            bloss = criterion(b_output.squeeze(0), bass)
            loss = hloss + bloss
            loss.backward()
            optimizer.step()
            CELoss += loss.item()

        avg_loss = CELoss / len(encodedData)

        # Useful for tracking training progress when training neural network
        # Cross-Entropy Loss of < 1.3 is achieved with DEFAULT_EPOCH
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if avg_loss > anchor_loss:
            overfit_count += 1
        else:
            overfit_count = 0

        anchor_loss = avg_loss

    return model


# Generates harmonies given a melody sequence
def generate_harmony(melody, genre):
    path = f"./neural_network/{genre}/pretrained_metadata.py"

    if os.path.isfile(path):
        metadata = importlib.import_module(
            f"neural_network.{genre}.pretrained_metadata"
        )
        model = load_model(metadata)

        return predict_notes(model, melody)
    else:
        print(f"'{genre}' is an invalid genre.")


def squeeze_note_from_chord_permutations(chord):
    note = encodedNotes["D4"]
    if chord and isinstance(chord, tuple):
        if chord in encodedNotes:
            note = encodedNotes[chord]
        else:
            for i in range(len(chord), 0, -1):
                for combo in combinations(chord, i):
                    if combo in encodedNotes:
                        note = encodedNotes[combo]
                        break

    # Is single note, not chord
    elif chord in encodedNotes:
        note = encodedNotes[chord]

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

        # Invert k,v pairs of encodedNotes to derive final pitches
        encodedNotesInverse = {i: note for note, i in encodedNotes.items()}
        return [
            encodedNotesInverse[i.item()] if encodedNotesInverse[i.item()] else "C4"
            for i in harmony_predicted
        ], [encodedNotesInverse[i.item()] for i in bass_predicted]


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
