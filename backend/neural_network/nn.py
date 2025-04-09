import os
import torch
import torch.nn as nn
import torch.optim as optim
import music21 as m21

import pretrained_metadata as metadata

# Load built-in corpus
chorales = m21.corpus.chorales.Iterator()

# Notes are encoded in tuples of three possible and unique notes
encodedNotes = {}

DEFAULT_EPOCH = 250


# Simple LSTM-based model to predict harmony from melody
class HarmonyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super(HarmonyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# Save recently trained model to pretrained path
def save_model(model):
    torch.save(model.state_dict(), "./pretrained")
    with open("./pretrained_metadata.py", "w") as file:
        file.write(f"CORPUS_ENCODED_SIZE={len(encodedNotes)}\n")
        file.write(f"NOTE_TO_IDX={encodedNotes}\n")
        file.write("""PRETRAINED_PATH='./pretrained'\n""")


# Load pretrained model and model metadata
def load_model():
    model = HarmonyModel(metadata.CORPUS_ENCODED_SIZE)
    if os.path.exists(metadata.PRETRAINED_PATH):
        model.load_state_dict(torch.load(metadata.PRETRAINED_PATH))
        model.eval()

        global encodedNotes
        encodedNotes = metadata.NOTE_TO_IDX
    else:
        print("No pretrained model found. Training new model.")
        model = train_network()
    return model


# Extracts melody-harmony pairs from the music21 corpus with harmony as chords
def process_corpus_data():
    sequences = []

    for chorale in chorales:
        mNotes = []
        hNotes = []

        # Store parts by name
        parts = {"Soprano": None, "Alto": None, "Tenor": None, "Bass": None}

        # Tranpose Note objects from music21 corpus
        for part in chorale.parts:
            if part.partName:
                for voice in parts.keys():
                    if voice in part.partName:
                        parts[voice] = part.flatten().notes

        # Tranpose melody-harmony from Note objects
        if parts["Soprano"]:
            for note in parts["Soprano"]:
                if note.isNote:
                    mNotes.append(note.nameWithOctave)

                    # Harmony = Alto + Tenor + Bass
                    chord = []
                    for voice in ["Alto", "Tenor", "Bass"]:
                        if parts[voice]:
                            matching_notes = parts[voice].getElementsByOffset(
                                note.offset, mustBeginInSpan=False
                            )
                            (
                                chord.append(
                                    matching_notes[0].nameWithOctave
                                    if matching_notes
                                    else "Rest"
                                )
                            )

                    hNotes.append(tuple(chord))

        if mNotes and hNotes and len(mNotes) == len(hNotes):
            sequences.append((mNotes, hNotes))

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
        )
        for melody, harmony in sequences
    ]

    return encodedData


# Train a new PyTorch model with hyperparams and optimizers per the tutorials
def train_network():
    # Generate encoded training data from music21 corpus
    sequences = process_corpus_data()
    encodedData = encode_sequences(sequences)

    # Hyperparameters per the PyTorch tutorials
    vocab_size = len(encodedNotes)
    model = HarmonyModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop by # of epochs
    epochs = DEFAULT_EPOCH
    for epoch in range(epochs):
        CELoss = 0
        for melody, harmony in encodedData:
            optimizer.zero_grad()
            output = model(melody.unsqueeze(0))
            loss = criterion(output.squeeze(0), harmony)
            loss.backward()
            optimizer.step()
            CELoss += loss.item()

        # Useful for tracking training progress when training neural network
        # Cross-Entropy Loss of < 1.3 is achieved with DEFAULT_EPOCH
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {CELoss / len(encodedData):.4f}")

    return model


# Generates harmonies given a melody sequence
def generate_harmony(model, melodySequence):
    melodyTensor = torch.tensor(
        [encodedNotes[n] for n in melodySequence], dtype=torch.long
    ).unsqueeze(0)
    model.eval()

    # Predict harmony notes from sequence of melody notes
    with torch.no_grad():
        output = model(melodyTensor).squeeze(0)
        predictedNotes = torch.argmax(output, dim=1)

        # Invert k,v pairs of encodedNotes to derive final pitches
        encodedNotesInverse = {i: note for note, i in encodedNotes.items()}
        return [encodedNotesInverse[i.item()] for i in predictedNotes]
