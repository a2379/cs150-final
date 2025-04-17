import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import music21 as m21


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
    path = f"./{genre}/pretrained_metadata.py"

    if os.path.isfile(path):
        metadata = importlib.import_module(f"{genre}.pretrained_metadata")
        model = load_model(metadata)
        print(genre)

        # melody = m21.stream.Part()
        # melody.insert(0, m21.instrument.Piano())
        # harmony = m21.stream.Part()
        # harmony.insert(0, m21.instrument.Piano())
        # bass = m21.stream.Part()
        # bass.insert(0, m21.instrument.Piano())
        # bass.append(m21.clef.BassClef())

        # convert_to_music21(sections, melody_pitches, harmony_pitches, bass_pitches)

        harmony, bass = predict_notes(model, melody)
        return (melody, harmony, bass)
    else:
        print(f"'{genre}' is an invalid genre.")


def predict_notes(model, melody):
    melodyTensor = torch.tensor(
        [encodedNotes[n] for n in melody], dtype=torch.long
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
        return [encodedNotesInverse[i.item()] for i in harmony_predicted], [
            encodedNotesInverse[i.item()] for i in bass_predicted
        ]


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
