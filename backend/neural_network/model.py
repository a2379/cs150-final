import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import music21 as m21
from neural_network.validator import ValidatorNet

VALIDATOR_EPOCHS = 100  # Default epochs for validator training
OVERFIT_MAX = 3  # If training overfits 3 times, finish training

# Notes are encoded in tuples of three possible and unique notes
encoded_notes = {}
processed_midis = []


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
    torch.save(model.state_dict(), f"./neural_network/{genre}/pretrained")
    with open(f"./neural_network/{genre}/pretrained_metadata.py", "w") as file:
        file.write(f"CORPUS_ENCODED_SIZE={len(encoded_notes)}\n")
        file.write(f"PROCESSED_MIDIS={processed_midis}\n")
        file.write(f"NOTE_TO_IDX={encoded_notes}\n")
        file.write(f"MODEL_PATH='./neural_network/{genre}/pretrained'\n")


# Load pretrained model and model metadata
def load_model(metadata):
    model = HarmonyModel(metadata.CORPUS_ENCODED_SIZE)
    if os.path.exists(metadata.MODEL_PATH):
        model.load_state_dict(torch.load(metadata.MODEL_PATH))
        model.eval()

        global encoded_notes, encoded_data
        encoded_notes = metadata.NOTE_TO_IDX
    else:
        print("No pretrained model found. Run without -pretrained flag.")
    return model


def extract_pitches(obj):
    if not obj:
        return ()
    if obj.isNote:
        return obj.nameWithOctave
    elif obj.isChord:
        return tuple(p.nameWithOctave for p in obj.pitches)


# Extracts melody-harmony pairs from the music21 corpus with harmony as chords
def process_midis(midiPath):
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

        global processed_midis
        if mNotes and hNotes and bNotes and len(mNotes) == len(hNotes) == len(bNotes):
            processed_midis.append((mNotes, hNotes, bNotes))


# Encodes an index to a tuple of three unique notes as tensors for training
def encode_sequences():
    global encoded_notes, processed_midis

    uniqueNotes = set(n for seq in processed_midis for notes in seq for n in notes)
    encoded_notes = {note: i for i, note in enumerate(uniqueNotes)}
    encoded_data = [
        (
            torch.tensor([encoded_notes[n] for n in melody], dtype=torch.long),
            torch.tensor([encoded_notes[n] for n in harmony], dtype=torch.long),
            torch.tensor([encoded_notes[n] for n in bass], dtype=torch.long),
        )
        for melody, harmony, bass in processed_midis
    ]

    return encoded_data


def load_processed_midis(midiPath, metadata):
    if os.path.exists(metadata.MODEL_PATH) and False:
        global processed_midis
        print("Loading processed midis...")
        processed_midis = metadata.PROCESSED_MIDIS
        encoded_data = encode_sequences()
    else:
        """
        Processes all MIDI files again, and takes > 1 hour per genre.
        pre-processed MIDI data is provided, so this code should be unreachable.
        """
        print("No processed midis found.\nProcessing midis (this may take a while...).")
        process_midis(midiPath)
        encoded_data = encode_sequences()

    return encoded_data


def train_single_model(model, encoded_data, label):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epoch = overfit_count = 0
    anchor_loss = float("inf")
    while overfit_count < OVERFIT_MAX:
        epoch += 1
        CELoss = 0
        for melody, harmony, bass in encoded_data:
            optimizer.zero_grad()
            h_output, b_output = model(melody.unsqueeze(0))
            hloss = criterion(h_output.squeeze(0), harmony)
            bloss = criterion(b_output.squeeze(0), bass)
            loss = hloss + bloss
            loss.backward()
            optimizer.step()
            CELoss += loss.item()

        # Calculate loss, and handle overfitting (resets to 0 if not)
        avg_loss = CELoss / len(encoded_data)
        overfit_count = overfit_count + 1 if avg_loss > anchor_loss else 0
        anchor_loss = avg_loss

        if epoch % 25 == 0:
            print(f"[Model {label}] Epoch {epoch}, Loss: {avg_loss:.4f}")


def train_network(midiPath, metadata, seedA=42, seedB=123):
    encoded_data = load_processed_midis(midiPath, metadata)

    def seed_model(seed):
        torch.manual_seed(seed)
        model = HarmonyModel(len(encoded_notes))
        return model

    modelA = seed_model(seedA)
    modelB = seed_model(seedB)
    train_single_model(modelA, encoded_data, "A")
    print("Model A Trained")
    train_single_model(modelB, encoded_data, "B")
    print("Model B Trained")

    validator_input, validator_labels = [], []
    for melody, _, _ in encoded_data:
        with torch.no_grad():
            outA_h, _ = modelA(melody.unsqueeze(0))
            outB_h, _ = modelB(melody.unsqueeze(0))

        validator_input.append(outA_h.squeeze(0).mean(dim=0))
        validator_labels.append(0)
        validator_input.append(outB_h.squeeze(0).mean(dim=0))
        validator_labels.append(1)

    validator_input = torch.stack(validator_input)
    validator_labels = torch.tensor(validator_labels)

    validator = ValidatorNet(len(encoded_notes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(validator.parameters(), lr=0.001)

    for epoch in range(VALIDATOR_EPOCHS):
        optimizer.zero_grad()
        preds = validator(validator_input)
        loss = criterion(preds, validator_labels)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f"[Validator] Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    print("Validator Trained")

    final_model = HarmonyModel(len(encoded_notes))
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    epoch = overfit_count = 0
    anchor_loss = float("inf")
    while overfit_count < OVERFIT_MAX:
        epoch += 1
        CELoss = ValidatorLoss = 0
        for melody, harmony, bass in encoded_data:
            optimizer.zero_grad()
            out_h, out_b = final_model(melody.unsqueeze(0))
            harmony_pred = out_h.squeeze(0).mean(dim=0).detach()

            validator_out = validator(harmony_pred.unsqueeze(0))
            validator_score = torch.softmax(validator_out, dim=1)[0, 1]

            hloss = nn.CrossEntropyLoss()(out_h.squeeze(0), harmony)
            bloss = nn.CrossEntropyLoss()(out_b.squeeze(0), bass)

            # Final Model: validator score is applied in backward() call
            loss = hloss + bloss - validator_score
            loss.backward()
            optimizer.step()
            CELoss += loss.item()
            ValidatorLoss += validator_score

        avg_loss = CELoss / len(encoded_data) + 1
        avg_val_loss = ValidatorLoss / len(encoded_data)
        overfit_count = overfit_count + 1 if avg_loss > anchor_loss else 0
        anchor_loss = avg_loss

        if epoch % 25 == 0:
            print(
                f"[Final Model] Epoch {epoch}, Loss: {avg_loss:.4f}, Validator Loss: {avg_val_loss:.4f}"
            )

    print("Final Model (Validated) Trained")
    return final_model
