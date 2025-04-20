#! /usr/bin/env python3
"""CS 150 final project"""

import time
import argparse
import random
from automata.automata import grid_to_stream
import neural_network.nn as nn
import rhythm.rhythm as rhythm
from music21 import *
from m21_helper import convert_to_music21
from dataclasses import dataclass
from flask import Flask, request, jsonify


# Global state
@dataclass
class UiState:
    grid1: list
    grid2: list
    bpm: int = 120
    genre: str = "jazz"


# Default Random Grid State (cli usage)
random.seed(22)
ui_state = UiState(
    [random.random() for _ in range(16)], [random.random() for _ in range(16)]
)
app = Flask(__name__)


# API endpoints
@app.route("/")
def default():
    return "No default endpoint - use the API methods."


@app.route("/api/play", methods=["POST"])
def play():
    ui_state.bpm = request.json["bpm"]
    ui_state.genre = request.json["genre"]
    ui_state.grid1 = request.json["grid1"]
    ui_state.grid2 = request.json["grid2"]
    stream = generate(True)
    stream.show()
    return jsonify({"message": "Music Generated"}), 200


def generate(pretrained):
    melody = grid_to_stream(ui_state.grid1, ui_state.grid2, ui_state.bpm)
    harmony, bass = nn.generate_harmony(melody, ui_state.genre, pretrained)
    rhythm_gen = rhythm.RhythmGenerator()
    final_melody = rhythm_gen.arrange_piece(melody)
    final_harmony = rhythm_gen.arrange_piece(harmony)
    final_bass = rhythm_gen.arrange_piece(bass)
    return convert_to_music21(final_melody, final_harmony, final_bass, ui_state.bpm)


# Argument parser and entry point
def parse_args():
    parser = argparse.ArgumentParser(description="Run the MelodyLab server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Change server port (not usually required)",
    )
    parser.add_argument(
        "-i",
        "--host",
        type=str,
        default="127.0.0.1",
        help="Run server on host IP (not usually required)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Show Flask debug info"
    )
    parser.add_argument(
        "-m", "--midi", action="store_true", help="Play song with MIDI and exit"
    )
    parser.add_argument(
        "-s", "--sheet", action="store_true", help="Show sheet music and exit"
    )
    parser.add_argument(
        "-t", "--text", action="store_true", help="Show text representation and exit"
    )
    parser.add_argument("-b", "--bpm", type=int, default=120, help="Override BPM")
    parser.add_argument(
        "-genre", type=str, default="jazz", help="Neural Network genre (default: jazz)"
    )
    parser.add_argument(
        "-pretrained", action="store_true", help="User Pretrained Neural Network Models"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ui_state.bpm = args.bpm
    if args.midi or args.sheet or args.text or args.pretrained:
        start_time = time.time()
        stream = generate(args.pretrained)
        end_time = time.time()
        print(f"{(end_time - start_time) / 60:.2f} minutes")
        if args.midi:
            stream.show("midi")
        elif args.text:
            stream.show("text")
        else:
            stream.show()
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
