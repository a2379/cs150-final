#! /usr/bin/env python3
"""CS 150 final project"""

import argparse
from dataclasses import dataclass
import subprocess
import itertools

from flask import Flask, request

from automata import grid_to_stream

# Global state

@dataclass
class UiState:
    grid: list
    bpm: int = 120
    proc = None


ui_state = UiState([[0]*64]*64)
app = Flask(__name__)

# API endpoints

@app.route("/")
def default():
    return "No default endpoint - use the API methods."


@app.route("/api/update-grid", methods=["POST"])
def update_grid():
    grid = request.json["grid_state"] # 2D list of lists
    for i, lst in enumerate(grid):
        for j, val in enumerate(lst):
            ui_state.grid[i][j] = val


@app.route("/api/set-speed", methods=["POST"])
def set_bpm():
    ui_state.bpm = request.json["bpm"]


@app.route("/api/play", methods=["GET"])
def play():
    strm = grid_to_stream(ui_state.grid, ui_state.bpm)
    ui_state.proc = Process(target=strm.show, args=("midi",))
    ui_state.proc.start()


@app.route("/api/stop", methods=["GET"])
def stop():
    if ui_state.proc:
        if ui_state.proc.is_alive():
            ui_state.proc.terminate()
            ui_state.proc = None
        else:
            ui_state.proc.join()
            ui_state.proc = None


# Argument parser and entry point

def show(cli_args) -> None:
    strm = grid_to_stream(ui_state.grid, ui_state.bpm)
    if cli_args.midi:
        strm.show("midi")
    elif cli_args.sheet:
        strm.show()
    elif cli_args.text:
        strm.show("text")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MelodyLab server")
    parser.add_argument("-p", "--port", type=int, default=5000,
                        help="Change server port (not usually required)")
    parser.add_argument("-i", "--host", type=str, default="127.0.0.1",
                        help="Run server on host IP (not usually required)")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Show Flask debug info")
    parser.add_argument("-m", "--midi", action='store_true',
                        help="Play song with MIDI and exit")
    parser.add_argument("-s", "--sheet", action='store_true',
                        help="Show sheet music" and exit)
    parser.add_argument("-t", "--text", action='store_true',
                        help="Show text representation and exit")
    parser.add_argument("-b", "--bpm", type=int, default=120,
                        help="Override BPM")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ui_state.bpm = args.bpm
    if args.midi or args.sheet or args.text:
        show(args)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
