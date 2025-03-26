#! /usr/bin/env python3
"""CS 150 final project"""

import argparse

from flask import Flask, request

# Algorithmic stuff

# API endpoints

app = Flask(__name__)

@app.route("/")
def default():
    return "WIP"

@app.route("/api/update-base-melody", methods=["POST"])
def update_base_melody():
    pass

@app.route("/api/add-algorithm", methods=["POST"])
def add_algorithm():
    pass

@app.route("/api/delete-algorithm", methods=["POST"])
def delete_algorithm():
    pass

@app.route("/api/set-bpm", methods=["POST"])
def set_bpm():
    pass

@app.route("/api/play", methods=["GET"])
def play():
    pass

@app.route("/api/stop", methods=["GET"])
def stop():
    pass

# Argument parser and entry point

def parse_args():
    parser = argparse.ArgumentParser(description="Run the MelodyLab server")
    parser.add_argument("-p", "--port", type=int, default=5000,
                        help="Change server port (not usually required)")
    parser.add_argument("-i", "--host", type=str, default="0.0.0.0",
                        help="Run server on host IP (not usually required)")
    parser.add_argument("-b", "--browser", action='store_true',
                        help="Try to launch specified browser using xdg-open")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Show Flask debug info")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.browser:
        subprocess.run(["xdg-open", f"https://{args.host}:{args.port}"])
    app.run(host=args.host, port=args.port, debug=args.debug)
