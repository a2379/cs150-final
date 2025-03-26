#! /usr/bin/env python3
"""CS 150 final project"""

from flask import Flask, request

app = Flask(__name__)

@app.route("/api/update-base-melody", methods=["POST"])
def update_base_melody():
    pass

@app.route("/api/add-algorithm", methods=["POST"])
def add_algorithm():
    pass

if __name__ == "__main__":
    app.run()
