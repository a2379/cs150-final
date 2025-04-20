# CS 150 final project

## Team members

- Antero Mejr, UTLN amejr01
- Easton Kang, UTLN ekang05
- Chaz Beauchamp, UTLN cbeauc01

## Usage (GUI)

- Run `./run-project.sh` to:
    - Install backend (pip3), frontend (pnpm) dependencies
    - Run backend, frontend servers
- Navigate your web browser to the URL `http://localhost:5173`
- Generated music will be output to MuseScore (stream.show())

## Usage (cli)


## Composition approach

The project provides a “Groovebox” or graphical step-sequencer program that
uses higher order multidimensional cellular automata and neural networks to
produce a song. The algorithmic method is a hybrid approach that starts with the
user setting the on/off status of each "cell" in the step sequencer grid. This
is translated into the starting state of the Cellular Automata. The Automata is
run for a certain number of cycles (depending on the length of the song), then
that input is passed into a pre-trained neural network, which adds additional
harmony parts using a network. The output of the network is then translated into
Music21 data structures which can be played.
