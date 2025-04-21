# CS 150 final project

## Team members

- Antero Mejr, UTLN amejr01
- Easton Kang, UTLN ekang05
- Chaz Beauchamp, UTLN cbeauc01

## Usage (GUI)

- Run `./run-project.sh` to:
    - Install backend (pip3), frontend (pnpm) dependencies
    - Run backend (port 5001), frontend servers (port 5173)
- Navigate your web browser to the URL `http://localhost:5173`
- Generated music will be output as midi or sheet

## Usage (cli)

- Navigate to `cd backend`
- Run with `./project.py [options]`

***Remove -pretrained flag, if you wish to train a new neural network.***<br>
***Please review 'Runtimes' section to view estimates of training times (very long).***

| Option                     | Description                                               | Default         |
|----------------------------|-----------------------------------------------------------|-----------------|
| `-s`, `--sheet`            | Show sheet music and exit                                 | `True`          |
| `-m`, `--midi`             | Play song with MIDI and exit                              | `False`         |
| `-t`, `--text`             | Show text representation and exit                         | `False`         |
| `-b`, `--bpm <int>`        | Override BPM of the song                                  | `120`           |
| `-genre <str>`             | Neural Network genre to use                               | `jazz`          |
| `-pretrained`              | Use pretrained Neural Network models                      | `False`         |

### Runtimes (cli)

| Genre   | -pretrained | Runtime                   | stdout phases |
|---------|-------------|---------------------------|---------------|
| gospel  | True        | < 1 second                | No            |
| jazz    | True        | < 2 seconds               | No            |
| rock    | True        | < 2 seconds               | No            |
| gospel  | False       | ~ 6 minutes               | Yes           |
| jazz    | False       | ~ 20 minutes              | Yes           |
| rock    | False       | ~ 64 minutes              | Yes           |

*Shortest training runtime is **6 minutes** (gospel).*<br>
*There are plenty of print outputs to guide you between training phases.*

### Examples (cli)
Generate **sheet** music using the **pretrained gospel** model
- `./project.py -pretrained -s -genre gospel`

**Train new** jazz model, and generate **midi** music
- `./project.py -m -genre jazz`

## Composition approach

<<<<<<< HEAD
The project provides a “Groovebox” or graphical step-sequencer program that
uses higher order multidimensional cellular automata and neural networks to
produce a song. The algorithmic method is a hybrid approach that starts with the
user setting the on/off status of each "cell" in the step sequencer grid. This
is translated into the starting state of the Cellular Automata. The Automata is
run for a certain number of cycles (depending on the length of the song), then
that input is passed into a pre-trained neural network, which adds additional
harmony parts using a network. Next, the output from these parts is modified
via a stochastic binary subdivision algorithm with 'emotional scoring' to
add variety to the rhythm. The resulting composition is then translated into
Music21 data structures which can be played.
=======
The project provides a “Groovebox” or graphical step-sequencer program that uses
second-order multidimensional cellular automata and neural networks to produce a
song. The algorithmic method is a hybrid approach that starts with the user
setting the on/off status of each "cell" in the step sequencer grid. This is
translated into the starting state of the Cellular Automata. The Automata is run
for a certain number of cycles (depending on the length of the song), then that
input is passed into a pre-trained neural network, which adds additional harmony
parts using a network. Then, all parts have their note durations adjusted using
emotion-based stochastic binary subdivision. The output of the network is then
translated into Music21 data structures which can be played.
>>>>>>> 0c8bae839cf2d1c659afbaf58c98bb366a3b27fb
