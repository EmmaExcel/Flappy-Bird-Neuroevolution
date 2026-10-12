# Flappy Bird Neuroevolution

A Pygame Flappy Bird environment with genetic algorithm training for populations of neural networks.

## Features

- Population of 100 birds evaluated concurrently
- Per-bird input vector with normalized position, velocity, and upcoming pipe coordinates
- Environmental dynamics including horizontal wind and variable gravity
- Vertical pipe oscillation
- Genetic algorithm with tournament selection, uniform crossover, Gaussian mutation, and elitism
- Real-time statistics overlay and live fitness graph
- Headless trainer for high-throughput headless execution
- SQLite generation audit log and JSON model export

## Requirements

Python 3.10 or higher.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Visual Mode

Run the game loop with live rendering:

```bash
python main.py
```

Optional flags:

```bash
python main.py --speed 5                # Fast-forward simulation
python main.py --no-wind               # Disable wind drift
python main.py --no-moving             # Disable oscillating pipes
python main.py --no-vargrav            # Keep gravity constant
python main.py --resume best_bird.json # Seed population from saved model
```

Keyboard controls:
- Up Arrow: Increase simulation speed
- Down Arrow: Decrease simulation speed
- Escape: Exit application

### Headless Training

Train headlessly at full CPU throughput:

```bash
python train.py
```

Optional flags:

```bash
python train.py --gens 500             # Stop after specified generation count
python train.py --db evolution.db      # Custom SQLite database target
python train.py --json-out model.json  # Custom export target
python train.py --resume-json model.json
```

### Champion Playback

Load and watch a trained bird play:

```bash
python main.py --play best_bird.json
```

Or load the champion recorded in the SQLite database:

```bash
python main.py --play-champion
```

### Benchmarks

Run headless evaluation comparing different network layer topologies:

```bash
python main.py --compare-topologies --compare-gens 50
```

## Tests

Execute test suite:

```bash
python tests/test_core.py
```

## Project Structure

```
.
├── bird.py            Physics calculations and decision logic
├── config.py          Physics constants, screen metrics, hyperparameters
├── environment.py     Wind drift and gravity variation state
├── game.py            Frame progression, collision checks, score handling
├── hud.py             HUD metrics panel and fitness chart
├── main.py            Visual runner, playback mode, CLI arguments
├── network.py         Feedforward network with NumPy vectorization
├── neuroevolution.py  Topology benchmark routine
├── pipe.py            Pipe spawn calculations and boundary geometry
├── population.py      Selection, crossover, mutation, elitism
├── renderer.py        Pygame canvas drawing
├── storage.py         JSON serialization and SQLite persistence
├── train.py           Headless runner for fast training
└── tests/
    └── test_core.py   Unit tests
```
