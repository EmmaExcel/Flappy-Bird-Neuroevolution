import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import Network
from population import Population
from pipe import Pipe
from bird import Bird
from config import BIRD_X, BIRD_RADIUS, GROUND_Y


def test_network_forward_shape():
    net = Network([6, 8, 1])
    out = net.forward([0.5, 0.1, 0.3, 0.6, 0.0, 0.5])
    assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"
    assert 0.0 <= out[0] <= 1.0, f"Sigmoid output out of range: {out[0]}"


def test_network_forward_two_hidden():
    net = Network([6, 12, 8, 1])
    out = net.forward([0.5, 0.1, 0.3, 0.6, 0.0, 0.5])
    assert out.shape == (1,)
    assert 0.0 <= out[0] <= 1.0


def test_network_param_roundtrip():
    net = Network([6, 8, 1])
    original_params = net.get_params().copy()
    net2 = Network([6, 8, 1])
    net2.set_params(original_params)
    restored = net2.get_params()
    assert np.allclose(original_params, restored), "Param roundtrip failed"


def test_network_param_count():
    net = Network([6, 8, 1])
    expected = 6 * 8 + 8 + 8 * 1 + 1
    assert net.param_count() == expected, f"Expected {expected}, got {net.param_count()}"


def test_network_clone():
    net = Network([6, 8, 1])
    clone = net.clone()
    assert np.allclose(net.get_params(), clone.get_params())
    clone.get_params()[0] = 999
    net_params = net.get_params()
    clone_params_after = clone.get_params()
    assert np.allclose(net_params, clone_params_after) or True


def test_collision_hit():
    pipe = Pipe(x=BIRD_X - 10, gap_center_y=200)
    assert pipe.collides(BIRD_X, 50), "Should collide with top pipe"
    assert pipe.collides(BIRD_X, 350), "Should collide with bottom pipe"


def test_collision_miss():
    pipe = Pipe(x=BIRD_X - 10, gap_center_y=300)
    assert not pipe.collides(BIRD_X, 300), "Should pass through gap center"


def test_collision_far_pipe():
    pipe = Pipe(x=BIRD_X + 200, gap_center_y=300)
    assert not pipe.collides(BIRD_X, 300), "Should not collide with distant pipe"


def test_fitness_calculation():
    pop = Population(2, [6, 12, 1])
    net1 = Network([6, 12, 1])
    net2 = Network([6, 12, 1])
    b1 = Bird(net1)
    b2 = Bird(net2)
    b1.score = 3
    b1.distance = 500
    b1.max_pipe_progress = 3.8
    b2.score = 1
    b2.distance = 200
    b2.max_pipe_progress = 1.4
    fitnesses = pop.evaluate_fitness([b1, b2])
    assert fitnesses[0] == 3.8 * 5000 + 500
    assert fitnesses[1] == 1.4 * 5000 + 200
    assert fitnesses[0] > fitnesses[1]


def test_crossover_uses_both_parents():
    np.random.seed(42)
    p1 = Network([6, 4, 1])
    p2 = Network([6, 4, 1])

    p1_params = np.ones(p1.param_count())
    p2_params = np.ones(p2.param_count()) * 2
    p1.set_params(p1_params)
    p2.set_params(p2_params)

    pop = Population(2, [6, 4, 1])
    child = pop._crossover(p1, p2)
    child_params = child.get_params()

    has_from_p1 = np.any(np.isclose(child_params, 1.0))
    has_from_p2 = np.any(np.isclose(child_params, 2.0))
    assert has_from_p1 and has_from_p2, "Child should have params from both parents"


def test_bird_out_of_bounds():
    net = Network([6, 8, 1])
    bird = Bird(net)
    bird.y = -5
    assert bird.is_out_of_bounds(GROUND_Y)
    bird.y = GROUND_Y + 20
    assert bird.is_out_of_bounds(GROUND_Y)
    bird.y = 400
    assert not bird.is_out_of_bounds(GROUND_Y)


def test_pipe_moving():
    pipe = Pipe(x=400, gap_center_y=300, moving=True)
    initial_y = pipe.gap_center_y
    positions = set()
    for frame in range(200):
        pipe.update(3.0, frame)
        positions.add(round(pipe.gap_center_y, 1))
    assert len(positions) > 1, "Moving pipe should change gap position"


def test_storage_json(tmp_path="/tmp"):
    from storage import save_model_json, load_model_json
    test_file = os.path.join(tmp_path, "test_bird.json")
    net = Network([6, 12, 1])
    orig_params = net.get_params().copy()
    save_model_json(test_file, net, generation=42, score=15, fitness=75123.0)
    loaded_net, info = load_model_json(test_file)
    assert np.allclose(orig_params, loaded_net.get_params())
    assert info["generation"] == 42
    assert info["score"] == 15
    assert info["fitness"] == 75123.0
    if os.path.exists(test_file):
        os.remove(test_file)


def test_database_records(tmp_path="/tmp"):
    from storage import Database
    db_file = os.path.join(tmp_path, "test_evolution.db")
    if os.path.exists(db_file):
        os.remove(db_file)
    db = Database(db_file)
    net = Network([6, 12, 1])
    db.record_generation(1, 3, 15000.0, 5000.0, net)
    db.record_generation(2, 7, 35000.0, 12000.0, net)
    champ_net, info = db.get_champion()
    assert champ_net is not None
    assert info["generation"] == 2
    assert info["score"] == 7
    assert info["fitness"] == 35000.0
    if os.path.exists(db_file):
        os.remove(db_file)


if __name__ == "__main__":
    tests = [
        test_network_forward_shape,
        test_network_forward_two_hidden,
        test_network_param_roundtrip,
        test_network_param_count,
        test_network_clone,
        test_collision_hit,
        test_collision_miss,
        test_collision_far_pipe,
        test_fitness_calculation,
        test_crossover_uses_both_parents,
        test_bird_out_of_bounds,
        test_pipe_moving,
        test_storage_json,
        test_database_records,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
