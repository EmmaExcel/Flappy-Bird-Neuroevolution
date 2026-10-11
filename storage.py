import json
import sqlite3
import os

from network import Network


def save_model_json(filepath, network, generation=0, score=0, fitness=0.0):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True) if os.path.dirname(filepath) else None
    data = {
        "topology": network.topology,
        "params": network.get_params().tolist(),
        "generation": generation,
        "score": score,
        "fitness": float(fitness),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_model_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    net = Network(data["topology"])
    net.set_params(data["params"])
    return net, data


class Database:
    def __init__(self, db_path="evolution.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    generation INTEGER PRIMARY KEY,
                    best_score INTEGER,
                    best_fitness REAL,
                    avg_fitness REAL,
                    topology TEXT,
                    best_params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS champion (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    generation INTEGER,
                    score INTEGER,
                    fitness REAL,
                    topology TEXT,
                    params TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def record_generation(self, generation, best_score, best_fitness, avg_fitness, network):
        topo_str = json.dumps(network.topology)
        params_str = json.dumps(network.get_params().tolist())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO generations 
                (generation, best_score, best_fitness, avg_fitness, topology, best_params)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (generation, best_score, float(best_fitness), float(avg_fitness), topo_str, params_str))

            cursor.execute("SELECT fitness FROM champion WHERE id = 1")
            row = cursor.fetchone()
            if row is None or float(best_fitness) > row[0]:
                cursor.execute("""
                    INSERT OR REPLACE INTO champion (id, generation, score, fitness, topology, params, updated_at)
                    VALUES (1, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (generation, best_score, float(best_fitness), topo_str, params_str))
            conn.commit()

    def get_champion(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT generation, score, fitness, topology, params FROM champion WHERE id = 1")
            row = cursor.fetchone()
            if row is None:
                return None, None
            generation, score, fitness, topo_str, params_str = row
            topology = json.loads(topo_str)
            params = json.loads(params_str)
            net = Network(topology)
            net.set_params(params)
            info = {
                "generation": generation,
                "score": score,
                "fitness": fitness,
                "topology": topology,
            }
            return net, info
