"""SQLite store for ML training data (stdlib sqlite3, no server).

A *dataset* fixes the X (inputs) and y (outputs) column lists; samples
are rows of that dataset. Each X column also carries an editable
[min, max] exploration range (used by the LHS generator and freely
editable in the ML tab). A sample starts as "pending" (X known, y
unknown - a hole to fill by running the CFD case), becomes "filled"
once its results.json values are recorded, or "failed" with an error
message. Failed rows are retried by the next fill run.

The database file lives at the repo root (ml_data.db) so datasets
accumulate across GUI sessions and optimization runs.
"""

import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "ml_data.db"


class MLStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS datasets ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "name TEXT UNIQUE, x_columns TEXT, y_columns TEXT,"
                "created TEXT)")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS samples ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "dataset_id INTEGER, x TEXT, y TEXT,"
                "status TEXT, case_name TEXT, error TEXT, created TEXT)")
            try:                      # per-column LHS bounds, added later
                self.conn.execute(
                    "ALTER TABLE datasets ADD COLUMN x_bounds TEXT")
            except sqlite3.OperationalError:
                pass                  # column already exists

    # ------------------------------------------------------------ datasets
    def create_dataset(self, name, x_columns, y_columns, x_bounds=None):
        """Create a dataset; returns its id. Names are unique - creating
        an existing name returns the existing dataset's id. x_bounds is
        an optional {x column: (min, max)} map persisted with the
        dataset so the LHS exploration ranges survive restarts."""
        row = self.conn.execute(
            "SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
        if row:
            return int(row["id"])
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO datasets (name, x_columns, y_columns,"
                " x_bounds, created) VALUES (?, ?, ?, ?, ?)",
                (name, json.dumps(list(x_columns)),
                 json.dumps(list(y_columns)),
                 json.dumps({c: list(v) for c, v in (x_bounds or {}).items()}),
                 time.strftime("%Y-%m-%d %H:%M:%S")))
        return int(cur.lastrowid)

    def datasets(self):
        """All datasets as dicts with decoded column lists."""
        rows = self.conn.execute(
            "SELECT * FROM datasets ORDER BY id").fetchall()
        return [self._dataset(r) for r in rows]

    def dataset(self, ds_id):
        row = self.conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (ds_id,)).fetchone()
        return self._dataset(row) if row else None

    def _dataset(self, row):
        try:
            bounds = {c: tuple(float(b) for b in v) for c, v in
                      json.loads(row["x_bounds"] or "{}").items()}
        except (TypeError, ValueError):
            bounds = {}
        return {"id": int(row["id"]), "name": row["name"],
                "x_columns": json.loads(row["x_columns"]),
                "y_columns": json.loads(row["y_columns"]),
                "x_bounds": bounds,
                "created": row["created"]}

    def set_x_bounds(self, ds_id, bounds):
        """Persist the LHS exploration ranges {x column: (min, max)}."""
        with self.conn:
            self.conn.execute(
                "UPDATE datasets SET x_bounds = ? WHERE id = ?",
                (json.dumps({c: [float(v[0]), float(v[1])]
                             for c, v in bounds.items()}), ds_id))

    def delete_dataset(self, ds_id):
        """Delete a dataset and all of its samples. Returns True when a
        dataset was removed."""
        with self.conn:
            self.conn.execute("DELETE FROM samples WHERE dataset_id = ?",
                              (ds_id,))
            cur = self.conn.execute("DELETE FROM datasets WHERE id = ?",
                                    (ds_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------- samples
    def add_samples(self, ds_id, xs, ys=None):
        """Insert rows; `xs` is a list of {col: value} dicts (missing
        columns are stored as null), `ys` an optional parallel list of
        {col: value} dicts (None -> pending status)."""
        ds = self.dataset(ds_id)
        xcols, ycols = ds["x_columns"], ds["y_columns"]
        ids = []
        with self.conn:
            for k, xv in enumerate(xs):
                yv = ys[k] if ys else None
                if yv is None:
                    yv = {}
                status = "filled" if any(
                    yv.get(c) is not None for c in ycols) else "pending"
                cur = self.conn.execute(
                    "INSERT INTO samples (dataset_id, x, y, status, case_name,"
                    " error, created) VALUES (?, ?, ?, ?, ?, NULL, ?)",
                    (ds_id, _pack(xv, xcols), _pack(yv, ycols),
                     status, None, time.strftime("%H:%M:%S")))
                ids.append(int(cur.lastrowid))
        return ids

    def samples(self, ds_id):
        rows = self.conn.execute(
            "SELECT * FROM samples WHERE dataset_id = ? ORDER BY id",
            (ds_id,)).fetchall()
        return [self._sample(r) for r in rows]

    def _sample(self, row):
        return {"id": int(row["id"]), "x": json.loads(row["x"]),
                "y": json.loads(row["y"]), "status": row["status"],
                "case": row["case_name"], "error": row["error"]}

    def pending(self, ds_id):
        """Rows that still need a CFD run (pending or previously failed)."""
        return [s for s in self.samples(ds_id)
                if s["status"] in ("pending", "failed")]

    def filled(self, ds_id):
        return [s for s in self.samples(ds_id) if s["status"] == "filled"]

    def record_result(self, sample_id, y, case):
        with self.conn:
            self.conn.execute(
                "UPDATE samples SET y = ?, status = 'filled',"
                " case_name = ?, error = NULL WHERE id = ?",
                (json.dumps(y), case, sample_id))

    def record_failure(self, sample_id, error, case):
        with self.conn:
            self.conn.execute(
                "UPDATE samples SET status = 'failed', case_name = ?,"
                " error = ? WHERE id = ?", (case, str(error), sample_id))


def _pack(values, columns):
    """{col: value} -> JSON object with the dataset's column order and
    nulls for anything missing, so rows stay comparable."""
    return json.dumps({c: values.get(c) for c in columns})
