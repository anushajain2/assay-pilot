"""Local campaign storage: SQLite plus files, append-only by design.

The store keeps campaigns, their rounds of measurements, the plans produced,
and the scorecards those plans earned when results came back.

The central rule is that **history is never rewritten**. Adding a round
creates a new campaign *version*; it does not modify the previous one. Runs
and their selections are inserted once and never updated, so a plan exported
three months ago still reads exactly as it did then, including the predictions
that turned out to be wrong.

Everything stays on the machine it was created on. Nothing here contacts a
network service.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .campaign import Campaign
from .data.loaders import MeasurementSet, load_measurements
from .errors import CampaignError
from .pipeline import ExperimentPlan
from .reporting.manifest import RunManifest
from .scoring import RoundScorecard

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id     TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    current_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_versions (
    campaign_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    definition  TEXT NOT NULL,
    PRIMARY KEY (campaign_id, version),
    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
);

CREATE TABLE IF NOT EXISTS measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    round_id     TEXT,
    variant_id   TEXT NOT NULL,
    sequence     TEXT NOT NULL,
    seq_hash     TEXT NOT NULL,
    mutations    TEXT NOT NULL,
    value        REAL NOT NULL,
    value_raw    REAL NOT NULL,
    censoring    TEXT NOT NULL DEFAULT 'none',
    condition_id TEXT NOT NULL,
    replicate_id TEXT,
    plate_id     TEXT,
    imported_at  TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
);

CREATE INDEX IF NOT EXISTS measurements_campaign
    ON measurements (campaign_id, round_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    campaign_id    TEXT NOT NULL,
    version        INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    method         TEXT NOT NULL,
    evidence_level TEXT NOT NULL,
    policy         TEXT NOT NULL,
    batch_size     INTEGER NOT NULL,
    n_selected     INTEGER NOT NULL,
    manifest       TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
);

CREATE TABLE IF NOT EXISTS selections (
    run_id          TEXT NOT NULL,
    design_id       TEXT NOT NULL,
    candidate_id    TEXT NOT NULL,
    selection_rank  INTEGER NOT NULL,
    sequence        TEXT NOT NULL,
    mutations       TEXT NOT NULL,
    score           REAL,
    predicted_value REAL,
    reason          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, design_id),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS scorecards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    n_matched   INTEGER NOT NULL,
    payload     TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""


@dataclass(frozen=True)
class StoredRun:
    """A plan as recorded in the store."""

    run_id: str
    campaign_id: str
    version: int
    created_at: str
    method: str
    evidence_level: str
    manifest: RunManifest


class CampaignStore:
    """SQLite-backed campaign history.

    Use as a context manager::

        with CampaignStore("campaigns.db") as store:
            store.create_campaign(campaign)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        with self.connection:
            self.connection.executescript(_SCHEMA)
            self.connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------ campaigns

    def create_campaign(self, campaign: Campaign, *, note: str = "initial version") -> int:
        """Store a new campaign and return its version number (1)."""
        existing = self.connection.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign.campaign_id,)
        ).fetchone()
        if existing:
            raise CampaignError(
                f"campaign {campaign.campaign_id!r} already exists in {self.path.name};"
                " use add_round() to extend it or choose another campaign_id"
            )
        with self.connection:
            self.connection.execute(
                "INSERT INTO campaigns (campaign_id, created_at, current_version) VALUES (?, ?, 1)",
                (campaign.campaign_id, campaign.created_at),
            )
            self.connection.execute(
                "INSERT INTO campaign_versions (campaign_id, version, created_at, note,"
                " definition) VALUES (?, 1, ?, ?, ?)",
                (
                    campaign.campaign_id,
                    campaign.created_at,
                    note,
                    json.dumps(campaign.to_dict()),
                ),
            )
        return 1

    def get_campaign(self, campaign_id: str, *, version: int | None = None) -> Campaign:
        """Load a campaign definition, latest version by default."""
        if version is None:
            row = self.connection.execute(
                "SELECT current_version FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if row is None:
                raise CampaignError(f"no campaign {campaign_id!r} in {self.path.name}")
            version = int(row["current_version"])
        row = self.connection.execute(
            "SELECT definition FROM campaign_versions WHERE campaign_id = ? AND version = ?",
            (campaign_id, version),
        ).fetchone()
        if row is None:
            raise CampaignError(f"campaign {campaign_id!r} has no version {version}")
        return Campaign.from_dict(json.loads(row["definition"]))

    def update_campaign(self, campaign: Campaign, *, note: str) -> int:
        """Record a changed campaign definition as a new version."""
        current = self.connection.execute(
            "SELECT current_version FROM campaigns WHERE campaign_id = ?",
            (campaign.campaign_id,),
        ).fetchone()
        if current is None:
            raise CampaignError(f"no campaign {campaign.campaign_id!r} in {self.path.name}")
        version = int(current["current_version"]) + 1
        with self.connection:
            self.connection.execute(
                "INSERT INTO campaign_versions (campaign_id, version, created_at, note,"
                " definition) VALUES (?, ?, datetime('now'), ?, ?)",
                (campaign.campaign_id, version, note, json.dumps(campaign.to_dict())),
            )
            self.connection.execute(
                "UPDATE campaigns SET current_version = ? WHERE campaign_id = ?",
                (version, campaign.campaign_id),
            )
        return version

    def list_campaigns(self) -> pd.DataFrame:
        """Every campaign in the store, with its round and run counts."""
        query = """
            SELECT c.campaign_id,
                   c.created_at,
                   c.current_version,
                   (SELECT COUNT(DISTINCT round_id) FROM measurements m
                     WHERE m.campaign_id = c.campaign_id) AS n_rounds,
                   (SELECT COUNT(*) FROM measurements m
                     WHERE m.campaign_id = c.campaign_id) AS n_measurements,
                   (SELECT COUNT(*) FROM runs r
                     WHERE r.campaign_id = c.campaign_id) AS n_runs
              FROM campaigns c
             ORDER BY c.created_at
        """
        return pd.read_sql_query(query, self.connection)

    def campaign_versions(self, campaign_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT version, created_at, note FROM campaign_versions"
            " WHERE campaign_id = ? ORDER BY version",
            self.connection,
            params=(campaign_id,),
        )

    # --------------------------------------------------------------- measurements

    def add_round(
        self,
        campaign_id: str,
        measurements: MeasurementSet,
        *,
        round_id: str | None = None,
        source: str = "",
        note: str | None = None,
    ) -> int:
        """Append a round of measurements and create a new campaign version.

        Existing measurements, runs, and selections are untouched. The new
        version number is returned.
        """
        campaign = self.get_campaign(campaign_id)
        frame = measurements.frame
        if frame.empty:
            raise CampaignError("no usable measurements to add")

        version = (
            int(
                self.connection.execute(
                    "SELECT current_version FROM campaigns WHERE campaign_id = ?", (campaign_id,)
                ).fetchone()["current_version"]
            )
            + 1
        )

        rows: list[tuple[Any, ...]] = []
        for row in frame.itertuples(index=False):
            rows.append(
                (
                    campaign_id,
                    version,
                    str(round_id)
                    if round_id is not None
                    else (str(row.round_id) if row.round_id is not None else None),
                    row.variant_id,
                    row.sequence,
                    row.seq_hash,
                    row.mutations,
                    float(row.value),
                    float(row.value_raw),
                    row.censoring,
                    row.condition_id,
                    row.replicate_id,
                    row.plate_id,
                    source,
                )
            )

        with self.connection:
            self.connection.executemany(
                "INSERT INTO measurements (campaign_id, version, round_id, variant_id,"
                " sequence, seq_hash, mutations, value, value_raw, censoring, condition_id,"
                " replicate_id, plate_id, imported_at, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)",
                rows,
            )
            self.connection.execute(
                "INSERT INTO campaign_versions (campaign_id, version, created_at, note,"
                " definition) VALUES (?, ?, datetime('now'), ?, ?)",
                (
                    campaign_id,
                    version,
                    note
                    or f"added {len(rows)} measurement(s)"
                    + (f" for round {round_id}" if round_id else ""),
                    json.dumps(campaign.to_dict()),
                ),
            )
            self.connection.execute(
                "UPDATE campaigns SET current_version = ? WHERE campaign_id = ?",
                (version, campaign_id),
            )
        return version

    def measurements_frame(
        self, campaign_id: str, *, rounds: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Raw stored measurement rows for a campaign."""
        query = "SELECT * FROM measurements WHERE campaign_id = ?"
        params: list[Any] = [campaign_id]
        if rounds:
            placeholders = ",".join("?" for _ in rounds)
            query += f" AND round_id IN ({placeholders})"
            params.extend(rounds)
        query += " ORDER BY id"
        return pd.read_sql_query(query, self.connection, params=params)

    def load_measurements(
        self, campaign_id: str, *, rounds: Sequence[str] | None = None
    ) -> MeasurementSet:
        """Stored measurements, re-validated against the current campaign definition.

        Re-validating on the way out means a campaign whose reference sequence
        was corrected cannot silently keep variants that no longer match it.
        """
        campaign = self.get_campaign(campaign_id)
        frame = self.measurements_frame(campaign_id, rounds=rounds)
        if frame.empty:
            return load_measurements(
                pd.DataFrame(columns=["variant_id", "sequence", "value", "condition_id"]),
                campaign,
                strict=True,
            )
        return load_measurements(frame, campaign, strict=False)

    # ----------------------------------------------------------------------- runs

    def save_run(self, plan: ExperimentPlan) -> str:
        """Record a plan. Runs are inserted once and never updated."""
        version = int(
            self.connection.execute(
                "SELECT current_version FROM campaigns WHERE campaign_id = ?",
                (plan.campaign.campaign_id,),
            ).fetchone()["current_version"]
        )
        selections = [
            (
                plan.run_id,
                row.design_id,
                row.candidate_id,
                int(row.selection_rank),
                row.sequence,
                row.mutations,
                float(row.score) if pd.notna(row.score) else None,
                float(row.predicted_value) if pd.notna(row.predicted_value) else None,
                getattr(row, "selection_reason", ""),
            )
            for row in plan.selected.itertuples(index=False)
        ]
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs (run_id, campaign_id, version, created_at, method,"
                " evidence_level, policy, batch_size, n_selected, manifest)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    plan.run_id,
                    plan.campaign.campaign_id,
                    version,
                    plan.manifest.created_at,
                    plan.method_key,
                    plan.evidence_level,
                    plan.outcome.policy,
                    int(plan.outcome.requested),
                    int(len(plan.selected)),
                    plan.manifest.to_json(indent=0),
                ),
            )
            self.connection.executemany(
                "INSERT INTO selections (run_id, design_id, candidate_id, selection_rank,"
                " sequence, mutations, score, predicted_value, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                selections,
            )
        return plan.run_id

    def list_runs(self, campaign_id: str | None = None) -> pd.DataFrame:
        query = (
            "SELECT run_id, campaign_id, version, created_at, method, evidence_level,"
            " policy, batch_size, n_selected FROM runs"
        )
        params: tuple[Any, ...] = ()
        if campaign_id:
            query += " WHERE campaign_id = ?"
            params = (campaign_id,)
        query += " ORDER BY created_at"
        return pd.read_sql_query(query, self.connection, params=params)

    def get_run(self, run_id: str) -> StoredRun:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise CampaignError(f"no run {run_id!r} in {self.path.name}")
        manifest_data = json.loads(row["manifest"])
        known = {f for f in RunManifest.__dataclass_fields__}
        manifest = RunManifest(**{k: v for k, v in manifest_data.items() if k in known})
        return StoredRun(
            run_id=row["run_id"],
            campaign_id=row["campaign_id"],
            version=int(row["version"]),
            created_at=row["created_at"],
            method=row["method"],
            evidence_level=row["evidence_level"],
            manifest=manifest,
        )

    def selections(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM selections WHERE run_id = ? ORDER BY selection_rank",
            self.connection,
            params=(run_id,),
        )

    def latest_run(self, campaign_id: str) -> StoredRun | None:
        row = self.connection.execute(
            "SELECT run_id FROM runs WHERE campaign_id = ? ORDER BY created_at DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        return self.get_run(row["run_id"]) if row else None

    # ----------------------------------------------------------------- scorecards

    def save_scorecard(self, scorecard: RoundScorecard) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO scorecards (run_id, campaign_id, created_at, n_matched, payload)"
                " VALUES (?, ?, datetime('now'), ?, ?)",
                (
                    scorecard.run_id,
                    scorecard.campaign_id,
                    scorecard.n_matched,
                    json.dumps(scorecard.to_dict()),
                ),
            )

    def scorecards(self, campaign_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT run_id, created_at, n_matched, payload FROM scorecards"
            " WHERE campaign_id = ? ORDER BY created_at",
            self.connection,
            params=(campaign_id,),
        )

    def history(self, campaign_id: str) -> pd.DataFrame:
        """One row per run: what was planned, and how it scored when results arrived."""
        runs = self.list_runs(campaign_id)
        if runs.empty:
            return runs
        cards = self.scorecards(campaign_id)
        if cards.empty:
            runs["n_matched"] = 0
            runs["spearman"] = float("nan")
            return runs
        parsed = []
        for row in cards.itertuples(index=False):
            payload = json.loads(row.payload)
            parsed.append(
                {
                    "run_id": row.run_id,
                    "n_matched": payload.get("n_matched"),
                    "spearman": payload.get("metrics", {}).get("spearman"),
                    "hit_rate": payload.get("metrics", {}).get("hit_rate"),
                    "best_observed": payload.get("metrics", {}).get("best_observed"),
                }
            )
        return runs.merge(pd.DataFrame(parsed), on="run_id", how="left")

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CampaignStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_store(path: str | Path) -> CampaignStore:
    """Open (or create) a campaign store at ``path``."""
    return CampaignStore(path)


def iter_rows(frame: pd.DataFrame) -> Iterable[dict[str, Any]]:  # pragma: no cover - helper
    yield from (row._asdict() for row in frame.itertuples(index=False))
