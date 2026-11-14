from datetime import UTC, datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest

from common.storage import LocalStore, S3Store, open_store
from tests.conftest import make_raw_issue
from transform.clean import clean_issues
from transform.lake import partition_key, read_partition, read_processed, upsert, write_quarantine

T0 = datetime(2026, 1, 10, tzinfo=UTC)
NOW = datetime(2026, 3, 1, tzinfo=UTC)


@pytest.fixture(params=["local", "s3"])
def store(request, local_store, s3_store):
    return local_store if request.param == "local" else s3_store


def test_bytes_json_gzip_roundtrip(store):
    assert store.get_bytes("missing") is None
    assert store.get_json("missing.json") is None
    store.put_json("a/b.json", {"x": 1})
    store.put_json("a/c.json.gz", [1, 2, 3])
    assert store.get_json("a/b.json") == {"x": 1}
    assert store.get_json("a/c.json.gz") == [1, 2, 3]
    assert store.list_keys("a/") == ["a/b.json", "a/c.json.gz"]
    store.delete("a/b.json")
    assert store.list_keys("a/") == ["a/c.json.gz"]


def test_parquet_helpers(store):
    df = pd.DataFrame({"a": [1, 2]})
    store.put_parquet("p/x.parquet", df)
    pd.testing.assert_frame_equal(store.get_parquet("p/x.parquet"), df)
    assert store.read_parquet_prefix("p/").shape == (2, 1)
    assert store.read_parquet_prefix("nothing/").empty
    assert store.get_parquet("p/none.parquet") is None


def test_open_store_parses_uris(tmp_path, monkeypatch):
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    assert isinstance(open_store(f"file://{tmp_path}"), LocalStore)
    s = open_store("s3://bucket/some/prefix")
    assert isinstance(s, S3Store) and s.bucket == "bucket" and s.prefix == "some/prefix"
    assert s.uri == "s3://bucket/some/prefix"
    monkeypatch.setenv("DATA_URI", str(tmp_path / "env"))
    assert open_store().uri.endswith("env")


def test_s3_prefix_is_transparent(s3_store):
    prefixed = S3Store(s3_store.bucket, "team/a", client=s3_store.s3)
    prefixed.put_json("k.json", {"v": 1})
    assert prefixed.list_keys("") == ["k.json"]
    assert s3_store.list_keys("team/") == ["team/a/k.json"]


def _batch(items, at=NOW):
    return clean_issues(items, "o/r", at)


def test_upsert_insert_update_unchanged_and_idempotent(store):
    first = _batch([make_raw_issue(1, T0), make_raw_issue(2, T0 + timedelta(days=40))])
    s1 = upsert(store, first)
    assert s1 == {"partitions_written": 2, "rows_inserted": 2, "rows_updated": 0, "rows_unchanged": 0}
    assert store.list_keys("processed/") == [
        "processed/issues/repo=o__r/created_month=2026-01/part-0.parquet",
        "processed/issues/repo=o__r/created_month=2026-02/part-0.parquet",
    ]

    later = NOW + timedelta(hours=1)
    second = _batch(
        [
            make_raw_issue(1, T0, closed=T0 + timedelta(days=3), updated=T0 + timedelta(days=3)),  # update
            make_raw_issue(3, T0 + timedelta(days=1)),  # insert, same partition
        ],
        at=later,
    )
    s2 = upsert(store, second)
    assert s2["rows_inserted"] == 1 and s2["rows_updated"] == 1

    s3 = upsert(store, second)  # replay (e.g. Lambda retry) is a no-op
    assert s3["rows_inserted"] == 0 and s3["rows_updated"] == 0 and s3["rows_unchanged"] == 2

    df = read_processed(store).set_index("number").sort_index()
    assert df.index.tolist() == [1, 2, 3]
    assert df.loc[1, "state"] == "closed" and df.loc[1, "resolution_days"] == 3.0
    assert df.loc[1, "first_ingested_at"] == pd.Timestamp(NOW)  # preserved across updates
    assert df.loc[1, "last_ingested_at"] == pd.Timestamp(later)


def test_parquet_uses_microsecond_timestamps_for_athena(local_store):
    upsert(local_store, _batch([make_raw_issue(1, T0)]))
    key = partition_key("o/r", "2026-01")
    schema = pq.read_schema(local_store.root / key)
    assert str(schema.field("created_at").type) == "timestamp[us, tz=UTC]"
    assert str(schema.field("label_names").type).startswith("list<")
    assert read_partition(local_store, key).loc[0, "label_names"] == []


def test_upsert_empty_and_read_empty(store):
    assert upsert(store, _batch([]))["partitions_written"] == 0
    assert read_processed(store).empty


def test_quarantine_written_only_when_needed(store):
    assert write_quarantine(store, pd.DataFrame(), "run", NOW) is None
    bad = _batch([make_raw_issue(1, T0)]).assign(dq_reasons="x;")
    key = write_quarantine(store, bad, "run1", NOW)
    assert key == "quarantine/issues/date=2026-03-01/run1.json"
    assert store.get_json(key)[0]["dq_reasons"] == "x;"
