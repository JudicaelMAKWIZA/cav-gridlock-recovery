"""Écritures atomiques et identités des expériences, sans modifier les sources."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_file(path):
    return json.loads(Path(path).read_text("utf-8"))


@contextmanager
def single_writer(folder):
    """Refuse un second écrivain ; un verrou mort reste tracé avant reprise."""
    import psutil

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "writer.lock"
    if lock.exists():
        previous = json_file(lock)
        try:
            process = psutil.Process(previous["pid"])
            alive = abs(process.create_time() - previous["created"]) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError(f"Un écrivain est déjà actif : PID {previous['pid']}")
        lock.rename(folder / f"interrupted-lock-{time.time_ns()}.json")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                dict(pid=os.getpid(), created=psutil.Process().create_time()), stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def safe_destination(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "Les résultats doivent être hors du corpus immuable, sans dossier parent commun comme cible."
        )
    return output


def verify_cache_files(folder):
    """Vérifie les octets du cache avant une consultation ou une reprise explicite."""
    folder = Path(folder)
    manifest = json_file(folder / "complete.json")
    for relative, expected in manifest["files"].items():
        if file_hash(folder / relative) != expected:
            raise ValueError(f"Cache modifié ou incomplet : {relative}")
    return manifest


def protect_frozen_dataset(output):
    """Empêche les commandes du laboratoire d'écrire dans le corpus gelé."""
    dataset = Path(
        os.environ.get("CGR_M02_ROOT", "D:/cav-recovery-data/metadrive-dense-20261018")
    ).resolve()
    safe_destination(dataset, output)
