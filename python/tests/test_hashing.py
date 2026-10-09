import hashlib
import os

from modelport.hashing import sha256_file


def test_small_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_file_larger_than_one_chunk(tmp_path):
    data = os.urandom(3 * (1 << 20) + 5)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
