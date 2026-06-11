"""The tensor parallelism demo must agree with the unsharded reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


def test_sharded_block_matches_reference():
    import tensor_parallel

    assert tensor_parallel.main() is True
