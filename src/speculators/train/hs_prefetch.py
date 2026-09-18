"""Read upcoming hidden-state files through a mounted filesystem.

This warms rclone's VFS disk cache without copying files to a second path.
The caller must pass batches in the same order used by the DataLoader.
"""

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from numpy.typing import NDArray

_READ_SIZE = 8 * 1024 * 1024


def _warm_batch(
    batch: NDArray, hidden_states_path: Path, file_index_offset: int
) -> None:
    buffer = bytearray(_READ_SIZE)
    for index in batch:
        path = hidden_states_path / f"hs_{int(index) + file_index_offset}.safetensors"
        with path.open("rb", buffering=0) as stream:
            while stream.readinto(buffer):
                pass


def iter_prefetched_batches(
    batches: Sequence[NDArray],
    hidden_states_path: Path,
    ahead_batches: int,
    workers: int,
    file_index_offset: int = 0,
    prewarmed_prefix: int = 0,
) -> Iterator[NDArray]:
    """Yield each batch only after its HS files have been read to EOF.

    At most ahead_batches future batches are submitted at a time. The
    background readers run in the main process while DataLoader workers
    consume earlier batches. Nothing is copied or deleted by this code;
    rclone owns its VFS cache and eviction policy.
    """
    if ahead_batches < 1 or workers < 1:
        raise ValueError("HS prefetch batches and workers must both be positive")
    if not 0 <= prewarmed_prefix <= len(batches):
        raise ValueError("prewarmed_prefix must be within the batch list")

    pending: deque[tuple[NDArray, Future[None]]] = deque()
    next_index = prewarmed_prefix
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="hs-prefetch"
    ) as executor:
        while next_index < min(prewarmed_prefix + ahead_batches, len(batches)):
            batch = batches[next_index]
            pending.append(
                (
                    batch,
                    executor.submit(
                        _warm_batch, batch, hidden_states_path, file_index_offset
                    ),
                )
            )
            next_index += 1

        for batch in batches[:prewarmed_prefix]:
            yield batch

        while pending:
            batch, future = pending.popleft()
            future.result()
            if next_index < len(batches):
                upcoming = batches[next_index]
                pending.append(
                    (
                        upcoming,
                        executor.submit(
                            _warm_batch,
                            upcoming,
                            hidden_states_path,
                            file_index_offset,
                        ),
                    )
                )
                next_index += 1
            yield batch


def warm_initial_batches(
    batches: Sequence[NDArray],
    hidden_states_path: Path,
    workers: int,
    file_index_offset: int = 0,
) -> None:
    """Warm the first training window before DataLoader starts iterating."""
    if not batches:
        return
    for _ in iter_prefetched_batches(
        batches, hidden_states_path, len(batches), workers, file_index_offset
    ):
        pass
