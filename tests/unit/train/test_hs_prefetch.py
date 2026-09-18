"""Tests for read-through hidden-state cache prewarming."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader, Dataset

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.hs_prefetch import iter_prefetched_batches
from speculators.train.trainer import Trainer


class HiddenStatePrefetchTests(unittest.TestCase):
    def test_reads_files_before_yield_with_split_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hs_10.safetensors").write_bytes(b"a" * 37)
            (root / "hs_11.safetensors").write_bytes(b"b" * 41)
            batches = [np.array([0]), np.array([1])]
            result = list(iter_prefetched_batches(batches, root, 1, 2, 10))
            self.assertEqual([batch.tolist() for batch in result], [[0], [1]])

    def test_rolling_prefetch_yields_ready_batch_without_full_window(self):
        blocked = Event()
        release = Event()
        batches = [np.array([0]), np.array([1]), np.array([2])]

        def warm(batch, _root, _offset):
            if int(batch[0]) == 2:
                blocked.set()
                if not release.wait(3):
                    raise TimeoutError("later batch remained blocked")

        with patch("speculators.train.hs_prefetch._warm_batch", side_effect=warm):
            stream = iter_prefetched_batches(
                batches, Path("/unused"), 2, 2, prewarmed_prefix=1
            )
            try:
                self.assertEqual(next(stream).tolist(), [0])
                self.assertTrue(blocked.wait(2))
                with ThreadPoolExecutor(max_workers=1) as caller:
                    ready = caller.submit(next, stream)
                    self.assertEqual(ready.result(timeout=2).tolist(), [1])
            finally:
                release.set()
            self.assertEqual(next(stream).tolist(), [2])

    def test_missing_file_is_reported_before_batch_is_yielded(self):
        with tempfile.TemporaryDirectory() as tmp:
            batches = [np.array([3])]
            stream = iter_prefetched_batches(batches, Path(tmp), 1, 1)
            with self.assertRaises(FileNotFoundError):
                next(stream)

    def test_dataloader_only_dispatches_warmed_samples(self):
        warmed: set[int] = set()

        class CheckedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                assert index in warmed
                return index

        def record_warm(batch, _root, _offset):
            warmed.update(int(index) for index in batch)

        sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=2,
            lengths=[1] * 4,
            num_replicas=1,
            rank=0,
            hs_prefetch_path=Path("/unused"),
            hs_prefetch_batches=2,
            hs_prefetch_workers=2,
        )
        with patch(
            "speculators.train.hs_prefetch._warm_batch", side_effect=record_warm
        ):
            loader = DataLoader(
                CheckedDataset(), batch_sampler=sampler, num_workers=0
            )
            rows = [batch.tolist() for batch in loader]
        self.assertEqual(sum(len(row) for row in rows), 4)

    def test_startup_prefetch_warms_one_batch_during_model_setup(self):
        sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=2,
            lengths=[1] * 4,
            num_replicas=1,
            rank=0,
            hs_prefetch_path=Path("/unused"),
            hs_prefetch_batches=16,
            hs_prefetch_workers=1,
        )
        trainer = Trainer.__new__(Trainer)
        trainer.train_loader = SimpleNamespace(batch_sampler=sampler)
        trainer.config = SimpleNamespace(num_epochs=1)
        trainer.current_epoch = 0
        trainer._resume_local_step = 0
        trainer._initial_hs_executor = None
        trainer._initial_hs_future = None
        started = Event()
        release = Event()

        def pause_warming(*_args):
            started.set()
            if not release.wait(2):
                raise TimeoutError("test warmup timed out")

        with patch(
            "speculators.train.trainer.warm_initial_batches",
            side_effect=pause_warming,
        ):
            trainer._start_initial_hs_prefetch()
            try:
                self.assertTrue(started.wait(1))
                self.assertFalse(trainer._initial_hs_future.done())
            finally:
                release.set()
            trainer._wait_initial_hs_prefetch()
        self.assertEqual(sampler._prewarmed_batch_count, 1)

    def test_startup_prefetch_uses_remaining_batches_on_resume(self):
        sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=2,
            lengths=[1] * 8,
            num_replicas=1,
            rank=0,
            hs_prefetch_path=Path("/unused"),
            hs_prefetch_batches=2,
            hs_prefetch_workers=1,
        )
        trainer = Trainer.__new__(Trainer)
        trainer.train_loader = SimpleNamespace(batch_sampler=sampler)
        trainer.config = SimpleNamespace(num_epochs=3)
        trainer.current_epoch = 1
        trainer._resume_local_step = 1
        trainer._initial_hs_executor = None
        trainer._initial_hs_future = None

        with patch("speculators.train.trainer.warm_initial_batches") as warm:
            trainer._start_initial_hs_prefetch()
            trainer._wait_initial_hs_prefetch()

        expected = sampler._generate_batches(1)[1:2]
        actual = warm.call_args.args[0]
        self.assertEqual(
            [item.tolist() for item in actual],
            [item.tolist() for item in expected],
        )
        self.assertIsNone(trainer._initial_hs_future)
        self.assertEqual(sampler._prewarmed_batch_count, 1)

        # The resume path slices skipped batches before DataLoader iteration.
        all_batches = sampler._generate_batches(1)
        sampler._cached_generated_batches = (1, all_batches[1:])
        sampler.set_epoch(1)
        with patch("speculators.train.hs_prefetch._warm_batch") as read:
            remaining = [batch.tolist() for batch in sampler]
        self.assertEqual(remaining, [batch.tolist() for batch in all_batches[1:]])
        self.assertEqual(read.call_count, 2)
        self.assertEqual(read.call_args.args[0].tolist(), all_batches[-1].tolist())

    def test_sampler_uses_new_epoch_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(8):
                (root / f"hs_{i}.safetensors").write_bytes(bytes([i]))
            sampler = MultipackDistributedBatchSamplerV2(
                batch_max_length=4,
                lengths=[1] * 8,
                num_replicas=1,
                rank=0,
                hs_prefetch_path=root,
                hs_prefetch_batches=2,
                hs_prefetch_workers=2,
            )
            for epoch in (0, 1):
                sampler.set_epoch(epoch)
                expected = [
                    batch.tolist() for batch in sampler._generate_batches(epoch)
                ]
                actual = [batch.tolist() for batch in sampler]
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
