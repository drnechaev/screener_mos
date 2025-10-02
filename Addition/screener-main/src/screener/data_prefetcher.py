from typing import Any, Literal
import threading
import multiprocessing
import random
import time

from torch.utils.data import Dataset


class DataPrefetcher(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            num_samples_per_epoch: int,
            num_workers: int,
            buffer_size: int,
            clone_factor: int,
            backend: Literal['threading', 'multiprocessing'] = 'threading'
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.buffer_size = buffer_size
        self.clone_factor = clone_factor

        match backend:
            case 'threading':
                worker_cls = threading.Thread
                lock_cls = threading.Lock
                self.stop_event = threading.Event()
            case 'multiprocessing':
                worker_cls = multiprocessing.Process
                lock_cls = multiprocessing.Lock
                self.stop_event = multiprocessing.Event()
            case _:
                raise ValueError(backend)

        self.buffers = [[] for _ in range(num_workers)]
        self.locks = [lock_cls() for _ in range(num_workers)]
        self.workers = []
        for worker_id in range(num_workers):
            worker = worker_cls(target=self._worker, args=(worker_id,), daemon=True)
            worker.start()
            self.workers.append(worker)

    def _worker(self, worker_id: int) -> None:
        while not self.stop_event.is_set():
            data = self.dataset[random.randint(0, len(self.dataset) - 1)]

            with self.locks[worker_id]:
                if len(self.buffers[worker_id]) < self.buffer_size:
                    self.buffers[worker_id].append((data, 0))
                else:
                    time.sleep(0.01)

    def __getitem__(self, index: int) -> Any:
        for worker_id in random.sample(range(len(self.workers)), k=len(self.workers)):
            with self.locks[worker_id]:
                if not self.buffers[worker_id]:
                    time.sleep(0.01)
                    continue

                i = random.randrange(len(self.buffers[worker_id]))
                clone, clones_count = self.buffers[worker_id][i]
                clones_count += 1
                if clones_count >= self.clone_factor:
                    del self.buffers[worker_id][i]
                else:
                    self.buffers[worker_id][i] = (clone, clones_count)
                return clone

        return self.dataset[random.randint(0, len(self.dataset) - 1)]

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def __del__(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
