import torch


class BufferedDataLoader:
    def __init__(self, dataloader, buffer_times, combined_filter, accelerator):
        self._dataloader = dataloader
        self._buffer_times = buffer_times
        self._combined_filter = combined_filter

        self._batch_size = self._dataloader.batch_size
        self._dataloader = accelerator.prepare(self._dataloader)
        self._device = accelerator.device

        self._buffers = None

    def __iter__(self):
        self._data_iterator = iter(self._dataloader)
        self._buffers = None
        return self

    def __next__(self):
        if self._buffers is None or all(len(buffer) == 0 for buffer in self._buffers):
            self._buffers = ([], [], [])
            try:
                while len(self._buffers[0]) < self._buffer_times * self._batch_size:
                    batch = next(self._data_iterator)
                    filtered_batch = self._combined_filter(batch.to(self._device))
                    for i, tensor in enumerate(filtered_batch):
                        self._buffers[i].append(tensor)
            except StopIteration:
                if not any(self._buffers):
                    raise StopIteration
            self._buffers = [torch.cat(buffers, dim=0) for buffers in self._buffers]

        if len(self._buffers[0]) == 0:
            raise StopIteration

        indices = torch.randperm(len(self._buffers[0]))
        self._buffers = [buffer[indices] for buffer in self._buffers]

        batch_end = min(self._batch_size, len(self._buffers[0]))
        yield_batch = tuple(buffer[:batch_end] for buffer in self._buffers)
        self._buffers = [buffer[batch_end:] for buffer in self._buffers]

        if not any(len(buffer) >= self._batch_size for buffer in self._buffers):
            try:
                next_batch = next(self._data_iterator)
                filtered_next_batch = self._combined_filter(next_batch.to(self._device))
                for i, tensor in enumerate(filtered_next_batch):
                    self._buffers[i] = torch.cat((self._buffers[i], tensor), dim=0)
            except StopIteration:
                pass

        return yield_batch

    def __len__(self):
        return len(self._dataloader)
