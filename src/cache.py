from typing import Optional
from collections import OrderedDict


class LRUCache:
    """
    A fixed-capacity Least‑Recently‑Used (LRU) cache: stores key→value pairs and evicts
    the oldest entry when full.
    """

    def __init__(self, capacity: int):
        """
        Initialize an LRU cache with the given maximum capacity.
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key) -> Optional[int]:
        """
        Return the value for key if present, marking it as recently used; otherwise
        return None.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value) -> None:
        """
        Insert or update a key-value pair, evicting the least-recently-used entry if
        capacity is exceeded.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """
        Remove all entries from the cache, resetting it to empty.
        """
        self.cache.clear()
