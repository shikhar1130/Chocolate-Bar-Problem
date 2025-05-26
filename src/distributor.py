import time
import math
import heapq
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

from .cache import LRUCache
from .exceptions import InputError
from .utils import (
    DEFAULT_CACHE_SIZE,
    MAX_INPUT_VALUE,
    DEFAULT_BRUTEFORCE_THRESHOLD,
    DEFAULT_COMPLEMENT_THRESHOLD,
)


class ChocolateDistributor:
    """
    Distribute chocolate bars among children to meet exact demands with minimal cuts,
    using greedy strategies, LRU caching, and optional validation.
    """

    def __init__(
        self,
        cache_size: int = DEFAULT_CACHE_SIZE,
        enable_validation: bool = True,
        enable_sanitization: bool = True,
        bruteforce_threshold: int = DEFAULT_BRUTEFORCE_THRESHOLD,
        complement_threshold: int = DEFAULT_COMPLEMENT_THRESHOLD,
    ):
        """
        Initialize distributor with an LRU cache and feature flags for validation,
        sanitization, and complement look‑ahead.
        """
        self.cache = LRUCache(cache_size)
        self.size_heap = []

        # Configuration flags
        self.enable_validation = enable_validation
        self.enable_sanitization = enable_sanitization
        self.bruteforce_threshold = bruteforce_threshold
        self.complement_threshold = complement_threshold

    def __str__(self) -> str:
        """
        Return a formatted snapshot of the distributor’s configuration, heap state, and
        cache statistics.
        """
        # Heap summary
        if self.size_heap:
            smallest, largest = min(self.size_heap), max(self.size_heap)
            heap_summary = f"{len(self.size_heap)} pieces (sizes {smallest}–{largest})"
        else:
            heap_summary = "empty"

        # Cache summary
        cache = self.cache.cache
        if cache:
            sample_keys = list(cache)[:3]
            cache_summary = (
                f"{len(cache)}/{self.cache.capacity} entries; "
                f"sample keys: {sample_keys}"
            )
        else:
            cache_summary = "empty"

        return (
            "ChocolateDistributor(\n"
            f"  validation={'on' if self.enable_validation else 'off'},\n"
            f"  sanitization={'on' if self.enable_sanitization else 'off'},\n"
            f"  brute-force  threshold={self.bruteforce_threshold}\n"
            f"  complement threshold={self.complement_threshold}\n"
            ")\n"
            f"Heap: {heap_summary}\n"
            f"Cache: {cache_summary}"
        )

    def sanitize_input(self, value: Any) -> Optional[int]:
        """
        Ensure a single input value is a valid positive integer (or convertible),
        raising InputError otherwise.
        """
        try:
            if value is None:
                raise InputError("Value is None")
            if isinstance(value, str):
                if value.strip().isdigit():
                    return int(value)
                raise InputError(f"Invalid string input: '{value}'")
            if isinstance(value, float):
                if math.isinf(value) or math.isnan(value):
                    raise InputError(f"Invalid float: {value}")
                if value.is_integer():
                    return int(value)
                raise InputError(f"Non-integer float: {value}")
            if isinstance(value, int):
                if value <= 0:
                    raise InputError(
                        f"Bar/requirement must be positive, Current input: {value}"
                    )
                if value > MAX_INPUT_VALUE:
                    raise InputError(f"Unrealistically large value: {value}")
                return value
            raise InputError(f"Unrecognized input type: {type(value)}")
        except Exception as e:
            raise InputError(f"Sanitization failed for '{value}': {str(e)}")

    def validate_and_convert_inputs(
        self, bars: List[Any], children: List[Any]
    ) -> Tuple[List[int], List[int]]:
        """
        Convert and validate lists of bars and children’s demands into positive integer
        lists.
        """
        if not self.enable_validation:
            return ([int(b) for b in bars], [int(c) for c in children])

        # Validate that 'items' is a non-empty list or tuple; raise error if not
        def convert_list(items: List[Any], name: str) -> List[int]:
            if not isinstance(items, (list, tuple)):
                raise InputError(f"{name} must be a list or tuple")
            if not items:
                raise InputError(f"Empty {name} provided")

            # Sanitize and convert each item
            converted_list = []
            for item in items:
                if self.enable_sanitization:
                    value = self.sanitize_input(item)
                else:
                    value = int(item)
                if value is None or value <= 0:
                    raise InputError(f"Invalid {name.rstrip('s')} value: {item}")
                converted_list.append(value)
            return converted_list

        return (convert_list(bars, "bars"), convert_list(children, "children"))

    def find_exact_match(self, inventory: Dict[int, int], requirement: int) -> bool:
        """
        Attempt to satisfy a child by consuming a piece that exactly matches their
        requirement.
        """
        if requirement in inventory and inventory[requirement] > 0:
            inventory[requirement] -= 1  # Decrease count of the piece
            if inventory[requirement] == 0:  # If no pieces left, remove from inventory
                del inventory[requirement]
            return True
        return False

    def brute_force_cuts(
        self, children: List[int], inventory: Dict[int, int]
    ) -> Optional[int]:
        """
        Recursively try all combinations of pieces.
        Brute force solution for small inputs (Number of childen < Threshold)
        """

        def try_all_combinations(
            demands: List[int], inventory: Dict[int, int], cuts: int
        ) -> Optional[int]:
            if not demands:
                return cuts
            requirement = demands[0]
            min_cuts = float("inf")

            # Try each piece
            for size, count in list(inventory.items()):
                if count <= 0:
                    continue
                # Exact match
                if size == requirement:
                    inventory[size] -= 1
                    result = try_all_combinations(demands[1:], inventory, cuts)
                    inventory[size] += 1
                    if result is not None:
                        min_cuts = min(min_cuts, result)
                # Larger piece
                elif size > requirement:
                    inventory[size] -= 1
                    remainder = size - requirement
                    inventory[remainder] = inventory.get(remainder, 0) + 1
                    result = try_all_combinations(demands[1:], inventory, cuts + 1)
                    inventory[size] += 1
                    if remainder in inventory:
                        inventory[remainder] -= 1
                        if inventory[remainder] == 0:
                            del inventory[remainder]
                    if result is not None:
                        min_cuts = min(min_cuts, result)

            return min_cuts if min_cuts != float("inf") else None

        return try_all_combinations(children, inventory.copy(), 0)

    def find_doublet_match(
        self,
        inventory: Dict[int, int],
        requirement: int,
        remaining_children: set,
        current_index: int,
        child_indices: Dict[int, List[int]],
    ) -> Tuple[bool, Optional[int]]:
        """
        Locate two children whose combined demands exactly match an available piece.
        Gives a more accurate answer (but takes more time).
        """
        # Check if the requirement is larger than the largest piece
        for piece_size, count in list(inventory.items()):
            if count <= 0:
                continue
            # Check if piece size is larger than requirement
            if piece_size > requirement:
                complement = piece_size - requirement
                # Check if complement exists in remaining children's values
                if complement in child_indices:
                    # Get the first valid index for this complement
                    for matching_index in child_indices[complement]:
                        if (
                            matching_index > current_index
                            and matching_index in remaining_children
                        ):
                            return True, matching_index
        return False, None

    def find_triplet_match(
        self,
        inventory: Dict[int, int],
        requirement: int,
        current_index: int,
        child_indices: Dict[int, List[int]],
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Find three children whose demands sum to an available piece size.
        Gives a more accurate answer (but takes more time).
        """
        # Build sorted list of (value, index) pairs for remaining children
        remaining_demands = [
            (value, index)
            for value, indices in child_indices.items()
            for index in indices
            if index > current_index
        ]
        remaining_demands.sort()  # Sort by value

        # Try larger pieces first to minimize leftovers
        for piece_size in sorted(inventory.keys(), reverse=True):
            if inventory[piece_size] <= 0 or piece_size <= requirement:
                continue

            remaining = piece_size - requirement

            # Two-pointer search through remaining demands
            left, right = 0, len(remaining_demands) - 1
            while left < right:
                sum_pair = remaining_demands[left][0] + remaining_demands[right][0]
                if sum_pair == remaining:
                    idx1, idx2 = remaining_demands[left][1], remaining_demands[right][1]
                    if idx1 != idx2:  # So that we don't use same child twice
                        return True, (idx1, idx2)
                    left += 1
                elif sum_pair < remaining:
                    left += 1
                else:
                    right -= 1

        return False, None

    def find_larger_piece(
        self, inventory: Dict[int, int], requirement: int
    ) -> Optional[int]:
        """
        Retrieve the smallest available piece larger than the requirement, cleaning out
        exhausted or too‑small entries.
        """
        while self.size_heap:
            smallest_piece = self.size_heap[0]

            # Size exhausted – discard
            if inventory.get(smallest_piece, 0) == 0:
                heapq.heappop(self.size_heap)
                continue

            # Size ≤ requirement – discard
            if smallest_piece <= requirement:
                heapq.heappop(self.size_heap)
                continue

            return smallest_piece  # valid answer
        return None  # nothing larger exists

    def combine_smaller_pieces(
        self, inventory: Dict[int, int], requirement: int
    ) -> Tuple[int, bool]:
        """
        Greedily merge smaller pieces to meet a requirement, returning extra cuts used
        and success status.
        """
        current_sum = 0
        pieces_used = []
        # Sort available pieces in descending order
        available_sizes = sorted(
            [size for size in inventory.keys() if inventory[size] > 0], reverse=True
        )

        # Iterate over available sizes and combine pieces
        for size in available_sizes:
            while inventory[size] > 0 and current_sum < requirement:
                current_sum += size
                pieces_used.append(size)
                inventory[size] -= 1

            if current_sum >= requirement:
                break
        if current_sum < requirement:
            return 0, False

        # Remove used pieces and add remainder
        remainder = current_sum - requirement
        if remainder > 0:
            inventory[remainder] = inventory.get(remainder, 0) + 1

        return 0, True

    def remove_served_child(
        self,
        child_index: int,
        remaining_children: set,
        child_indices: Dict[int, List[int]],
        children: List[int],
    ) -> None:
        """
        Remove a served child from tracking structures.
        """
        # Remove child from remaining children set and child indices
        remaining_children.remove(child_index)
        demand = children[child_index]
        if demand in child_indices:
            child_indices[demand].remove(child_index)
            if not child_indices[demand]:  # Remove empty list
                del child_indices[demand]

    def compute_minimum_cuts(
        self, bars: List[int], children: List[int]
    ) -> Tuple[int, float, str]:
        """
        Compute and return the minimal cuts, execution time (ms), and status message
        for distributing bars to children.
        """
        start_time = time.perf_counter()
        try:
            # 1. Validate and convert inputs
            bars, children = self.validate_and_convert_inputs(bars, children)

            # Check total chocolate sufficiency
            if sum(bars) < sum(children):
                return 0, 0, "Insufficient chocolate"

            # Cache check
            cache_key = (
                tuple(sorted(bars)),
                tuple(sorted(children)),
            )
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                end_time = time.perf_counter()
                return cached_result, (end_time - start_time) * 1000, "Cache hit"

            # 2. Single bar/child case
            if len(bars) == 1 and len(children) == 1:
                result = 1 if bars[0] > children[0] else 0
                self.cache.put(cache_key, result)
                end_time = time.perf_counter()
                return result, (end_time - start_time) * 1000, "Single case"

            # Preprocessing
            bars.sort(reverse=True)
            children.sort(reverse=True)
            inventory: Dict[int, int] = {}
            for bar in bars:
                inventory[bar] = inventory.get(bar, 0) + 1

            # 3. Bulk exact‑match elimination
            child_need = Counter(children)
            for size in list(child_need):
                match = min(child_need[size], inventory.get(size, 0))
                if match:
                    child_need[size] -= match
                    if child_need[size] == 0:
                        del child_need[size]
                    inventory[size] -= match
                    if inventory[size] == 0:
                        del inventory[size]

            # Rebuild child list (still sorted in descending order)
            children = sorted(
                [s for s, c in child_need.items() for _ in range(c)], reverse=True
            )

            # One‑off heap construction
            self.size_heap = list(inventory.keys())
            heapq.heapify(self.size_heap)

            # Initialize remaining children set and child indices
            remaining_children_set = set(range(len(children)))
            child_indices = {}
            for index, demand in enumerate(children):
                if demand not in child_indices:
                    child_indices[demand] = []
                child_indices[demand].append(index)

            # 4. Try brute force for small inputs
            if len(children) <= self.bruteforce_threshold:
                bruteforce_inventory = Counter(inventory)
                if (
                    result := self.brute_force_cuts(
                        sorted(children, reverse=True), bruteforce_inventory
                    )
                ) is not None:
                    self.cache.put(cache_key, result)
                    end_time = time.perf_counter()
                    return result, (end_time - start_time) * 1000, "Brute force"

            # Initialize cuts counter
            cuts = 0
            child_index = 0
            used_complement = False
            # Main loop for distributing chocolate
            while child_index < len(children):
                if child_index not in remaining_children_set:
                    child_index += 1
                    continue

                requirement = children[child_index]

                # Try exact match first
                if self.find_exact_match(inventory, requirement):
                    self.remove_served_child(
                        child_index, remaining_children_set, child_indices, children
                    )
                    child_index += 1
                    continue

                # 5. Check if complement look-ahead should be used if
                use_complement = len(children) < self.complement_threshold
                if use_complement:
                    found_match, matching_index = self.find_doublet_match(
                        inventory,
                        requirement,
                        remaining_children_set,
                        child_index,
                        child_indices,
                    )

                    if found_match:
                        # Perform balanced cut
                        piece_size = requirement + children[matching_index]
                        cuts += 1
                        used_complement = True

                        # Remove the original piece
                        inventory[piece_size] -= 1
                        if inventory[piece_size] == 0:
                            del inventory[piece_size]

                        # Mark both children as processed
                        self.remove_served_child(
                            child_index, remaining_children_set, child_indices, children
                        )
                        self.remove_served_child(
                            matching_index,
                            remaining_children_set,
                            child_indices,
                            children,
                        )

                        child_index += 1
                        continue

                    if not found_match:  # Try triplet if pair not found
                        triplet_result = self.find_triplet_match(
                            inventory,
                            requirement,
                            child_index,
                            child_indices,
                        )
                        if triplet_result[0]:  # Check if match found
                            found_match, (index1, index2) = triplet_result
                            piece_size = (
                                requirement + children[index1] + children[index2]
                            )
                            cuts += 2  # Need two cuts for three pieces
                            used_complement = True
                            inventory[piece_size] -= 1
                            if inventory[piece_size] == 0:
                                del inventory[piece_size]

                            self.remove_served_child(
                                child_index,
                                remaining_children_set,
                                child_indices,
                                children,
                            )
                            self.remove_served_child(
                                index1, remaining_children_set, child_indices, children
                            )
                            self.remove_served_child(
                                index2, remaining_children_set, child_indices, children
                            )
                            child_index += 1
                            continue

                # 6. If no balanced cut found, try larger piece
                larger_piece = self.find_larger_piece(inventory, requirement)
                if larger_piece:
                    cuts += 1
                    inventory[larger_piece] -= 1
                    # If none left, remove
                    if inventory[larger_piece] == 0:
                        del inventory[larger_piece]
                    remainder = larger_piece - requirement
                    # Add remainder back to heap
                    if remainder > 0:
                        inventory[remainder] = inventory.get(remainder, 0) + 1
                        heapq.heappush(self.size_heap, remainder)
                    self.remove_served_child(
                        child_index, remaining_children_set, child_indices, children
                    )
                    child_index += 1
                    continue

                # 7. Try combining smaller pieces
                additional_cuts, success = self.combine_smaller_pieces(
                    inventory, requirement
                )
                if not success:
                    return 0, 0, "Cannot satisfy requirement"
                cuts += additional_cuts
                self.remove_served_child(
                    child_index, remaining_children_set, child_indices, children
                )
                child_index += 1

            # Cache the result
            self.cache.put(cache_key, cuts)
            end_time = time.perf_counter()
            status = "Complement method" if used_complement else "Greedy method"
            return cuts, (end_time - start_time) * 1000, status

        # Handle exceptions and return appropriate messages
        except InputError as e:
            return 0, 0, f"Input Error: {str(e)}"
        except Exception as e:
            return 0, 0, f"Unexpected Error: {str(e)}"
