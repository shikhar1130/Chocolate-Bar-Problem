from src import ChocolateDistributor


def run_comprehensive_tests():
    distributor = ChocolateDistributor(
        enable_validation=True,
        enable_sanitization=True,
        bruteforce_threshold=10,
        complement_threshold=200
    )
    test_cases = [
        ([15, 10, 5], [8, 7, 6, 5, 4]),
        ([1, 2, 3, 2, 10, 6, 6, 9, 7, 8], [4, 5, 4, 5, 3, 5, 5, 4, 2, 4]),
        # ([2, 500, 7], [4, 30, 8, 34, 5, 2 ,34 , 2, 1]),
        # ([10, 8, 6, 4], [7, 6, 5, 4, 3]),
        # ([5, 5, 5], [5, 5, 5]),
        # ([100], [1]*99),
        # ([2]*1000, [1]*1000),
        # ([3**20], [2**19]*2),
        # ([float('inf')], [float('inf')]),
        # ([2**-21], [2**-22]),
        # ([None], [2]),
        # ([2], ['1']),
        # ([], []),  # Empty input
        # ([0], [1]),  # Zero bar length
        # ([5], [0]),  # Zero requirement
        # ([1], [2]),  # Insufficient chocolate
    ]

    for i, (bars, children) in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Bars: {bars}")
        print(f"Children: {children}")
        try:
            result, time_taken, message = distributor.compute_minimum_cuts  (
                bars, children
            )
        except Exception as e:
            result, time_taken, message = None, 0, f"Unhandled Test Error: {str(e)}"
        print(f"Minimum number of cuts: {result}")
        print(f"Time taken: {time_taken:.4f} ms")
        print(f"Message: {message}")
        print("-" * 50)

if __name__ == "__main__":
    run_comprehensive_tests()