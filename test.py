import unittest
import numpy as np
from main import CSO


def fitness_Himmelblau(coords):
    x = coords[0]
    y = coords[1]

    return ((x**2 + y - 11) ** 2) + (((x + y**2 - 7) ** 2))


def test_function(fitness_function, expected_results):
    s = CSO(
        fitness_function,
        P=150,
        n=2,
        pa=0.25,
        beta=1.5,
        bound=None,
        plot=True,
        min=True,
        verbose=True,
        Tmax=300,
    )
    optimal_coords = s.execute()

    flag = False

    compare_function = lambda d: d < 0.001
    compare_function = np.vectorize(compare_function)

    for expected_result in expected_results:
        print(f"comparing {expected_result} and {optimal_coords}")

        if np.any(compare_function(np.abs(optimal_coords - expected_result))):
            flag = True
            break

    assert flag == True


class TestCSO(unittest.TestCase):
    def test_himmelblau(self):
        test_function(
            fitness_function=fitness_Himmelblau,
            expected_results=[
                np.array([-2.805118, 3.131312]),
                np.array([-3.779310, -3.283186]),
                np.array([3.584428, -1.848126]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
