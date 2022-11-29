import unittest

from crossovers import MultiplePlaceCrossover, UniformCrossover


class TestMultiplePlaceCrossOver(unittest.TestCase):

    def test_multiple_place_crossover(self):
        child1, child2 = MultiplePlaceCrossover(n_crossover_places=4, probability=1)._crossover_multiple_places(
            cross_fields=[2, 5, 6, 7], parent1=[1, 2, 3, 4, 5, 6, 7, 8], parent2=[4, 3, 2, 1, 8, 7, 6, 5])

        self.assertEqual(child1, [1, 2, 2, 1, 8, 6, 6, 8])
        self.assertEqual(child2, [4, 3, 3, 4, 5, 7, 7, 5])


if __name__ == '__main__':
    unittest.main()


class TestUniformCrossover(unittest.TestCase):

    def test_uniform_crossover(self):
        # given
        crossover = UniformCrossover(probability=1)
        parents = [[1, 2, 3, 4], [5, 6, 7, 8]]
        # when
        child1, child2 = crossover._crossover_multiple_places(cross_fields=[1, 2, 3, 4], parent1=parents[0],
                                                       parent2=parents[1])

        # then
        self.assertEqual(child1, [1, 6, 3, 8])
        self.assertEqual(child2, [5, 2, 7, 4])
