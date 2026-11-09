"""Test Part1 of probability review lab assignment."""

import pytest
import numpy as np

import probability_review.discrete_probability_core as dpc
from probability_review import part1


class TestFunctionEvaluatePMF():
    """Test the evaluate_pmf function."""

    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that a ValueError is thrown when prob. do not sum to one."""
        class MockExperiment1(dpc.DiscreteExperiment):
            """Fake Experiment with invalid probabilities."""

            def enumerate_outcomes(self):
                """Return outcomes with negative probability."""
                return [(1, 0.5), (2, 0.5), (3, 0.5)]

        def mock_rv(outcome):
            """Return fake value."""
            return outcome

        with pytest.raises(ValueError) as excinfo:
            part1.evaluate_pmf(MockExperiment1, mock_rv)
        assert "sum to one" in str(excinfo.value)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        class MockExperiment1(dpc.DiscreteExperiment):
            """Fake Experiment with negative outcomes."""

            def enumerate_outcomes(self):
                """Return outcomes with negative probability."""
                return [(1, 0.5), (2, -0.5), (3, 0.5), (4, 0.5)]

        def mock_rv(outcome):
            """Return fake value."""
            return outcome

        with pytest.raises(ValueError) as excinfo:
            part1.evaluate_pmf(MockExperiment1, mock_rv)
        assert "negative" in str(excinfo.value)

    def test_correct_for_two_dice_sum(self):
        """Test for the correct pmf for the sum of two 6-sided dice."""
        pmf = part1.evaluate_pmf(part1.TwoDiceRoll, part1.two_dice_sum, dice_sides=4)

        assert np.isclose(pmf[2], 0.0625)
        assert np.isclose(pmf[3], 0.125)
        assert np.isclose(pmf[4], 0.1875)
        assert np.isclose(pmf[5], 0.25)
        assert np.isclose(pmf[6], 0.1875)
        assert np.isclose(pmf[7], 0.125)
        assert np.isclose(pmf[8], 0.0625)

    def test_correct_for_two_dice_diff(self):
        """Test for the correct pmf for the difference of two 6-sided dice."""
        pmf = part1.evaluate_pmf(part1.TwoDiceRoll, part1.two_dice_difference, dice_sides=4)

        assert np.isclose(pmf[0], 0.25)
        assert np.isclose(pmf[1], 0.375)
        assert np.isclose(pmf[2], 0.25)
        assert np.isclose(pmf[3], 0.125)

    def test_correct_for_rolling_doubles(self):
        """Test for the correct pmf for the indicator of rolling doubles."""
        pmf = part1.evaluate_pmf(part1.TwoDiceRoll, part1.doubles_rolled, dice_sides=4)

        assert np.isclose(pmf[1], 0.25)
        assert np.isclose(pmf[0], 0.75)


class TestFunctionExpectedValue():
    """Test the expected_value function."""

    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that a ValueError is thrown when prob. do not sum to one."""
        pmf = {1: 0.2, 2: 0.4, 3: 0.3}

        with pytest.raises(ValueError) as excinfo:
            part1.expected_value(pmf)
        assert "sum to one" in str(excinfo.value)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        pmf = {1: 0.2, 2: 0.4, 3: 0.4, 4: -0.2, 5: 0.2}

        with pytest.raises(ValueError) as excinfo:
            part1.expected_value(pmf)
        assert "negative" in str(excinfo.value)

    def test_throws_when_passed_empty_pmf(self):
        """Test that ValueError is thrown when passed an empty pmf."""
        pmf = {}
        with pytest.raises(ValueError) as excinfo:
            part1.expected_value(pmf)
        assert "Empty pmf" in str(excinfo.value)

    def test_correct_for_two_dice_sum(self):
        """Test for correct expected value for the sum of two 6-sided dice."""
        pmf = {2: 0.0625, 3: 0.125, 4: 0.1875, 5: 0.25, 6: 0.1875, 7: 0.125, 8: 0.0625}
        assert np.isclose(part1.expected_value(pmf), 5.0)

    def test_correct_for_two_dice_diff(self):
        """Test for correct expected value for the diff of two 6-sided dice."""
        pmf = {0: 0.25, 1: 0.375, 2: 0.25, 3: 0.125}
        assert np.isclose(part1.expected_value(pmf), 1.25)


class TestFunctionVariance():
    """Test the variance function."""

    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that a ValueError is thrown when prob. do not sum to one."""
        pmf = {1: 0.2, 2: 0.4, 3: 0.3}

        with pytest.raises(ValueError) as excinfo:
            part1.variance(pmf)
        assert "sum to one" in str(excinfo.value)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        pmf = {1: 0.2, 2: 0.4, 3: 0.4, 4: -0.2, 5: 0.2}

        with pytest.raises(ValueError) as excinfo:
            part1.variance(pmf)
        assert "negative" in str(excinfo.value)

    def test_throws_when_passed_empty_pmf(self):
        """Test that ValueError is thrown when passed an empty pmf."""
        pmf = {}
        with pytest.raises(ValueError) as excinfo:
            part1.variance(pmf)
        assert "Empty pmf" in str(excinfo.value)

    def test_correct_for_two_dice_sum(self):
        """Test for correct variance for the sum of two 6-sided dice."""
        pmf = {2: 0.0625, 3: 0.125, 4: 0.1875, 5: 0.25, 6: 0.1875, 7: 0.125, 8: 0.0625}
        assert np.isclose(part1.variance(pmf), 2.5)

    def test_correct_for_two_dice_diff(self):
        """Test for correct variance for the diff of two 6-sided dice."""
        pmf = {0: 0.25, 1: 0.375, 2: 0.25, 3: 0.125}
        assert np.isclose(part1.variance(pmf), 0.9375)
