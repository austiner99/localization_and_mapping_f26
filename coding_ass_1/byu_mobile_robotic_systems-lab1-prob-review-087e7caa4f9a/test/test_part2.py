"""Test Part 2 of probability review lab assignment."""

import pytest
import numpy as np

import probability_review.discrete_probability_core as dpc
from probability_review import part1, part2

class TestFunctionEvaluateJointPMF():
    """Test the evaluate_joint_pmf function."""

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
            part2.evaluate_joint_pmf(MockExperiment1, mock_rv, mock_rv)
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
            part2.evaluate_joint_pmf(MockExperiment1, mock_rv, mock_rv)
        assert "negative" in str(excinfo.value)

    def test_correct_for_two_dice_sum_and_two_dice_difference(self):
        """Test for the correct joint pmf for the sum of two 4-sided dice and the difference of two 6-sided dice."""
        sum_vals, diff_vals, sumdiff_pmf_dic = part2.evaluate_joint_pmf(part1.TwoDiceRoll, part1.two_dice_sum, part1.two_dice_difference, dice_sides=4)
        sum_vals = np.sort(sum_vals) # sum_vals = np.sort(np.unique(sum_vals))
        diff_vals = np.sort(diff_vals) # diff_vals = np.sort(np.unique(diff_vals))
        # Test values from two dice sum
        assert np.isclose(sum_vals[0], 2)
        assert np.isclose(sum_vals[1], 3)
        assert np.isclose(sum_vals[2], 4)
        assert np.isclose(sum_vals[3], 5)
        assert np.isclose(sum_vals[4], 6)
        assert np.isclose(sum_vals[5], 7)
        assert np.isclose(sum_vals[6], 8)
        # Test values from two dice diff
        assert np.isclose(diff_vals[0], 0)
        assert np.isclose(diff_vals[1], 1)
        assert np.isclose(diff_vals[2], 2)
        assert np.isclose(diff_vals[3], 3)
        # Test Joint PMF
        assert np.isclose(sumdiff_pmf_dic[(2, 0)], 0.0625)
        assert np.isclose(sumdiff_pmf_dic[(3, 1)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(4, 2)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(5, 3)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(4, 0)], 0.0625)
        assert np.isclose(sumdiff_pmf_dic[(5, 1)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(6, 2)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(6, 0)], 0.0625)
        assert np.isclose(sumdiff_pmf_dic[(7, 1)], 0.125)
        assert np.isclose(sumdiff_pmf_dic[(8, 0)], 0.0625)

    def test_correct_for_two_dice_sum_and_two_dice_sum_plus_one(self):
        """Test for the correct joint pmf for the sum of two 4-sided dice and the sum of two 4-sided dice plus one."""
        sum_vals, sum_plus_one_vals, sumsum_plus_one_pmf_dic = part2.evaluate_joint_pmf(part1.TwoDiceRoll, part1.two_dice_sum, part1.two_dice_sum_plus_one, dice_sides=4)
        sum_vals = np.sort(sum_vals)
        sum_plus_one_vals = np.sort(sum_plus_one_vals)
        # Test values from two dice sum
        assert np.isclose(sum_vals[0], 2)
        assert np.isclose(sum_vals[1], 3)
        assert np.isclose(sum_vals[2], 4)
        assert np.isclose(sum_vals[3], 5)
        assert np.isclose(sum_vals[4], 6)
        assert np.isclose(sum_vals[5], 7)
        assert np.isclose(sum_vals[6], 8)
        # Test values from two dice sum
        assert np.isclose(sum_plus_one_vals[0], 3)
        assert np.isclose(sum_plus_one_vals[1], 4)
        assert np.isclose(sum_plus_one_vals[2], 5)
        assert np.isclose(sum_plus_one_vals[3], 6)
        assert np.isclose(sum_plus_one_vals[4], 7)
        assert np.isclose(sum_plus_one_vals[5], 8)
        assert np.isclose(sum_plus_one_vals[6], 9)
        # Test Joint PMF
        assert np.isclose(sumsum_plus_one_pmf_dic[(2, 3)], 0.0625)
        assert np.isclose(sumsum_plus_one_pmf_dic[(3, 4)], 0.125)
        assert np.isclose(sumsum_plus_one_pmf_dic[(4, 5)], 0.1875)
        assert np.isclose(sumsum_plus_one_pmf_dic[(5, 6)], 0.25)
        assert np.isclose(sumsum_plus_one_pmf_dic[(6, 7)], 0.1875)
        assert np.isclose(sumsum_plus_one_pmf_dic[(7, 8)], 0.125)
        assert np.isclose(sumsum_plus_one_pmf_dic[(8, 9)], 0.0625)

    def test_correct_for_two_dice_sum_and_weighted_two_dice_sum(self):
        """Test for the correct joint pmf for the sum of two 4-sided dice and the weighted sum of two 4-sided dice."""
        sum_vals, weighted_sum_vals, sumweighted_sum_pmf_dic = part2.evaluate_joint_pmf(part1.TwoDiceRoll, part1.two_dice_sum, part1.weighted_two_dice_sum, dice_sides=4)
        sum_vals = np.sort(sum_vals)
        weighted_sum_vals = np.sort(weighted_sum_vals)
        # Test values from two dice sum
        assert np.isclose(sum_vals[0], 2)
        assert np.isclose(sum_vals[1], 3)
        assert np.isclose(sum_vals[2], 4)
        assert np.isclose(sum_vals[3], 5)
        assert np.isclose(sum_vals[4], 6)
        assert np.isclose(sum_vals[5], 7)
        assert np.isclose(sum_vals[6], 8)
        # Test values from two dice sum
        assert np.isclose(weighted_sum_vals[0], 3.5)
        assert np.isclose(weighted_sum_vals[1], 4.5)
        assert np.isclose(weighted_sum_vals[2], 5.5)
        assert np.isclose(weighted_sum_vals[3], 6.0)
        assert np.isclose(weighted_sum_vals[4], 6.5)
        assert np.isclose(weighted_sum_vals[5], 7.0)
        assert np.isclose(weighted_sum_vals[6], 8.0)
        assert np.isclose(weighted_sum_vals[7], 8.5)
        assert np.isclose(weighted_sum_vals[8], 9.0)
        assert np.isclose(weighted_sum_vals[9], 9.5)
        assert np.isclose(weighted_sum_vals[10], 10.5)
        assert np.isclose(weighted_sum_vals[11], 11.0)
        assert np.isclose(weighted_sum_vals[12], 11.5)
        assert np.isclose(weighted_sum_vals[13], 12.0)
        assert np.isclose(weighted_sum_vals[14], 13.0)
        assert np.isclose(weighted_sum_vals[15], 14.0)
        # Test Joint PMF
        assert np.isclose(sumweighted_sum_pmf_dic[(2, 3.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(3, 6.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(4, 8.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(5, 11.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(3, 4.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(4, 7.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(5, 9.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(6, 12.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(4, 5.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(5, 8.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(6, 10.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(7, 13.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(5, 6.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(6, 9.0)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(7, 11.5)], 0.0625)
        assert np.isclose(sumweighted_sum_pmf_dic[(8, 14.0)], 0.0625)

class TestFunctionExpectedValue():
    """Test the expected_value function."""
    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that a ValueError is thrown when prob. do not sum to one."""
        pmf = {(1, 0): 0.2, (2, 4): 0.4, (3, 1): 0.3}

        with pytest.raises(ValueError) as excinfo:
            part2.expected_value(pmf)
        assert "sum to one" in str(excinfo.value)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        pmf = {(1, 0): 0.2, (2, 4): 0.4, (3, 1): 0.4, (4, 7): -0.2, (5, 6): 0.2}

        with pytest.raises(ValueError) as excinfo:
            part2.expected_value(pmf)
        assert "negative" in str(excinfo.value)

    def test_throws_when_passed_empty_pmf(self):
        """Test that ValueError is thrown when passed an empty pmf."""
        pmf = {}
        with pytest.raises(ValueError) as excinfo:
            part2.expected_value(pmf)
        assert "Empty pmf" in str(excinfo.value)
    def test_correct_for_two_dice_sum_and_two_dice_difference(self):
        """Test for the correct expected value for the sum of two 4-sided dice and the difference of two 6-sided dice."""
        pmf = {(2, 0): 0.0625, (3, 1): 0.125,
               (4, 2): 0.125, (5, 3): 0.125,
               (4, 0): 0.0625, (5, 1): 0.125,
               (6, 2): 0.125, (6, 0): 0.0625,
               (7, 1): 0.125, (8, 0): 0.0625}
        ev = part2.expected_value(pmf)
        assert np.isclose(ev[0], 5.)
        assert np.isclose(ev[1], 1.25)
    def test_correct_for_two_dice_sum_and_two_dice_sum_plus_one(self):
        """Test for the correct expected value for the sum of two 4-sided dice and the sum of two 4-sided dice plus one."""
        pmf = {(2, 3): 0.0625, (3, 4): 0.125,
               (4, 5): 0.1875, (5, 6): 0.25,
               (6, 7): 0.1875, (7, 8): 0.125,
               (8, 9): 0.0625}
        ev = part2.expected_value(pmf)
        assert np.isclose(ev[0], 5.)
        assert np.isclose(ev[1], 6.)
    def test_correct_for_two_dice_sum_and_weighted_two_dice_sum(self):
        """Test for the correct expected value for the sum of two 4-sided dice and the weighted sum of two 4-sided dice."""
        pmf = {(2, 3.5): 0.0625, (3, 6.0): 0.0625,
               (4, 8.5): 0.0625, (5, 11.0): 0.0625,
               (3, 4.5): 0.0625, (4, 7.0): 0.0625,
               (5, 9.5): 0.0625, (6, 12.0): 0.0625,
               (4, 5.5): 0.0625, (5, 8.0): 0.0625,
               (6, 10.5): 0.0625, (7, 13.0): 0.0625,
               (5, 6.5): 0.0625, (6, 9.0): 0.0625,
               (7, 11.5): 0.0625, (8, 14.0): 0.0625}
        ev = part2.expected_value(pmf)
        assert np.isclose(ev[0], 5.)
        assert np.isclose(ev[1], 8.75)

class TestFunctionCovariance():
    """Test the covariance function."""
    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that a ValueError is thrown when prob. do not sum to one."""
        pmf = {(1, 0): 0.2, (2, 4): 0.4, (3, 1): 0.3}

        with pytest.raises(ValueError) as excinfo:
            part2.covariance(pmf)
        assert "sum to one" in str(excinfo.value)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        pmf = {(1, 0): 0.2, (2, 4): 0.4, (3, 1): 0.4, (4, 7): -0.2, (5, 6): 0.2}

        with pytest.raises(ValueError) as excinfo:
            part2.covariance(pmf)
        assert "negative" in str(excinfo.value)

    def test_throws_when_passed_empty_pmf(self):
        """Test that ValueError is thrown when passed an empty pmf."""
        pmf = {}
        with pytest.raises(ValueError) as excinfo:
            part2.covariance(pmf)
        assert "Empty pmf" in str(excinfo.value)
    def test_correct_for_two_dice_sum_and_two_dice_difference(self):
        """Test for the correct covariance for the sum of two 4-sided dice and the difference of two 6-sided dice."""
        pmf = {(2, 0): 0.0625, (3, 1): 0.125,
               (4, 2): 0.125, (5, 3): 0.125,
               (4, 0): 0.0625, (5, 1): 0.125,
               (6, 2): 0.125, (6, 0): 0.0625,
               (7, 1): 0.125, (8, 0): 0.0625}
        cov = part2.covariance(pmf)
        assert np.isclose(cov[0,0], 2.5)
        assert np.isclose(cov[0,1], 0.)
        assert np.isclose(cov[1,0], 0.)
        assert np.isclose(cov[1,1], 0.9375)
    def test_correct_for_two_dice_sum_and_two_dice_sum_plus_one(self):
        """Test for the correct covariance for the sum of two 4-sided dice and the sum of two 4-sided dice plus one."""
        pmf = {(2, 3): 0.0625, (3, 4): 0.125,
               (4, 5): 0.1875, (5, 6): 0.25,
               (6, 7): 0.1875, (7, 8): 0.125,
               (8, 9): 0.0625}
        cov = part2.covariance(pmf)
        assert np.isclose(cov[0,0], 2.5)
        assert np.isclose(cov[0,1], 2.5)
        assert np.isclose(cov[1,0], 2.5)
        assert np.isclose(cov[1,1], 2.5)
    def test_correct_for_two_dice_sum_and_weighted_two_dice_sum(self):
        """Test for the correct covariance for the sum of two 4-sided dice and the weighted sum of two 4-sided dice."""
        pmf = {(2, 3.5): 0.0625, (3, 6.0): 0.0625,
               (4, 8.5): 0.0625, (5, 11.0): 0.0625,
               (3, 4.5): 0.0625, (4, 7.0): 0.0625,
               (5, 9.5): 0.0625, (6, 12.0): 0.0625,
               (4, 5.5): 0.0625, (5, 8.0): 0.0625,
               (6, 10.5): 0.0625, (7, 13.0): 0.0625,
               (5, 6.5): 0.0625, (6, 9.0): 0.0625,
               (7, 11.5): 0.0625, (8, 14.0): 0.0625}
        cov = part2.covariance(pmf)
        assert np.isclose(cov[0,0], 2.5)
        assert np.isclose(cov[0,1], 4.375)
        assert np.isclose(cov[1,0], 4.375)
        assert np.isclose(cov[1,1], 9.0625)