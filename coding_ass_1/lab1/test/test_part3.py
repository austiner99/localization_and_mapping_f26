"""Test Part 3 of probability review lab assignment."""

import pytest
import numpy as np

import probability_review.discrete_probability_core as dpc
from probability_review import part3

class TestFunctionMarginalizeOut():
    """Test the marginalize_out function."""

    def test_throws_when_invalid_indicator_passed_in(self):
        """Test that a ValueError is thrown when an invalid indicator for the variable to marginalize out is passed in."""
        mock_vals1 = [2, 3, 4, 5]
        mock_vals2 = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.25, (3, 1): 0.25,
                     (4, 2): 0.25, (5, 3): 0.25}
        joint_pmf = dpc.JointProbabilityMassFunction(mock_vals1, mock_vals2, joint_pmf_dic, "Mock1", "Mock2")

        with pytest.raises(ValueError) as excinfo:
            part3.marginalize_out(joint_pmf, 0)
        assert "Invalid variable_to_marginalize indicator" in str(excinfo)

        with pytest.raises(ValueError) as excinfo:
            part3.marginalize_out(joint_pmf, 3)
        assert "Invalid variable_to_marginalize indicator" in str(excinfo)

        with pytest.raises(ValueError) as excinfo:
            part3.marginalize_out(joint_pmf, -4)
        assert "Invalid variable_to_marginalize indicator" in str(excinfo)

    def test_throws_when_invalid_joint_pmf_received(self):
        """Test that a ValueError is thrown when an invalid joint PMF is passed in."""
        mock_vals1 = [2, 3, 4, 5]
        mock_vals2 = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.25, (3, 1): 0.25,
                     (4, 2): 0.25, (5, 3): 0.25}

        with pytest.raises(ValueError) as excinfo:
            part3.marginalize_out(joint_pmf_dic, 1)
        assert "Invalid joint PMF" in str(excinfo)

    def test_correct_for_marginalize_out_difference_from_sum_and_difference(self):
        """Test for the correct marginal when removing the difference of two dice from the joint pmf of the sum of two dice and the difference of two dice, using 4-sided dice."""
        sum_vals = [2, 3, 4, 5, 6, 7, 8]
        diff_vals = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.0625, (3, 1): 0.125,
                     (4, 2): 0.125, (5, 3): 0.125,
                     (4, 0): 0.0625, (5, 1): 0.125,
                     (6, 2): 0.125, (6, 0): 0.0625,
                     (7, 1): 0.125, (8, 0): 0.0625}
        joint_pmf = dpc.JointProbabilityMassFunction(sum_vals, diff_vals, joint_pmf_dic, "Sum", "Diff")
        marginal = part3.marginalize_out(joint_pmf, 2)

        assert np.isclose(marginal.probabilities[2], 0.0625)
        assert np.isclose(marginal.probabilities[3], 0.125)
        assert np.isclose(marginal.probabilities[4], 0.1875)
        assert np.isclose(marginal.probabilities[5], 0.25)
        assert np.isclose(marginal.probabilities[6], 0.1875)
        assert np.isclose(marginal.probabilities[7], 0.125)
        assert np.isclose(marginal.probabilities[8], 0.0625)

    def test_correct_for_marginalize_out_sum_from_sum_and_difference(self):
        """Test for the correct marginal when removing the sum of two dice from the joint pmf of the sum of two dice and the difference of two dice, using 4-sided dice."""
        sum_vals = [2, 3, 4, 5, 6, 7, 8]
        diff_vals = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.0625, (3, 1): 0.125,
                     (4, 2): 0.125, (5, 3): 0.125,
                     (4, 0): 0.0625, (5, 1): 0.125,
                     (6, 2): 0.125, (6, 0): 0.0625,
                     (7, 1): 0.125, (8, 0): 0.0625}
        joint_pmf = dpc.JointProbabilityMassFunction(sum_vals, diff_vals, joint_pmf_dic, "Sum", "Diff")
        marginal = part3.marginalize_out(joint_pmf, 1)

        assert np.isclose(marginal.probabilities[0], 0.25)
        assert np.isclose(marginal.probabilities[1], 0.375)
        assert np.isclose(marginal.probabilities[2], 0.25)
        assert np.isclose(marginal.probabilities[3], 0.125)

class TestFunctionConditionAgainst():
    """Test the condition_against function."""

    def test_throws_when_invalid_indicator_passed_in(self):
        """Test that a ValueError is thrown when an invalid indicator for the variable to marginalize out is passed in."""
        mock_vals1 = [2, 3, 4, 5]
        mock_vals2 = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.25, (3, 1): 0.25,
                     (4, 2): 0.25, (5, 3): 0.25}
        joint_pmf = dpc.JointProbabilityMassFunction(mock_vals1, mock_vals2, joint_pmf_dic, "Mock1", "Mock2")

        with pytest.raises(ValueError) as excinfo:
            part3.condition_against(joint_pmf, 0)
        assert "Invalid known_random_variable indicator" in str(excinfo)

        with pytest.raises(ValueError) as excinfo:
            part3.condition_against(joint_pmf, 3)
        assert "Invalid known_random_variable indicator" in str(excinfo)

        with pytest.raises(ValueError) as excinfo:
            part3.condition_against(joint_pmf, -4)
        assert "Invalid known_random_variable indicator" in str(excinfo)

    def test_throws_when_invalid_joint_pmf_received(self):
        """Test that a ValueError is thrown when an invalid joint PMF is passed in."""
        mock_vals1 = [2, 3, 4, 5]
        mock_vals2 = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.25, (3, 1): 0.25,
                     (4, 2): 0.25, (5, 3): 0.25}

        with pytest.raises(ValueError) as excinfo:
            part3.condition_against(joint_pmf_dic, 1)
        assert "Invalid joint PMF" in str(excinfo)

    def test_correct_for_condition_against_difference_from_sum_and_difference(self):
        """Test for the correct result when conditioning the joint pmf of the sum of two dice and the difference of two dice, using 4-sided dice, against the difference."""
        sum_vals = [2, 3, 4, 5, 6, 7, 8]
        diff_vals = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.0625, (3, 1): 0.125,
                     (4, 2): 0.125, (5, 3): 0.125,
                     (4, 0): 0.0625, (5, 1): 0.125,
                     (6, 2): 0.125, (6, 0): 0.0625,
                     (7, 1): 0.125, (8, 0): 0.0625}
        joint_pmf = dpc.JointProbabilityMassFunction(sum_vals, diff_vals, joint_pmf_dic, "Sum", "Diff")
        sum_given_diff = part3.condition_against(joint_pmf, 2)

        assert np.isclose(sum_given_diff[0].probabilities[2], 0.25)        
        assert np.isclose(sum_given_diff[0].probabilities[4], 0.25)
        assert np.isclose(sum_given_diff[0].probabilities[6], 0.25)
        assert np.isclose(sum_given_diff[0].probabilities[8], 0.25)
        assert np.isclose(sum_given_diff[1].probabilities[3], 0.3333333333333333)
        assert np.isclose(sum_given_diff[1].probabilities[5], 0.3333333333333333)
        assert np.isclose(sum_given_diff[1].probabilities[7], 0.3333333333333333)
        assert np.isclose(sum_given_diff[2].probabilities[4], 0.5)
        assert np.isclose(sum_given_diff[2].probabilities[6], 0.5)
        assert np.isclose(sum_given_diff[3].probabilities[5], 1.0)

    def test_correct_for_condition_against_sum_from_sum_and_difference(self):
        """Test for the correct result when conditioning the joint pmf of the sum of two dice and the difference of two dice, using 4-sided dice, against the sum."""
        sum_vals = [2, 3, 4, 5, 6, 7, 8]
        diff_vals = [0, 1, 2, 3]
        joint_pmf_dic = {(2, 0): 0.0625, (3, 1): 0.125,
                     (4, 2): 0.125, (5, 3): 0.125,
                     (4, 0): 0.0625, (5, 1): 0.125,
                     (6, 2): 0.125, (6, 0): 0.0625,
                     (7, 1): 0.125, (8, 0): 0.0625}
        joint_pmf = dpc.JointProbabilityMassFunction(sum_vals, diff_vals, joint_pmf_dic, "Sum", "Diff")
        diff_given_sum = part3.condition_against(joint_pmf, 1)
        
        assert np.isclose(diff_given_sum[2].probabilities[0], 1.0)        
        assert np.isclose(diff_given_sum[3].probabilities[1], 1.0)
        assert np.isclose(diff_given_sum[4].probabilities[2], 0.6666666666666666)
        assert np.isclose(diff_given_sum[4].probabilities[0], 0.3333333333333333)
        assert np.isclose(diff_given_sum[5].probabilities[3], 0.5)
        assert np.isclose(diff_given_sum[5].probabilities[1], 0.5)
        assert np.isclose(diff_given_sum[6].probabilities[2], 0.6666666666666666)
        assert np.isclose(diff_given_sum[6].probabilities[0], 0.3333333333333333)
        assert np.isclose(diff_given_sum[7].probabilities[1], 1.0)
        assert np.isclose(diff_given_sum[8].probabilities[0], 1.0)