"""Test the discrete_probability_core classes."""

import pytest
import numpy as np

import probability_review.discrete_probability_core as dpc

class TestProbabilityMassFunction():
    """Test the ProbabilityMassFunction class."""

    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that ValueError is thrown when prob. do not sum to one."""
        pmf_dic = {1: 0.5, 2: 0.5, 3: 0.5}

        with pytest.raises(ValueError) as excinfo:
            dpc.ProbabilityMassFunction(pmf_dic)
        assert "add up to one" in str(excinfo)

        pmf_dic = {1: 0.2, 2: 0.2, 3: 0.2}

        with pytest.raises(ValueError) as excinfo:
            dpc.ProbabilityMassFunction(pmf_dic)
        assert "add up to one" in str(excinfo)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        pmf_dic = {1: 0.5, 2: -0.25, 3: 0.75}
        
        with pytest.raises(ValueError) as excinfo:
            dpc.ProbabilityMassFunction(pmf_dic)
        assert "cannot be negative" in str(excinfo)


class TestJointProbabilityMassFunction():
    """Test the JointProbabilityMassFunction class."""
    def test_throws_when_probabilities_do_not_sum_to_one(self):
        """Test that ValueError is thrown when prob. do not sum to one."""
        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): 0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (4, 6): 0.75}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "sum to one" in str(excinfo)

        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): 0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (4, 6): 0.15}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "sum to one" in str(excinfo)

    def test_throws_when_passed_negatively_likely_outcomes(self):
        """Test that ValueError is thrown for negative experiment outcomes."""
        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): -0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (4, 6): 0.75}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "negative" in str(excinfo)

    def test_throws_when_passed_invalid_value(self):
        """Test that ValueError is thrown for invalid values in dictionary."""
        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): 0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (4, 7): 0.25}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "invalid value" in str(excinfo)

        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): 0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (9, 6): 0.25}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "invalid value" in str(excinfo)

        val1 = [1, 2, 3, 4]
        val2 = [5, 6]
        joint_pmf_dic = {(1, 5): 0.25, (2, 6): 0.25,
                         (3, 5): 0.25, (9, 9): 0.25}

        with pytest.raises(ValueError) as excinfo:
            dpc.JointProbabilityMassFunction(val1, val2, joint_pmf_dic)
        assert "invalid value" in str(excinfo)
