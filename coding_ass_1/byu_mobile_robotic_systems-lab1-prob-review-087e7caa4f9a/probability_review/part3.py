#!/usr/bin/env python3

"""Module defining code for part3 of probability review lab assignment.

Functions to be implemented:
marginalize_out - marginalize out a single random variable from a joint
    pmf containing it.
condition_against - conditionalize a joint pmf against against a given value for
    one of its variables.

Main function (script operation):
main - evaluate and output conditionalized and marginalized pmfs 
    from various joint pmfs.

"""

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import probability_review.discrete_probability_core as dpc
from probability_review.part1 import TwoDiceRoll
from probability_review.part1 import two_dice_sum
from probability_review.part1 import two_dice_difference
from probability_review.part2 import evaluate_joint_pmf

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"


###########################################################################
# TODO: Complete the following function that marginalizes out a single
# random variable from a joint probability mass function containing it.
#
# Ex: marginalize_out(joint_pmf, 1)
# should return the marginal PMF of random variable 2 after marginalizing
# out random variable 1.
###########################################################################

def marginalize_out(joint_pmf, variable_to_marginalize):
    """Marginalize the random variable out of the joint_pmf.

    Parameters:
    joint_pmf: A JointProbabilityMassFunction object representing the joint
        PMF of two discrete random variables.
    variable_to_marginalize: A number either 1 or 2 designating which random
        variable is to be marginalized out.

    Returns:
    marginalized_pmf (ProbabilityMassFunction): A ProbablityMassFunction 
        object representing the PMF of the remaining random variable after
        the specified random variable has been marginalized out. (Note that 
        mathmatically, if the joint pmf passed in to this function contained
        N>2 variables, the resulting "marginalized_pmf" would also be a 
        joint pmf containing N-1 variables (as opposed to a PMF of a single 
        variable). However, because in this exercise we are only working with 
        Joint PMFs of 2 variables, the resulting "marginalized_pmf" contains 
        only a single variable.)

    Exceptions:
    ValueError("Invalid joint PMF recieved."): This function raises a 
      ValueError exception with the above message if the argument joint_pmf
      is not of the class JointProbabilityMassFunction.
    ValueError("Invalid variable_to_marginalize indicator (must be 1 or 2)."):
      This function raises a ValueError exception with the above message 
      if the argument variable_to_marginalize is not equal to 1 or 2.

    """
    if(not (variable_to_marginalize == 1 or
            variable_to_marginalize == 2)):
        msg = "Invalid variable_to_marginalize indicator (must be 1 or 2)."
        raise ValueError(msg)

    if(not isinstance(joint_pmf, dpc.JointProbabilityMassFunction)):
        raise ValueError("Invalid joint PMF received.")

    var1_values = joint_pmf.values_var1
    var2_values = joint_pmf.values_var2
    pmf = joint_pmf.probabilities

    sorted_var1_values = sorted(set(var1_values))
    sorted_var2_values = sorted(set(var2_values))

    marginalized_pmf_dic = {}
    
    if variable_to_marginalize == 1:
        for val1 in sorted_var1_values:
            for val2 in sorted_var2_values:
                if (val1, val2) not in pmf:
                    continue
                if val2 in marginalized_pmf_dic:
                    marginalized_pmf_dic[val2] += pmf[val1, val2]
                else:
                    marginalized_pmf_dic[val2] = pmf[val1, val2]
    else:
        for val1 in sorted_var1_values:
            for val2 in sorted_var2_values:
                if (val1, val2) not in pmf:
                    continue
                if val1 in marginalized_pmf_dic:
                    marginalized_pmf_dic[val1] += pmf[val1, val2]
                else:
                    marginalized_pmf_dic[val1] = pmf[val1, val2]

    marginalized_pmf = dpc.ProbabilityMassFunction(marginalized_pmf_dic)
    return marginalized_pmf


###########################################################################
# TODO: Complete the following function to evaluate the set of conditional
# pmfs from a joint probability mass function given that one of its
# variables is known
#
# Ex: conditionalize_against(joint_pmf, 1)
# should return a dictionary that maps "known" values for random variable 1
# to pmfs for the value of random variable 2. 
###########################################################################

def condition_against(joint_pmf, known_random_variable):
    """Condition a joint_pmf against the random variable specified.

    Parameters:
    joint_pmf: A JointProbabilityMassFunction object representing the joint
        PMF of two discrete random variables.
    known_random_variable: A number either 1 or 2 designating which random
        variable is known.

    Returns:
    conditional_pmf (dic): A dictionary mapping all possible values of the 
        known random variable to a ProbabilityMassFunction object that 
        represents the PMF of the unknown random variable given that the 
        known random variable takes on the known value.

    Exceptions:
    ValueError("Invalid joint PMF recieved."): This function raises a 
      ValueError exception with the above message if the argument joint_pmf
      is not of the class JointProbabilityMassFunction.
    ValueError("Invalid known_random_variable indicator (must be 1 or 2)."):
      This function raises a ValueError exception with the above message 
      if the argument known_random_variable is not equal to 1 or 2.

    """
    if(not (known_random_variable == 1 or
            known_random_variable == 2)):
        msg = "Invalid known_random_variable indicator (must be 1 or 2)."
        raise ValueError(msg)

    if(not isinstance(joint_pmf, dpc.JointProbabilityMassFunction)):
        raise ValueError("Invalid joint PMF received.")

    if known_random_variable == 1:
        marginalized_variable = 2
    else:
        marginalized_variable = 1
    marginalized_pmf = marginalize_out(joint_pmf, marginalized_variable)
    marginalized_pmf_dic = marginalized_pmf.probabilities
    
    var1_values = joint_pmf.values_var1
    var2_values = joint_pmf.values_var2
    pmf = joint_pmf.probabilities
    
    sorted_var1_values = sorted(list(set(var1_values)))
    sorted_var2_values = sorted(list(set(var2_values)))
    
    conditional_pmf = {}

    if known_random_variable == 1:
        for var1 in sorted_var1_values:
            probabilities = {}
            for var2 in sorted_var2_values:
                if (var1, var2) in pmf:
                    probabilities[var2] = (pmf[(var1, var2)] / marginalized_pmf_dic[var1])
            conditional_pmf[var1] = dpc.ProbabilityMassFunction(probabilities)
    else:
        for var2 in sorted_var2_values:
            probabilities = {}
            for var1 in sorted_var1_values:
                if (var1, var2) in pmf:
                    probabilities[var1] = (pmf[(var1, var2)] / marginalized_pmf_dic[var2])
            conditional_pmf[var2] = dpc.ProbabilityMassFunction(probabilities)

    return conditional_pmf

def main():
    """Marginalize and Condition various Joint PMFs."""
    num_dice_sides = 6
    #Evaluate Joint PMF for the sum and difference of two six-sided dice
    sum_vals, diff_vals, sumdiff_pmf_dic = evaluate_joint_pmf(
        TwoDiceRoll, two_dice_sum, two_dice_difference, dice_sides=num_dice_sides)
    sumdiff_pmf = dpc.JointProbabilityMassFunction(
        sum_vals,
        diff_vals,
        sumdiff_pmf_dic,
        "Sum",
        "Diff ({0}-sided Dice)".format(num_dice_sides))

    fig, ax = sumdiff_pmf.plot_pmf()

    #Evaluate marginal of Sum and Diff individually
    sum_marginal = marginalize_out(sumdiff_pmf, 2)
    fig, ax = sum_marginal.plot_pmf()
    ax.set_title("Marginal PMF of the Sum of Two ({0}-sided) Dice:".format(num_dice_sides))
    ax.set_ylabel("p(X=x)")
    ax.set_xlabel("x")

    diff_marginal = marginalize_out(sumdiff_pmf, 1)
    fig, ax = diff_marginal.plot_pmf()
    ax.set_title("Marginal PMF of the Diff. of Two ({0}-sided) Dice:".format(num_dice_sides))
    ax.set_ylabel("p(Y=y)")
    ax.set_xlabel("y")

    #Conditionalize Sum on Difference
    sum_given_diff = condition_against(sumdiff_pmf, 2)
    sum_given_diff_is_2 = sum_given_diff[2]
    fig, ax = sum_given_diff_is_2.plot_pmf()
    ax.set_title("Conditionalized PMF of Sum (Given that Diff is 2)")
    ax.set_ylabel("p(X=x | Y=2)")
    ax.set_xlabel("x")


    #Conditionalize Difference on Sum
    diff_given_sum = condition_against(sumdiff_pmf, 1)
    diff_given_sum_is_4 = diff_given_sum[4]
    fig, ax = diff_given_sum_is_4.plot_pmf()
    ax.set_title("Conditionalized PMF of Diff (Given that Sum is 4)")
    ax.set_ylabel("p(Y=y | X=4)")
    ax.set_xlabel("x")
                 
    plt.show()

    

if __name__ == "__main__":
    main()
