#!/usr/bin/env python3

"""Module defining code for part 1 of probabilty review lab assignment.

Defined classes:
DiceRoll - An experiment representing the roll of a die.
TwoDiceRoll - An experiment representing the roll of two dice.

Defined functions:
two_dice_sum - a random variable that equals the sum of two dice.
two_dice_difference - a random variable that equals the diff. of two dice.
doubles_rolled - an indicator random variable for rolling doubles.

Functions to be implemented:
evaluate_pmf - evaluate the pmf of a given random variable
expected_value - calculate the expected value of a given random variable.
variance - calculate the variance of a random variable

Main function (script operation):
main - evaluate and output the pmf, expected value, and variance of
    several random variables.

"""

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import probability_review.discrete_probability_core as dpc

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

########################
# Example Experiments
########################


class DiceRoll(dpc.DiscreteExperiment):
    """An experiment representing the roll of a fairly six-sided die."""

    def __init__(self, num_sides=6):
        super().__init__()
        if num_sides >= 3:
            self.num_sides = num_sides
        else:
            self.num_sides = 3

    def enumerate_outcomes(self):
        """Enumerate all possible outcomes of the experiment.

        Returns:
        outcomes: a list of tuples where the first element in each tuple
        is the outcome and the second is the associated probability.

        """
        outcomes = []
        for roll in range(1, self.num_sides+1):
            outcomes.append((roll, 1.0/self.num_sides))
        return outcomes


class TwoDiceRoll(dpc.DiscreteExperiment):
    """An experiment representing the roll of two fair six-sided dice."""

    def __init__(self, num_sides=6):
        super().__init__()
        if num_sides >= 3:
            self.num_sides = num_sides
        else:
            self.num_sides = 3

    def enumerate_outcomes(self):
        """Enumerate all possible outcomes of the experiment.

        Returns:
        outcomes: a list of tuples where the first element in each tuple
        is the outcome and the second is the associated probability.

        """
        outcomes = []
        for roll1 in range(1, self.num_sides+1):
            for roll2 in range(1, self.num_sides+1):
                outcomes.append(((roll1, roll2), 1.0/self.num_sides**2))

        return outcomes

###############################
# Example Ramdom Variables
###############################


def two_dice_sum(outcome):
    """Compute the sum of two dice.""" 
    return outcome[0] + outcome[1]

def two_dice_difference(outcome):
    """Compute the difference of two dice."""
    return max(outcome[0], outcome[1]) - min(outcome[0], outcome[1])

def doubles_rolled(outcome):
    """Return 1 if doubles is rolled, 0 otherwise."""
    if(outcome[0] == outcome[1]):
        return 1
    else:
        return 0

def two_dice_sum_plus_one(outcome):
    """Compute the sum of two dice plus one.""" 
    return outcome[0] + outcome[1] + 1

def weighted_two_dice_sum(outcome):
    """Compute a weighted sum of two dice.""" 
    return (1.0*outcome[0] + 2.5*outcome[1])

def two_dice_true_difference(outcome):
    """Compute the true_difference of two dice."""
    return outcome[1] - outcome[0]


###########################################################################
# TODO: Complete the following function to evaluate the probability
# mass function of a specified random variable under a given experiment
#
# Ex: evaluate_pmf(TwoDiceRoll, two_dice_sum) should return a dictionary
# that represents the pmf of the sum of a roll of two six-sided dice.
# This dictionary can then be passed into the constructor of the
# ProbabilityMassFunction class to create an object that can do various
# actions such as plot the pmf.
#
# Don't forget to check if the probabilities assigned to the outcomes of
# the experiment passed into the function are valid.
#
# Note: In python, classes and functions can be passed as arguments by
# using the name of the function.
###########################################################################


def evaluate_pmf(experiment_initializer, random_variable, dice_sides=6):
    """Evaluate PMF of a specified random variable for a given experiment.

    Parameters:
    experiment_initializer: Class constructor (derived from DiscreteExperiment)
        that constructs an experiment object for enumerating the possible
        experiment outcomes.
    random_variable: A function object that takes an experiment outcome and
        maps it to a real-number value.

    Returns:
    p_x: A dictionary object mapping all possible values of the random variable
      to their associated probability.

    Exceptions:
    ValueError("Outcomes with negative probabilities recieved."): This function
      raises a ValueError exception with the above message if any of the
      probability values returned by the experiment are negative.
    ValueError("Probabilties do not sum to one."): This function returns a
      value error with the above message if the probabilities for all outcomes
      of the experiment do not sum to one.

    """
    experiment = experiment_initializer(dice_sides)
    p_x = {}

    for outcome, probability in experiment.enumerate_outcomes():
        if probability < 0:
            raise ValueError("Outcomes with negative probabilities recieved.")
        rv_value = random_variable(outcome)
        p_x[rv_value] = p_x.get(rv_value, 0) + probability

    if sum(p_x.values()) < 0.9999999 or sum(p_x.values()) > 1.0000001:
        print(f"Sum of probabilities: {sum(p_x.values())}")
        raise ValueError("Probabilties do not sum to one.")
    
    return p_x

###########################################################################
# TODO: Implement the following function to evaluate the expected value
# of a given random variable.
###########################################################################


def expected_value(rv_pmf):
    """Evaluate the expected value of the given random variable.

    Parameters:
    rv_pmf: The PMF of the specified random variable.

    Returns:
    e_value: The expected value of the given random variable.

    Exceptions:
    ValueError("PMF with negative probabilities recieved."): This function
      raises a ValueError exception with the above message if any of the
      probability values in the passed in pmf are negative.
    ValueError("Probabilties do not sum to one."): This function raises a
      ValueError with the above message if the passed in probabilities do
      not sum to one.
    ValueError("Empty pmf."): This function raises a ValueError with the
      above message if the passed in pmf has zero values.

    """
    if not rv_pmf:
        raise ValueError("Empty pmf.")

    pmf = {}
    if(isinstance(rv_pmf, dpc.ProbabilityMassFunction)):
        pmf = rv_pmf.probabilities
    else:
        pmf = rv_pmf

    for value, probability in pmf.items():
        if probability < 0:
            raise ValueError("PMF with negative probabilities recieved.")
    if sum(pmf.values()) < 0.9999999 or sum(pmf.values()) > 1.0000001:
        raise ValueError("Probabilties do not sum to one.")
        
    e_value = 0.0

    for value, probability in pmf.items():
        e_value += value * probability

    return e_value


###########################################################################
# TODO: Implement the following function to evaluate the variance of the
# given random variable
###########################################################################


def variance(rv_pmf):
    """Evaluate the variance of the given random variable.

    Parameters:
    rv_pmf: The PMF of the specified random variable.

    Returns:
    var: The variance of the given random variable.

    Exceptions:
    ValueError("PMF with negative probabilities recieved."): This function
      raises a ValueError exception with the above message if any of the
      probability values in the passed in pmf are negative.
    ValueError("Probabilties do not sum to one."): This function raises a
      ValueError with the above message if the passed in probabilities do
      not sum to one.
    ValueError("Empty pmf."): This function raises a ValueError with the
      above message if the passed in pmf has zero values.

    """
    if not rv_pmf:
        raise ValueError("Empty pmf.")
    
    if(isinstance(rv_pmf, dpc.ProbabilityMassFunction)):
        pmf = rv_pmf.probabilities
    else:
        pmf = rv_pmf

    for value, probability in pmf.items():
        if probability < 0:
            raise ValueError("PMF with negative probabilities recieved.")
    if sum(pmf.values()) < 0.9999999 or sum(pmf.values()) > 1.0000001:
        raise ValueError("Probabilties do not sum to one.")
    
    var = 0.0
    
    e_value = expected_value(rv_pmf)

    for value, probability in pmf.items():
        var += (value - e_value) ** 2 * probability

    return var


def main():
    """Evaluate the PMF, Expected Value, and Variance for several RVs."""
    # Evaluate PMF for the sum of a roll of two six-sided dice
    num_dice_sides = 6
    x_pmf_dic = evaluate_pmf(TwoDiceRoll, two_dice_sum, dice_sides=num_dice_sides)
    x_pmf = dpc.ProbabilityMassFunction(x_pmf_dic)

    print('Expected Value of the Sum of Two ({0}-sided) Dice:'.format(num_dice_sides), expected_value(x_pmf))
    print('Variance of the Sum of Two ({0}-sided) Dice:'.format(num_dice_sides), variance(x_pmf))

    fig, ax = x_pmf.plot_pmf()
    ax.set_title("PMF of the Sum of Two ({0}-sided) Dice:".format(num_dice_sides))
    ax.set_ylabel("p(X=x)")
    ax.set_xlabel("x")

    # Evaluate PMF for the difference of a roll of two six-sided dice
    y_pmf_dic = evaluate_pmf(TwoDiceRoll, two_dice_difference, dice_sides=num_dice_sides)
    y_pmf = dpc.ProbabilityMassFunction(y_pmf_dic)

    print('Expected Value of the Diff. of Two ({0}-sided) Dice:'.format(num_dice_sides), expected_value(y_pmf))
    print('Variance of the Difference of Two ({0}-sided) Dice:'.format(num_dice_sides), variance(y_pmf))

    fig, ax = y_pmf.plot_pmf()
    ax.set_title("PMF of the Difference of Two ({0}-sided) Dice:".format(num_dice_sides))
    ax.set_ylabel("p(Y=y)")
    ax.set_xlabel("y")

    # Evaluate PMF for the indicator of the event that doubles are rolled
    z_pmf_dic = evaluate_pmf(TwoDiceRoll, doubles_rolled, dice_sides=num_dice_sides)
    z_pmf = dpc.ProbabilityMassFunction(z_pmf_dic)

    print('Expected Value of the Indicator RV for Rolling Doubles ({0}-sided Dice):'.format(num_dice_sides),
          expected_value(z_pmf))

    fig, ax = z_pmf.plot_pmf()
    ax.set_title("PMF of the Indicator RV for Rolling Doubles with {0}-sided Dice".format(num_dice_sides))
    ax.set_ylabel("p(Z=z)")
    ax.set_xlabel("z")

    plt.show()


if __name__ == "__main__":
    main()
