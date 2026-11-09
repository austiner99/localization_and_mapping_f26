#!/usr/bin/env python

"""Module defining discrete probability base objects for probability review.

Defined Classes:
DiscreteExperiment - Base class for discrete experiment objects
ProbabilityMassFunction - Class representing a PMF for 1 discrete RV
JointProbabilityMassFunction - Class representing a joint PMF over 2 RVs

Exceptions:
ValueError - The constructors for ProbabilityMassFunction and
JointProbabilityMassFunction will throw ValueError exceptions if
initialized with probability values that are negative or that do not sum to
one.
"""

from abc import abstractmethod

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"


class DiscreteExperiment():
    """A base class for an experiment with discrete outcomes."""

    @abstractmethod
    def __init__(self, foo=None):
        pass
    
    def enumerate_outcomes(self):
        """Enumerate all possible outcomes of the experiment.

        Returns:
        outcomes: a list of tuples where the first element in each tuple
        is the outcome and the second is the associated probability.

        """
        pass


class ProbabilityMassFunction():
    """A class representing the PMF of a discrete random variable."""

    def __init__(self, probabilities):
        """Create a probability mass function object.

        Parameters:
        probabilities (dic): A dictionary mapping all possible values the
            random variable can take on to its associated probability.

        Exceptions:
        ValueError: This function will raise a ValueError exception if
          any of the passed in probability values are negative or if they
          do not sum to one.

        """
        if(not np.isclose(sum(probabilities.values()), 1.0)):
            raise ValueError("Probability values must add up to one.")

        if(any(p < 0 for p in probabilities.values())):
            raise ValueError("Probabilities cannot be negative.")

        self.probabilities = probabilities

    def get_values(self):
        """Return the list of possible values the variable can take."""
        return self.probabilities.keys()

    def p(self, value):
        """Return the probability that the RV will take on a value."""
        if(not value in self.probabilities.keys()):
            return 0
        else:
            return self.probabilities[value]
        
    def plot_pmf(self):
        """Plot the pmf as a stem plot.

        Returns:
        fig - the figure object for the pmf
        ax - the axis object for the pmf

        """
        fig, ax = plt.subplots()

        plt.stem(self.probabilities.keys(),
                 self.probabilities.values(),
                 use_line_collection=True)

        ax.set_ylim([0, 1])
        left, right = ax.get_xlim()
        ax.set_xlim([left-0.8, right+0.8])

        return fig, ax


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    
def set_xy_axes_equal(ax):
    '''Make the xy axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = 0#abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    #ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    
class JointProbabilityMassFunction():
    """A class representing the joint PMF of two discrete random variables."""

    def __init__(self,
                 values_var1,
                 values_var2,
                 probabilities,
                 var1_name="x",
                 var2_name="y"):
        """Create a joint probability mass function object.

        Parameters:
        values_var1: A numpy array of the possible values var1 can take.
        values_var2: A numpy array of the possible values var2 can take.
        probabilities (dic): A dictionary mapping all possible pairs of
          values the two random variables can take on (stored as a two element
          tuple with the value of the 1st random variable 1st) to the associated 
          probability that the pair of possible values will occur.
        var1_name (optional string): Name of of variable 1.
        var2_name (optional string): Name of of variable 2.


        Exceptions:
        ValueError("Probabilities cannot be negative."): This
          function raises a ValueError exception with the above message if any
          of the probability values returned by the experiment are negative.
        ValueError("Probabilties do not sum to one."): This function returns a
          value error with the above message if the passed in probabilities do
          not sum to one.
        ValueError("Probability given for invalid value."): This function raises
          a ValueError exception with the above message if the passed in
          dictionary includes a value for the random variable that does not
          match values_var1 and values_var2.
        
        """
        if(not np.isclose(sum(probabilities.values()), 1.0)):
            raise ValueError("Probability do not sum to one.")

        if(any(p < 0 for p in probabilities.values())):
            raise ValueError("Probabilities cannot be negative.")

        for value in probabilities.keys():
            if(not value[0] in values_var1 or
               not value[1] in values_var2):
                raise ValueError("Probability given for invalid value.")

        self.var1_name = var1_name
        self.var2_name = var2_name        
        self.values_var1 = values_var1
        self.values_var2 = values_var2
        self.probabilities = probabilities

    def p(self, value):
        """Return the probability that the RV will take on a value."""
        if(not value in self.probabilities.keys()):
            return 0
        else:
            return self.probabilities[value]

    def get_values(self):
        """Return the list of possible values the variable can take."""
        return self.probabilities.keys()

    def plot_pmf(self):
        """Plot the pmf as a stem plot.

        Returns:
        fig - the figure object for the pmf
        ax - the axis object for the pmf

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        dz = []

        sorted_var1_vals = np.sort(self.values_var1)
        sorted_var2_vals = np.sort(self.values_var2)

        for i in np.arange(len(self.values_var1)):
            for j in np.arange(len(self.values_var2)):
                val = (sorted_var1_vals[i], sorted_var2_vals[j])
                if val in self.get_values():
                    x.append(i-0.5)
                    y.append(j-0.5)
                    z.append(0)
                    dz.append(self.p(val))

        dx = np.ones(len(dz))
        dy = np.ones(len(dz))

        ax.bar3d(x, y, z, dx, dy, dz, color='#00ceaa')

        left, right = ax.get_xlim()
        ax.set_xlim([left-0.8, right+0.8])
        left, right = ax.get_ylim()        
        ax.set_ylim([left-0.8, right+0.8])
        ax.set_zlim([0, 1.0])

        ax.set_xticks(np.arange(len(self.values_var1)))
        ax.set_xticklabels(sorted_var1_vals)
        ax.set_yticks(np.arange(len(self.values_var2)))
        ax.set_yticklabels(sorted_var2_vals)

        ax.set_xlabel(self.var1_name)
        ax.set_ylabel(self.var2_name)
        ax.set_zlabel("p(" + self.var1_name + ", " + self.var2_name + ")")
        ax.set_title("Joint PMF of " + self.var1_name + " and " +
                     self.var2_name)
        set_xy_axes_equal(ax)
        
        return fig, ax    
