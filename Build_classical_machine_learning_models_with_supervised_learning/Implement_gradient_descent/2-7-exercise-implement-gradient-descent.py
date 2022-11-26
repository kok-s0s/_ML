#!/usr/bin/env python
# coding: utf-8

# # Exercise: Gradient descent
#
# Previously, we identified trends in winter temperatures by fitting a linear regression model to weather data. Here, we'll repeat this process by focusing on the optimizer. Specifically, we'll work with batch gradient descent and explore how changing the learning rate can alter its behavior.
#
# The model we'll be working with will be the same linear regression model that we've used in other units. The principles we learn, however, also apply to much more complex models.
#
# ## Loading data and preparing our model
#
# Let's load up our weather data from Seattle, filter to January temperatures, and make slight adjustments so that the dates are mathematically interpretable.

# In[ ]:


from datetime import datetime
import pandas
import graphing  # Custom graphing code. See our GitHub repository

# Load a file that contains weather data for Seattle
data = pandas.read_csv("seattleWeather_1948-2017.csv", parse_dates=["date"])

# Remove all dates after July 1 because we have to to plant onions before summer begins
data = data[[d.month < 7 for d in data.date]].copy()


# Convert the dates into numbers so we can use them in our models
# We make a year column that can contain fractions. For example,
# 1948.5 is halfway through the year 1948
data["year"] = [(d.year + d.timetuple().tm_yday / 365.25) for d in data.date]

# Let's take a quick look at our data
print("Visual Check:")
graphing.scatter_2D(
    data, label_x="year", label_y="min_temperature", title="Temperatures over time (°F)"
)


# ## Fitting a model automatically
#
# Let's fit a line to this data well by using an existing library.

# In[ ]:


import statsmodels.formula.api as smf

# Perform linear regression to fit a line to our data
# NB OLS uses the sum or mean of squared differences as a cost function,
# which we're familiar with from our last exercise
model = smf.ols(formula="min_temperature ~ year", data=data).fit()

# Print the model
intercept = model.params[0]
slope = model.params[1]

print(f"The model is: y = {slope:0.3f} * X + {intercept:0.3f}")


# Ooh, some math! Don't let that bother you. It's quite common for labels and features to be referred to as `Y` and `X`, respectively.
# Here:
# * `Y` is temperature (°F).
# * `X` is year.
# * -83 is a _model parameter_ that acts as the line offset.
# * 0.063 is a _model parameter_ that defines the line slope (in °F per year).
#
# So this little equation states that the model estimates temperature by multiplying the year by `0.063` and then subtracting `83`.
#
# How did the library calculate these values? Let's go through the process.
#
# ## Model selection
#
# The first step is always selecting a model. Let's reuse the model that we used in previous exercises.

# In[ ]:


class MyModel:
    def __init__(self):
        """
        Creates a new MyModel
        """
        # Straight lines described by two parameters:
        # The slope is the angle of the line
        self.slope = 0
        # The intercept moves the line up or down
        self.intercept = 0

    def predict(self, date):
        """
        Estimates the temperature from the date
        """
        return date * self.slope + self.intercept

    def get_summary(self):
        """
        Returns a string that summarises the model
        """
        return f"y = {self.slope} * x + {self.intercept}"


print("Model class ready")


# ## Fitting our model with gradient descent
#
# The automatic method used the _ordinary least squares_ (OLS) method, which is the standard way to fit a line. OLS uses the mean (or sum) of square differences as a cost function. (Recall our experimentation with the sum of squared differences in the last exercise.) Let's replicate the preceding line fitting, and break down each step so we can watch it in action.
#
# Recall that for each iteration, our training conducts three steps:
#
# 1. Estimation of `Y` (temperature) from `X` (year).
#
# 2. Calculation of the cost function and its slope.
#
# 3. Adjustment of our model according to this slope.
#
# Let's implement this now to watch it in action. Note that *to keep things simple, we'll focus on estimating one parameter (line slope) for now*.
#
# ### Visualizing the error function
#
# First, let's look at the error function for this data. Normally we don't know this in advance, but for learning purposes, let's calculate it now for different potential models.

# In[ ]:


import numpy as np

x = data.year
temperature_true = data.min_temperature

# We'll use a prebuilt method to show a 3D plot
# This requires a range of x values, a range of y values,
# and a way to calculate z
# Here, we set:
#   x to a range of potential model intercepts
#   y to a range of potential model slopes
#   z as the cost for that combination of model parameters

# Choose a range of intercepts and slopes values
intercepts = np.linspace(-100, -70, 10)
slopes = np.linspace(0.060, 0.07, 10)


# Set a cost function. This will be the mean of squared differences
def cost_function(temperature_estimate):
    """
    Calculates cost for a given temperature estimate
    Our cost function is the mean of squared differences (a.k.a. mean squared error)
    """
    # Note that with NumPy to square each value, we use **
    return np.mean((temperature_true - temperature_estimate) ** 2)


def predict_and_calc_cost(intercept, slope):
    """
    Uses the model to make a prediction, then calculates the cost
    """

    # Predict temperature by using these model parameters
    temperature_estimate = x * slope + intercept

    # Calculate cost
    return cost_function(temperature_estimate)


# Call the graphing method. This will use our cost function,
# which is above. If you want to view this code in detail,
# then see this project's GitHub repository
graphing.surface(
    x_values=intercepts,
    y_values=slopes,
    calc_z=predict_and_calc_cost,
    title="Cost for Different Model Parameters",
    axis_title_x="Model intercept",
    axis_title_y="Model slope",
    axis_title_z="Cost",
)


# The preceding graph is interactive. Try clicking and dragging the mouse to rotate it.
#
# Notice how cost changes with both intercept and line slope. This is because our model has a slope and an intercept, which both will affect how well the line fits the data. A consequence is that the gradient of the cost function must also be described by two numbers: one for intercept and one for line slope.
#
# Our lowest point on the graph is the location of the best line equation for our data: a slope of 0.063 and an intercept of -83. Let's try to train a model to find this point.
#
# ### Implementing gradient descent
#
# To implement gradient descent, we need a method that can calculate the gradient of the preceding curve.

# In[ ]:


def calculate_gradient(temperature_estimate):
    """
    This calculates the gradient for a linear regession
    by using the Mean Squared Error cost function
    """

    # The partial derivatives of MSE are as follows
    # You don't need to be able to do this just yet, but
    # it's important to note that these give you the two gradients
    # that we need to train our model
    error = temperature_estimate - temperature_true
    grad_intercept = np.mean(error) * 2
    grad_slope = (x * error).mean() * 2

    return grad_intercept, grad_slope


print("Function is ready!")


# Now all we need is a starting guess, and a loop that will update this guess with each iteration.

# In[ ]:


def gradient_descent(learning_rate, number_of_iterations):
    """
    Performs gradient descent for a one-variable function.

    learning_rate: Larger numbers follow the gradient more aggressively
    number_of_iterations: The maximum number of iterations to perform
    """

    # Our starting guess is y = 0 * x - 83
    # We're going to start with the correct intercept so that
    # only the line's slope is estimated. This is just to keep
    # things simple for this exercise
    model = MyModel()
    model.intercept = -83
    model.slope = 0

    for i in range(number_of_iterations):
        # Calculate the predicted values
        predicted_temperature = model.predict(x)

        # == OPTIMIZER ===
        # Calculate the gradient
        _, grad_slope = calculate_gradient(predicted_temperature)
        # Update the estimation of the line
        model.slope -= learning_rate * grad_slope

        # Print the current estimation and cost every 100 iterations
        if i % 100 == 0:
            estimate = model.predict(x)
            cost = cost_function(estimate)
            print("Next estimate:", model.get_summary(), f"Cost: {cost}")

    # Print the final model
    print(f"Final estimate:", model.get_summary())


# Run gradient descent
gradient_descent(learning_rate=1e-9, number_of_iterations=1000)


# Our model found the correct answer, but it took a number of steps. Looking at the printout, we can see how it progressively stepped toward the correct solution.
#
# Now, what happens if we make the learning rate faster? This means taking larger steps.

# In[ ]:


gradient_descent(learning_rate=1e-8, number_of_iterations=200)


# Our model appears to have found the solution faster. If we increase the rate even more, however, things don't go so well:

# In[ ]:


gradient_descent(learning_rate=5e-7, number_of_iterations=500)


# Notice how the cost is getting worse each time.
#
# This is because the steps that the model was taking were too large. Although it would step toward the correct solution, it would step too far and actually get worse with each attempt.
#
# For each model, there's an ideal learning rate. It requires experimentation.
#
# ## Fitting multiple variables simultaneously
#
# We've just fit one variable here to keep things simple. Expanding this to fit multiple variables requires only a few small code changes:
#
# * We need to update more than one variable in the gradient descent loop.
#
# * We need to do some preprocessing of the data, which we alluded to in an earlier exercise. We'll cover how and why in later learning material.
#
# ## Summary
#
# Well done! In this unit, we:
#
# * Watched gradient descent in action.
#
# * Saw how changing the learning rate can improve a model's training speed.
#
# * Learned that changing the learning rate can also result in unstable models.
#
# You might have noticed that where the cost function stopped and the optimizer began became a little blurred here. Don't let that bother you. This is happens commonly, simply because they're conceptually separate and their mathematics sometimes can become intertwined.
