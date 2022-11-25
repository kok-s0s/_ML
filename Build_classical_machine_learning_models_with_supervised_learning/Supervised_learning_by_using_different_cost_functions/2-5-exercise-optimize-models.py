#!/usr/bin/env python
# coding: utf-8

# # Exercise: Supervised learning by using different cost functions
#
# In this exercise, we'll have a deeper look at how cost functions can change:
#
# * How well models appear to have fit data.
# * The kinds of relationships a model represents.
#
# ## Loading the data
#
# Let's start by loading the data. To make this exercise simpler, we'll use only a few datapoints this time.

# In[ ]:


import pandas
from datetime import datetime

# Load a file that contains our weather data
dataset = pandas.read_csv("seattleWeather_1948-2017.csv", parse_dates=["date"])

# Convert the dates into numbers so we can use them in our models
# We make a year column that can contain fractions. For example,
# 1948.5 is halfway through the year 1948
dataset["year"] = [(d.year + d.timetuple().tm_yday / 365.25) for d in dataset.date]


# For the sake of this exercise, let's look at February 1 for the following years:
desired_dates = [
    datetime(1950, 2, 1),
    datetime(1960, 2, 1),
    datetime(1970, 2, 1),
    datetime(1980, 2, 1),
    datetime(1990, 2, 1),
    datetime(2000, 2, 1),
    datetime(2010, 2, 1),
    datetime(2017, 2, 1),
]

dataset = dataset[dataset.date.isin(desired_dates)].copy()

# Print the dataset
dataset


# ## Comparing two cost functions
#
# Let's compare two common cost functions: the _sum of squared differences_ (SSD) and the _sum of absolute differences_ (SAD). They both calculate the difference between each predicted value and the expected value. The distinction is simply:
#
# * SSD squares that difference and sums the result.
# * SAD converts differences into absolute differences and then sums them.
#
# To see these cost functions in action, we need to first implement them:

# In[ ]:


import numpy


def sum_of_square_differences(estimate, actual):
    # Note that with NumPy, to square each value we use **
    return numpy.sum((estimate - actual) ** 2)


def sum_of_absolute_differences(estimate, actual):
    return numpy.sum(numpy.abs(estimate - actual))


# They're very similar. How do they behave? Let's test with some fake model estimates.
#
# Let's say that the correct answers are `1` and `3`, but the model estimates `2` and `2`:

# In[ ]:


actual_label = numpy.array([1, 3])
model_estimate = numpy.array([2, 2])

print("SSD:", sum_of_square_differences(model_estimate, actual_label))
print("SAD:", sum_of_absolute_differences(model_estimate, actual_label))


# We have an error of `1` for each estimate, and both methods have returned the same error.
#
# What happens if we distribute these errors differently? Let's pretend that we estimated the first value perfectly but were off by `2` for the second value:

# In[ ]:


actual_label = numpy.array([1, 3])
model_estimate = numpy.array([1, 1])

print("SSD:", sum_of_square_differences(model_estimate, actual_label))
print("SAD:", sum_of_absolute_differences(model_estimate, actual_label))


# SAD has calculated the same cost as before, because the average error is still the same (`1 + 1 = 0 + 2`). According to SAD, the first and second set of estimates were equally good.
#
# By contrast, SSD has given a higher (worse) cost for the second set of estimates ( $1^2 + 1^2 < 0^2 + 2^2 $ ). When we use SSD, we encourage models to be both accurate and consistent in their accuracy.
#
#
# ## Differences in action
#
# Let's compare how our two cost functions affect model fitting.
#
# First, fit a model by using the SSD cost function:

# In[ ]:


from microsoft_custom_linear_regressor import MicrosoftCustomLinearRegressor
import graphing

# Create and fit the model
# We use a custom object that we've hidden from this notebook, because
# you don't need to understand its details. This fits a linear model
# by using a provided cost function

# Fit a model by using sum of square differences
model = MicrosoftCustomLinearRegressor().fit(
    X=dataset.year, y=dataset.min_temperature, cost_function=sum_of_square_differences
)

# Graph the model
graphing.scatter_2D(
    dataset, label_x="year", label_y="min_temperature", trendline=model.predict
)


# Our SSD method normally does well, but here it did a poor job. The line is a far distance from the values for many years. Why? Notice that the datapoint at the lower left doesn't seem to follow the trend of the other datapoints. 1950 was a very cold winter in Seattle, and this datapoint is strongly influencing our final model (the blue line). What happens if we change the cost function?
#
# ### Sum of absolute differences
#
# Let's repeat what we've just done, but using SAD.

# In[ ]:


# Fit a model with SSD
# Fit a model by using sum of square differences
model = MicrosoftCustomLinearRegressor().fit(
    X=dataset.year, y=dataset.min_temperature, cost_function=sum_of_absolute_differences
)

# Graph the model
graphing.scatter_2D(
    dataset, label_x="year", label_y="min_temperature", trendline=model.predict
)


# It's clear that this line passes through the majority of points much better than before, at the expense of almost ignoring the measurement taken in 1950.
#
# In our farming scenario, we're interested in how average temperatures are changing over time. We don't have much interest in 1950 specifically, so for us, this is a better result. In other situations, of course, we might consider this result worse.
#
#
# ## Summary
#
# In this exercise, you learned about how changing the cost function that's used during fitting can result in different final results.
#
# You also learned how this behavior happens because these cost functions describe the "best" way to fit a model. Although from a data analyst's point of view, there can be drawbacks no matter which cost function is chosen.
