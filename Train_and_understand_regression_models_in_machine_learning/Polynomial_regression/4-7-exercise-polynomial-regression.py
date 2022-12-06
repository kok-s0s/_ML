#!/usr/bin/env python
# coding: utf-8

# # Exercise: Fitting a Polynomial Curve

# In this exercise, we'll have a look at a different type of regression called _polynomial regression_.
# In contrast to _linear regression_ which models relationships as straight lines, _polynomial regression_ models relationships as curves.
#
# Recall in our previous exercise how the relationship between `core_temperature` and `protein_content_of_last_meal` could not be properly explained using a straight line. In this exercise, we'll use _polynomial regression_ to fit a curve to the data instead.
#
# ## Data visualisation
#
# Let's start this exercise by loading in and having a look at our data.

# In[ ]:


import pandas

# Import the data from the .csv file
dataset = pandas.read_csv("doggy-illness.csv", delimiter="\t")

# Let's have a look at the data
dataset


# # Simple Linear Regression
#
# Let's quickly jog our memory by performing the same _simple linear regression_ as we did in the previous exercise using the `temperature` and `protein_content_of_last_meal` columns of the dataset.
#

# In[ ]:


import statsmodels.formula.api as smf
import graphing  # custom graphing code. See our GitHub repo for details

# Perform linear regression. This method takes care of
# the entire fitting procedure for us.
simple_formula = "core_temperature ~ protein_content_of_last_meal"
simple_model = smf.ols(formula=simple_formula, data=dataset).fit()

# Show a graph of the result
graphing.scatter_2D(
    dataset,
    label_x="protein_content_of_last_meal",
    label_y="core_temperature",
    trendline=lambda x: simple_model.params[1] * x + simple_model.params[0],
)


# Notice how the relationship between the two variables is not truly linear. Looking at the plot, it's fairly clear to see that the points tend more heavily towards one side of the line, especially for the higher `core-temperature` and `protein_content_of_last_meal` values.
# A straight line might not be the best way to describe this relationship.
#
# Let's have a quick look at the model's R-Squared score:

# In[ ]:


print("R-squared:", simple_model.rsquared)


# That is quite a reasonable R-Squared score, but let's see if we can get an even better one!
#
# ## Simple Polynomial Regression
#
# Let's fit a _simple polynomial regression_ this time. Similarly to a _simple linear regression_, a _simple polynomial regression_ models the relationship between a label and a single feature. Unlike a _simple linear regression_, a _simple polynomial regression_ can explain relationships that aren't simply straight lines.
#
# In our example, we are going to use a three parameter polynomial.

# In[ ]:


# Perform polynomial regression. This method takes care of
# the entire fitting procedure for us.
polynomial_formula = "core_temperature ~ protein_content_of_last_meal + I(protein_content_of_last_meal**2)"
polynomial_model = smf.ols(formula=polynomial_formula, data=dataset).fit()

# Show a graph of the result
graphing.scatter_2D(
    dataset,
    label_x="protein_content_of_last_meal",
    label_y="core_temperature",
    # Our trendline is the equation for the polynomial
    trendline=lambda x: polynomial_model.params[2] * x**2
    + polynomial_model.params[1] * x
    + polynomial_model.params[0],
)


# That looks a lot better already. Let's confirm by having a quick look at the R-Squared score:

# In[ ]:


print("R-squared:", polynomial_model.rsquared)


# That's a better R-Squared_ score than the one obtained from the previous model - great! We can now confidently tell our vet to prioritize dogs who ate a high protein diet the night before.

# Let's chart our model as a 3D chart. We'll view $X$ and $X^2$ as two separate parameters. Notice that if you rotate the visual just right, our regression model is still a flat plane. This is why polynomial models are still considered to be `linear models`.

# In[ ]:


import numpy as np

fig = graphing.surface(
    x_values=np.array(
        [
            min(dataset.protein_content_of_last_meal),
            max(dataset.protein_content_of_last_meal),
        ]
    ),
    y_values=np.array(
        [
            min(dataset.protein_content_of_last_meal) ** 2,
            max(dataset.protein_content_of_last_meal) ** 2,
        ]
    ),
    calc_z=lambda x, y: polynomial_model.params[0]
    + (polynomial_model.params[1] * x)
    + (polynomial_model.params[2] * y),
    axis_title_x="x",
    axis_title_y="x2",
    axis_title_z="Core temperature",
)
# Add our datapoints to it and display
fig.add_scatter3d(
    x=dataset.protein_content_of_last_meal,
    y=dataset.protein_content_of_last_meal**2,
    z=dataset.core_temperature,
    mode="markers",
)
fig.show()


# ## Extrapolating
#
# Let's see what happens if we extroplate our data. We would like to see if dogs that ate meals even higher in protein are expected to get even sicker.
#
# Let's start with the _linear regression_. We can set what range we would like to extrapolate our data over by using the `x_range` argument in the plotting function. Let's extrapolate over the range `[0,100]`:
#

# In[ ]:


# Show an extrapolated graph of the linear model
graphing.scatter_2D(
    dataset,
    label_x="protein_content_of_last_meal",
    label_y="core_temperature",
    # We extrapolate over the following range
    x_range=[0, 100],
    trendline=lambda x: simple_model.params[1] * x + simple_model.params[0],
)


# Next, we extrapolate the _polynomial regression_ over the same range:

# In[ ]:


# Show an extrapolated graph of the polynomial model
graphing.scatter_2D(
    dataset,
    label_x="protein_content_of_last_meal",
    label_y="core_temperature",
    # We extrapolate over the following range
    x_range=[0, 100],
    trendline=lambda x: polynomial_model.params[2] * x**2
    + polynomial_model.params[1] * x
    + polynomial_model.params[0],
)


# These two graphs predict two very different things!
#
# The extrapolated _polynolmial regression_ expects `core_temperature` to go down, while the extrapolated _linear regression_ expects linear expects `core_temperature` to go up.
# A quick look at the graphs obtained in the previous exercise confirms that we should expect the `core_temeprature` to be rising as the `protein_content_of_last_meal` increases, not falling.
#
# In general, it's not recommended to extrapolate from a _polynomial regression_ unless you have an a-priori reason to do so (which is only very rarely the case, so it's best to err on the side of caution, and never extrapolate from  _polynomial regressions_!)
#
# ## Summary
#
# We covered the following concepts in this exercise:
#
# - Build _simple linear regression_ and _simple polynomial regression_ models.
# - Compare the performance of both models by plotting them, and looking at R-Squared values.
# - Extrapolated the models over a wider range of values.

# %%
