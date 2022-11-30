#!/usr/bin/env python
# coding: utf-8

# # Exercise: Titanic Dataset - Visualising Different Types of Data
#
# To build better machine learning models we have to understand the data we are working with. This usually means both:
#
# 1. visualising the data
#
# 2. understanding the kind of data we are working with
#
# In this module we'll practice cleaning our Titanic dataset and visualising different kinds of data, particularly those that are continuous, ordinal, categorical, or simply identity columns.
#
# ## A Note On Our Graphing
#
# In this exercise we'll be using a custom python script to create our graphs. This uses a well known graphing library called Plotly.
#
# We use this script to help you focus on learning data exploration rather than getting bogged down in understanding Plotly. If you would like to look at the code for this custom script, you can find it in our GitHub repository.
#
# ## First Inspection
#
# Let's reload the Titanic Dataset and reacquaint ourselves with its data:
#

# In[ ]:


import pandas as pd

# Load data from our dataset file into a pandas dataframe
dataset = pd.read_csv("titanic.csv", index_col=False, sep=",", header=0)

# Let's take a look at the data
dataset.head()


# Take a careful look at the columns and see if you can identify which columns hold continuous, ordinal, categorical, or identity data.
#
# We can display a brief summary of the _dataypes_ by using panda's `info()` method:

# In[ ]:


dataset.info()


# We can see that several columns are stored as numerical data (those with `int64` or `float64` types), while others contain more complex data types (those with `object` as Dtype)
#
# ## Visualising Ordinal Data
#
# Let's start by visualising some ordinal data. We have available:
#
# 1. `Pclass` - the ticket class
# 2. `Parch` - the number of parents or children on the ship
# 3. `sibsp` - the number of siblings or spouses on the ship
#
# Ordinal data can be viewed in almost any kind of graph.
# Let's start with a simple histogram that describing relationships between the ticket class and the likelihood of survival.

# In[ ]:


import graphing

graphing.histogram(
    dataset, label_x="Pclass", label_y="Survived", histfunc="avg", include_boxplot=True
)


# The box and whisker plot (top) shows that at least half the people had third-class tickets - notice how the median and maximum of the plot both sit at `Pclass = 3`.
#
# The histogram shows us that people in second and third class tended not to survive the wreck.
#
# Let's look at how survival varies, depending on whether you had parents or children on the ship

# In[ ]:


graphing.multiple_histogram(
    dataset,
    label_x="Pclass",  # group by ticket class
    label_group="Parch",  # colour by no parents or children
    label_y="Survived",
    histfunc="avg",
)


# For first and second class ticket holders, people in larger family groups appear to have had better rates of survival. This doesn't seem to be the case for third class passengers, though.
#
# Lastly, let's see if those with different ticket types tended to be in different sized families. A box and whisker is a nice alternative to histograms when we want to look at the spread of data.
#

# In[ ]:


graphing.box_and_whisker(dataset, label_x="Pclass", label_y="SibSp")


# Most values are zero. This tells us most people were travelling without siblings and without a partner. There are no obvious differences in this value between the different ticket classes.
#
# ## Visualising Continuous Data
#
# _Continuous_ data are usually best viewed using either:
#
# 1. An XY scatter plot, particularly for relationships between two continuous features
# 2. Histograms or Box and Whisker plots, to look at the spread of data
#
# Our dataset has `Age` and `Fare` as continuous data columns. Let's view them:

# In[ ]:


graphing.scatter_2D(dataset, label_x="Age", label_y="Fare")


# There isn't an obvious relationship between `Age` and `Fare`.
#
# Does the cost of a fare or the person's age have any relationship with likelihood of survival?

# In[ ]:


# Plot Fare vs Survival
graphing.histogram(
    dataset,
    label_x="Fare",
    label_y="Survived",
    histfunc="avg",
    nbins=30,
    title="Fare vs Survival",
    include_boxplot=True,
    show=True,
)

# Plot Age vs Survival
graphing.histogram(
    dataset,
    label_x="Age",
    label_y="Survived",
    histfunc="avg",
    title="Age vs Survival",
    nbins=30,
    include_boxplot=True,
)


# Our first figure's boxplot (top) shows us that most people held tickets costing less than £25, and the histogram shows us that people with more expensive tickets tended to survive.
#
# Our second figure indicates passengers were about 30yo on average, and that most children under 10yo survived, unlike most adults.
#
#
# ## Visualising Categorical Data
#
# Our Titanic dataset have the following _categorical_ columns:
# * `Sex` (Male, Female)
# * `Embarked` - the port of ambarkation (C, Q, or S)
# * `Cabin` (many options)
# * `Survival` (0 = no, 1 = yes)
#
# Categorical data are usually viewable in a similar way to ordinal data, but with data viewed as order-less groups. Alternatively, categories appear as colours or groups in other kinds of plots.
#
# Plotting categorical data against other categorical data lets us see how data is clustered. This is little more than a coloured table. Let's do this now:

# In[ ]:


import plotly.graph_objects as go
import numpy as np

# Create some simple functions
# Read their descriptions to find out more
def get_rows(sex, port):
    """Returns rows that match in terms of sex and embarkment port"""
    return dataset[(dataset.Embarked == port) & (dataset.Sex == sex)]


def proportion_survived(sex, port):
    """Returns the proportion of people meeting criteria who survived"""
    survived = get_rows(sex, port).Survived
    return np.mean(survived)


# Make two columns of data - together these represent each combination
# of sex and embarkment port
sexes = ["male", "male", "male", "female", "female", "female"]
ports = ["C", "Q", "S"] * 2

# Calculate the number of passengers at each port + sex combination
passenger_count = [len(get_rows(sex, port)) for sex, port in zip(sexes, ports)]

# Calculate the proportion of passengers from each port + sex combination who survived
passenger_survival = [proportion_survived(sex, port) for sex, port in zip(sexes, ports)]

# Combine into a single data frame
table = pd.DataFrame(
    dict(
        sex=sexes,
        port=ports,
        passenger_count=passenger_count,
        passenger_survival_rate=passenger_survival,
    )
)

# Make a bubble plot
# This is just a scatter plot but each entry in the plot
# has a size and colour. We set colour to passenger_survival
# and size to the number of passengers
graphing.scatter_2D(
    table,
    label_colour="passenger_survival_rate",
    label_size="passenger_count",
    size_multiplier=0.3,
    title="Bubble Plot of Categorical Data",
)


# Looks like women have a much higher survival rate than men, but there were more men on the ship.
#
# We can also see that most people boarded at Port `S` ("Southampton"). It does seem that there is a weak relationship between the port of boarding and survival.
#
# ## Summary
#
# You've learned about different types of data and practiced exploring data through graphs.
#
# Through these, we've discovered that some features are related to others, and that survival rate seems to be influenced by many features.

# %%
