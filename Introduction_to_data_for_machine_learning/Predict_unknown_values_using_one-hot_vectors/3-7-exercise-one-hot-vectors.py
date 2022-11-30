#!/usr/bin/env python
# coding: utf-8

# # Exercise: Titanic Dataset - One-Hot Vectors
#
# In this Unit we'll build a model to predict who survived the Titanic disaster.
#
# While doing so, we'll practice transforming data between numerical and categorical types, including using one-hot vectors.
#
# ## Preparing data
#
# Let's start by opening and quickly cleaning up our dataset, like we did in the last unit:
#

# In[ ]:


import pandas


# Open our dataset from file
dataset = pandas.read_csv("titanic.csv", index_col=False, sep=",", header=0)

# Fill missing cabin information with 'Unknown'
dataset["Cabin"].fillna("Unknown", inplace=True)

# Remove rows missing Age information
dataset.dropna(subset=["Age"], inplace=True)

# Remove the Name, PassengerId, and Ticket fields
# This is optional and only to make it easier to read our print-outs
dataset.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

dataset.head()


# ## About Our Model
#
# we'll be training a type of model called Logistic Regression, which will predict who survives the Titanic disaster.
#
# You don't need to understand logistic regression to understand this exercise, so we have put the implementation of outside of this notebook in a method called `train_logistic_regression`. If you're curious, you can read this method in our GitHub repository.
#
# `train_logistic_regression`:
#
# 1. Accepts our data frame and a list of features to include in the model.
# 2. Trains the model
# 3. Returns a number stating how well the model performs predicting survival. **Smaller numbers are better.**
#
# ## Numerical Only
#
# Let's create a model, only using the numerical features.
#
# First, we'll use `Pclass` here as a ordinal feature, rather than a one-hot encoded categorical feature.

# In[ ]:


from m0c_logistic_regression import train_logistic_regression

features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
loss_numerical_only = train_logistic_regression(dataset, features)

print(f"Numerical-Only, Log-Loss (cost): {loss_numerical_only}")


# We have our starting point. Let's see if we can improve the model using categorical features.
#
# ## Binary Categorical Features
#
# Categorical features that have just two potential values can be encoded in a single column as `0` and `1`.
#
# Let's convert `Sex` values into `IsFemale` - a `0` for male and `1` for female - and include that in our model.

# In[ ]:


# Swap male / female with numerical values
# We can do this because there are only two categories
dataset["IsFemale"] = dataset.Sex.replace({"male": 0, "female": 1})

# Print out the first few rows of the dataset
print(dataset.head())

# Run and test the model, also using IsFemale this time
features = ["Age", "Pclass", "SibSp", "Parch", "Fare", "IsFemale"]
loss_binary_categoricals = train_logistic_regression(dataset, features)

print(f"\nNumerical + Sex, Log-Loss (cost): {loss_binary_categoricals}")


# Our loss (error) has decreased! This means this model performs better than the previous.
#
# ## One-Hot Encoding
#
# Ticket class (`Pclass`) is an Ordinal feature. That means that its potential values (1, 2 & 3) are treated as having an order and being equally spaced. It's possible that this even spacing is simply not correct though - in stories we have heard about the Titanic, the third-class passengers were treated much worse than those in 1st and 2nd class.
#
# Let's convert `Pclass` into a categorical feature using one-hot encoding:

# In[ ]:


# Get all possible categories for the "PClass" column
print(f"Possible values for PClass: {dataset['Pclass'].unique()}")

# Use Pandas to One-Hot encode the PClass category
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass"], drop_first=False)

# Add back in the old Pclass column, for learning purposes
dataset_with_one_hot["Pclass"] = dataset.Pclass

# Print out the first few rows
dataset_with_one_hot.head()


# See how `Pclass` has been converted into three values: `Pclass_1`, `Pclass_2` and `Pclass_3`.
#
# Rows with `Pclass` of 1 have a value in the `Pclass_1` column. The same pattern is there for values of 2 and 3.
#
# Lets now re-run our model treating `Pclass` values as a categorical, rather than ordinal.

# In[ ]:


# Run and test the model, also using Pclass as a categorical feature this time
features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "IsFemale",
    "Pclass_1",
    "Pclass_2",
    "Pclass_3",
]

loss_pclass_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(
    f"\nNumerical, Sex, Categorical Pclass, Log-Loss (cost): {loss_pclass_categorical}"
)


# This seems to have made things slightly worse!
#
# Let's move on.
#
# ## Including Cabin
#
# Recall that many passengers had `Cabin` information. `Cabin` is a categorical feature and should be a good predictor of survival, because people in lower cabins probably had little time to escape during the sinking.
#
# Let's encode cabin using one-hot vectors and include it in a model. There are so many cabins this time that we won't print them all out. If you would like to practice printing them out, feel free to edit the code as practice.

# In[ ]:


# Use Pandas to One-Hot encode the Cabin and Pclass categories
dataset_with_one_hot = pandas.get_dummies(
    dataset, columns=["Pclass", "Cabin"], drop_first=False
)

# Find cabin column names
cabin_column_names = list(
    c for c in dataset_with_one_hot.columns if c.startswith("Cabin_")
)

# Print out how many cabins there were
print(len(cabin_column_names), "cabins found")

# Make a list of features
features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "IsFemale",
    "Pclass_1",
    "Pclass_2",
    "Pclass_3",
] + cabin_column_names

# Run the model and print the result
loss_cabin_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(
    f"\nNumerical, Sex, Categorical Pclass, Cabin, Log-Loss (cost): {loss_cabin_categorical}"
)


# That's our best result so far!
#
# ## Improving Power
#
# Including very large numbers of categorical classes - such as 135 Cabins - is often not the best way to train a model. This is because the model only has a few examples of each category class to learn from.
#
# Models can sometimes be improved by simplifying features. `Cabin` was probably useful because it indicated which floor of the titanic people were probably situated in: those in lower decks would have had their quarters flooded first.
#
# Using deck information might be simpler than categorizing people into Cabins.
#
# Let's simplify what we have run, replacing the 135 `Cabin` categories with a simpler `Deck` category, that has only 9 values: A - G, T, and U (Unknown)

# In[ ]:


# We have cabin names, like A31, G45. The letter refers to the deck that
# the cabin was on. Extract just the deck and save it to a column.
dataset["Deck"] = [c[0] for c in dataset.Cabin]

print("Decks: ", sorted(dataset.Deck.unique()))

# Create one-hot vectors for:
# Pclass - the class of ticket. (This could be treated as ordinal or categorical)
# Deck - the deck that the cabin was on
dataset_with_one_hot = pandas.get_dummies(
    dataset, columns=["Pclass", "Deck"], drop_first=False
)

# Find the deck names
deck_of_cabin_column_names = list(
    c for c in dataset_with_one_hot.columns if c.startswith("Deck_")
)

features = [
    "Age",
    "IsFemale",
    "SibSp",
    "Parch",
    "Fare",
    "Pclass_1",
    "Pclass_2",
    "Pclass_3",
    "Deck_A",
    "Deck_B",
    "Deck_C",
    "Deck_D",
    "Deck_E",
    "Deck_F",
    "Deck_G",
    "Deck_U",
    "Deck_T",
]

loss_deck = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nSimplifying Cabin Into Deck, Log-Loss (cost): {loss_deck}")


# ## Comparing Models
#
# Let's compare the `loss` for these models:

# In[ ]:


# Use a dataframe to create a comparison table of metrics
# Copy metrics from previous Unit

l = [
    ["Numeric Features Only", loss_numerical_only],
    ["Adding Sex as Binary", loss_binary_categoricals],
    ["Treating Pclass as Categorical", loss_pclass_categorical],
    ["Using Cabin as Categorical", loss_cabin_categorical],
    ["Using Deck rather than Cabin", loss_deck],
]

pandas.DataFrame(l, columns=["Dataset", "Log-Loss (Low is better)"])


# We can see that including categorical features can both improve and harm how well a model works. Often, experimentation is the best way to find the best model.
#
# ## Summary
#
# In this unit you learned how to use One-Hot encoding to address categorical data.
#
# We also explored how sometimes thinking critically about the problem you're trying can improve a solution better than simply including all possible features in a model.

# %%
