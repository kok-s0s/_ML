#!/usr/bin/env python
# coding: utf-8

# # Exercise: Using a Trained Model on New Data
#
# In Unit 3, we created a basic model that let us find the relationship between a dog's harness size and their boot size. We showed how this model could then be used to make a prediction about a new, previously unseen dog.
#
# It's common to build, train, then use a model while we are just learning about machine learning; but in the real world, we don't want to train the model _every time_ we want to make a prediction.
#
# Consider our avalanche-dog equipment store scenario:
#
# * We want to train the model just once, then load that model onto the server that runs our online store.
# * Although the model is _trained_ on a dataset we downloaded from the internet, we actually want to _use_ it to estimate the boot size of our customers' dogs who are not in this dataset!
#
# How can we do this?
#
# Here, we'll:
#
# 1. Create a basic model
# 2. Save it to disk
# 3. Load it from disk
# 4. Use it to make predictions about a dog who was not in the training dataset
#
# ## Load the dataset
#
# Let's begin by opening the dataset from file.

# In[ ]:


import pandas


# Load a file containing dog's boot and harness sizes
data = pandas.read_csv("doggy-boot-harness.csv")

# Print the first few rows
data.head()


# ## Create and train a model
#
# As we've done before, we'll create a simple Linear Regression model and train it on our dataset.

# In[ ]:


import statsmodels.formula.api as smf

# Fit a simple model that finds a linear relationship
# between boot size and harness size, which we can use later
# to predict a dog's boot size, given their harness size
model = smf.ols(formula="boot_size ~ harness_size", data=data).fit()

print("Model trained!")


# ## Save and load a model
#
# Our model is ready to use, but we don't need it yet. Let's save it to disk.

# In[ ]:


import joblib

model_filename = "./avalanche_dog_boot_model.pkl"
joblib.dump(model, model_filename)

print("Model saved!")


# Loading our model is just as easy:

# In[ ]:


model_loaded = joblib.load(model_filename)

print("We have loaded a model with the following parameters:")
print(model_loaded.params)


# ## Put it together
#
# On our website, we'll want to take the harness of our customer's dog, then calculate their dog's boot size using the model that we've already trained.
#
# Let's put everything here together to make a function that loads the model from disk, then uses it to predict our customer's dog's boot size height.

# In[ ]:


# Let's write a function that loads and uses our model
def load_model_and_predict(harness_size):
    """
    This function loads a pretrained model. It uses the model
    with the customer's dog's harness size to predict the size of
    boots that will fit that dog.

    harness_size: The dog harness size, in cm
    """

    # Load the model from file and print basic information about it
    loaded_model = joblib.load(model_filename)

    print("We've loaded a model with the following parameters:")
    print(loaded_model.params)

    # Prepare data for the model
    inputs = {"harness_size": [harness_size]}

    # Use the model to make a prediction
    predicted_boot_size = loaded_model.predict(inputs)[0]

    return predicted_boot_size


# Practice using our model
predicted_boot_size = load_model_and_predict(45)

print("Predicted dog boot size:", predicted_boot_size)


# ## Real world use
#
# We've done it; we can predict an avalanche dog's boot size based on the size of their harness. Our last step is to use this to warn people if they might be buying the wrong sized doggy boots.
#
# As an example, we'll make a function that accepts the harness size, the size of the boots selected, and returns a message for the customer. We would integrate this function into our online store.

# In[ ]:


def check_size_of_boots(selected_harness_size, selected_boot_size):
    """
    Calculates whether the customer has chosen a pair of doggy boots that
    are a sensible size. This works by estimating the dog's actual boot
    size from their harness size.

    This returns a message for the customer that should be shown before
    they complete their payment

    selected_harness_size: The size of the harness the customer wants to buy
    selected_boot_size: The size of the doggy boots the customer wants to buy
    """

    # Estimate the customer's dog's boot size
    estimated_boot_size = load_model_and_predict(selected_harness_size)

    # Round to the nearest whole number because we don't sell partial sizes
    estimated_boot_size = int(round(estimated_boot_size))

    # Check if the boot size selected is appropriate
    if selected_boot_size == estimated_boot_size:
        # The selected boots are probably OK
        return f"Great choice! We think these boots will fit your avalanche dog well."

    if selected_boot_size < estimated_boot_size:
        # Selected boots might be too small
        return (
            "The boots you have selected might be TOO SMALL for a dog as "
            f"big as yours. We recommend a doggy boots size of {estimated_boot_size}."
        )

    if selected_boot_size > estimated_boot_size:
        # Selected boots might be too big
        return (
            "The boots you have selected might be TOO BIG for a dog as "
            f"small as yours. We recommend a doggy boots size of {estimated_boot_size}."
        )


# Practice using our new warning system
check_size_of_boots(selected_harness_size=55, selected_boot_size=39)


#
# Change `selected_harness_size` and `selected_boot_size` in the preceding example and re-run the cell to see this in action.
#
# ## Summary
#
# Well done! We've put together a system that can predict if customers are buying doggy boots that may not fit their avalanche dog, based solely on the size of harness they're purchasing.
#
# In this exercise, we practiced:
#
# 1. Creating basic models
# 2. Training, then saving them to disk
# 3. Loading them from disk
# 4. Making predictions with them using new data sets
