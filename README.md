# Space Titanic ML Project
<div style='text-align: left;'>
    <img src='./imgs/spaceshipTitanic.jpg' alt='Spaceship Titanic' width='50%'/>
</div>

## Project Overview

This project focuses on developing a machine learning model to predct whether a passenger was transported off of a spaceship in a simulated diasaster event. The primary objective is to achieve a classification accuracy of over 80% on the prediction dataset. Through this project i aim to deepen my understanding of scikit-learn, and develop custom classes and functions for use in future machine learning workflows.

## Problem Statement

The problem is a binary classification task, in which the model predicts the liklihood of a passenger being transported off of a spaceship in a freak disaster event, based on various passenger-related features.

## Project Goals

1. Build a classification model that achieves over 80% accuracy on the prediction dataset
2. Develop and use custom classes and functions to structure machine learning projects for efficient preprocessing, feature engineering, and evaluation
3. Continue developing [MLToolz](https://github.com/mailliwJ/MLToolz) and [VizToolz](https://github.com/mailliwJ/VizToolz) scripts which have now been turned into packages for easier access and use in future projects.

## Dataset Description

Two datasets were available for this project via [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data)
; a training set and a predictions set. Both datasets include the same features, except for the prediction set no having the target feature, 'Transported'. The features are all based on passenger-related information. The table below outlines the features. The training set is fairly clean, however all columns except `PassengerId` and `Transported` have missing values that require addressing.

|Feature|Description|
|-|-|
|`PassengerId`|A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always|
|`HomePlanet`|The planet the passenger departed from, typically their planet of permanent residence|
|`CryoSleep`|Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins|
|`Cabin`|The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard|
|`Destination`|The planet the passenger will be debarking to|
|`Age`|The age of the passenger|
|`VIP`|Whether the passenger has paid for special VIP service during the voyage|
|`RoomService`|Amount the passenger has billed for room service|
|`FoodCourt`|Amount the passenger has billed at the food court|
|`ShoppingMall`|Amount the passenger has billed at the shopping mall|
|`Spa`|Amount the passenger has blled at the spa|
|`VR Deck`|Amount the passenger has billed at the VR deck|
|`Name`|The first and last name of the passenger|
|`Transported`|Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict|

## Key Feature Creation

Some of the features do not contribute much insight directly, however they can be transformed into more useful features. I have create the following features:

## Data Preprocessing and Feature Engineering

After a brief exploration of the training dataset I devised a strategy for imputing missing values. I decided to use a combination of methods
use of custom classes to carry out imputation
porportional imputer funciton
KNN Imputer

Pipeline set-up
the development of a working pipeline structure took several iterations to overcome various errors. My final pipeline solution includes a 
why its important

use of cutom classes and column transformers
- power transformations
- scaling
- encoding

## Model Selection and Evaluation

Which algorithms chosen to investigate initially
how these were evaluated (including discussion about metrics)
functions to modularize visualization and presentation of performance metrics
- classification report
- confusion matrix

## Model Tuning
- exploration of hyperparameter tuning using gridsearchCV, randomizedsearchCV and bayesian methods using optuna

## Voting and Stacking methods
- Sideline investigation into potential for a stacked and or voting classifier

## Results

Show score in submission on Kaggle

## Project Structure

- `data/`: Folder containing raw and processed datasets
- `imgs/`: Folder continaing images used in notebooks
- `models/`: Folder containing saved models to 'save' time retraining
- `notebooks/`: Jupyter notebooks documenting data exploration, preprocessing, feature engineering, model construction and evaluation stored as episode in this saga of a space project
- `utils/`: Python scripts for custom functions, preprocessing classes and pipeline setup
- `README.md`: Project overivew and documentation
- `projectBrief.md`: Summary of the project task
