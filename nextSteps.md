ERROR in Stacking Classifier
    - Not sure what new behaviour is all about!

1. Feature Selection Methods:
    - Consider best methods for regression based and tree based algorithms
    - Investigate the use / implementation of the following:
        a) SeletKBest using mutual_info_score
        b) Recursive Feature Elimination check [here](https://machinelearningmastery.com/rfe-feature-selection-in-python/) and [here](https://medium.com/@hsu.lihsiang.esth/feature-selection-with-recursive-feature-elimination-rfe-for-parisian-bike-count-data-23f0ce9db691#:~:text=Firstly%2C%20unlike%20SelectKBest%2C%20RFECV%20does,the%20number%20of%20features%20dynamically.)

        ---> RFE (select=5) made the model worse. try rerunning in pipeline on all algorithms
        -> still need to investigate slectKbest

2. Reconsider Pipeline:
    - Is there a better way to construct the final pipeline so that bianry variables do not have to be one-hot encoded?
        - Investigate adding parameters to the custom classes and focus on returning only the columns wanted as input for final model.
        - Potentially using using FunctionTransformers in a pipeline that is then wrapped in a column transformer
            - Need to investigate how the column transformer will behave in this scenario and if i can pass the transformations and feature creations onto the next step in the pipeline

3. Ensure understanding of stacking and voting classifiers

4. EDA
    - Use PhiK matrix for checking features for colinearity and correlations with target

5. CatBooost issues:
    - see [here](https://github.com/kinir/catboost-with-pipelines/blob/master/sklearn-pandas-catboost.ipynb)

6. Add a cleaned version of dataset to /data directory