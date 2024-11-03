## Base algorithm cross validation:

**With feature selection**:
- n_features = 8

|Model|Balanced Accuracy|F1 Score|ROC-AUC|
|-|-|-|-|
LogisticRegression|0.747348|0.745064|0.797293|
|SVC|0.505689|0.506007|0.508350|
|RandomForestClassifier|0.790880|0.798937|0.852683|
|LGBMClassifier|0.796176|0.807406|0.854283|
|XGBClassifier|0.786807|0.798518|0.843949|

**Without feature selection**:

|Model|Balanced Accuracy|F1 Score|ROC-AUC|
|-|-|-|-|
|LogisticRegression|0.753861|0.753811|0.832223|
|SVC|0.502068|0.502747|0.505785|
|RandomForestClassifier|0.801845|0.801898|0.888602|
|LGBMClassifier|0.803691|0.807208|0.899289|
|XGBClassifier|0.798819|0.800827|0.895854|

- Getting slightly better scores without recursive feature elimination (RFE, n_features=8)
- also tried with 5 and the results were even worse
- looks like it a better idea to have an 'all in' model or more feature engineering is required