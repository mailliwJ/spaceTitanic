import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Transform 'PassengerId'
def transform_passengerId(df):
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df['PassengerNumber'] = df['PassengerId'].str.split('_').str[1].astype(float)
    group_counts = df['GroupId'].value_counts()
    df['GroupSize'] = df['GroupId'].map(group_counts)
    df['InGroup'] = np.where(df['GroupSize'] > 1, 1, 0)
    return df

# Transform 'Cabin'
def transform_Cabin(df):
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNumber'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]
    bin_edges = np.linspace(df['CabinNumber'].min(), df['CabinNumber'].max(), 5)
    df['CabinPosition'] = pd.cut(df['CabinNumber'],
                                 bins=bin_edges,
                                 labels=['Front','Second','Third','Back'],
                                 include_lowest=True)
    return df

# Imputations for NaNs in 'HomePlanet'
def impute_homePlanet(df):
    group_modes = df.groupby('GroupId')['HomePlanet'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = group_modes[df['HomePlanet'].isna()]

    deck_modes = df.groupby('Deck')['HomePlanet'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = deck_modes[df['HomePlanet'].isna()]

    if 'VIP' in df.columns:
        vip_mode_homePlanet = df.loc[df['VIP'] == True, 'HomePlanet'].mode().iloc[0]
        df.loc[df['VIP'] & df['HomePlanet'].isna(), 'HomePlanet'] = vip_mode_homePlanet

    df['HomePlanet'].fillna(df['HomePlanet'].mode().iloc[0], inplace=True)

    return df

# Proportional imputer for categorical columns
def proportional_imputer(df, impute_cols):
    for col in impute_cols:
        proportions = df.groupby('HomePlanet')[col].value_counts(normalize=True)

        def impute_values(row):
            if pd.isna(row[col]):
                group = row['HomePlanet']
                if pd.notna(group) and group in proportions.index:
                    group_proportions = proportions.loc[group].dropna()
                    return np.random.choice(group_proportions.index, p=group_proportions.values)
            return row[col]
        
        # Apply the impute function to each column
        df[col] = df.apply(impute_values, axis=1)
    return df

# KNN imputation for numerical columns
def knn_imputer(df, columns):
    imputer = KNNImputer(n_neighbors=5)
    df[columns] = imputer.fit_transform(df[columns])
    return df

# Create 'TotalSpent' feature
def create_totalSpent(df):
    df['TotalSpent'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
    return df

# Convert specific columns to integers
def convert_to_int(df):
    for col in ['InGroup', 'CryoSleep', 'VIP', 'Transported']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# Drop unwanted columns
def drop_cols(df):
    droppers = ['PassengerNumber','GroupId','Cabin','CabinNumber','Name']
    df.drop(droppers, axis=1, inplace=True)
    return df

# Main function to process DataFrame in order
def process_dataframe(df):
    df = transform_passengerId(df)
    df = transform_Cabin(df)
    df = impute_homePlanet(df)
    df = proportional_imputer(df, impute_cols=['Destination', 'Deck', 'Side', 'CabinPosition', 'VIP', 'CryoSleep'])
    df = knn_imputer(df, columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
    df = create_totalSpent(df)
    df = convert_to_int(df)
    return df

#---------------------------------------------------

# Transform 'PassengerId'
def transform_passengerId(df):
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df['PassengerNumber'] = df['PassengerId'].str.split('_').str[1].astype(float)
    group_counts = df['GroupId'].value_counts()
    df['GroupSize'] = df['GroupId'].map(group_counts)
    df['InGroup'] = np.where(df['GroupSize'] > 1, 1, 0)
    return df

# Transform 'Cabin'
def transform_Cabin(df):
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNumber'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]
    bin_edges = np.linspace(df['CabinNumber'].min(), df['CabinNumber'].max(), 5)
    df['CabinPosition'] = pd.cut(df['CabinNumber'],
                                 bins=bin_edges,
                                 labels=['Front','Second','Third','Back'],
                                 include_lowest=True)
    return df

# Imputations for NaNs in 'HomePlanet'
def impute_homePlanet(df):
    group_modes = df.groupby('GroupId')['HomePlanet'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = group_modes[df['HomePlanet'].isna()]

    deck_modes = df.groupby('Deck')['HomePlanet'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = deck_modes[df['HomePlanet'].isna()]

    if 'VIP' in df.columns:
        vip_mode_homePlanet = df.loc[df['VIP'] == True, 'HomePlanet'].mode().iloc[0]
        df.loc[df['VIP'] & df['HomePlanet'].isna(), 'HomePlanet'] = vip_mode_homePlanet

    df['HomePlanet'].fillna(df['HomePlanet'].mode().iloc[0], inplace=True)

    return df

# Proportional imputer for categorical columns
def proportional_imputer(df, impute_cols):
    for col in impute_cols:
        proportions = df.groupby('HomePlanet')[col].value_counts(normalize=True)

        def impute_values(row):
            if pd.isna(row[col]):
                group = row['HomePlanet']
                if pd.notna(group) and group in proportions.index:
                    group_proportions = proportions.loc[group].dropna()
                    return np.random.choice(group_proportions.index, p=group_proportions.values)
            return row[col]
        
        # Apply the impute function to each column
        df[col] = df.apply(impute_values, axis=1)
    return df

# KNN imputation for numerical columns
def knn_imputer(df, columns):
    imputer = KNNImputer(n_neighbors=5)
    df[columns] = imputer.fit_transform(df[columns])
    return df

# Create 'TotalSpent' feature
def create_totalSpent(df):
    df['TotalSpent'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
    return df

# Create 'service_used' and 'big_spender' binary features
def create_serviceSpenders(df):

    num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent'] 
    iqr_limits = {col: df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)) for col in num_cols}

    for col in num_cols:
        if col != 'TotalSpent':
            df[f'{col}_used'] = df[col].apply(lambda x: 1 if x > 0 else 0).astype(int)

        outlier_limit = iqr_limits[col]
        df[f'{col}_big_spender'] = df[col].apply(lambda x: 1 if x > outlier_limit else 0).astype(int)
    
    return df

# Convert specific columns to integers
def convert_to_int(df):
    for col in ['InGroup', 'CryoSleep', 'VIP', 'Transported']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# Drop unwanted columns
def drop_cols(df):
    droppers = ['PassengerNumber','GroupId','Cabin','CabinNumber','Name']
    df.drop(droppers, axis=1, inplace=True)
    return df

# Main function to process DataFrame in order
def process_dataframe(df):
    df = transform_passengerId(df)
    df = transform_Cabin(df)
    df = impute_homePlanet(df)
    df = proportional_imputer(df, impute_cols=['Destination', 'Deck', 'Side', 'CabinPosition', 'VIP', 'CryoSleep'])
    df = knn_imputer(df, columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
    df = create_totalSpent(df)
    df = create_serviceSpenders(df)
    df = convert_to_int(df)
    df = drop_cols(df)
    return df