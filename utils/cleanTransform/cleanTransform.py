import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Transform 'PassengerId'
def transform_passengerId(df):
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df['PassengerNumber'] = df['PassengerId'].str.split('_').str[1].astype(float)
    group_counts = df['GroupId'].value_counts()
    df['GroupSize'] = df['GroupId'].map(group_counts)
    df['InGroup'] = np.where(df['GroupSize'] > 1, 1, 0)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Transform 'Cabin'
def transform_Cabin(df):
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNumber'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]
    bin_edges = np.linspace(df['CabinNumber'].min(), df['CabinNumber'].max(), 5)
    df['CabinPosition'] = pd.cut(df['CabinNumber'], bins=bin_edges, labels=['Front','Second','Third','Back'], include_lowest=True)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Imputations for NaNs in 'HomePlanet'
def impute_homePlanet(df):
    # Calculate modes for each group
    group_modes = df.groupby('GroupId')['HomePlanet'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    deck_modes = df.groupby('Deck')['HomePlanet'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    global_mode = df['HomePlanet'].mode().iloc[0]
    # Apply group modes
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = (df.loc[df['HomePlanet'].isna(), 'GroupId'].map(group_modes))
    # Apply deck modes
    df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = (df.loc[df['HomePlanet'].isna(), 'Deck'].map(deck_modes))
    # Fill remaining missing values with global mode
    df['HomePlanet'].fillna(global_mode, inplace=True)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Proportional imputer for categorical columns
def proportional_imputer(df, impute_cols, group_col='HomePlanet'):
    # Store proportions for each column
    proportions = {}
    for col in impute_cols:
        proportions[col] = (df.groupby(group_col)[col].value_counts(normalize=True).dropna())

    # Impute missing values
    for col in impute_cols:
        col_proportions = proportions.get(col, None)
        if col_proportions is not None:
            def map_group(row):
                group = row[group_col]
                if pd.isna(row[col]) and pd.notna(group) and group in col_proportions.index:
                    group_proportions = col_proportions.loc[group]
                    return np.random.choice(group_proportions.index, p=group_proportions.values)
                return row[col]
            df[col] = df.apply(map_group, axis=1)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# KNN imputation for numerical columns
def knn_imputer(df, columns):
    imputer = KNNImputer(n_neighbors=5)
    df[columns] = imputer.fit_transform(df[columns])
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create 'TotalSpent' feature
def create_totalSpent(df):
    df['TotalSpent'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create servicesUsed columns
def createServicesUsed(df):
    services = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in services:
        df[f'{col}_used'] = df[col].apply(lambda x: 1 if x > 0 else 0).astype(int)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create bigSpender columns
def createBigSpenders(df):
    num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent'] 
    iqr_limits = {col: df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)) for col in num_cols}
    for col in num_cols:
        outlier_limit = iqr_limits[col]
        df[f'{col}_big_spender'] = df[col].apply(lambda x: 1 if x > outlier_limit else 0).astype(int)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert specific columns to integers
def convert_to_int(df):
    for col in ['InGroup', 'CryoSleep', 'VIP', 'Transported']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def oneHot(df, oh_cols=None):
    df = pd.get_dummies(df, columns=oh_cols)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def numPipe(df, num_cols=None):
    pt = PowerTransformer(method='yeo-johnson')
    scaler = StandardScaler()

    df[num_cols] = pt.fit_transform(df[num_cols])
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Drop unwanted columns
def drop_cols(df, drop_cols=None):
    droppers = drop_cols
    df.drop(droppers, axis=1, inplace=True)
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function to process DataFrame in order
def process_dataframe(df):
    df = transform_passengerId(df)
    df = transform_Cabin(df)
    df = impute_homePlanet(df)
    df = proportional_imputer(df, impute_cols=['Destination', 'Deck', 'Side', 'CabinPosition', 'VIP', 'CryoSleep'])
    df = knn_imputer(df, columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
    df = create_totalSpent(df)
    df = createServicesUsed(df)
    df = createBigSpenders(df)
    df = oneHot(df, oh_cols=['HomePlanet','Destination','Deck','Side','CabinPosition','GroupSize'])
    df = numPipe(df, num_cols=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalSpent'])
    df = convert_to_int(df)
    df = drop_cols(df, drop_cols=['PassengerNumber','GroupId','Cabin','CabinNumber','Name'])
    return df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------