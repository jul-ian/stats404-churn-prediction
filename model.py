## Importing libraries to be used and data
import numpy as np
import pandas as pd
import xgboost as xgb

def get_lists_of_dtypes(df):
    """Helper function to create list of features by type."""
    strings = list()
    integers = list()
    floats = list()

    for x in df.columns:
        if x != "Churn":
            if str(df[x].dtype)[:3] in 'obj':
                strings.append(x)
            elif str(df[x].dtype)[:3] in 'int':
                integers.append(x)
            elif str(df[x].dtype)[:3] in 'flo':
                floats.append(x)
            else:
                continue
        else:
                continue
    return strings, integers, floats

def yn_recode(s):
    """Helper function to recode 'yes' to 1 and 'no' to 0"""
    if s.lower() == "yes":
        return 1
    elif s.lower() == "no":
        return 0
    else:
        return np.nan

def prep_data(df):     
	"""Helper function to do basic feature engineering for training and input data""" 
	drop_cols = ["AgeHH1", "AgeHH2", "NewCellphoneUser", "NotNewCellphoneUser", 
	"CustomerID", "HandsetPrice", "ServiceArea", "RetentionOffersAccepted"]
    
	df = df.drop(columns=drop_cols).dropna()
    
	my_str, my_int, my_flo = get_lists_of_dtypes(df)
    
	df = pd.get_dummies(df, columns=my_str, drop_first=True)
	return df
        

def churn_prob(data_dict):
    """Function trains model and scored input data based on trained model"""
    chdata = pd.read_csv("churn_data.csv")
    chdata = prep_data(chdata)     
    chdata["Churn"] = chdata["Churn"].apply(yn_recode)

    x_train = chdata.drop(columns="Churn")
    y_train = chdata["Churn"]
    
    test_df = pd.DataFrame({k: [v] for k, v in data_dict.items()})
    test_df = prep_data(test_df)
    
    mis_col = set(x_train) - set(test_df)
    for col in mis_col:
        test_df[col] = 0
    test_df = test_df[x_train.columns]
    #use_label_encoder=False
    xgb_mod = xgb.XGBClassifier(eval_metric="logloss", learning_rate=0.01)
    
    xgb_mod.fit(x_train, y_train)
    
    y_pred_xg = xgb_mod.predict(test_df)
    
    test_df["prob"] = xgb_mod.predict_proba(test_df)[:, 1]
    
    return {"churn_probability":list(test_df["prob"])}


if __name__ == "__main__":

    my_input = {'CustomerID': 3000030,
     'MonthlyRevenue': 38.05,
     'MonthlyMinutes': 682.0,
     'TotalRecurringCharge': 52.0,
     'DirectorAssistedCalls': 0.25,
     'OverageMinutes': 0.0,
     'RoamingCalls': 0.0,
     'PercChangeMinutes': 148.0,
     'PercChangeRevenues': -3.1,
     'DroppedCalls': 9.0,
     'BlockedCalls': 1.7,
     'UnansweredCalls': 13.0,
     'CustomerCareCalls': 0.7,
     'ThreewayCalls': 0.0,
     'ReceivedCalls': 42.2,
     'OutboundCalls': 6.7,
     'InboundCalls': 0.0,
     'PeakCallsInOut': 33.3,
     'OffPeakCallsInOut': 53.0,
     'DroppedBlockedCalls': 10.7,
     'CallForwardingCalls': 0.0,
     'CallWaitingCalls': 0.7,
     'MonthsInService': 53,
     'UniqueSubs': 1,
     'ActiveSubs': 1,
     'ServiceArea': 'OKCTUL918',
     'Handsets': 3.0,
     'HandsetModels': 2.0,
     'CurrentEquipmentDays': 231.0,
     'AgeHH1': 28.0,
     'AgeHH2': 0.0,
     'ChildrenInHH': 'No',
     'HandsetRefurbished': 'No',
     'HandsetWebCapable': 'Yes',
     'TruckOwner': 'No',
     'RVOwner': 'No',
     'Homeownership': 'Known',
     'BuysViaMailOrder': 'No',
     'RespondsToMailOffers': 'No',
     'OptOutMailings': 'No',
     'NonUSTravel': 'Yes',
     'OwnsComputer': 'No',
     'HasCreditCard': 'Yes',
     'RetentionCalls': 0,
     'RetentionOffersAccepted': 0,
     'NewCellphoneUser': 'Yes',
     'NotNewCellphoneUser': 'No',
     'ReferralsMadeBySubscriber': 0,
     'IncomeGroup': 1,
     'OwnsMotorcycle': 'No',
     'AdjustmentsToCreditRating': 1,
     'HandsetPrice': '30',
     'MadeCallToRetentionTeam': 'No',
     'CreditRating': '3-Good',
     'PrizmCode': 'Other',
     'Occupation': 'Other',
     'MaritalStatus': 'Yes'}
    
    my_prob = churn_prob(my_input)
    print(my_prob)