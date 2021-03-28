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
	drop_cols = ["AgeHH1", "AgeHH2", "NewCellphoneUser", "NotNewCellphoneUser", "CustomerID", "HandsetPrice", "ServiceArea", "RetentionOffersAccepted"]
	for t in drop_cols:
		if t in list(df.columns):
			df.drop(columns=t)
		else:
			continue

	my_str, my_int, my_flo = get_lists_of_dtypes(df)
    
	df = pd.get_dummies(df, columns=my_str, drop_first=True)
	
	return df
        

def churn_prob(data_dict, xgb_mod):
	"""Function trains model and scored input data based on trained model"""
	
	test_df = pd.DataFrame(data_dict, index=[0])
	test_df = prep_data(test_df)
    
	train_cols = xgb_mod.get_booster().feature_names
	mis_col = list(set(train_cols) - set(test_df.columns))
	if len(mis_col) > 0:
		for col in mis_col:
			test_df[col] = np.nan
		test_df = test_df[train_cols]
		
	y_pred_xg = xgb_mod.predict(test_df)
    
	test_df["prob"] = xgb_mod.predict_proba(test_df)[:, 1]
    
	return {"churn_probability":list(test_df["prob"])}


if __name__ == "__main__":
	
	import s3fs
	import joblib
	
	## Same input used in the Docker container
	my_input = {"MonthlyRevenue": 49.99, "MonthlyMinutes": 650.0, "TotalRecurringCharge": 50.0, "MonthsInService": 53}
	
	s3_fs = s3fs.S3FileSystem(anon=False)
	bucket_name = "stats404-project-ja"
	key_name = "xgb_model.joblib"
	
	with s3_fs.open(f"{bucket_name}/{key_name}","rb") as file:
		my_model = joblib.load(file)
    
	my_pred = churn_prob(my_input, my_model)
	
	print(my_pred)