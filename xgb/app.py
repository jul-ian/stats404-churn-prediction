from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def xgb_predict():
	req = request.get_json()
	
	model = joblib.load('xgb_model.joblib')
	
	df_req = pd.DataFrame(req, index=[0])
	
	drop_cols = ["AgeHH1", "AgeHH2", "NewCellphoneUser", "NotNewCellphoneUser", "CustomerID", "HandsetPrice", "ServiceArea", "RetentionOffersAccepted"]
	for t in drop_cols:
		if t in list(df_req.columns):
			df_req.drop(columns=t)
		else:
			continue

	strings = list()
	integers = list()
	floats = list()

	for x in df_req.columns:
		if str(df_req[x].dtype)[:3] in 'obj':
			strings.append(x)
		elif str(df_req[x].dtype)[:3] in 'int':
			integers.append(x)
		elif str(df_req[x].dtype)[:3] in 'flo':
			floats.append(x)
		else:
			continue
    
	df1 = pd.get_dummies(df_req, columns=strings, drop_first=True)

	train_cols = model.get_booster().feature_names
	mis_col = list(set(train_cols) - set(df_req.columns))
	if len(mis_col) > 0:
		for col in mis_col:
			df_req[col] = np.nan
		df_req = df_req[train_cols]

	df_req["prob"] = model.predict_proba(df_req)[:, 1]
	
	result = {"Churn_Probability" :list(df_req["prob"])}
	
	return jsonify(result)

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
    
    
