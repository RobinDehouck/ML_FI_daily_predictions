import yfinance as yf
import utils.rsi
import utils.trading_instruments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def get_clean_df(yf_token):
	data = yf.Ticker(yf_token + "-USD")
	df = data.history(period="1y", interval = '1d')
	rsidata = utils.rsi.rsi(df)
	temp = rsidata.to_frame()
	df['rsi'] = temp
	df['sma'] = utils.trading_instruments.awesome_oscillator(df)
	df['bop'] = utils.trading_instruments.bop(df)
	df['pgo'] = utils.trading_instruments.pgo(df)
	df['ao'] = utils.trading_instruments.awesome_oscillator(df)
	df['cmo'] = utils.trading_instruments.cmo(df)
	df['apo'] = utils.trading_instruments.apo(df)
	del df['Open']
	del df['High']
	del df['Low']
	del df['Dividends']
	del df['Stock Splits']

	df_v2 = pd.DataFrame()
	df_v2['decile_rsi']= pd.qcut(df['rsi'],q = 20, labels = False)
	df_v2['decile_volume']= pd.qcut(df['Volume'],q = 20, labels = False)
	df_v2['decile_sma']= pd.qcut(df['sma'],q = 20, labels = False)
	df_v2['decile_bop']= pd.qcut(df['bop'],q = 20, labels = False)
	df_v2['decile_ao']= pd.qcut(df['ao'],q = 20, labels = False)
	df_v2['decile_apo']= pd.qcut(df['apo'],q = 20, labels = False)
	df_v2['decile_cmo']= pd.qcut(df['cmo'],q = 20, labels = False)
	df_v2['decile_price']= pd.qcut(df['Close'],q = 20, labels = False)
	df_v2['decile_price_tomorrow'] = df_v2['decile_price'].shift(-1)
	df_v2 = df_v2.dropna()

	return (df_v2)

total_acc = 0
for i in range(100):
	train = get_clean_df("BTC")
	list_var_cat = train.select_dtypes('float').columns.tolist()
	list_var_cat += train.select_dtypes('int').columns.tolist()
	for i in list_var_cat:
		le = LabelEncoder()
		train[i] = le.fit_transform(train[i])

	train, test = train_test_split(train, test_size = 0.2)
	var_to_use = train.columns.tolist()
	var_to_use.remove('decile_price_tomorrow')
	var_to_use.remove('decile_ao')
	var_to_use.remove('decile_apo')
	var_to_use.remove('decile_cmo')


	rf = RandomForestClassifier()
	rf.fit(train[var_to_use], train['decile_price_tomorrow'])

	pred = rf.predict(test[var_to_use])
	test['pred'] = pred
	acc = accuracy_score(test['decile_price_tomorrow'], pred)
	# joblib.dump(rf, "./random_forest_vol_rsi_v2.joblib")
	# print(acc)
	# print(test)
	total_acc+=acc
print(var_to_use)
print(total_acc/100)