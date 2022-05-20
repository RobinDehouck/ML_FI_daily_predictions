import yfinance as yf
import utils.rsi
import utils.trading_instruments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import joblib

def get_clean_df(yf_token):
	data = yf.Ticker(yf_token + "-USD")
	df = data.history(period="2y", interval = '1d')
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
	df_v2.to_csv('btc_usd_05-17-2022.csv')
	return (df_v2)

# print(get_clean_df("BTC").columns)
total_acc = 0
tokens = ['BNB', 'SOL', 'DOGE', 'AVAX', 'ADS', 'ILV', 'WAXP']
for token in tokens:
	total_acc_bis = 0
	df = get_clean_df(token) #BNB 66 SOL 67 DOGE 69 AVAX 65 ADS 73 ILV 65 WAXP 67
	for i in range(1):
		train = df
		for i in train.columns:
			le = LabelEncoder()
			train[i] = le.fit_transform(train[i])
		train, test = train_test_split(train, test_size = 0.2)
		var_to_use = train.columns.tolist()
		var_to_use.remove('decile_price_tomorrow')
		var_to_use.remove('decile_price')

		rf = RandomForestRegressor()
		rf.fit(train[var_to_use], train['decile_price_tomorrow'])

		pred = rf.predict(test[var_to_use])
		pred = pred.round()
		test['pred'] = pred
		acc = accuracy_score(test['decile_price_tomorrow'], pred)
		joblib.dump(rf, "models/random_forest_daily_" + token + ".joblib")
		j = 0
		count_good = 0
		count_bad = 0
		while j < len(test):
			if test['decile_price'][j] != test['pred'][j] and test['decile_price'][j] != test['decile_price_tomorrow'][j]:
				print(test.loc[[test.index[j]]])
				if test['decile_price_tomorrow'][j] > test['decile_price'][j] and test['pred'][j] > test['decile_price'][j]:
					count_good+=1
				elif test['decile_price_tomorrow'][j] < test['decile_price'][j] and test['pred'][j] < test['decile_price'][j]:
					count_good+=1
				else:
					count_bad+=1
			j+=1
		total_acc+=acc
		# print(acc)
		total_acc_bis+= count_good / (count_good + count_bad)
		# print(count_good, count_bad, count_good / (count_good + count_bad))
	# print(var_to_use)
	print(token, total_acc_bis/1)