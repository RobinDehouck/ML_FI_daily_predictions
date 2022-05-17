import pandas as pd
import pandas_ta as ta
import yfinance as yf

def awesome_oscillator(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.ao())

def simple_moving_average(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.sma())

def bop(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.bop())

def pgo(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.pgo())

def apo(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.apo())

def cmo(df):
	df.set_index(pd.DatetimeIndex(df.index), inplace=True)
	return(df.ta.cmo())

# name = 'ao'
# token = yf.Ticker('BTC-USD')
# df = token.history(period="4y")
# df[name] = instru(df)
# df['decile_price']= pd.qcut(df['Close'],q = 15, labels = False)
# df['decile_' + name]= pd.qcut(df[name],q = 15, labels = False)
# print(df.tail(50))
# print(df['decile_price'].corr(df['decile_' + name]))

#cmo 0.16
#apo 0.16
#ao 0.16