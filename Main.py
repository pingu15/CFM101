import yfinance as yf

hist = yf.Ticker('BAC').history(period='1y')
volume = hist['Volume'].resample('MS').sum()
print(volume)