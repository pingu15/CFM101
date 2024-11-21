import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import threading
import sys
import numpy as np

attributes = ['sector', 'exchange', 'currency', 'marketCap']
start_date = '2023-10-01'
end_date = '2024-10-01'


# Each industry is mapped to (% share of S&P500, % share of TSX60, S&P industry ticker, TSX60 capped industry ticker)
# To obtain values for % share, run market_by_sector('SP500') and market_by_sector('TSX60'), respectively (see next cell)
# Since % share changes quarterly, we don't need to run this every time
sectors = {
    'Basic Materials': (0.0171, 0.0849, '^SP500-15', '^GSPTTMT'),
    'Industrials': (0.0719, 0.1311, '^SP500-20', '^GSPTTIN'),
    'Consumer Cyclical': (0.1075, 0.0531, '^SP500-25', '^GSPTTCD'),
    'Consumer Defensive': (0.0576, 0.0509, '^SP500-30', '^GSPTTCS'),
    'Healthcare': (0.1014, 0.0000, '^SP500-35', '^GSPTTHC'),
    'Financial Services': (0.1303, 0.3387, '^SP500-40', '^SPTTFS'),
    'Technology': (0.3045, 0.0963, '^SP500-45', '^SPTTTK'),
    'Communication Services': (0.1340, 0.0304, '^SP500-50', '^GSPTTTS'),
    'Utilities': (0.0235, 0.0318, '^SP500-55', '^GSPTTUT'),
    'Real Estate': (0.0207, 0.0062, '^SP500-60', '^GSPTTRE'),
    'Energy': (0.0315, 0.1766, '^SP500-1010', '^SPTTEN')
}

# adds ticker info to data df
def get_data(ticker, data, history, filter):
    yf_data = yf.Ticker(ticker).info
    if(not filter):
        for att in attributes:
            if(att not in yf_data):
                print(ticker, 'missing', att)
                continue
            data[ticker, att] = yf_data[att]
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
        history[ticker] = hist['Close'].pct_change().dropna()
        return
    # check if stock is CAD or USD
    if('currency' not in yf_data or yf_data['currency'] not in ['USD', 'CAD']):
        data.drop(ticker, inplace=True)
        print('Dropped', ticker)
        return
    for att in attributes:
        if(att not in yf_data):
            print(ticker, 'missing', att)
            continue
        data.loc[ticker, att] = yf_data[att]
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
    history[ticker] = hist['Close'].pct_change().dropna()
    volume = hist['Volume'].resample('MS').sum()
    # Take all months with >= 18 trading days for volume calculation
    volume.drop([month for month in volume.index if hist.resample('MS').size().loc[month] < 18], inplace=True)
    # check if stock has at least 100,000 average monthly volume
    if(volume.mean() < 1e5):
        data.drop(ticker, inplace=True)

# returns df containing all ticker info
def get_tickers(file_name='Tickers.csv', filter=True):
    with threading.Lock():
        tickers = pd.read_csv(file_name, header=None)
        data = pd.DataFrame(index=[ticker for ticker in tickers[0]], columns=attributes)
        history = {}
        threads = [threading.Thread(target=get_data, args=(ticker,data,history,filter)) for ticker in tickers[0]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return (data, history)

def weighted_max_bipartite_matching(N, M, A, C):
    """
    Finds the weighted maximum bipartite matching for sectors and stocks.
    
    Args:
    - N: Number of sectors.
    - M: Number of stocks.
    - A: List of length N where A[i] is the number of stocks required by sector i.
    - C: 2D list (N x M) of weights (correlations) between sectors and stocks.
    
    Returns:
    - match: List of tuples (sector, stock) representing the matching.
    - total_weight: Total weight of the matching.
    """
    # Expand the graph: create dummy nodes for each sector demand
    total_sectors = sum(A)
    expanded_C = [[-sys.maxsize] * M for _ in range(total_sectors)]
    
    sector_mapping = []
    index = 0
    for i in range(N):
        for _ in range(A[i]):
            expanded_C[index] = C[i]
            sector_mapping.append(i)  # Map expanded sector to original sector
            index += 1

    # Hungarian algorithm for max-weight matching
    match = [-1] * M  # Stores which sector is assigned to each stock
    sector_label = [0] * total_sectors
    stock_label = [0] * M
    slack = [0] * M
    slack_x = [-1] * M
    parent = [-1] * M
     
    def dfs(x, visited_x, visited_y):
        visited_x[x] = True
        for y in range(M):
            if visited_y[y]:
                continue
            delta = sector_label[x] + stock_label[y] - expanded_C[x][y]
            if delta == 0:  # Tight edge
                visited_y[y] = True
                if match[y] == -1 or dfs(match[y], visited_x, visited_y):
                    match[y] = x
                    return True
            else:  # Update slack
                if slack[y] > delta:
                    slack[y] = delta
                    slack_x[y] = x
        return False

    # Initialize labels
    for x in range(total_sectors):
        sector_label[x] = max(expanded_C[x])

    # Augmenting path search
    for x in range(total_sectors):
        slack = [sys.maxsize] * M
        slack_x = [-1] * M
        while True:
            visited_x = [False] * total_sectors
            visited_y = [False] * M
            if dfs(x, visited_x, visited_y):
                break
            # Update labels
            delta = min(slack[y] for y in range(M) if not visited_y[y])
            for i in range(total_sectors):
                if visited_x[i]:
                    sector_label[i] -= delta
            for y in range(M):
                if visited_y[y]:
                    stock_label[y] += delta
                else:
                    slack[y] -= delta

    # Extract results
    total_weight = 0
    final_match = []
    for y in range(M):
        if match[y] != -1:
            sector_idx = sector_mapping[match[y]]
            final_match.append((sector_idx, y))
            total_weight += C[sector_idx][y]

    return final_match, total_weight

# returns function of stocks to sectors as given by f
# 0 for S&P500, 1 for TSX60
def calc(data, history, f, index):
    sector_metric = {stock:{} for stock in data.index}
    for sector in sectors:
        for stock in data.index:
            if(sectors[sector][index] == 0):
                continue
            df = pd.DataFrame({stock: history[stock], sector: history[sectors[sector][2+index]]}).dropna()
            # calculate metric given a function f
            sector_metric[stock][sector] = f(df, stock, sector)
    return sector_metric

def beta(df, stock, sector):
    return df[stock].cov(df[sector])/df[sector].var()

def corr(df, stock, sector):
    return df[stock].corr(df[sector])

# returns df containing history for each sector in TSX60
# since historical data for individual TSX60 sectors is unavailable, we take the weighted average of all stocks in each sector
def tsx_sectors():
    data, history = get_tickers('TSX60.csv', False)
    sector_history = pd.DataFrame({sectors[sector][3]: 0 for sector in sectors}, index=history[data.index[0]].index)
    total_market_cap = {sectors[sector][3]: 0 for sector in sectors}
    for stock in history:
        total_market_cap[sectors[data['sector'].loc[stock]][3]] += data['marketCap'].loc[stock]
    for stock in history:
        sector = sectors[data['sector'].loc[stock]][3]
        sector_history[sector] += history[stock]*data['marketCap'].loc[stock]/total_market_cap[sector]
    return sector_history

# returns df containing history for each sector in S&P500
def sp_sectors():
    history = {sectors[sector][2]: yf.Ticker(sectors[sector][2]).history(start=start_date, end=end_date)['Close'].pct_change().dropna() for sector in sectors}
    return pd.DataFrame(history, index=list(history.values())[0].index)

# returns sector percent change since start date
def aggregate_pct_change(history, stock):
    result = pd.Series(index=history[stock].index)
    prev = 1
    for day in history[stock].index:
        result[day] = prev*(1+history[stock][day])
        prev = result[day]
    return result

# binary search for optimal max percentage of a single stock such that we can have 24 stocks in our portfolio
def max_percentage(min_pct):
    low = min_pct
    high = 1.0
    while(low < high):
        mid = (low+high)/2
        sum = 0
        for sector in sectors:
            sum += min(max(1, sectors[sector][0]/2//mid), sectors[sector][0]/2//min_pct) + min(max(1, sectors[sector][1]/2//mid), sectors[sector][1]/2//min_pct)
        if(sum > 24):
            low = mid+0.0001
        else:
            high = mid
    return round(low, 4)

def create_portfolio(sector_corr, min_pct, max_pct):
    portfolio = {stock: 0 for stock in data.index}
    SECTOR_IDX = {}
    IDX_SECTOR = {}
    STOCK_IDX = {}
    IDX_STOCK = {}
    idx = 0
    for sector in sectors:
        SECTOR_IDX[sectors[sector][2]] = idx
        IDX_SECTOR[idx] = sectors[sector][2]
        SECTOR_IDX[sectors[sector][3]] = idx+1
        IDX_SECTOR[idx+1] = sectors[sector][3]
        idx += 2
    idx = 0
    for stock in data.index:
        STOCK_IDX[stock] = idx
        IDX_STOCK[idx] = stock
        idx += 1
    CORR = [[] for _ in SECTOR_IDX]
    NUM_STOCKS = [0 for _ in SECTOR_IDX]
    for sector in sectors:
        for i in range(2):
            if(sectors[sector][i] < min_pct): # also checks if sector has no percentage
                continue
            for j in range(len(IDX_STOCK)):
                CORR[SECTOR_IDX[sectors[sector][2+i]]].append(int(sector_corr[i][IDX_STOCK[j]][sector]*1000)+10000)
            NUM_STOCKS[SECTOR_IDX[sectors[sector][2+i]]] = int(max(1, sectors[sector][i]/2//max_pct))
    result, max_corr = weighted_max_bipartite_matching(len(SECTOR_IDX), len(IDX_STOCK), NUM_STOCKS, CORR)
    print(result)
    return portfolio

MAX_STOCKS = 24
MIN_PCT = 1/(2*MAX_STOCKS)
MAX_PCT = max_percentage(MIN_PCT)
data, history = get_tickers()
tsx_by_sector = tsx_sectors()
for sector in tsx_by_sector:
    history[sector] = tsx_by_sector[sector]
sp_by_sector = sp_sectors()
for sector in sp_by_sector:
    history[sector] = sp_by_sector[sector]
sector_corr = [calc(data, history, corr, 0), calc(data, history, corr, 1)]
create_portfolio(sector_corr, MIN_PCT, MAX_PCT)