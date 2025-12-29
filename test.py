import yfinance as yf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt


#這測多個的收益

# --- NEWMA 計算函式 ---
def NEWMA(series_close, series_volume, days):
    days = int(max(1, days))
    pv = series_close * series_volume
    sum_pv = pv.rolling(window=days).sum()
    sum_v = series_volume.rolling(window=days).sum()
    return sum_pv / sum_v

# --- 適應值函數 ---
def get_fitness(close_price, volume, short_ma_days, long_ma_days):
    if short_ma_days >= long_ma_days: return -9999
    if short_ma_days < 3 or long_ma_days > 240: return -9999

    ma_short = NEWMA(close_price, volume, short_ma_days)
    ma_long = NEWMA(close_price, volume, long_ma_days)

    sig = (ma_short > ma_long).astype(int)
    pct_change = close_price.pct_change().fillna(0)
    strategy_ret = sig.shift(1) * pct_change
    strategy_ret = strategy_ret.fillna(0)
    
    if len(strategy_ret) == 0: return -9999
    cum_ret = (1 + strategy_ret).cumprod()
    if len(cum_ret) == 0: return -9999
        
    return cum_ret.iloc[-1] - 1

# --- AE-QTS  ---
class AE_QTS_Engine:
    def __init__(self, num_qubits=16):
        self.num_qubits = num_qubits
        self.qubits = np.full((num_qubits, 2), 1/np.sqrt(2))
        
    def observe(self):
        binary_solution = []
        for i in range(self.num_qubits):
            prob_one = self.qubits[i][1] ** 2
            if random.random() < prob_one: binary_solution.append(1)
            else: binary_solution.append(0)
        return binary_solution

    def decode(self, binary_solution):
        params = []
        for i in range(2): 
            chunk = binary_solution[i*8 : (i+1)*8]
            val = 0
            for bit in chunk: val = (val << 1) | bit
            if val < 3: val = 3
            if val > 240: val = 240
            params.append(val)
        return params

    def update(self, best_binary, worst_binary):
        delta_theta = 0.05 * np.pi 
        for i in range(self.num_qubits):
            best_bit = best_binary[i]
            worst_bit = worst_binary[i]
            theta = 0
            if best_bit == 0 and worst_bit == 1: theta = -delta_theta
            elif best_bit == 1 and worst_bit == 0: theta = delta_theta
            else: continue
            alpha = self.qubits[i][0]
            beta = self.qubits[i][1]
            self.qubits[i][0] = alpha * np.cos(theta) - beta * np.sin(theta)
            self.qubits[i][1] = alpha * np.sin(theta) + beta * np.cos(theta)

# ==========================================
# 2. 批次執行主程式
# ==========================================

stock_list = ["2330.TW", "2317.TW", "2454.TW"] 
results_summary = []

# 初始化總資金池
total_initial_capital = 0
total_final_capital = 0

print(f"開始執行多股票 AE-QTS 回測 (資金各 100 萬)...")

for stock_id in stock_list:
    print(f"\n>>> 正在測試: {stock_id} ...")

    # A. 資料處理
    csv_file = f"{stock_id}_data.csv"
    if os.path.exists(csv_file):
        try: os.remove(csv_file) 
        except: pass
            
    try:
        df = yf.download(stock_id, period="3y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['Close', 'Volume'], inplace=True)
        
        if len(df) < 200:
            print(f"資料不足，跳過 {stock_id}")
            continue
            
    except Exception as e:
        print(f"下載失敗 {stock_id}: {e}")
        continue

    # B. AE-QTS 動態回測
    training_window = 120
    testing_days = 20
    total_days = len(df)
    
    initial_capital = 1_000_000
    funds = initial_capital
    position = 0
    stock_qty = 0
    current_idx = training_window
    
    while current_idx < total_days - testing_days:
        # 準備訓練資料
        train_start = current_idx - training_window
        t_close = df['Close'].iloc[train_start:current_idx].copy()
        t_volume = df['Volume'].iloc[train_start:current_idx].copy()
        
        if len(t_close) < 10:
            current_idx += testing_days
            continue

        # AE-QTS 找參數
        engine = AE_QTS_Engine(num_qubits=16)
        best_params = [5, 20]
        best_score = -9999
        generations = 5   
        population_size = 8
        
        for gen in range(generations):
            pop = []
            for _ in range(population_size):
                binary = engine.observe()
                params = engine.decode(binary)
                score = get_fitness(t_close, t_volume, params[0], params[1])
                pop.append({'binary': binary, 'params': params, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            pop.sort(key=lambda x: x['score'], reverse=True)
            if pop[0]['score'] > pop[-1]['score']:
                engine.update(pop[0]['binary'], pop[-1]['binary'])
        
        best_s, best_l = best_params

        # 測試期交易
        test_end = min(current_idx + testing_days, total_days)
        ma_s_full = NEWMA(df['Close'], df['Volume'], best_s)
        ma_l_full = NEWMA(df['Close'], df['Volume'], best_l)
        
        for i in range(current_idx, test_end):
            price = df['Close'].iloc[i]
            ms = ma_s_full.iloc[i]
            ml = ma_l_full.iloc[i]
            if pd.isna(ms) or pd.isna(ml): continue
            
            if ms > ml and position == 0:
                stock_qty = funds / price
                funds = 0
                position = 1
            elif ms < ml and position == 1:
                funds = stock_qty * price
                stock_qty = 0
                position = 0
        
        current_idx += testing_days

    # C. 單一股票結算
    final_price = df['Close'].iloc[-1]
    if position == 1:
        funds = stock_qty * final_price
    
    total_ret = (funds - initial_capital) / initial_capital * 100
    print(f"  -> {stock_id} 最終資產: {int(funds)} (報酬率: {total_ret:.2f}%)")
    
    # 加入總資金池
    total_initial_capital += initial_capital
    total_final_capital += funds

    results_summary.append({
        'Stock': stock_id,
        'Final Capital': int(funds),
        'Return': total_ret
    })

# ==========================================
# 3. 投資組合總結報告
# ==========================================
print("\n" + "="*40)
print(f"【投資組合總損益報告】")
print(f"{'='*40}")

grand_total_profit = total_final_capital - total_initial_capital
portfolio_return = (grand_total_profit / total_initial_capital) * 100

print(f"投入本金: {total_initial_capital:,}")
print(f"最終資產: {int(total_final_capital):,}")
print(f"總獲利金額: {int(grand_total_profit):,} 元")
print(f"總報酬率: {portfolio_return:.2f}%")
print("-" * 40)
print("個別表現:")
for res in results_summary:
    print(f"{res['Stock']}: {res['Return']:.2f}% (獲利 {res['Final Capital'] - 1000000:,})")
