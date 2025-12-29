import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# ==========================================
# 1. 核心類別與函式定義 (AE-QTS 32bit + Tabu)
# ==========================================

def NEWMA(series_close, series_volume, days):
    """ 計算新型成交量權重移動平均線 (New MA) """
    days = int(max(1, days))
    pv = series_close * series_volume 
    sum_pv = pv.rolling(window=days).sum()
    sum_volume = series_volume.rolling(window=days).sum()
    return sum_pv / sum_volume

def get_fitness(close_price, volume, Ta, Tb, Tx, Ty):
    """
    適應值函數 (4參數: 買進 Ta, Tb / 賣出 Tx, Ty)
    """
    # 邊界與合理性檢查
    if any(p > 240 for p in [Ta, Tb, Tx, Ty]): return -9999
    if Ta >= Tb or Tx >= Ty: return -9999
    if any(p < 3 for p in [Ta, Tb, Tx, Ty]): return -9999

    # 計算四條 MA
    ma_buy_s = NEWMA(close_price, volume, Ta)
    ma_buy_l = NEWMA(close_price, volume, Tb)
    ma_sell_s = NEWMA(close_price, volume, Tx)
    ma_sell_l = NEWMA(close_price, volume, Ty)

    # 產生訊號
    buy_signal = (ma_buy_s > ma_buy_l).to_numpy()
    sell_signal = (ma_sell_s < ma_sell_l).to_numpy()
    prices = close_price.to_numpy()
    
    # 快速回測
    position = 0 
    buy_price = 0
    total_profit = 1.0 
    
    for i in range(1, len(prices)):
        # 黃金交叉買進
        if position == 0 and buy_signal[i] and not buy_signal[i-1]:
            position = 1
            buy_price = prices[i]
        # 死亡交叉賣出
        elif position == 1 and sell_signal[i] and not sell_signal[i-1]:
            position = 0
            if buy_price > 0:
                profit = (prices[i] - buy_price) / buy_price
                total_profit *= (1 + profit)
            
    # 強制結算
    if position == 1 and buy_price > 0:
        profit = (prices[-1] - buy_price) / buy_price
        total_profit *= (1 + profit)
        
    return total_profit - 1

class AE_QTS:
    """ AE-QTS 引擎 (32 Qubits) """
    def __init__(self, num_qubits=32): 
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
        for i in range(4): # 解碼出 4 個參數
            chunk = binary_solution[i*8 : (i+1)*8]
            val = 0
            for bit in chunk: val = (val << 1) | bit
            if val == 0: val = 1 
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
# 2. 多股票批次執行主程式
# ==========================================

# 設定要測試的股票清單 (建議包含不同股性的股票)
# 例如: 台積電(趨勢)、鴻海(黏著)、聯發科(波動)、精金(震盪)、鼎炫(隨機)
stock_list = ["2330.TW", "2317.TW", "2454.TW", "3049.TW", "8499.TW"]

results_summary = []
total_initial_capital = 0
total_final_capital = 0

print(f"開始執行多股票 AE-QTS (32bit+Tabu) 測試...")
print(f"測試名單: {stock_list}")

for stock_id in stock_list:
    print(f"\n{'-'*20}")
    print(f">>> 正在測試: {stock_id} ...")
    
    # --- 下載資料 ---
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
            print("資料不足跳過")
            continue
    except Exception as e:
        print(f"下載錯誤: {e}")
        continue

    # --- 參數設定 ---
    training_window = 120
    testing_days = 20
    total_days = len(df)
    
    initial_capital = 1_000_000
    funds = initial_capital
    position = 0
    stock_qty = 0
    current_idx = training_window
    
    # --- 滑動視窗 ---
    while current_idx < total_days - testing_days:
        train_start = current_idx - training_window
        t_close = df['Close'].iloc[train_start:current_idx].copy()
        t_volume = df['Volume'].iloc[train_start:current_idx].copy()
        
        if len(t_close) < 10:
            current_idx += testing_days
            continue

        # AE-QTS 初始化
        engine = AE_QTS(num_qubits=32)
        best_params_global = [5, 20, 5, 20]
        best_score_global = -9999
        
        generations = 8    # 為了批次跑快一點，代數設 8
        population_size = 10
        tabu_list = []
        tabu_size = 5

        # 演化
        for gen in range(generations):
            pop = []
            attempts = 0
            while len(pop) < population_size and attempts < population_size * 2:
                attempts += 1
                binary = engine.observe()
                if tuple(binary) in tabu_list: continue # Tabu check
                
                tabu_list.append(tuple(binary))
                if len(tabu_list) > tabu_size: tabu_list.pop(0)
                
                params = engine.decode(binary)
                score = get_fitness(t_close, t_volume, *params)
                pop.append({'binary': binary, 'params': params, 'score': score})
                
                if score > best_score_global:
                    best_score_global = score
                    best_params_global = params
            
            if not pop: continue
            pop.sort(key=lambda x: x['score'], reverse=True)
            if pop[0]['score'] > pop[-1]['score']:
                engine.update(pop[0]['binary'], pop[-1]['binary'])

        # 測試期交易
        best_Ta, best_Tb, best_Tx, best_Ty = best_params_global
        test_end = min(current_idx + testing_days, total_days)
        
        # 計算 MA
        ma_buy_s = NEWMA(df['Close'], df['Volume'], best_Ta)
        ma_buy_l = NEWMA(df['Close'], df['Volume'], best_Tb)
        ma_sell_s = NEWMA(df['Close'], df['Volume'], best_Tx)
        ma_sell_l = NEWMA(df['Close'], df['Volume'], best_Ty)
        
        for i in range(current_idx, test_end):
            price = df['Close'].iloc[i]
            
            # 買進: Ta > Tb
            buy_sig = ma_buy_s.iloc[i] > ma_buy_l.iloc[i]
            # 賣出: Tx < Ty
            sell_sig = ma_sell_s.iloc[i] < ma_sell_l.iloc[i]
            
            if pd.isna(ma_buy_s.iloc[i]): continue

            if buy_sig and position == 0:
                stock_qty = funds / price
                funds = 0
                position = 1
            elif sell_sig and position == 1:
                funds = stock_qty * price
                stock_qty = 0
                position = 0
        
        current_idx += testing_days
        # print(f"進度: {current_idx}/{total_days}...")

    # 單一股票結算
    final_price = df['Close'].iloc[-1]
    if position == 1:
        funds = stock_qty * final_price
    
    # 紀錄結果
    ret = (funds - initial_capital) / initial_capital * 100
    print(f"  -> {stock_id} 最終資產: {int(funds)} (報酬率: {ret:.2f}%)")
    
    total_initial_capital += initial_capital
    total_final_capital += funds
    results_summary.append({
        'Stock': stock_id,
        'Final Capital': funds,
        'Return': ret
    })

# ==========================================
# 3. 投資組合總結算 (加入你要求的這段)
# ==========================================

# 先計算總獲利與總報酬率
grand_total_profit = total_final_capital - total_initial_capital
portfolio_return = 0
if total_initial_capital > 0:
    portfolio_return = (grand_total_profit / total_initial_capital) * 100

print("\n" + "="*40)
print(f"【投資組合總損益報告】")
print("="*40)
print(f"投入本金: {total_initial_capital:,}")
print(f"最終資產: {int(total_final_capital):,}")
print(f"總獲利金額: {int(grand_total_profit):,} 元")
print(f"總報酬率: {portfolio_return:.2f}%")
print("-" * 40)
print("個別表現:")
for res in results_summary:
    # 計算單檔獲利金額
    profit = res['Final Capital'] - 1000000
    print(f"{res['Stock']}: {res['Return']:.2f}% (獲利 {int(profit):,})")
print("="*40)