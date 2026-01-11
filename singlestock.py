import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# ==========================================
# 1. 設定與資料準備
# ==========================================

# 設定要測試的股票代碼
stock_id = "3049.TW" 
# stock_id = "2330.TW"  # 台積電

# 檢查是否有本地檔案，若無則下載
csv_file = f"{stock_id}_data.csv"
if os.path.exists(csv_file):
    try:
        print(f"讀取本地檔案: {csv_file}")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"讀取本地檔案錯誤: {e}") 
        exit()
else:
    print(f"下載 {stock_id} 資料中...")
    df = yf.download(stock_id, period="3y", interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(csv_file)
    print("資料下載並儲存完成。")

# ==========================================
# 2. 核心類別與函式定義
# ==========================================

def NEWMA(series_close, series_volume, days):
    """
    計算新型成交量權重移動平均線 (New MA) 
    """
    days = int(max(1, days))
    pv = series_close * series_volume 
    sum_pv = pv.rolling(window=days).sum()
    sum_volume = series_volume.rolling(window=days).sum()
    return sum_pv / sum_volume

def get_fitness(close_price, volume, Ta, Tb, Tx, Ty):
    # ... (前面的邊界檢查不變) ...
    if any(p > 240 for p in [Ta, Tb, Tx, Ty]): return -9999
    if Ta >= Tb or Tx >= Ty: return -9999
    if any(p < 3 for p in [Ta, Tb, Tx, Ty]): return -9999

    ma_buy_s = NEWMA(close_price, volume, Ta)
    ma_buy_l = NEWMA(close_price, volume, Tb)
    ma_sell_s = NEWMA(close_price, volume, Tx)
    ma_sell_l = NEWMA(close_price, volume, Ty)

    buy_signal = (ma_buy_s > ma_buy_l).to_numpy()
    sell_signal = (ma_sell_s < ma_sell_l).to_numpy()
    prices = close_price.to_numpy()
    
    position = 0 
    buy_price = 0
    total_profit = 1.0 
    
    # 【修正】設定交易成本 (手續費 + 證交稅)
    # 台灣約 0.00585 (來回)，保守估計設 0.006 (0.6%)
    cost = 0.006 

    for i in range(1, len(prices)):
        if position == 0 and buy_signal[i] and not buy_signal[i-1]:
            position = 1
            buy_price = prices[i]
        elif position == 1 and sell_signal[i] and not sell_signal[i-1]:
            position = 0
            if buy_price > 0:
                # 扣除成本後的報酬率
                profit = ((prices[i] - buy_price) / buy_price) - cost
                total_profit *= (1 + profit)
            
    if position == 1 and buy_price > 0:
        profit = ((prices[-1] - buy_price) / buy_price) - cost
        total_profit *= (1 + profit)
        
    return total_profit - 1


# # === Debug 專用 Fitness Function ===
# def get_fitness(close_price, volume, Ta, Tb, Tx, Ty):
#     # ... (前面的參數檢查邏輯保持不變) ...
#     if any(p > 240 for p in [Ta, Tb, Tx, Ty]): return -9999
#     if Ta >= Tb or Tx >= Ty: return -9999
#     if any(p < 3 for p in [Ta, Tb, Tx, Ty]): return -9999

#     # 計算 MA
#     ma_buy_s = NEWMA(close_price, volume, Ta)
#     ma_buy_l = NEWMA(close_price, volume, Tb)
#     ma_sell_s = NEWMA(close_price, volume, Tx)
#     ma_sell_l = NEWMA(close_price, volume, Ty)

#     buy_signal = ((ma_buy_s > ma_buy_l) & (close_price > ma_buy_l)).to_numpy()
#     sell_signal = (ma_sell_s < ma_sell_l).to_numpy()
#     prices = close_price.to_numpy()
#     dates = close_price.index # 取得日期以便 Debug
    
#     position = 0 
#     buy_price = 0
#     total_profit = 1.0 
    
#     print(f"\n--- DEBUG: 測試參數 Buy({Ta},{Tb}) Sell({Tx},{Ty}) ---")
    
#     trade_count = 0
    
#     for i in range(1, len(prices)):
#         # 買進
#         if position == 0 and buy_signal[i] and not buy_signal[i-1]:
#             position = 1
#             buy_price = prices[i]
#             print(f"[{dates[i].date()}] 買進 @ {buy_price:.2f} (短MA:{ma_buy_s.iloc[i]:.2f} > 長MA:{ma_buy_l.iloc[i]:.2f})")
            
#         # 賣出
#         elif position == 1 and sell_signal[i] and not sell_signal[i-1]:
#             position = 0
#             if buy_price > 0:
#                 profit = (prices[i] - buy_price) / buy_price
#                 total_profit *= (1 + profit)
#                 trade_count += 1
#                 print(f"[{dates[i].date()}] 賣出 @ {prices[i]:.2f} | 單次損益: {profit*100:.2f}% | 目前淨值: {total_profit*100:.2f}%")
            
#     # 最後結算
#     if position == 1 and buy_price > 0:
#         profit = (prices[-1] - buy_price) / buy_price
#         total_profit *= (1 + profit)
#         print(f"[結算] 強制賣出 @ {prices[-1]:.2f} | 最終淨值: {total_profit*100:.2f}%")
        
#     if trade_count == 0 and position == 0:
#         print("沒有發生任何交易 (正確的空手策略)")
        
#     return total_profit - 1


class AE_QTS:
    """
    AE-QTS 引擎 (32 Qubits + 4 Parameters)
    """
    def __init__(self, num_qubits=32): 
        # 【修正】改為 32 Qubits (4個參數 x 8 bits)
        self.num_qubits = num_qubits
        self.qubits = np.full((num_qubits, 2), 1/np.sqrt(2))
        
    def observe(self):
        binary_solution = []
        for i in range(self.num_qubits):
            prob_one = self.qubits[i][1] ** 2
            if random.random() < prob_one:
                binary_solution.append(1)
            else:
                binary_solution.append(0)
        return binary_solution

    def decode(self, binary_solution):
        """
        解碼：將 32 bits 轉成 [Ta, Tb, Tx, Ty] 4 個參數
        """
        params = []
        # 改為 4 個參數
        for i in range(4): 
            chunk = binary_solution[i*8 : (i+1)*8]
            val = 0
            for bit in chunk:
                val = (val << 1) | bit
            
            # 【修正】這裡只做最小限制，最大值 240 交給 fitness function 判斷
            # 這樣符合論文 "超出範圍視為無效解" 的邏輯
            if val == 0: val = 1 
            
            params.append(val)
        return params

    def update(self, best_binary, worst_binary):
        delta_theta = 0.05 * np.pi 
        
        for i in range(self.num_qubits):
            best_bit = best_binary[i]
            worst_bit = worst_binary[i]
            
            theta = 0
            if best_bit == 0 and worst_bit == 1:
                theta = -delta_theta
            elif best_bit == 1 and worst_bit == 0:
                theta = delta_theta
            else:
                continue
                
            alpha = self.qubits[i][0]
            beta = self.qubits[i][1]
            
            new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
            new_beta  = alpha * np.sin(theta) + beta * np.cos(theta)
            
            self.qubits[i][0] = new_alpha
            self.qubits[i][1] = new_beta

# ==========================================
# 3. 主程式：防禦力點滿版
# ==========================================

print("開始 AE-QTS (32bit + Tabu + 強制停損 + 斜率濾網) 測試...")

# 計算 60MA (季線)
df['MA60'] = df['Close'].rolling(window=60).mean()

# 計算 60MA 的斜率 (今天 - 昨天)
# 斜率 > 0 代表均線向上，斜率 < 0 代表均線向下
df['MA60_Slope'] = df['MA60'].diff()

# 設定回測參數
training_window = 120
testing_days = 20
total_days = len(df)

initial_capital = 1_000_000
funds = initial_capital
position = 0 
stock_qty = 0
buy_price = 0 # 紀錄買入成本
history = []

current_idx = training_window

while current_idx < total_days - testing_days:
    
    curr_date = df.index[current_idx].date()
    
    # === A. 訓練階段 ===
    train_start = current_idx - training_window
    train_end = current_idx
    
    t_close = df['Close'].iloc[train_start:train_end].copy()
    t_volume = df['Volume'].iloc[train_start:train_end].copy()
    
    if len(t_close) < 10:
        current_idx += testing_days
        continue

    # --- AE-QTS 找參數 ---
    engine = AE_QTS(num_qubits=32) 
    best_params_global = [5, 20, 5, 20] 
    best_score_global = -9999
    
    generations = 8 
    population_size = 10
    tabu_list = []
    
    for gen in range(generations):
        pop = []
        attempts = 0
        while len(pop) < population_size and attempts < population_size*2:
            attempts += 1
            binary = engine.observe()
            if tuple(binary) in tabu_list: continue
            tabu_list.append(tuple(binary))
            if len(tabu_list) > 5: tabu_list.pop(0)
            
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
            
    if best_score_global < 0.08: # 建議改成 0.08 (8%)
        current_idx += testing_days
            
        # (選用) 印出為什麼空手，方便你觀察
        # print(f"{stock_id} {curr_date}: 訓練績效僅 {best_score_global:.2%} < 8%，放棄交易")
        continue

    best_Ta, best_Tb, best_Tx, best_Ty = best_params_global

    # === B. 測試階段 (Testing) ===
    test_end = min(current_idx + testing_days, total_days)

    ma_buy_s = NEWMA(df['Close'], df['Volume'], best_Ta)
    ma_buy_l = NEWMA(df['Close'], df['Volume'], best_Tb)
    ma_sell_s = NEWMA(df['Close'], df['Volume'], best_Tx)
    ma_sell_l = NEWMA(df['Close'], df['Volume'], best_Ty)

    for i in range(current_idx, test_end):
        date = df.index[i]
        price = df['Close'].iloc[i]
        ma60 = df['MA60'].iloc[i]
        ma60_slope = df['MA60_Slope'].iloc[i]
        
        if pd.isna(ma_buy_s.iloc[i]): continue

        # 訊號
        buy_sig = ma_buy_s.iloc[i] > ma_buy_l.iloc[i]
        sell_sig = ma_sell_s.iloc[i] < ma_sell_l.iloc[i]

        # ==========================================
        # 【濾網 2】: 趨勢濾網 (季線向上 + 股價在季線上)
        # 這是避免「假突破」的最強濾網
        # ==========================================
        is_uptrend = (price > ma60) and (ma60_slope > 0)

        # 1. 買進邏輯
        if buy_sig and position == 0 and is_uptrend:
            stock_qty = funds / price
            buy_price = price # 紀錄成本價
            funds = 0
            position = 1
            history.append(f"{date.date()} 買入 @ {price:.1f}")
            
        # 2. 賣出邏輯 (正常訊號)
        elif sell_sig and position == 1:
            funds = stock_qty * price
            stock_qty = 0
            position = 0
            history.append(f"{date.date()} 賣出 @ {price:.1f}")

        # ==========================================
        # 【濾網 3】: 強制停損 (Hard Stop Loss)
        # 如果虧損超過 7% (0.93)，無條件砍單，不要凹單！
        # ==========================================
        elif position == 1 and price < (buy_price * 0.93):
            funds = stock_qty * price
            stock_qty = 0
            position = 0
            history.append(f"{date.date()} 停損(-7%) @ {price:.1f}")

    # === C. 滑動 ===
    current_idx += testing_days
    current_asset = funds + (stock_qty * df['Close'].iloc[current_idx-1] if position else 0)
    print(f"進度: {curr_date} -> 下一站 | 資產: {int(current_asset)}")

# ==========================================
# 4. 結算與輸出
# ==========================================

final_price = df['Close'].iloc[-1]
if position == 1:
    funds = stock_qty * final_price

total_return = (funds - initial_capital) / initial_capital * 100

print("\n" + "="*30)
print(f"【AE-QTS (32bit+Tabu)】最終結果")
print(f"測試股票: {stock_id}")
print(f"初始資金: {initial_capital}")
print(f"最終資產: {int(funds)}")
print(f"總報酬率: {total_return:.2f}%")
print("="*30)

print("最近 5 筆交易紀錄:")
for i in history[-5:]:
    print(i)