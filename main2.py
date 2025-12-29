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
stock_id = "2330.TW"

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
    """
    適應值函數 (升級版：支援買賣分離 4 參數)
    輸入：股價、成交量、買進參數(Ta, Tb)、賣出參數(Tx, Ty)
    """
    # 1. 參數合理性與邊界檢查 (論文設定)
    # 若參數大於 240，視為無效解 (給予極低分)
    if any(p > 240 for p in [Ta, Tb, Tx, Ty]): return -9999
    # 確保短天期 < 長天期
    if Ta >= Tb or Tx >= Ty: return -9999
    # 確保參數至少為 3
    if any(p < 3 for p in [Ta, Tb, Tx, Ty]): return -9999

    # 2. 計算四條 MA 線
    # 買進參考線
    ma_buy_s = NEWMA(close_price, volume, Ta)
    ma_buy_l = NEWMA(close_price, volume, Tb)
    # 賣出參考線
    ma_sell_s = NEWMA(close_price, volume, Tx)
    ma_sell_l = NEWMA(close_price, volume, Ty)

    # 3. 產生訊號
    # 買進訊號: 短 > 長
    buy_signal = (ma_buy_s > ma_buy_l).to_numpy()
    # 賣出訊號: 短 < 長
    sell_signal = (ma_sell_s < ma_sell_l).to_numpy()
    
    prices = close_price.to_numpy()
    
    # 4. 快速回測 (模擬持有狀態)
    position = 0 # 0: 空手, 1: 持有
    buy_price = 0
    total_profit = 1.0 # 複利計算初始值
    
    # 使用迴圈精確模擬買賣狀態 (比向量化更適合非對稱策略)
    for i in range(1, len(prices)):
        # 如果空手 且 出現買進訊號 (由無訊號轉為有訊號，即黃金交叉)
        if position == 0 and buy_signal[i] and not buy_signal[i-1]:
            position = 1
            buy_price = prices[i]
            
        # 如果持有 且 出現賣出訊號 (由無訊號轉為有訊號，即死亡交叉)
        elif position == 1 and sell_signal[i] and not sell_signal[i-1]:
            position = 0
            # 避免除以 0
            if buy_price > 0:
                profit = (prices[i] - buy_price) / buy_price
                total_profit *= (1 + profit)
            
    # 如果最後還持有，以最後一天價格結算
    if position == 1 and buy_price > 0:
        profit = (prices[-1] - buy_price) / buy_price
        total_profit *= (1 + profit)
        
    return total_profit - 1 # 回傳總報酬率

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
# 3. 主程式：滑動視窗動態回測
# ==========================================

print("開始 AE-QTS (32bit + Tabu) 動態滑動回測...")

# 設定回測參數
training_window = 120
testing_days = 20
total_days = len(df)

initial_capital = 1_000_000
funds = initial_capital
position = 0 
stock_qty = 0
history = []

current_idx = training_window

while current_idx < total_days - testing_days:
    
    curr_date = df.index[current_idx].date()
    
    # === A. 訓練階段 (Training) ===
    train_start = current_idx - training_window
    train_end = current_idx
    
    t_close = df['Close'].iloc[train_start:train_end].copy()
    t_volume = df['Volume'].iloc[train_start:train_end].copy()
    
    if len(t_close) < 10:
        current_idx += testing_days
        continue

    # --- AE-QTS 演算法 (32bit) --- 
    
    # 1. 建立 32 Qubits 引擎
    engine = AE_QTS(num_qubits=32) 
    
    best_params_global = [5, 20, 5, 20] # 預設 [Ta, Tb, Tx, Ty]
    best_score_global = -9999
    
    generations = 10 
    population_size = 10
    
    # 【新增】禁忌表 (Tabu List)
    tabu_list = []
    tabu_size = 5 # 記憶最近 5 個走過的解

    # 開始演化
    for gen in range(generations):
        population = [] 
        
        # 嘗試產生 population_size 個不重複的解
        attempts = 0
        while len(population) < population_size and attempts < population_size * 2:
            attempts += 1
            
            # 2. 觀測
            binary_code = engine.observe()
            
            # 【新增】禁忌搜尋檢查
            # 將 list 轉成 tuple 才能比對
            binary_tuple = tuple(binary_code)
            
            if binary_tuple in tabu_list:
                continue # 如果在禁忌表中，跳過這個解 (重新觀測)
            
            # 加入禁忌表
            tabu_list.append(binary_tuple)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0) # 移除最舊的
            
            # 解碼與評估
            params = engine.decode(binary_code) 
            # 這裡 params 有 4 個值: Ta, Tb, Tx, Ty
            score = get_fitness(t_close, t_volume, params[0], params[1], params[2], params[3])
            
            population.append({
                'binary': binary_code,
                'params': params,
                'score': score
            })
            
            if score > best_score_global:
                best_score_global = score
                best_params_global = params
        
        # 如果這一代沒有產生有效解 (很少見)，就跳過
        if not population: continue

        # 3. 排序
        population.sort(key=lambda x: x['score'], reverse=True)
        
        best_sol = population[0] 
        worst_sol = population[-1]
        
        # 4. 量子更新
        if best_sol['score'] > worst_sol['score']:
            engine.update(best_sol['binary'], worst_sol['binary'])
            
    # 訓練結束，取得最佳 4 參數
    best_Ta, best_Tb, best_Tx, best_Ty = best_params_global
    
    # === B. 測試階段 (Testing) ===
    test_end = current_idx + testing_days
    if test_end > total_days: test_end = total_days

    # 準備四條 MA 線
    ma_buy_s = NEWMA(df['Close'], df['Volume'], best_Ta)
    ma_buy_l = NEWMA(df['Close'], df['Volume'], best_Tb)
    ma_sell_s = NEWMA(df['Close'], df['Volume'], best_Tx)
    ma_sell_l = NEWMA(df['Close'], df['Volume'], best_Ty)

    for i in range(current_idx, test_end):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        # 買進訊號判定: Ta > Tb
        buy_cond = ma_buy_s.iloc[i] > ma_buy_l_full.iloc[i] if 'ma_buy_l_full' in locals() else ma_buy_s.iloc[i] > ma_buy_l.iloc[i]
        # 修正變數名稱避免混淆，直接用上面算好的
        buy_cond = ma_buy_s.iloc[i] > ma_buy_l.iloc[i]
        
        # 賣出訊號判定: Tx < Ty
        sell_cond = ma_sell_s.iloc[i] < ma_sell_l.iloc[i]
        
        if pd.isna(ma_buy_s.iloc[i]) or pd.isna(ma_buy_l.iloc[i]): continue
        
        # 交易執行
        if buy_cond and position == 0: 
            stock_qty = funds / price
            funds = 0
            position = 1
            history.append(f"{date.date()} 買入({best_Ta},{best_Tb}) @ {price:.1f}")
            
        elif sell_cond and position == 1: 
            funds = stock_qty * price
            stock_qty = 0
            position = 0
            history.append(f"{date.date()} 賣出({best_Tx},{best_Ty}) @ {price:.1f}")

    # === C. 滑動視窗 ===
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
for h in history[-5:]:
    print(h)