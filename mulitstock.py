import yfinance as yf
import pandas as pd
import os
import numpy as np
import random

# ==========================================
# 1. 核心邏輯優化版 (Numpy + 預計算)
# ==========================================

def precompute_all_newma(close_series, volume_series, max_period=240):
    """
    【加速關鍵 1】一次性預計算所有天期的 MA
    回傳一個矩陣，Shape = (241, Data長度)
    Row i 代表 i 日的 NewMA 數列
    """
    n = len(close_series)
    # 初始化一個大矩陣來存所有 MA
    all_mas = np.zeros((max_period + 1, n))
    
    # 轉換成 Numpy 運算比較快
    p = close_series.to_numpy()
    v = volume_series.to_numpy()
    pv = p * v
    
    # 利用 Pandas 的快速 rolling 預算
    # 雖然這裡用 Pandas，但因為只跑一次 (1~240 loop)，所以沒關係
    series_pv = pd.Series(pv)
    series_v = pd.Series(v)
    
    print("正在預計算技術指標矩陣 (1~240)...")
    for d in range(1, max_period + 1):
        # 計算 d 日 NewMA
        sum_pv = series_pv.rolling(d).sum().to_numpy()
        sum_v = series_v.rolling(d).sum().to_numpy()
        
        # 避免除以 0，使用舊值填補 (bfill邏輯的 numpy 實作較繁瑣，這裡用簡化版)
        with np.errstate(divide='ignore', invalid='ignore'):
            ma_d = sum_pv / sum_v
        
        # 簡單的填補 NaN (用前一個有效值或 0)
        # 這裡為了速度，直接將 NaN 填為 0 (反正前幾天也不會交易)
        ma_d = np.nan_to_num(ma_d)
        
        all_mas[d] = ma_d
        
    return all_mas

def get_fitness_fast(ma_matrix, start_idx, end_idx, prices_segment, Ta, Tb, Tx, Ty, memo):
    """
    【加速關鍵 2 & 3】查表法 + 快取
    """
    # 1. 檢查快取
    param_key = (Ta, Tb, Tx, Ty)
    if param_key in memo:
        return memo[param_key]

    # 2. 邊界檢查
    if any(p > 240 for p in [Ta, Tb, Tx, Ty]) or any(p < 2 for p in [Ta, Tb, Tx, Ty]):
        return -9999
    if Ta >= Tb or Tx >= Ty: 
        return -9999

    # 3. 從預計算矩陣直接切片 (極速)
    # 注意：ma_matrix 是全域的，我們只取當前視窗 [start_idx : end_idx]
    ma_buy_s = ma_matrix[Ta, start_idx:end_idx]
    ma_buy_l = ma_matrix[Tb, start_idx:end_idx]
    ma_sell_s = ma_matrix[Tx, start_idx:end_idx]
    ma_sell_l = ma_matrix[Ty, start_idx:end_idx]

    # 4. Numpy 訊號計算
    buy_signal = (ma_buy_s > ma_buy_l)
    sell_signal = (ma_sell_s < ma_sell_l)
    
    # 5. 模擬交易 (純數字迴圈)
    position = 0 
    cash = 1.0
    stock_qty = 0
    cost = 0.004425
    
    # 使用迴圈跑這 120 天 (Python 對於小迴圈的 overhead 尚可接受，若要更快可用 Numba，但這裡先用純 Py)
    # 為了加速，我們找出訊號轉折點即可
    
    n_days = len(prices_segment)
    for i in range(1, n_days):
        # 買進
        if position == 0:
            # 黃金交叉: 今天 True, 昨天 False
            if buy_signal[i] and not buy_signal[i-1]:
                position = 1
                stock_qty = cash / prices_segment[i]
                cash = 0
        # 賣出
        elif position == 1:
            # 死亡交叉
            if sell_signal[i] and not sell_signal[i-1]:
                position = 0
                cash = stock_qty * prices_segment[i] * (1 - cost)
                stock_qty = 0
            
    if position == 1:
        cash = stock_qty * prices_segment[-1] * (1 - cost)
        
    total_return = cash - 1.0
    
    # 寫入快取
    memo[param_key] = total_return
    return total_return

class AE_QTS_Fast:
    def __init__(self, num_qubits=32): 
        self.num_qubits = num_qubits
        self.qubits = np.full((num_qubits, 2), 1/np.sqrt(2))
        
    def observe(self):
        # 向量化隨機抽樣
        probs = self.qubits[:, 1] ** 2
        # 產生 0 ~ 1 隨機數
        rands = np.random.rand(self.num_qubits)
        # 如果 rand < prob_one (1的機率)，則為 1
        binary_solution = (rands < probs).astype(int)
        return binary_solution

    def decode(self, binary_solution):
        # 二進制轉十進制 (位元運算)
        params = []
        for i in range(4): 
            chunk = binary_solution[i*8 : (i+1)*8]
            # 快速位元轉換
            val = 0
            for bit in chunk:
                val = (val << 1) | bit
            if val == 0: val = 1
            params.append(val)
        return params

    def update(self, best_binary, worst_binary):
        delta_theta = 0.05 * np.pi 
        # 向量化更新
        diff = best_binary - worst_binary # 1-0=1(Best是1), 0-1=-1(Best是0), 0-0=0
        
        # 建立旋轉角度陣列
        thetas = np.zeros(self.num_qubits)
        # Best=0, Worst=1 -> diff=-1 -> theta = -delta
        thetas[diff == -1] = -delta_theta
        # Best=1, Worst=0 -> diff=1  -> theta = delta
        thetas[diff == 1] = delta_theta
        
        c = np.cos(thetas)
        s = np.sin(thetas)
        
        alpha = self.qubits[:, 0]
        beta = self.qubits[:, 1]
        
        # 矩陣旋轉
        self.qubits[:, 0] = alpha * c - beta * s
        self.qubits[:, 1] = alpha * s + beta * c

# ==========================================
# 2. 主程式：極速版回測
# ==========================================

stock_list = ["2330.TW", "2317.TW", "2454.TW", "3049.TW", "2543.TW"] 

results_summary = []
total_initial_capital = 0
total_final_capital = 0

print(f"【 AE-QTS 回測測試】")

for stock_id in stock_list:
    print(f"\nProcessing: {stock_id} ...")
    
    # --- 下載資料 ---
    try:
        df = yf.download(stock_id, period="3y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['Close', 'Volume'], inplace=True)
        if len(df) < 200: 
            print("  -> 資料不足跳過")
            continue
    except Exception as e:
        print(f"  -> 下載錯誤: {e}")
        continue

    # --- 【步驟 1】 全域預計算 (只算一次) ---
    # 這會產生一個 shape 為 (241, len(df)) 的大矩陣
    ma_matrix = precompute_all_newma(df['Close'], df['Volume'])
    close_prices = df['Close'].to_numpy() # 轉成 numpy 加速讀取
    
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
        # 定義當前視窗的索引範圍
        train_start = current_idx - training_window
        train_end = current_idx
        
        # 切片出訓練用的價格 (Numpy Array)
        train_prices_seg = close_prices[train_start:train_end]
        
        # 初始化 AE-QTS
        engine = AE_QTS_Fast(num_qubits=32)
        best_params_local = [5, 20, 5, 20]
        best_score_local = -9999
        
        # 初始化快取 (每個視窗獨立快取，因為價格數據變了)
        memo = {}
        
        generations = 50     
        population_size = 20 
        
        # --- 演化迴圈 ---
        for gen in range(generations):
            pop = []
            
            # 這裡可以進一步平行化，但為了簡單起見維持單線程 (已經夠快了)
            for _ in range(population_size):
                binary = engine.observe()
                params = engine.decode(binary)
                
                # 呼叫加速版的 Fitness Function
                score = get_fitness_fast(
                    ma_matrix, train_start, train_end, train_prices_seg, 
                    *params, memo
                )
                
                pop.append({'binary': binary, 'params': params, 'score': score})
            
            pop.sort(key=lambda x: x['score'], reverse=True)
            
            if pop[0]['score'] > best_score_local:
                best_score_local = pop[0]['score']
                best_params_local = pop[0]['params']
            
            if pop[0]['score'] > pop[-1]['score']:
                engine.update(pop[0]['binary'], pop[-1]['binary'])

        # --- 測試期交易 ---
        best_Ta, best_Tb, best_Tx, best_Ty = best_params_local
        test_end = min(current_idx + testing_days, total_days)
        
        # 直接從預計算矩陣取值
        ma_buy_s = ma_matrix[best_Ta, current_idx:test_end]
        ma_buy_l = ma_matrix[best_Tb, current_idx:test_end]
        ma_sell_s = ma_matrix[best_Tx, current_idx:test_end]
        ma_sell_l = ma_matrix[best_Ty, current_idx:test_end]
        
        test_prices = close_prices[current_idx:test_end]
        
        # 前一日的指標值 (用於判斷第一天的交叉)
        # 注意邊界：current_idx - 1
        prev_ma_buy_s = ma_matrix[best_Ta, current_idx-1]
        prev_ma_buy_l = ma_matrix[best_Tb, current_idx-1]
        prev_ma_sell_s = ma_matrix[best_Tx, current_idx-1]
        prev_ma_sell_l = ma_matrix[best_Ty, current_idx-1]
        
        # 測試期迴圈
        for i in range(len(test_prices)):
            price = test_prices[i]
            
            # 當前指標
            curr_bs = ma_buy_s[i]
            curr_bl = ma_buy_l[i]
            curr_ss = ma_sell_s[i]
            curr_sl = ma_sell_l[i]
            
            # 上一期指標 (如果是迴圈第一天，用視窗外的昨收，否則用迴圈內的昨收)
            last_bs = ma_buy_s[i-1] if i > 0 else prev_ma_buy_s
            last_bl = ma_buy_l[i-1] if i > 0 else prev_ma_buy_l
            last_ss = ma_sell_s[i-1] if i > 0 else prev_ma_sell_s
            last_sl = ma_sell_l[i-1] if i > 0 else prev_ma_sell_l
            
            # 買進訊號: 黃金交叉
            buy_sig = (curr_bs > curr_bl) and (last_bs <= last_bl)
            
            # 賣出訊號: 死亡交叉
            sell_sig = (curr_ss < curr_sl) and (last_ss >= last_sl)
            
            if position == 0 and buy_sig:
                stock_qty = funds / price
                funds = 0
                position = 1
            
            elif position == 1 and sell_sig:
                funds = stock_qty * price * (1 - 0.004425)
                stock_qty = 0
                position = 0
        
        current_idx += testing_days

    # 結算
    final_price = close_prices[-1]
    if position == 1:
        funds = stock_qty * final_price * (1 - 0.004425)
    
    ret = (funds - initial_capital) / initial_capital * 100
    
    total_initial_capital += initial_capital
    total_final_capital += funds
    results_summary.append({
        'Stock': stock_id,
        'Final Capital': int(funds),
        'Return': ret
    })

# ==========================================
# 3. 輸出結果
# ==========================================
print("\n" + "="*40)
print(f"【 AE-QTS 回測報告】")
print("="*40)
print(f"投入本金: {total_initial_capital:,}")
print(f"最終資產: {int(total_final_capital):,}")
grand_profit = total_final_capital - total_initial_capital
pf_return = (grand_profit / total_initial_capital * 100) if total_initial_capital > 0 else 0
print(f"總獲利: {int(grand_profit):,} 元")
print(f"總報酬率: {pf_return:.2f}%")
print("-" * 40)
for res in results_summary:
    profit = res['Final Capital'] - 1000000
    print(f"{res['Stock']}: {res['Return']:.2f}% (損益 {int(profit):,})")
print("="*40)