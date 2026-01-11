import yfinance as yf
import pandas as pd
import os
import numpy as np
import random

# ==========================================
# 1. 核心邏輯：布林通道數據預計算
# ==========================================

def precompute_bollinger_data(close_series, max_period=80):
    """
    預計算所有可能週期的 MA 和 STD
    """
    n = len(close_series)
    all_mas = np.zeros((max_period + 1, n))
    all_stds = np.zeros((max_period + 1, n))
    
    p = close_series.to_numpy()
    series_p = pd.Series(p)
    
    # 只計算 5日 ~ 80日
    for d in range(5, max_period + 1):
        ma_d = series_p.rolling(d).mean().to_numpy()
        std_d = series_p.rolling(d).std().to_numpy()
        
        all_mas[d] = np.nan_to_num(ma_d)
        all_stds[d] = np.nan_to_num(std_d)
        
    return all_mas, all_stds, p

# ==========================================
# 2. 適應值函數 (%B 策略 + MA200 濾網)
# ==========================================

def get_fitness_bollinger_trend(ma_matrix, std_matrix, prices_segment, ma200_segment, start_idx, end_idx, 
                                p1, p2, p3, p4, memo):
    """
    參數:
    p1 -> N (週期)
    p2 -> K (寬度)
    p3 -> Buy_Th (買進)
    p4 -> Sell_Th (賣出)
    ma200_segment -> 對應區間的 MA200 數據 (用於趨勢過濾)
    """
    # 1. 參數解碼
    N = (p1 % 60) + 5            
    K = (p2 % 30 + 10) / 10.0    
    buy_th = (p3 / 100.0) - 0.5
    sell_th = (p4 / 100.0) - 0.5
    
    param_key = (N, K, buy_th, sell_th)
    if param_key in memo: return memo[param_key]

    if sell_th <= buy_th: return -9999
    
    # 2. 快速計算 %B
    ma = ma_matrix[N, start_idx:end_idx]
    std = std_matrix[N, start_idx:end_idx]
    p_seg = prices_segment
    # 注意：ma200_segment 已經是切片好的數據，長度應與 p_seg 相同
    ma200_seg = ma200_segment 
    
    lower = ma - (K * std)
    width = 2 * K * std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pb = (p_seg - lower) / width
    pb = np.nan_to_num(pb) 
    
    # 3. 交易回測
    cash = 1.0
    position = 0
    stock_qty = 0
    cost = 0.004425
    
    max_dd = 0.0
    peak_equity = 1.0
    
    n_days = len(p_seg)
    
    for i in range(1, n_days):
        current_pb = pb[i]
        price = p_seg[i]
        current_ma200 = ma200_seg[i]
        
        if position == 0:
            # 買進條件: 
            # 1. %B 低於買進門檻 
            # 2. 【新增】價格必須高於 MA200 (趨勢向上)
            if current_pb < buy_th and price > current_ma200:
                position = 1
                stock_qty = cash / price
                cash = 0
        
        elif position == 1:
            # 賣出條件: %B 高於賣出門檻
            if current_pb > sell_th:
                position = 0
                cash = stock_qty * price * (1 - cost)
                stock_qty = 0
                
                if cash > peak_equity: peak_equity = cash
                else:
                    dd = (peak_equity - cash) / peak_equity
                    if dd > max_dd: max_dd = dd

    # 強制結算
    if position == 1:
        final_val = stock_qty * p_seg[-1] * (1 - cost)
        if final_val > peak_equity: peak_equity = final_val
        else:
            dd = (peak_equity - final_val) / peak_equity
            if dd > max_dd: max_dd = dd
        cash = final_val

    total_return = cash - 1.0
    
    # 4. 分數計算
    if total_return > 0:
        score = total_return / (max_dd + 0.1)
    else:
        score = total_return * (1 + max_dd * 5)
        
    memo[param_key] = score
    return score

# ==========================================
# 3. AE-QTS 引擎
# ==========================================

class AE_QTS_Engine:
    def __init__(self, num_qubits=32): 
        self.num_qubits = num_qubits
        self.qubits = np.full((num_qubits, 2), 1/np.sqrt(2))
        
    def observe(self):
        probs = self.qubits[:, 1] ** 2
        rands = np.random.rand(self.num_qubits)
        binary_solution = (rands < probs).astype(int)
        return binary_solution

    def decode(self, binary_solution):
        params = []
        for i in range(4): 
            chunk = binary_solution[i*8 : (i+1)*8]
            val = 0
            for bit in chunk: val = (val << 1) | bit
            if val == 0: val = 1
            params.append(val)
        return params

    def update(self, best_binary, worst_binary):
        delta_theta = 0.05 * np.pi 
        diff = best_binary - worst_binary
        thetas = np.zeros(self.num_qubits)
        thetas[diff == -1] = -delta_theta
        thetas[diff == 1] = delta_theta
        c = np.cos(thetas)
        s = np.sin(thetas)
        alpha = self.qubits[:, 0]
        beta = self.qubits[:, 1]
        self.qubits[:, 0] = alpha * c - beta * s
        self.qubits[:, 1] = alpha * s + beta * c

# ==========================================
# 4. 主程式 (布林通道 + MA200 + 15% 停損)
# ==========================================

stock_list = ["2330.TW", "2317.TW", "2454.TW", "3049.TW", "2543.TW"] 

# 參數設定
GENERATIONS = 60       
POPULATION_SIZE = 30   
THRESHOLD_SCORE = 0.2  

results_summary = []
total_initial_capital = 0
total_final_capital = 0

print(f"【AE-QTS 終極策略組合】")
print(f"機制: 動態布林通道 + MA200 趨勢濾網 + 15% 硬性停損")
print(f"設定: Gen={GENERATIONS}, Pop={POPULATION_SIZE}, Threshold={THRESHOLD_SCORE}")

for stock_id in stock_list:
    print(f"\nProcessing: {stock_id} ...")
    
    # --- 下載資料 ---
    try:
        df = yf.download(stock_id, period="3y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 250: # 需要多一點資料算 MA200
            print("  -> 資料不足跳過")
            continue
    except Exception as e:
        print(f"  -> 下載錯誤: {e}")
        continue

    # --- 預計算 (MA & STD) ---
    ma_matrix, std_matrix, close_prices = precompute_bollinger_data(df['Close'], max_period=80)
    
    # --- 【新增】預計算 MA200 ---
    # 計算 200 日均線，並處理 NaN
    ma200_full = df['Close'].rolling(window=200).mean().to_numpy()
    ma200_full = np.nan_to_num(ma200_full) # 前 199 天會是 0
    
    # --- 初始設定 ---
    training_window = 120
    testing_days = 20
    total_days = len(df)
    
    initial_capital = 1_000_000
    funds = initial_capital
    position = 0
    stock_qty = 0
    
    entry_price = 0 
    cooldown_counter = 0 
    current_idx = training_window
    
    # 若 training_window 小於 200，前段 MA200 可能為 0，這會自然過濾掉(因為 Price > 0)
    # 建議 current_idx 至少從 200 開始比較準，但為了維持回測長度，我們保持原樣，
    # 只是前 200 天 MA200 為 0，Price > 0 恆成立 (也就是前 200 天沒濾網效果)
    # 如果要嚴謹，可以設 current_idx = max(training_window, 200)
    
    if current_idx < 200:
        current_idx = 200

    # --- 滑動視窗 ---
    while current_idx < total_days - testing_days:
        train_start = current_idx - training_window
        train_end = current_idx
        
        # 訓練資料切片
        train_prices_seg = close_prices[train_start:train_end]
        train_ma200_seg = ma200_full[train_start:train_end] # 【新增】切出 MA200
        
        engine = AE_QTS_Engine(num_qubits=32)
        best_params_raw = [20, 20, 50, 150] 
        best_score_local = -9999
        memo = {}
        
        # 1. 訓練期：AE-QTS 演化
        for gen in range(GENERATIONS):
            pop = []
            for _ in range(POPULATION_SIZE):
                binary = engine.observe()
                params_raw = engine.decode(binary) 
                
                # 傳入 MA200 進行有濾網的訓練
                score = get_fitness_bollinger_trend(
                    ma_matrix, std_matrix, train_prices_seg, train_ma200_seg,
                    train_start, train_end, 
                    *params_raw, memo
                )
                pop.append({'binary': binary, 'params': params_raw, 'score': score})
            
            pop.sort(key=lambda x: x['score'], reverse=True)
            
            if pop[0]['score'] > best_score_local:
                best_score_local = pop[0]['score']
                best_params_raw = pop[0]['params']
            
            if pop[0]['score'] > pop[-1]['score']:
                engine.update(pop[0]['binary'], pop[-1]['binary'])

        # 2. 測試期準備
        allow_new_entry = True
        if best_score_local < THRESHOLD_SCORE:
            allow_new_entry = False
            
        p1, p2, p3, p4 = best_params_raw
        N = (p1 % 60) + 5
        K = (p2 % 30 + 10) / 10.0
        buy_th = (p3 / 100.0) - 0.5
        sell_th = (p4 / 100.0) - 0.5
        
        test_end = min(current_idx + testing_days, total_days)
        test_prices = close_prices[current_idx:test_end]
        test_ma200 = ma200_full[current_idx:test_end] # 【新增】切出測試期 MA200
        
        ma_seg = ma_matrix[N, current_idx:test_end]
        std_seg = std_matrix[N, current_idx:test_end]
        
        lower_seg = ma_seg - (K * std_seg)
        width_seg = 2 * K * std_seg
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pb_seg = (test_prices - lower_seg) / width_seg
        pb_seg = np.nan_to_num(pb_seg)
        
        # 3. 測試期逐日交易
        for i in range(len(test_prices)):
            price = test_prices[i]
            current_pb = pb_seg[i]
            current_ma200 = test_ma200[i] # 【新增】
            
            # 冷靜期
            if cooldown_counter > 0:
                cooldown_counter -= 1
                if position == 1: 
                    funds = stock_qty * price * (1 - 0.004425)
                    stock_qty = 0
                    position = 0
                continue 
            
            # --- 硬性停損 (15%) ---
            if position == 1:
                current_pnl = (price - entry_price) / entry_price
                if current_pnl < -0.15: # 15% 停損
                    funds = stock_qty * price * (1 - 0.004425)
                    stock_qty = 0
                    position = 0
                    cooldown_counter = 10 
                    continue

            # --- 交易訊號 ---
            if position == 0:
                # 買進: %B < buy_th AND 股價 > MA200 (趨勢濾網)
                if current_pb < buy_th and allow_new_entry and price > current_ma200:
                    stock_qty = funds / price
                    funds = 0
                    position = 1
                    entry_price = price
            
            elif position == 1:
                # 賣出: %B > sell_th
                if current_pb > sell_th:
                    funds = stock_qty * price * (1 - 0.004425)
                    stock_qty = 0
                    position = 0
                    entry_price = 0
        
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
# 5. 輸出結果
# ==========================================
print("\n" + "="*40)
print(f"【AE-QTS 終極版回測報告】")
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