import yfinance as yf
import pandas as pd
import os
import numpy as np
import random

# ==========================================
# 1. 核心邏輯 (極速優化版)
# ==========================================

def precompute_all_newma(close_series, volume_series, max_period=240):
    """ 
    預計算所有 MA 
    回傳矩陣 shape: (241, DataLength)
    """
    n = len(close_series)
    all_mas = np.zeros((max_period + 1, n))
    
    p = close_series.to_numpy()
    v = volume_series.to_numpy()
    pv = p * v
    
    series_pv = pd.Series(pv)
    series_v = pd.Series(v)
    
    # print("正在預計算技術指標矩陣 (1~240)...")
    for d in range(1, max_period + 1):
        sum_pv = series_pv.rolling(d).sum().to_numpy()
        sum_v = series_v.rolling(d).sum().to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            ma_d = sum_pv / sum_v
        ma_d = np.nan_to_num(ma_d)
        all_mas[d] = ma_d
    return all_mas

def get_fitness_fast_event_based(ma_matrix, start_idx, end_idx, prices_segment, Ta, Tb, Tx, Ty, memo):
    """
    【極速版適應值函數】
    使用 Event-Based 邏輯與加重 MDD 懲罰
    """
    param_key = (Ta, Tb, Tx, Ty)
    if param_key in memo: return memo[param_key]

    # 1. 邊界檢查
    if any(p > 240 for p in [Ta, Tb, Tx, Ty]) or any(p < 2 for p in [Ta, Tb, Tx, Ty]):
        return -9999
    if Ta >= Tb or Tx >= Ty: 
        return -9999

    # 2. 取得指標切片
    ma_buy_s = ma_matrix[Ta, start_idx:end_idx]
    ma_buy_l = ma_matrix[Tb, start_idx:end_idx]
    ma_sell_s = ma_matrix[Tx, start_idx:end_idx]
    ma_sell_l = ma_matrix[Ty, start_idx:end_idx]

    # 3. 快速計算訊號
    b_cond = (ma_buy_s > ma_buy_l)
    s_cond = (ma_sell_s < ma_sell_l)
    
    buy_signals = b_cond[1:] & ~b_cond[:-1]
    sell_signals = s_cond[1:] & ~s_cond[:-1]
    
    buy_indices = np.flatnonzero(buy_signals) + 1
    sell_indices = np.flatnonzero(sell_signals) + 1
    
    # 4. 事件驅動回測
    cash = 1.0
    position = 0
    entry_price = 0.0
    peak_equity = 1.0
    max_dd = 0.0
    cost = 0.004425
    
    if len(buy_indices) == 0 and len(sell_indices) == 0:
        memo[param_key] = 0 
        return 0
        
    events = []
    for idx in buy_indices: events.append((idx, 1))
    for idx in sell_indices: events.append((idx, -1))
    events.sort(key=lambda x: x[0]) 
    
    for day, action in events:
        price = prices_segment[day]
        
        if position == 0 and action == 1: # Buy
            position = 1
            entry_price = price
            
        elif position == 1 and action == -1: # Sell
            position = 0
            ret = (price / entry_price) * (1 - cost) - 1
            cash *= (1 + ret)
            
            if cash > peak_equity:
                peak_equity = cash
            else:
                dd = (peak_equity - cash) / peak_equity
                if dd > max_dd: max_dd = dd

    # 強制結算
    if position == 1:
        price = prices_segment[-1]
        ret = (price / entry_price) * (1 - cost) - 1
        final_value = cash * (1 + ret)
        
        if final_value > peak_equity:
            peak_equity = final_value
        else:
            dd = (peak_equity - final_value) / peak_equity
            if dd > max_dd: max_dd = dd
        cash = final_value

    total_return = cash - 1.0

    # 5. 分數計算 (加重懲罰版)
    if total_return > 0:
        # 獲利時：報酬 / (2倍回撤 + 0.1)
        score = total_return / ( (max_dd * 2) + 0.1 ) 
    else:
        # 虧損時：虧損 * (1 + 5倍回撤) -> 讓分數變得非常負
        score = total_return * (1 + (max_dd * 5)) 

    memo[param_key] = score
    return score

class AE_QTS_Fast:
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
# 2. 主程式 (含停損防護)
# ==========================================

stock_list = ["2330.TW", "2317.TW", "2454.TW", "3049.TW"]
# 參數設定
GENERATIONS = 50       
POPULATION_SIZE = 20   
THRESHOLD_SCORE = 0.3 # 訓練期分數門檻

results_summary = []
total_initial_capital = 0
total_final_capital = 0

print(f"【AE-QTS 最終版】")

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

    # --- 預計算 ---
    ma_matrix = precompute_all_newma(df['Close'], df['Volume'])
    close_prices = df['Close'].to_numpy()
    
    # --- 初始設定 ---
    training_window = 120
    testing_days = 20
    total_days = len(df)
    
    initial_capital = 1_000_000
    funds = initial_capital
    position = 0
    stock_qty = 0
    
    # 新增變數：紀錄買入價格 (用於計算停損)
    entry_price = 0 
    # 新增變數：冷靜期計數器
    cooldown_counter = 0 
    
    current_idx = training_window
    
    # --- 滑動視窗 ---
    while current_idx < total_days - testing_days:
        train_start = current_idx - training_window
        train_end = current_idx
        
        train_prices_seg = close_prices[train_start:train_end]
        
        engine = AE_QTS_Fast(num_qubits=32)
        best_params_local = [5, 20, 5, 20]
        best_score_local = -9999
        memo = {}
        
        # 1. 訓練期：AE-QTS 演化
        for gen in range(GENERATIONS):
            pop = []
            for _ in range(POPULATION_SIZE):
                binary = engine.observe()
                params = engine.decode(binary)
                score = get_fitness_fast_event_based(
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

        # 2. 測試期準備
        
        # 檢查門檻：如果訓練分數太低，這回合不允許建立新倉位
        # 但如果原本有倉位，還是要繼續監控停損/賣出
        allow_new_entry = True
        if best_score_local < THRESHOLD_SCORE:
            allow_new_entry = False
            
        best_Ta, best_Tb, best_Tx, best_Ty = best_params_local
        test_end = min(current_idx + testing_days, total_days)
        test_prices = close_prices[current_idx:test_end]
        
        # 取得指標
        ma_buy_s = ma_matrix[best_Ta, current_idx:test_end]
        ma_buy_l = ma_matrix[best_Tb, current_idx:test_end]
        ma_sell_s = ma_matrix[best_Tx, current_idx:test_end]
        ma_sell_l = ma_matrix[best_Ty, current_idx:test_end]
        
        prev_idx = current_idx - 1
        prev_ma_buy_s = ma_matrix[best_Ta, prev_idx]
        prev_ma_buy_l = ma_matrix[best_Tb, prev_idx]
        prev_ma_sell_s = ma_matrix[best_Tx, prev_idx]
        prev_ma_sell_l = ma_matrix[best_Ty, prev_idx]

        # 3. 測試期逐日交易 (包含停損邏輯)
        for i in range(len(test_prices)):
            price = test_prices[i]
            
            # --- [停損機制] ---
            # 如果還在冷靜期，減少計數，並跳過當日交易
            if cooldown_counter > 0:
                cooldown_counter -= 1
                # 如果有持倉 (理論上冷靜期應該是空手，但防呆檢查)，強制賣出
                if position == 1:
                    funds = stock_qty * price * (1 - 0.004425)
                    stock_qty = 0
                    position = 0
                continue # 跳過今天
            
            # 檢查硬性停損 (Hard Stop-Loss)
            if position == 1:
                # 計算當前報酬率
                current_pnl = (price - entry_price) / entry_price
                
                # 如果虧損超過 7%
                if current_pnl < -0.07:
                    funds = stock_qty * price * (1 - 0.004425)
                    stock_qty = 0
                    position = 0
                    cooldown_counter = 10 # 觸發停損後，強制休息 10 天
                    # print(f"  [STOP] 觸發停損 {current_pnl:.2%}")
                    continue # 跳過今天剩下的邏輯

            # --- [訊號計算] ---
            curr_bs = ma_buy_s[i]
            curr_bl = ma_buy_l[i]
            curr_ss = ma_sell_s[i]
            curr_sl = ma_sell_l[i]
            
            last_bs = ma_buy_s[i-1] if i > 0 else prev_ma_buy_s
            last_bl = ma_buy_l[i-1] if i > 0 else prev_ma_buy_l
            last_ss = ma_sell_s[i-1] if i > 0 else prev_ma_sell_s
            last_sl = ma_sell_l[i-1] if i > 0 else prev_ma_sell_l
            
            buy_sig = (curr_bs > curr_bl) and (last_bs <= last_bl)
            sell_sig = (curr_ss < curr_sl) and (last_ss >= last_sl)
            
            # --- [交易執行] ---
            # 買進：要有訊號 + 空手 + 允許進場(過門檻) + 不在冷靜期
            if position == 0 and buy_sig and allow_new_entry:
                stock_qty = funds / price
                funds = 0
                position = 1
                entry_price = price # 紀錄成本價
            
            # 賣出：要有訊號 + 持倉 (不管門檻如何，該賣就要賣)
            elif position == 1 and sell_sig:
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
# 3. 輸出結果
# ==========================================
print("\n" + "="*40)
print(f"【AE-QTS 最終回測報告】")
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