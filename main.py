import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# ==========================================
# 1. 設定與資料準備
# ==========================================

# 設定要測試的股票代碼 (單一股票測試)
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
    
    # 處理多層索引 (若有的話)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 刪除重複欄位並存檔
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(csv_file)
    print("資料下載並儲存完成。")

# ==========================================
# 2. 核心類別與函式定義
# ==========================================

class AE_QTS:
    """
    AE-QTS 量子啟發式演算法引擎 
    負責產生參數、解碼參數，以及透過旋轉閘更新量子位元
    """
    def __init__(self, num_qubits=16): 
        # 改成 16 Qubits (只找 Short, Long 兩個參數，各用 8 bits)
        self.num_qubits = num_qubits
        # 初始化為疊加態 (機率各半: 1/sqrt(2))
        self.qubits = np.full((num_qubits, 2), 1/np.sqrt(2))
        
    def observe(self):
        """
        觀測 (Measurement)：塌縮成 0/1 
        """
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
        解碼：將 16 bits 轉成 [Short, Long] 參數 
        """
        params = []
        # 只有 2 個參數，所以迴圈跑 2 次
        for i in range(2): 
            chunk = binary_solution[i*8 : (i+1)*8]
            val = 0
            for bit in chunk:
                val = (val << 1) | bit
            
            # 限制範圍：避免出現 0 或超大數值
            # 論文參數範圍建議 1~240 
            if val < 3: val = 3
            if val > 240: val = 240
            
            params.append(val)
        return params

    def update(self, best_binary, worst_binary):
        """
        量子旋轉閘更新 (Quantum Rotation Gate)
        根據最佳解與最差解調整 Qubit 角度
        """
        delta_theta = 0.05 * np.pi 
        
        for i in range(self.num_qubits):
            best_bit = best_binary[i]
            worst_bit = worst_binary[i]
            
            theta = 0
            # 判斷旋轉方向
            if best_bit == 0 and worst_bit == 1:
                theta = -delta_theta
            elif best_bit == 1 and worst_bit == 0:
                theta = delta_theta
            else:
                continue # 兩個一樣就不轉
                
            alpha = self.qubits[i][0]
            beta = self.qubits[i][1]
            
            # 旋轉矩陣運算 
            new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
            new_beta  = alpha * np.sin(theta) + beta * np.cos(theta)
            
            self.qubits[i][0] = new_alpha
            self.qubits[i][1] = new_beta

def NEWMA(series_close, series_volume, days):
    """
    計算新型成交量權重移動平均線 (New MA) 
    公式: sum(P*V) / sum(V)
    """
    # 確保 days 至少為 1，避免錯誤
    days = int(max(1, days))
    
    pv = series_close * series_volume 
    sum_pv = pv.rolling(window=days).sum()      # 計算 N 天的 PV 總和
    sum_volume = series_volume.rolling(window=days).sum()  # 計算 N 天的 Volume 總和
    
    return sum_pv / sum_volume  # PV除Volume得到新MA

def get_fitness(close_price, volume, short_ma_days, long_ma_days):
    """
    適應值函數 (Fitness Function) 
    輸入：股價、成交量、短天期、長天期
    輸出：這組參數的回測報酬率
    """
    # 1. 參數合理性檢查
    if short_ma_days >= long_ma_days: 
        return -9999 # 短天期大於長天期，不合理
    if short_ma_days < 3 or long_ma_days > 240:
        return -9999

    # 2. 計算 MA (使用 NEWMA)
    ma_short = NEWMA(close_price, volume, short_ma_days)
    ma_long = NEWMA(close_price, volume, long_ma_days)

    # 3. 計算策略報酬 (向量化運算)
    # 訊號: 1=持有 (短>長), 0=空手
    sig = (ma_short > ma_long).astype(int)
    
    # 每日漲跌幅
    pct_change = close_price.pct_change().fillna(0)
    
    # 策略回報 = 昨天的訊號 * 今天的漲跌 (shift(1) 代表用昨天訊號操作)
    strategy_ret = sig.shift(1) * pct_change
    strategy_ret = strategy_ret.fillna(0)
    
    # 防呆檢查
    if len(strategy_ret) == 0: return -9999
    
    # 計算累積報酬率
    cum_ret = (1 + strategy_ret).cumprod()
    if len(cum_ret) == 0: return -9999
        
    final_return = cum_ret.iloc[-1] - 1
    return final_return

# ==========================================
# 3. 主程式：滑動視窗動態回測

print("開始 AE-QTS 動態滑動回測...")

# 設定回測參數
training_window = 120  # 訓練期長度 
testing_days = 20      # 測試期長度 
total_days = len(df)

# 初始化帳戶
initial_capital = 1_000_000
funds = initial_capital
position = 0 # 0=空手, 1=持有
stock_qty = 0
history = []

current_idx = training_window # 從資料足夠處開始

# 開始滑動視窗迴圈
while current_idx < total_days - testing_days:
    
    curr_date = df.index[current_idx].date()
    
    # === A. 訓練階段 (Training) ===
    train_start = current_idx - training_window
    train_end = current_idx
    
    # 提取訓練資料
    t_close = df['Close'].iloc[train_start:train_end].copy()
    t_volume = df['Volume'].iloc[train_start:train_end].copy()
    
    # 資料不足則跳過
    if len(t_close) < 10:
        current_idx += testing_days
        continue

    #  AE-QTS 演算法尋找最佳參數 --- 
    
    # 1. 建立AEQTS
    engine = AE_QTS(num_qubits=16) 
    
    best_params_global = [5, 20] # 預設最佳參數
    best_score_global = -9999    # 預設最佳分數
    
    # 設定演化參數
    generations = 10     # 演化代數
    population_size = 10 # 族群大小

    # 開始演化
    for gen in range(generations):
        population = [] # 這一代的族群
        
        # 2. 觀測與評估
        for _ in range(population_size):
            binary_code = engine.observe()     # 觀測 
            params = engine.decode(binary_code) # 解碼 
            
            s, l = params[0], params[1]
            
            # 計算適應值 (回測績效)
            score = get_fitness(t_close, t_volume, s, l)
            
            population.append({
                'binary': binary_code,
                'params': params,
                'score': score
            })
            
            # 更新全域最佳解
            if score > best_score_global:
                best_score_global = score
                best_params_global = params

        # 3. 排序 (分數高的排前面)
        population.sort(key=lambda x: x['score'], reverse=True)
        
        best_sol = population[0]   # 最好解
        worst_sol = population[-1] # 最差解
        
        # [cite_start]4. 量子更新 (學習) 
        if best_sol['score'] > worst_sol['score']:
            engine.update(best_sol['binary'], worst_sol['binary'])
            
    # 訓練結束，取得最佳參數
    best_s, best_l = best_params_global
    
    # === B. 測試階段 (Testing) ===
    test_end = current_idx + testing_days
    if test_end > total_days: test_end = total_days

    # 用找到的參數計算整段時間的 MA (避免邊界 NaN)
    ma_short_full = NEWMA(df['Close'], df['Volume'], best_s)
    ma_long_full = NEWMA(df['Close'], df['Volume'], best_l)

    # 逐日跑測試期交易
    for i in range(current_idx, test_end):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        ms = ma_short_full.iloc[i]
        ml = ma_long_full.iloc[i]
        
        if pd.isna(ms) or pd.isna(ml): continue
        
        # 交易訊號判斷
        if ms > ml and position == 0: # 黃金交叉 -> 買進
            stock_qty = funds / price
            funds = 0
            position = 1
            history.append(f"{date.date()} 買入({best_s},{best_l}) @ {price:.1f}")
            
        elif ms < ml and position == 1: # 死亡交叉 -> 賣出
            funds = stock_qty * price
            stock_qty = 0
            position = 0
            history.append(f"{date.date()} 賣出({best_s},{best_l}) @ {price:.1f}")

    # === C. 滑動視窗 ===
    current_idx += testing_days
    
    # 顯示進度
    current_asset = funds + (stock_qty * df['Close'].iloc[current_idx-1] if position else 0)
    print(f"進度: {curr_date} -> 下一站 | 資產: {int(current_asset)}")

# ==========================================
# 4. 結算與輸出
# ==========================================

# 最後一天強制平倉結算
final_price = df['Close'].iloc[-1]
if position == 1:
    funds = stock_qty * final_price

total_return = (funds - initial_capital) / initial_capital * 100

print("\n" + "="*30)
print(f"【AE-QTS 動態策略】最終結果")
print(f"測試股票: {stock_id}")
print(f"初始資金: {initial_capital}")
print(f"最終資產: {int(funds)}")
print(f"總報酬率: {total_return:.2f}%")
print("="*30)

# 印出最後 5 筆交易紀錄
print("最近 5 筆交易紀錄:")
for h in history[-5:]:
    print(h)