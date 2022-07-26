import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data():
    """提取VIX计算需要的数据"""
    # 期权数据
    opt_basic = pd.read_csv('data/opt_basic.csv')
    opt_price = pd.read_csv('data/opt_price.csv')
    opt_data = pd.merge(opt_price, opt_basic, on='code')

    # 转换为日期格式
    opt_data['date'] = pd.to_datetime(opt_data['date'])
    opt_data['exercise_date'] = pd.to_datetime(opt_data['exercise_date'])
    # 筛选需要的字段

    use_fields = ['date', 'close', 'contract_type', 'exercise_price', 'exercise_date']
    opt_data = opt_data[use_fields]

    # 重命名
    opt_data['contract_type'] = opt_data['contract_type'].map({'CO': 'call', 'PO': 'put'})
    opt_data['T-days'] = (opt_data['exercise_date'] - opt_data['date']).apply(lambda x: x.days)
    opt_data['TTM'] = opt_data['T-days'] / 365

    # 利率数据
    interest_rate = pd.read_csv('data/interest_rate.csv', index_col=0)

    # 转换格式
    interest_rate.index = pd.DatetimeIndex(interest_rate.index)
    interest_rate.columns = interest_rate.columns.astype(int)

    # 填充遗漏天数
    omit_date = list(set(opt_data['date']) - set(interest_rate.index))
    for d in omit_date:
        interest_rate.loc[d, :] = np.nan
    interest_rate = interest_rate.sort_index().ffill()
    return opt_data, interest_rate

opt_data, interest_rate = prepare_data()

for dt, data in opt_data.groupby('date'):
    break
data

# 筛选近月、次近月的合约存续时间
all_T = data[data['T-days'] > 7]['T-days'].unique()
all_T.sort()
# 以【日】为单位和以【年】为单位的存续时间
Tdays_fm, Tdays_sfm = all_T[:2]
T_fm, T_sfm = all_T[:2] / 365
# 以此筛选出用于计算的合约数据
data_fm, data_sfm = data[data['TTM'] == T_fm], data[data['TTM'] == T_sfm]
data_fm

def pivot_table_(data):
    data = data.pivot_table(index='exercise_price', columns='contract_type', values='close')
    data['call-put'] = data['call'] - data['put']
    data['abs-diff'] = np.abs(data['call-put'])
    return data


def get_params(data, dt, Tdays):
    """
    参数计算函数.
    data: 前面计算的数据透视表；
    dt: 计算VIX指数的时间；
    Tdays: 到期时间，单位为（日）；
    return:
        S: 无套利标的资产价格
        R: 对应时间的无风险利率
        F: 无套利远期价格
        K0: 离F最近的，但小于F的执行价
        Q: pandas.DataFrame, 里面包含计算参数Q, K_i, delta_K_i
    """
    S = data.sort_values(by='abs-diff').index[0]
    R = interest_rate.loc[dt, Tdays] / 100
    F = S + np.exp(Tdays / 365 * R) * data.loc[S, 'call-put']
    K0 = np.max(data.index[data.index < F])
    # 合约选择
    Q = data[['call', 'put']].copy()
    Q.loc[Q.index >= F, 'Q'] = Q.loc[Q.index >= F, 'call']
    Q.loc[Q.index < F, 'Q'] = Q.loc[Q.index < F, 'put']
    Q = Q['Q'].reset_index()
    # 一大波操作：重设K_i index，移动窗口方法计算delta_K_i
    Q['delta_K'] = Q['exercise_price']\
        .rolling(3, center=True)\
        .apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 2)\
        .ffill().bfill() # 填充两端的缺失值
    return S, R, F, K0, Q

data_fm_ = pivot_table_(data_fm.copy())
data_sfm_ = pivot_table_(data_sfm.copy())

S_fm, R_fm, F_fm, K0_fm, Q_fm = get_params(data_fm_, dt, Tdays_fm)
S_sfm, R_sfm, F_sfm, K0_sfm, Q_sfm = get_params(data_sfm_, dt, Tdays_sfm)

# S_fm, R_fm, F_fm, K0_fm = (2.3, 0.04996766666666667, 2.325754666938786, 2.3)
# S_sfm, R_sfm, F_sfm, K0_sfm = (2.3, 0.049402999999999996, 2.3140361211554956, 2.3)

def get_sigma(T, R, F, K0, Q):
    """
    计算式（1）sigma的函数。至于为什么拆分成两部分，或许是“激动的心，颤抖的手”。
    """
    sigma_part1 = (2/T) * np.sum(np.exp(R*T)*Q['Q']*Q['delta_K']/np.square(Q['exercise_price']))
    sigma_part2 = (1/T) * (F/K0 - 1)**2
    return sigma_part1 + sigma_part2

sigma1 = get_sigma(T_fm, R_fm, F_fm, K0_fm, Q_fm)
sigma2 = get_sigma(T_sfm, R_sfm, F_sfm, K0_sfm, Q_sfm)

# sigma1, sigma2 = (0.06465976520531855, 0.054218952375645016)

def get_VIX(sigma1, sigma2, T1, T2):
    """计算式（2）VIX计算函数，同样激动。"""
    vix_part1 = (T1 * sigma1) * (T2 - 30 / 365) / (T2 - T1)
    vix_part2 = (T2 * sigma2) * (30 / 365 - T1) / (T2 - T1)
    return np.sqrt((365 / 30) * (vix_part1 + vix_part2))

vix = get_VIX(sigma1, sigma2, T_fm, T_sfm)

# 设置一个空字典来储存数据
VIX = {}
for dt, data in opt_data.groupby('date'):
    # 筛选近月、次近月的合约存续时间
    all_T = data[data['T-days'] > 7]['T-days'].unique()
    all_T.sort()
    T_fm, T_sfm = all_T[:2] / 365
    Tdays_fm, Tdays_sfm = all_T[:2]
    # 以此筛选出用于计算的合约数据
    data_fm, data_sfm = data[data['T'] == T_fm], data[data['T'] == T_sfm]
    # 数据透视表操作
    data_fm_ = pivot_table_(data_fm.copy())
    data_sfm_ = pivot_table_(data_sfm.copy())
    # 获取计算参数
    S_fm, R_fm, F_fm, K0_fm, Q_fm = get_params(data_fm_, dt, Tdays_fm)
    S_sfm, R_sfm, F_sfm, K0_sfm, Q_sfm = get_params(data_sfm_, dt, Tdays_sfm)
    # 计算sigma
    sigma1 = get_sigma(T_fm, R_fm, F_fm, K0_fm, Q_fm)
    sigma2 = get_sigma(T_sfm, R_sfm, F_sfm, K0_sfm, Q_sfm)
    # 计算VIX指数
    VIX[dt] = get_VIX(sigma1, sigma2, T_fm, T_sfm)

# 里面有一些天数算不出来（但不多），估计是合约移仓换月的时候无数可算。
VIX = pd.Series(VIX, name='VIX').dropna()

fig, ax = plt.subplots(figsize=(15, 6))
VIX.plot(ax=ax)
ax.grid()