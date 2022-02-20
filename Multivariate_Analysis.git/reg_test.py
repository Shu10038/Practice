'''
 重回帰分析の試し
  Created on 2018/1/21
  @author: Shuichi OHSAWA
'''

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データの読み込み
file = 'reg_test_data.csv'
df = pd.read_csv(file)
df = df.set_index('val')
print(df)

# データを３次元プロットしてみる
fig = plt.figure()
ax = Axes3D(fig)

# 軸ラベルを設定する
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# データを無理やりarrayに変換する
xs = df.T.ix[:,0].values
ys = df.T.ix[:,1].values
zs = df.T.ix[:,2].values

# 描画
ax.scatter3D(xs, ys, zs)
plt.show()

# 相関係数を表示
df.T.corr()
print(df.T.corr())

# 散布図行列を描く
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df.T)
plt.show()

# 重回帰分析
import statsmodels.formula.api as sm
reg = "y ~ x1 + x2"
model = sm.ols(formula=reg, data=df.T)

# 回帰分析を実行する
result = model.fit()

print(result.summary())
