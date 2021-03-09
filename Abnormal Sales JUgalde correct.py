#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:35:45 2020

@author: Juli
"""

import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

financials = pd.read_csv("/Users/Juli/Documents/UCSD Classes/International Business/MGT 153/Pandas/Financials.csv")
print(financials.head(20))

pd.set_option('display.max_columns', 500)
print(financials.head(20))

print(financials.isnull().sum(axis=1))

financials['Num Nulls'] = financials.isnull().sum(axis=1)
financials.sort_values('Num Nulls', ascending=False)
print(financials.head(20))

print('Hello World')

financials = financials.drop(['IBE', 'OCF', 'PPE'], axis=1)
print(financials.head(20))

print(financials.isnull().sum(axis=1))
print(financials.isnull().sum())

financials.dropna(subset=['SALE'], inplace = True)
print(financials.isnull().sum())

financials.dropna(thresh=11, inplace=True)
financials.AP.fillna(financials.AP.mean(), inplace=True)
financials.REC.fillna(financials.REC.mean(), inplace=True)
financials.BV.fillna(financials.BV.mean(), inplace=True)
financials.MV.fillna(financials.groupby('2_digit_sic')['MV'].transform("mean"), inplace=True)
print(financials.describe())

financials = financials[financials.XOPR >= 0]
financials = financials[financials.COGS >= 0]
financials = financials[financials.BV > 0]
financials = financials[financials.SALE > 0]
financials = financials[financials.AT > 0]
print(financials.describe())

financials.EMP = (financials.EMP - financials.EMP.mean())/financials.EMP.std()
financials.COGS = (financials.COGS - financials.COGS.min())/(financials.COGS.max() - financials.COGS.min())
financials['Binned_SALE'] = pd.qcut(financials.SALE, 10, labels=False)
financials.SALE = np.where(financials.SALE < financials.SALE.quantile(q=0.02), financials.SALE.quantile(q=0.02), financials.SALE)
financials.SALE = np.where(financials.SALE > financials.SALE.quantile(q=0.98), financials.SALE.quantile(q=0.98), financials.SALE)
print(financials.dtypes)
print(financials.Binned_SALE)

financials.datadate = pd.to_datetime(financials.datadate, format='%Y%m%d')
print(financials.dtypes)

financials['Year'] = financials.datadate.dt.year
print(financials.describe())

financials.sort_values(by=['gvkey', 'datadate'], ascending=[True, True], inplace = True)
financials['prevSALE'] = financials.SALE.shift(1)
financials['prevAT'] = financials.AT.shift(1)
financials['prevEMP'] = financials.EMP.shift(1)
financials['prevCOGS'] = financials.COGS.shift(1)
financials['prevREC'] = financials.REC.shift(1)
financials['prevXOPR'] = financials.XOPR.shift(1)
financials['prevAP'] = financials.AP.shift(1)
financials = financials[((financials.Year-1 == financials.Year.shift(1)) & (financials.gvkey == financials.gvkey.shift(1)))]
financials = financials[(financials.AT > 0)]
financials = financials[(financials.prevAT > 0)]
financials['Scaled_Sales'] = financials.SALE / financials.AT
financials['Scaled_prevSales'] = financials.prevSALE / financials.prevAT
financials['Scaled_Emp'] = financials.EMP / financials.prevAT
financials['Scaled_EmpChange'] = (financials.EMP - financials.prevEMP)/financials.prevAT
financials['Scaled_COGS'] = financials.COGS/financials.prevAT
financials['Scaled_COGSChange'] = (financials.COGS - financials.prevCOGS) / financials.prevAT
financials['Scaled_Rec'] = financials.REC / financials.prevAT
financials['Scaled_RecChange'] = (financials.REC - financials.prevREC) / financials.prevAT
financials['Scaled_XOPR'] = financials.XOPR / financials.prevAT
financials['Scaled_XOPRChange'] = (financials.XOPR - financials.prevXOPR) / financials.prevAT
financials['Scaled_AP'] = financials.AP / financials.prevAT
financials['Scaled_APChange'] = (financials.AP - financials.prevAP) / financials.prevAT
financials['BookToMarket'] = financials.BV / financials.MV

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
sns.boxplot(x=financials.SALE)
plt.figure()
sns.violinplot(x=financials.SALE, color="0.25")
plt.figure()
sns.distplot(financials.SALE, kde=False, fit=stats.norm)
plt.figure()
financials.SALE = np.log(financials.SALE)
sns.distplot(financials.SALE, kde=False, fit=stats.norm)
plt.figure()
industry_average_sales = financials.groupby(['2_digit_sic', 'Year'])['SALE'].mean()
industry_average_sales.name = 'average_sales'
financials.groupby(['2_digit_sic', 'Year']).agg({'Scaled_prevSales': 'sum', 'Scaled_Emp': 'mean', 'Scaled_Rec': ['min', 'mean']})
financials = pd.merge(financials, industry_average_sales, how='inner', left_on=['2_digit_sic', 'Year'], right_on=['2_digit_sic', 'Year'])
financials.sort_values(by=['gvkey', 'Year'], ascending=[True, True], inplace=True)
financials['SALE_Industry_Mean'] = financials.groupby(['2_digit_sic', 'Year'])['SALE'].transform("mean")
financials.sort_values(by=['gvkey','datadate'], ascending=[True, True], inplace = True)
financials['prevSALE2'] = financials.groupby('gvkey')['SALE'].shift(1)
print(financials.head(20))
financials.dropna(inplace=True)
model_results = sm.ols(formula='Scaled_Sales ~ Scaled_prevSales + Scaled_Emp + Scaled_EmpChange + Scaled_COGS + Scaled_COGSChange + Scaled_Rec + Scaled_RecChange + Scaled_XOPR + Scaled_XOPRChange + Scaled_AP + Scaled_APChange + BookToMarket', data=financials).fit()
model_results = sm.ols(formula='Scaled_Sales ~ Scaled_prevSales + Scaled_Emp + Scaled_EmpChange + Scaled_COGS + Scaled_COGSChange + Scaled_Rec + Scaled_RecChange + Scaled_XOPR + Scaled_XOPRChange + Scaled_AP + Scaled_APChange + BookToMarket', data=financials).fit()
print(model_results.summary())

#c. Code for running mice (bug)
#from statsmodels.imputation import mice
#financials_regression_columns = financials[['Scaled_Sales', 'Scaled_prevSales', 'Scaled_Emp']]
#imp = mice.MICEData(financials_regression_columns)
#mice_results = mice.MICE('Scaled_Sales ~ Scaled_prevSales + Scaled_Emp', sm.ols, imp).fit()
#print(mice_results.summary())

#robust
robust_result = model_results.get_robustcov_results(cov_type='cluster', use_t=None, groups=financials['2_digit_sic'])
print(robust_result.summary())

#Check OLS regression assumptions and other potential problems
financials["residuals"] = model_results.resid
financials["predicted"] = model_results.fittedvalues
plt.scatter(financials.predicted, financials.residuals)
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.figure()

#filter data
financials_subset = financials[(financials.predicted<20)]
plt.scatter(financials_subset.predicted, financials_subset.residuals)
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.figure()

#Predicted vs. Actual
plt.scatter(financials_subset.predicted, financials_subset.Scaled_Sales)
plt.title("Actual by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actuals")

#Heteroskedacity robust regression using new OLS
new_results = sm.ols(formula='Scaled_Sales ~ Scaled_prevSales + Scaled_Emp + Scaled_EmpChange + Scaled_COGS + Scaled_COGSChange + Scaled_Rec + Scaled_RecChange + Scaled_XOPR + Scaled_XOPRChange + Scaled_AP + Scaled_APChange + BookToMarket', data=financials).fit(cov_type='HC3', use_t=None)
print(new_results.summary())

#Normally distributed errors
sns.distplot(financials.residuals, kde=False, fit=stats.norm)
plt.figure()

#multicollinearity
import statsmodels.stats.outliers_influence as sm_influence
myX = financials[['Scaled_prevSales', 'Scaled_Emp', 'Scaled_EmpChange', 'Scaled_COGS', 'Scaled_COGSChange', 'Scaled_Rec', 'Scaled_RecChange', 'Scaled_XOPR', 'Scaled_XOPRChange', 'Scaled_AP', 'Scaled_APChange', 'BookToMarket']]
myX = myX.dropna()
#VIF results
vif = pd.DataFrame()
vif["VIF Factor"] = [sm_influence.variance_inflation_factor(myX.values, i) for i in range(myX.shape[1])]
#add new column
vif["Variable"] = myX.columns
print(vif.round(2))

#Outliers
financials["CooksD"] = model_results.get_influence().summary_frame().filter(["cooks_d"])
financials = financials[(financials.CooksD<(4/financials.residuals.count()))]
#RLM
from patsy import dmatrices
import statsmodels.api as sm_non_formula

y, X = dmatrices('Scaled_Sales ~ Scaled_prevSales + Scaled_Emp + Scaled_EmpChange + Scaled_COGS + Scaled_RecChange + Scaled_XOPRChange + Scaled_APChange + BookToMarket', data=financials, return_type='dataframe')
robust_results = sm_non_formula.RLM(y, X, M=sm_non_formula.robust.norms.HuberT()).fit()
print(robust_results.summary())
