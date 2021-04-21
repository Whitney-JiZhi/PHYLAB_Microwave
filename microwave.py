# PHY294 Lab - Microwave
# Zhi(Whitney) Ji and Chunsheng(Jason) Zuo


# -*- coding: utf-8 -*-

# filename="mydata.txt"

import pandas as pd
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import loadtxt
from IPython.display import display, HTML

# def momentum(ma,va,mb,vb,p_max) -> float:
#     M = ma*va + mb*vb
#     return M, p_max/100*M

# def energy(ma,va,mb,vb,p_max) -> float:
#     E = 0.5*ma*va**2 + 0.5*mb*vb**2
#     return E, p_max/100*E

import os
os.chdir('# put the working directory here #')  # put the working directory path inside the quotation mark


f_name = '4.3 data' # use the name of the file you want to read

data = pd.read_excel(f'{f_name}.xlsx')
# data = data.rename(columns={data.columns[3]: "I(A)", data.columns[5]: "V(V)"}, errors="raise")
print(data)


f_name2 = 'attenuation data'    # use the name of the file you want to read

data2 = pd.read_excel(f'{f_name2}.xlsx')
# data = data.rename(columns={data.columns[3]: "I(A)", data.columns[5]: "V(V)"}, errors="raise")
print(data2)


import math
def round_first(u):
    q = []
    for i in range(u.shape[0]):
        q.append(1 - int(math.floor(math.log10(abs(u.iloc[i])))) -1)
        u.iloc[i] = round(u.iloc[i],q[i])
    return u,q

def round_q(e,q):
    for i in range(e.shape[0]):
        e.iloc[i] = round(e.iloc[i],q[i]) 
    return e

# round(123,round_first(123))

print(data['Attenuation(dB)'].values)


what_graph = 'Attenuation(dB) vs Attenuation(mm)'
xdata = data['Attenuation(mm)'].values
ydata = data['Attenuation(dB)'].values
xerror = data['uamm'].values
yerror = data['uadb'].values

x2data = data2['Attenuation(mm)'].values
y2data = data2['Attenuation(dB)'].values
x2error = data2['uamm'].values
y2error = data2['uadb'].values


def linear(x,b):
    return b*x 

def linear2(x,b,c):
    return b*x + c

def exponent1(x,a):
    return a**x

def exponent2(x,a,b):
    return a**x+b

def log2(x,a,b):
    return a*np.log10(x)+b

def quadratic2(x,a,b):
    return a*x**2+b

def quadratic3(x,a,b,c):
    return a*x**2+b*x+c

def cube2(x,a,b):
    return a*x**(3)+b

def cubic3(x,a,b,c):
    return a*x**3+b*x**2

def quadro3(x,a,b,c):
    return a*x**4+c

def fitfunction1(x,popt):
    return popt[0]*np.exp(x)

def fitfunction2(x,popt):
    return popt[0]*np.exp(x) + popt[1]

def fitfunction_quadratic2(x,popt):
    return popt[0]*x**2 + popt[1]

def fitfunction_quadratic3(x,popt):
    return popt[0]*x**2 + popt[1]*x +popt[2]

def fit_wrap(func):
    def fit_n(x,popt):
        return func(x,*popt)
    return fit_n

def ff_linear2(x,popt):
    return popt[0]*x + popt[1]


#fitfunction(x) gives you your ideal fitted function, i.e. the line of best fit    

func = quadratic2
fitfunc = fit_wrap(func)
# popt, pcov = optimize.curve_fit(exponent1, xdata, ydata)
popt, pcov = optimize.curve_fit(func, xdata, ydata)
popt2, pcov2 = optimize.curve_fit(func, x2data, y2data)

# we have the best fit values in popt[], while pcov[] tells us the uncertainties


# make both data share the same range
start=min(xdata)
stop=max(max(xdata),max(x2data))    

xs=np.arange(start,stop,(stop-start)/1000)
curve =fitfunc(xs,popt)

xs2=np.arange(start,stop,(stop-start)/1000)
curve2 =fitfunc(xs2,popt2)


# (xs,curve) is the line of best fit for the data in (xdata,ydata) 


def chi(r,r_pred,u):
    return np.sum((r-r_pred)**2/(u**2))/(r.shape[0]-2)

def syx_square (r,r_pred):
    return 1/(r.shape[0]-2) * np.sum((r-r_pred)**2)
def sm (syx_square,xdata,delta):
    return (xdata.shape[0]*syx_square/delta)**0.5
def delta(x):
    return x.shape[0]*sum(x**2) - sum(x)**2
# for c in curves:
#     B,xs,curve,xdata,ydata,xerror,yerror,popt,pcov = c
# sm(syx_square(y2data,fitfunc(x2data,popt2)),x2data,delta(x2data))
print(chi(y2data,fitfunc(x2data,popt2),y2error))
print(chi(ydata,fitfunc(xdata,popt),yerror))


from scipy.stats import chisquare
print(chisquare(y2data, f_exp=fitfunc(x2data,popt2),ddof=2))
print(chisquare(ydata, f_exp=fitfunc(xdata,popt),ddof=2))


print(chisquare(ydata, f_exp=fitfunc(xdata,popt)))


def round_3(n):
    return round(n,3 - int(math.floor(math.log10(abs(n)))) -1)
print(round_3(1234))

print(pcov[0])

print(np.std([0.278,0.352,0.759,1.296,0.529]))

print(abs(pcov)**0.5)
print(abs(pcov2)**0.5)


column = ['Attenuation(mm)','Attenuation(dB)','Figure 4.1.1: Attenuation conversion from mm to dB' ] 
figure(num=None, figsize=(6,5), dpi=400, facecolor='w', edgecolor='k')
R_H = [] 
cl = 1
lg = []

plt.errorbar(xdata,ydata,yerr=yerror,xerr=xerror,fmt=".")
plt.plot(xs,curve)
print(f"Slope 1: {popt[0]} "+"+/-", f'{pcov[0,0]**0.5}')
# lg.append(f'Experimental Attenuation(dB) = {round_3(popt[0])}*Attenuation(mm)^2')
lg.append(f'Experimental Attenuation(dB) = {round_3(popt[0])}*Attenuation(mm)^2+{round_3(popt[1])}')

plt.errorbar(x2data,y2data,yerr=y2error,xerr=x2error,fmt=".")
plt.plot(xs2,curve2)
print(f"Slope 2: {popt2[0]} "+"+/-", f'{pcov2[0,0]**0.5}')
# lg.append(f'Theoretical Attenuation(dB) = {round_3(popt2[0])}*Attenuation(mm)^2')
lg.append(f'Preset Attenuation(dB) = {round_3(popt2[0])}*Attenuation(mm)^2+{round_3(popt2[1])}')



plt.xlabel(column[0])
plt.ylabel(column[1])
plt.title(column[2])
plt.legend(lg,fontsize=8)
# plt.text(0.016, 14,f'r$_i$ = {popt[0]:.2f} $\lambda$ ')
# plt.text(0.016, 14,f'r$_i$ = {popt[0]/100:.2f}*10$^8$$\lambda$ + {popt[1]:.2f} ')

# plt.text(0.016,13.5,f'm = {popt[0]/100:.2f}*10$^8$ \u00B1 {pcov[0,0]**0.5/100:.2f}*10$^8$')

plt.show()


# print(f"Slope: {popt[0]} "+"+/-", f'{pcov[0,0]**0.5}')


# import scipy.stats as stats
# stats.chisquare(ydata,fitfunction(xdata,popt))
figure(num=None, figsize=(6,5), dpi=400, facecolor='w', edgecolor='k')
R_H = [] 
cl = 1
lg = []

err,errc = optimize.curve_fit(linear2, xdata, ydata-fitfunc(xdata,popt))
plt.plot(xs,ff_linear2(xs,err),color = 'orange')
plt.errorbar(xdata,ydata-fitfunc(xdata,popt),yerr=yerror,xerr=xerror,fmt=".",color = 'orange')
lg.append(f'Experimental Attenuation error(dB) = {round_3(err[0])}*Attenuation(mm)^2+{round_3(err[1])}')

err2,errc2 = optimize.curve_fit(linear2, x2data, y2data-fitfunc(x2data,popt2))
plt.plot(xs,ff_linear2(xs2,err2),color='red')
plt.errorbar(x2data,y2data-fitfunc(x2data,popt2),yerr=y2error,xerr=x2error,fmt=".",color='red')
lg.append(f'Preset Attenuation error(dB) = {round_3(err2[0])}*Attenuation(mm)^2+{round_3(err2[1])}')

# plt.text(0.0160, 0.1,f'$\delta$r$_i$ = {int(err[0]*10**6)} $\delta$$\lambda$ + {err[1]:.4f}')


column = ['$\delta$Attenuation(nm)','$\delta$Attenuation(dB)','Figure 4.1.2: Residuals for Attenuation quadratic fitting'] 
plt.xlabel(column[0])
plt.ylabel(column[1])
plt.title(column[2])
plt.ylim([-5,5])
plt.legend(lg,fontsize=7)
plt.show()


column = ['$\lambda$(nm)','r$_o$(mm)','Figure 2.2: Electron Wavelength(nm) vs\n Outer Diffraction Ring Radius(mm)'] 
figure(num=None, figsize=(5,3), dpi=120, facecolor='w', edgecolor='k')
plt.errorbar(xdata,y2data,yerr=y2error,xerr=xerror,fmt=".")
plt.plot(xs,curve2,color = 'r')
plt.xlabel(column[0])
plt.ylabel(column[1])
plt.title(column[2])
# plt.text(0.016,24,f'r$_i$ = {popt2[0]/100:.2f}*10$^8$$\lambda$ + {popt2[1]:.2f} ')
plt.text(0.016,23,f'm = {popt2[0]/100:.2f}*10$^8$ \u00B1 {pcov2[0,0]**0.5/100:.2f}*10$^8$')
plt.text(0.016,24,f'r$_0$ = {popt2[0]:.2f} $\lambda$ ')

plt.show()
print(f"Slope: {popt2[0]:.5f} "+"+/-", f'{pcov2[0,0]**0.5:.5f}')
# stats.chisquare(y2data,fitfunction(xdata,popt2))
figure(num=None, figsize=(5,3), dpi=120, facecolor='w', edgecolor='k')
err2,errc2 = optimize.curve_fit(linear2, xdata, y2data-fitfunction(xdata,popt2))
plt.plot(xs,fitfunction2(xs,err2),color='r');
plt.scatter(xdata,y2data-fitfunction(xdata,popt2),color = 'g')
plt.errorbar(xdata,y2data-fitfunction(xdata,popt2),yerr=yerror,xerr=xerror,fmt=".")
plt.text(0.0160, 0.1,f'$\delta$r$_i$ = {int(err2[0]*10**6)} $\delta$$\lambda$ + {err2[1]:.4f}')
# plt.text(0.0160, 0.1,f'$\delta$r$_i$ = {err2[0]:.4f} $\delta$$\lambda$ ')

column = ['$\delta$$\lambda$(nm)','$\delta$r$_o$(mm)','Figure 3.2: Residual for the Outer Ring (y=mx)'] 
plt.xlabel(column[0])
plt.ylabel(column[1])
plt.title(column[2])
plt.show()


# figure(num=None, figsize=(3,3), dpi=120, facecolor='w', edgecolor='k')
# plt.errorbar(xdata,ydata,yerr=yerror,xerr=xerror,fmt=".")
# plt.plot(xs,curve,color = 'r')
# plt.xlabel(column[0])
# plt.ylabel(column[1])
# plt.title(column[1]+' vs '+column[0]+' plot')
# plt.show()


print(f"Slope: {popt[0]:.4f} "+"+/-", f'{pcov[0,0]**0.5:.4f}')
print(f"Slope: {popt2[0]:.4f} "+"+/-", f'{pcov2[0,0]**0.5:.4f}')
#     print("Intercept:", popt[1], "+/-", pcov[1,1]**0.5)
# print the slope with uncertainty, then the intercept with uncertainty


residual=ydata-fitfunction(xdata)
zeroliney=[0,0]
zerolinex=[start,stop]
#     figure(num=None, figsize=(122, 4), dpi=200, facecolor='w', edgecolor='k')
#     plt.errorbar(xdata,residual,yerr=yerror,xerr=xerror,fmt=".")
#     # plt.scatter(xdata,residual)
#     plt.plot(zerolinex,zeroliney)

#     plt.xlabel(column[0])
#     plt.ylabel("residuals of "+ column[1])
#     plt.title("Residuals of the fit")

#     plt.show()
if len(glist) == 2 and what_graph == '0':
    continue
import pandas as pd
# print(np.vstack([xdata,ydata]))
column.insert(1,'errori'+column[1][2:])
column.append('errorf'+column[1][2:]) 
df = pd.DataFrame(data=np.hstack([xdata.reshape(-1,1),xerror.reshape(-1,1),ydata.reshape(-1,1),yerror.reshape(-1,1)]).reshape(-1,4),columns = column)
display(HTML(df.to_html()))
# 757.2 471.7 0.54 0.18 0 0.55 3.7 3.6

p = df
print(p)
