#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:23:03 2023
Group plotting functions too have everything working out properly
@author: isabelle
"""

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%% import datas
init='uniform'

gaussian=pd.read_pickle(f'../results/reboot/Gaussian_results_{init}_clean.pkl')
rdwalk  =pd.read_pickle(f'../results/reboot/RdWalk_results_{init}_clean.pkl')
reversal=pd.read_pickle(f'../results/reboot/Reversal_results_{init}_clean.pkl')

learning=pd.read_pickle(f'../results/reboot/LearningPeriod_results_{init}_clean.pkl')
difficulty=pd.read_pickle(f'../results/reboot/Difficulty_results_{init}_clean.pkl')
rich_poor=pd.read_pickle(f'../results/reboot/Rich_Poor_results_{init}_clean.pkl')

maxi=pd.read_pickle('../results/reboot/Maxi_results_clean.pkl')

SP=pd.read_pickle('../results/reboot/SP_results_uniform_clean_test.pkl')
SP_all=pd.read_pickle('../results/reboot/SP_results_uniform.pkl')

#%% filter data and order
keys=[(0.15,0.85),(0.35,0.65),(0.05, 0.55), (0.15, 0.65), (0.35, 0.85), (0.45, 0.95)]
for k in keys:
    del difficulty[k]
difficulty=dict(sorted(difficulty.items()))
    
keys=[(0.15,0.65),(0.35,0.85), (0.05, 0.95), (0.15, 0.85), (0.35, 0.65), (0.45, 0.55)]
for k in keys:
    del rich_poor[k]
rich_poor=dict(sorted(rich_poor.items()))

keys=[(4,40),(16,10)]
for k in keys:
    del gaussian[k]
    del rdwalk[k]
    del reversal[k]
    del learning[k]
gaussian=dict(sorted(gaussian.items()))
rdwalk=dict(sorted(rdwalk.items()))
reversal=dict(sorted(reversal.items()))
learning=dict(sorted(learning.items()))

#%% figure esthetics
plt.rcParams['font.size'] = 12
# plt.rcParams['title.fontsize'] = 20

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = False

#%% plotting functions
def learningCurves(data,ax,k=False):
    if not k:
        for k in data.keys():
            acc=np.mean(data[k]['Cho'],axis=0)
            sns.lineplot(acc,ax=ax,label=k)
            ax.set_ylim(0.4,1)
            ax.set_xlim(0,len(acc))
    else:
        acc=np.mean(data[k]['Cho'],axis=0)
        sns.lineplot(acc,ax=ax,label=k)
        ax.set_ylim(0.4,1)
        ax.set_xlim(0,len(acc))
        
def alpha_box(data,ax):
    ix=0
    df=pd.DataFrame(columns=['Key',r'alpha+',r'alpha-'])
    for k in data.keys():
        # keys=[ix]*100
        ap=data[k]['params'][:,1]
        am=data[k]['params'][:,2]
        keys=[ix]*am.shape[0]
        dat={'Key':keys,r'alpha+':ap,r'alpha-':am}
        dat_ar=pd.DataFrame(data=dat)
        df=pd.concat((df,dat_ar))
        ix+=1
        
    alpha_df=df[['alpha+','alpha-']].stack().reset_index()
    key_df=df[['Key','Key']].stack().reset_index(drop=True)
    aldf=pd.concat((key_df,alpha_df),axis=1)
    aldf.columns=['Key','level_0','Alpha','Value']
    # sns.violinplot(data=aldf,y='Value',x='Key',hue='Alpha',split=True,ax=ax,palette=sns.color_palette(['darkseagreen','crimson']),inner=None)

    g=sns.stripplot(data=aldf,x="Key", y="Value", hue='Alpha',
                  jitter=True, ax = ax, alpha=0.3,dodge=True,
                  palette=sns.color_palette(['darkseagreen','pink']),
                  )
    all_x_values = [path.get_offsets()[:, 0] for path in g.collections]
    all_y_values = [path.get_offsets()[:, 1] for path in g.collections]
    for ix in range(len(all_y_values)//2):
        ax.plot(all_x_values[2*ix:2*ix+2],all_y_values[2*ix:2*ix+2],
                linewidth=.5,color='grey',alpha=0.3)
    # sns.catplot(data=aldf, kind="point",x="Alpha", y="Value", hue="Key", 
    #             color='grey',alpha=0.3, ax=ax)
    
    sns.pointplot(
        data=aldf, x="Key", y="Value", hue="Alpha",
        dodge=.4, linestyle="none", errorbar=('pi',95), capsize=.2,
        palette=sns.color_palette(['black']),
        marker=None, markeredgewidth=1,
        err_kws={'linewidth':1},legend=False,ax=ax
    )
    sns.pointplot(
        data=aldf, x="Key", y="Value", hue="Alpha",
        dodge=.4, linestyle="none", errorbar=('se',1), capsize=.3,
        palette=sns.color_palette(['darkgreen','crimson']),
        marker='D', markersize=5, legend=False,ax=ax,
        err_kws={'linewidth':1},zorder=10
    )
    ax.set_ylim(0,1)
    # plt.show()
    # ax.set_xlim(0,160)
    positivity_bias=df[df['alpha+']>df['alpha-']]
    for k in np.unique(positivity_bias['Key']):
        print(f'{k}: {positivity_bias[positivity_bias["Key"]==k].count()}')
    # print(f'Number of reboots with positivity bias: {positivity_bias}')
    # negativity_bias=df[df['alpha+']<df['alpha-']].count()
    # print(f'Number of reboots with negativity bias: {negativity_bias}')
    
def difference_box(data,ax):
    ix=0
    df=pd.DataFrame(columns=['Key','Alpha'])
    for k in data.keys():
        # keys=[ix]*100
        ap=data[k]['params'][:,1]
        am=data[k]['params'][:,2]
        keys=[ix]*am.shape[0]
        dat={'Key':keys,'Alpha':ap-am}
        dat_ar=pd.DataFrame(data=dat)
        df=pd.concat((df,dat_ar))
        ix+=1
        
    sns.violinplot(data=df,x='Key',y='Alpha',ax=ax,palette=sns.color_palette(['grey']),inner=None,legend=False)
    sns.pointplot(
        data=df, x="Key", y="Alpha",
        dodge=0, linestyle="none", errorbar=('pi',95), capsize=.2,
        palette=sns.color_palette(['black']),
        marker=None, markeredgewidth=1,
        err_kws={'linewidth':1},legend=False,ax=ax
    )
    sns.pointplot(
        data=df, x="Key", y="Alpha",
        dodge=0, linestyle="none", errorbar=('se',1), capsize=.3,
        palette=sns.color_palette(['black']),
        marker='D', markersize=5, legend=False,ax=ax,
        err_kws={'linewidth':1}
    )
    ax.set_ylim(-1,1)
    
    
def beta_box(data,ax):
    ix=0
    df=pd.DataFrame(columns=['Key','beta','phi'])
    for k in data.keys():
        # keys=[ix]*100
        be=data[k]['params'][:,0]
        ph=data[k]['params'][:,4]
        keys=[ix]*ph.shape[0]
        dat={'Key':keys,r'beta':be,r'phi':ph}
        dat_ar=pd.DataFrame(data=dat)
        df=pd.concat((df,dat_ar))
        ix+=1
        
    beta_df=df[['beta','phi']].stack().reset_index()
    key_df=df[['Key','Key']].stack().reset_index(drop=True)
    pbdf=pd.concat((key_df,beta_df),axis=1)
    pbdf.columns=['Key','level_0','Parameter','Value']#pbdf=pbdf.rename(columns={'Key':'Key','level_1':'Parameter',0:'Value'})
    
    # sns.violinplot(data=pbdf,y='Value',x='Key',hue='Parameter',split=True,ax=ax,palette=sns.color_palette(['grey','skyblue']),inner=None)
    # sns.boxplot(data=aldf, x="Key", y="Value", hue="Alpha", ax=ax,palette=sns.color_palette(['darkseagreen','crimson']))
    g=sns.stripplot(data=pbdf,x="Key", y="Value", hue='Parameter',
                  jitter=True, ax = ax, alpha=0.3,dodge=True,
                  palette=sns.color_palette(['grey','skyblue'])
                  )
    all_x_values = [path.get_offsets()[:, 0] for path in g.collections]
    all_y_values = [path.get_offsets()[:, 1] for path in g.collections]
    for ix in range(len(all_y_values)//2):
        ax.plot(all_x_values[2*ix:2*ix+2],all_y_values[2*ix:2*ix+2],
                linewidth=.5,color='grey',alpha=0.3)
    sns.pointplot(
        data=pbdf, x="Key", y="Value", hue="Parameter",
        dodge=.4, linestyle="none", errorbar=('pi',95), capsize=.2,
        palette=sns.color_palette(['black']),
        marker=None, markeredgewidth=1,
        err_kws={'linewidth':1},legend=False,ax=ax
    )
    sns.pointplot(
        data=pbdf, x="Key", y="Value", hue="Parameter",
        dodge=.4, linestyle="none", errorbar=('se',1), capsize=.3,
        palette=sns.color_palette(['black','cornflowerblue']),
        marker='D', markersize=5, legend=False,ax=ax,
        err_kws={'linewidth':1},zorder=10
    )
    ax.hlines(0,0,2,linestyle='dotted',color='k')
    ax.set_ylim(-20,20)
    
    
def alpha_gen(data,k,ax,gen=None,ngens=200):
    
    if gen is None:
        pgen=data[k]['pgen']
        sgen=data[k]['sgen']
    else:
        pgen=data[k][gen]['pgen']
        sgen=data[k][gen]['sgen']
        
    gen_palette=sns.blend_palette([(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)], n_colors=ngens)#sns.color_palette('bwr',200)  [(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)]
    ax.plot(np.linspace(0,1,ngens),np.linspace(0,1,ngens),'k',linestyle='dashed',label='',linewidth=1)
    
    # g = sns.FacetGrid(data[k]['pgen'],row=data[k]['pgen'][2], col=data[k]['pgen'][1], palette=gen_palette)
    # g = (g.map(plt.scatter, "total_bill", "tip").add_legend())
    ax.scatter(pgen[2],pgen[1],color=gen_palette,s=20)
    ax.vlines(pgen[2,-1],pgen[1,-1]-sgen[1,-1],pgen[1,-1]+sgen[1,-1],color=gen_palette[-1]) #
    ax.hlines(pgen[1,-1],pgen[2,-1]-sgen[2,-1],pgen[2,-1]+sgen[2,-1],color=gen_palette[-1])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    # sns.despine(offset=10,trim=True)
    # find a way to put the legend of the first and last generation
    
def beta_gen(data,k,ax,gen=None,ngens=200):
    if gen is None:
        pgen=data[k]['pgen']
        sgen=data[k]['sgen']
    else:
        pgen=data[k][gen]['pgen']
        sgen=data[k][gen]['sgen']
        
    gen_palette=sns.blend_palette([(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)], n_colors=ngens)#sns.color_palette('bwr',200)  [(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)]
    ax.hlines(0,0,40,color='k',linestyle='dashed',linewidth=1)
    for i in range (len(pgen[1])):
        if i==0:
            label='1st gen'
        elif i==len(pgen[1])-1:
            label='last gen'
        else:
            label=''
        ax.scatter(pgen[0][i],pgen[4][i],color=gen_palette[i],label=label,s=20)
        
    ax.vlines(pgen[0,-1],pgen[4,-1]-sgen[4,-1],pgen[4,-1]+sgen[4,-1],color=gen_palette[-1])
    ax.hlines(pgen[4,-1],pgen[0,-1]-sgen[0,-1],pgen[0,-1]+sgen[0,-1],color=gen_palette[-1])
    ax.set_xlim(0,20)
    ax.set_ylim(-20,20)
    # ax.legend()
    
def fitness(data,k,ax,gen=None):
    if gen is None:
        cur_fit=data[k]['crit']
    else:
        cur_fit=data[k][gen]['crit']
    if gen!=None:
        ax.axvline(x=gen,color='k',linestyle="dotted",linewidth=1)
    # ax.axvline(x=100,color='k',linestyle="dotted")
    sns.lineplot(cur_fit[2,:],color='k',label="All",ax=ax) #zeros rather than 2?
    sns.lineplot(cur_fit[1,:],color='orange',linestyle="solid",label="Worst 5%",ax=ax,linewidth=2)
    sns.lineplot(cur_fit[3,:],color="teal",linestyle='solid',label="Best 5%",ax=ax,linewidth=2)
    # ax.set_xlim(0,1)
    
    
def fit_density(data,abl,ax,ngens=200,npop=100,gen=None):
    if gen is None:
        cur_fit=data[abl]['Act']
    else:
        cur_fit=data[abl][gen]['Act']
    # cur_fit=data[abl][gen]['Act']
    bined_dat,x,y=np.histogram2d(np.tile(np.arange(0,ngens),npop),cur_fit.flatten(),bins=[ngens-1,50])
    sns.heatmap(bined_dat.T, ax=ax,cmap='BuPu',vmax=npop/4)
    if gen!=None:
        ax.axvline(x=gen,color='k',linestyle="dotted")
    ax.invert_yaxis()
    return x,y
    # ax.set_title(f"Population density per accuracy across generations ({leg_type[i-1]})",fontsize=20)
    # 
    # plt.yticks(fontsize=16, rotation=0)
    # plt.xticks(rotation=0)
    # plt.xlabel('Generation',fontsize=20)
    # plt.ylabel('Accuracy',fontsize=20)
    
def params_correlation(data,k,gen,reb=None):
    if reb==None:
        arr = np.empty((100,5))
        for r in range(100):
            arr[r,:]=data[k][r]['pgen'][:,gen]
        arr = arr.T
    else:
        arr=np.squeeze(data[k][reb]['full'][:,:,gen])
    cov=np.corrcoef(arr)
    return cov

def maxi_correlation(data,k,gen,reb=None):
    if reb==None:
        arr=np.squeeze(np.mean(data[k]['full_params'],axis=1)[:,gen,:])
    elif type(reb)==int:
        arr=np.squeeze(data[k]['full_params'][:,:,gen,reb])
    else:
        arr=np.reshape(data[k]['full_params'][:,:,gen,:],[5,-1])
    cov=np.corrcoef(arr)
    return cov

def alpha_phi(data,k,ax=False,reb=None,plot_type='params'):
    if reb != None:
        params=data[k][reb][plot_type]
    else:
        params=data[k][plot_type]
    alpha=params[:,1]-params[:,2]
    alpha=alpha.reshape(-1,1)
    phi = params[:,4].reshape(-1,1)
    lr = LinearRegression()
    lr.fit(phi,alpha)
    preds=lr.predict(phi)
    if ax:
        ax.scatter(phi,alpha)
        ax.plot(phi,preds,color='k')
    # print(phi.shape)
    return (lr.coef_, lr.intercept_)
    
def alpha_phi_gen(data,k,ax,gen=None,ngens=200):
    if gen is None:
        pgen=data[k]['pgen']
    else:
        pgen=data[k][gen]['pgen']
        
    gen_palette=sns.blend_palette([(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)], n_colors=ngens)#sns.color_palette('bwr',200)  [(45/255,5/255,176/255),(69/255,2/255,144/255),(187/255,31/255,79/255),(221/255,17/255,40/255)]
    ax.hlines(0,-20,20,color='k',linestyle='dotted')
    ax.vlines(0,-1,1,color='k',linestyle='dotted')
    for i in range (len(pgen[1])):
        ax.scatter(pgen[4][i],pgen[1][i]-pgen[2][i],color=gen_palette[i])
        
    ax.set_xlim(-20,20)
    ax.set_ylim(-1,1)
    ax.legend()

def plot_regressions(data,k,ax,plot_type='params'):
    coef,inter=[],[]
    x=np.linspace(-20,20,200)
    for reb in data[k].keys():
        c,i=alpha_phi(data,k,reb=reb,plot_type=plot_type)
        coef.append(c)
        inter.append(i)
        preds=c*x+i
        ax.plot(x,preds.T,color='grey')
    ax.set_ylim(-1,1)
    ax.set_xlim(-20,20)
    return (coef,inter)

#%% Initialization plot
# fig,ax=plt.subplots(3,5,figsize=(12,10), subplot_kw=dict(box_aspect=1))
# plt.subplots_adjust(hspace=0.4,wspace=.2)
# ix=0

# for k in SP.keys():
#     if ix==1 or ix==2:
#         ix=3-ix
#     fitness(SP,k,ax[0,ix])
#     alpha_gen(SP,k,ax[1,ix])
#     beta_gen(SP,k,ax[2,ix])
#     # if len(k)>5:
#     #     ax[0,ix].set_title(k)
#     # else:
#     #     ax[0,ix].set_title(fr'$\beta=${k[0]},' '\n' fr' $\alpha⁺=${k[1]}, $\alpha⁻=${k[2]},' '\n' fr'$\tau=${k[3]}, $\phi=${k[4]}')#, Init {ix+1}
#     if ix==1 or ix==2:
#         ix=3-ix
#     ix+=1
    
# #invert column order for better readability
# # ax1=ax[:,1]
# # ax2=ax[:,2]

# # ax[0,1]=ax[0,2]
# # ax[:,2]=ax1

# ax[0,0].get_legend().remove()
# ax[0,1].get_legend().remove()
# ax[0,2].get_legend().remove()
# ax[0,3].get_legend().remove()
# ax[0,4].get_legend().remove()
# # ax[2,0].get_legend().remove()
# # ax[2,1].get_legend().remove()
# # ax[2,2].get_legend().remove()
# # ax[2,3].get_legend().remove()


# # box = ax[0,4].get_position()
# # ax[0,4].legend(loc='upper center', bbox_to_anchor=(1.5, 0.8),
# #           fancybox=True, ncol=1)

# # box = ax[2,4].get_position()
# # ax[2,4].legend(loc='upper center', bbox_to_anchor=(1.5, 1.3),
# #           fancybox=True, ncol=1)

# ax[0,1].set_yticks([])
# ax[0,2].set_yticks([])
# ax[0,3].set_yticks([])
# ax[0,4].set_yticks([])
# ax[1,1].set_yticks([])
# ax[1,2].set_yticks([])
# ax[1,3].set_yticks([])
# ax[1,4].set_yticks([])
# ax[2,1].set_yticks([])
# ax[2,2].set_yticks([])
# ax[2,3].set_yticks([])
# ax[2,4].set_yticks([])

# # ax[1,0].set_xticks([])
# # ax[1,1].set_xticks([])
# # ax[1,2].set_xticks([])

# # ax[1,0].set_xlabel(None)
# # ax[1,1].set_xlabel(None)
# # ax[1,2].set_xlabel(r'$\alpha-$')
# # ax[2,0].set_xlabel(None)
# # ax[2,1].set_xlabel(None)
# # ax[2,2].set_xlabel(r'$\beta$')

# # ax[0,0].set_ylabel('Accuracy')
# # ax[1,0].set_ylabel(r'$\alpha+$')
# # ax[1,1].set_ylabel(None)
# # ax[1,2].set_ylabel(None)
# # ax[2,0].set_ylabel(r'$\phi$')
# # ax[2,1].set_ylabel(None)
# # ax[2,2].set_ylabel(None)

# for l in range(5):
#     ax[1,l].set_xticks([0,0.5,1],[]) #xaxis.set_major_formatter(FormatStrFormatter('%d'))0,0.5,1
#     ax[2,l].set_xticks([0,10,20],[])
#     ax[0,l].set_xticks([0,100,200],[])
    
# ax[1,0].set_yticks([0,0.5,1],[])
# ax[2,0].set_yticks([-20,0,20],[])
# ax[0,0].set_yticks([0.4,0.6,0.8,1],[])
# # ax[0,0].set_title('Difficulty')
# # ax[0,1].set_title('Richness')
# # ax[0,2].set_title('Learning period')

# sns.despine(offset=10,trim=True)

# sns.despine(ax=ax[0,1],left=True)
# sns.despine(ax=ax[0,2],left=True)
# sns.despine(ax=ax[0,3],left=True)
# sns.despine(ax=ax[0,4],left=True)
# # sns.despine(ax=ax[1,0],bottom=True)
# sns.despine(ax=ax[1,1],left=True)
# sns.despine(ax=ax[1,2],left=True)
# sns.despine(ax=ax[1,3],left=True)
# sns.despine(ax=ax[1,4],left=True)
# sns.despine(ax=ax[2,1],left=True)
# sns.despine(ax=ax[2,2],left=True)
# sns.despine(ax=ax[2,3],left=True)
# sns.despine(ax=ax[2,4],left=True)

# fig.savefig('./figures/v2/init_plot.png',dpi=300,bbox_inches='tight')
# fig.savefig('./figures/v2/init_plot_blank.pdf',dpi=300,bbox_inches='tight')
# fig.savefig('./figures/v2/init_plot_blank.svg',dpi=300,bbox_inches='tight')

#%% Factors plot
sns.set_palette('flare',3)
fig,ax=plt.subplots(3,3,figsize=(12,10))
plt.subplots_adjust(hspace=0.4,wspace=.2)

learningCurves(difficulty,ax[0,0])
learningCurves(rich_poor,ax[0,1])
learningCurves(learning,ax[0,2])

alpha_box(difficulty,ax[1,0])
alpha_box(rich_poor,ax[1,1])
alpha_box(learning,ax[1,2])

beta_box(difficulty,ax[2,0])
beta_box(rich_poor,ax[2,1])
beta_box(learning,ax[2,2])

# box = ax[2,2].get_position()
# ax[2,2].legend(loc='upper center', bbox_to_anchor=(0.5, 0.3),
#           fancybox=True, ncol=5)

# plt.show()
# box = ax[1,2].get_position()
# ax[1,2].legend(loc='upper center', bbox_to_anchor=(0.5, -.15), # on static: (0.5, 1)
#           fancybox=True, ncol=5)

# box = ax[0,2].get_position()
# ax[0,2].legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
#           fancybox=True, ncol=2)

# box = ax[0,1].get_position()
# ax[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
#           fancybox=True, ncol=2)

# box = ax[0,0].get_position()
# ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
#           fancybox=True, ncol=2)

ax[0,0].get_legend().remove()
ax[0,1].get_legend().remove()
ax[0,2].get_legend().remove()
ax[1,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[1,2].get_legend().remove()
ax[2,0].get_legend().remove()
ax[2,1].get_legend().remove()
ax[2,2].get_legend().remove()

ax[0,0].set_yticks([0.4,0.6,0.8,1],[])
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,0].set_yticks([0,0.5,1],[])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
ax[2,0].set_yticks([-20,-10,0,10,20],[])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])

ax[1,0].set_xticks([])
ax[1,1].set_xticks([])
ax[1,2].set_xticks([])

# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in difficulty.keys()]
ax[2,0].set_xticks(np.arange(0,len(difficulty.keys())),[]) #,labels=ticks
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in rich_poor.keys()]
ax[2,1].set_xticks(np.arange(0,len(rich_poor.keys())),[])
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in learning.keys()]
ax[2,2].set_xticks(np.arange(0,len(learning.keys())),[])

for l in range(3):
    ax[0,l].set_xticks([0,40,80,120,160],[])

ax[1,0].set_xlabel(None)
ax[1,1].set_xlabel(None)
ax[1,2].set_xlabel(None)
ax[2,0].set_xlabel(None)
ax[2,1].set_xlabel(None)
ax[2,2].set_xlabel(None)

ax[0,0].set_ylabel(None)
ax[1,0].set_ylabel(None)
ax[2,0].set_ylabel(None)
ax[1,1].set_ylabel(None)
ax[1,2].set_ylabel(None)
ax[2,1].set_ylabel(None)
ax[2,2].set_ylabel(None)

# ax[0,0].set_title('Difficulty')
# ax[0,1].set_title('Richness')
# ax[0,2].set_title('Learning period')

sns.despine(offset=10,trim=True)

sns.despine(ax=ax[0,1],left=True)
sns.despine(ax=ax[0,2],left=True)
sns.despine(ax=ax[1,0],bottom=True)
sns.despine(ax=ax[1,1],left=True,bottom=True)
sns.despine(ax=ax[1,2],left=True,bottom=True)
sns.despine(ax=ax[2,1],left=True)
sns.despine(ax=ax[2,2],left=True)


# fig.savefig(f'./figures/{init}/factors_plot.png',dpi=300,bbox_inches='tight')

# fig.savefig(f'./figures/v2/{init}_factors_strip_blank.pdf',dpi=300,bbox_inches='tight')
# fig.savefig(f'./figures/v2/{init}_factors_strip_blank.svg',dpi=300,bbox_inches='tight')

#%% volatility plot
sns.set_palette('flare',3)
fig,ax=plt.subplots(3,3,figsize=(12,10))
plt.subplots_adjust(hspace=0.4,wspace=.2)


learningCurves(reversal,ax[0,0])
learningCurves(gaussian,ax[0,1])
learningCurves(rdwalk,ax[0,2])

alpha_box(reversal,ax[1,0])
alpha_box(gaussian,ax[1,1])
alpha_box(rdwalk,ax[1,2])

beta_box(reversal,ax[2,0])
beta_box(gaussian,ax[2,1])
beta_box(rdwalk,ax[2,2])

# box = ax[2,2].get_position()
# ax[2,2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), #on static: (0.5, 1.2)
#           fancybox=False, ncol=5) #, frameon=False

# plt.show()
# if init=='static':
#     ax[1,1].get_legend().remove()
#     box = ax[1,2].get_position()
#     ax[1,2].legend(loc='upper center', bbox_to_anchor=(0.5, 1), #on static:(0.5, 1)
#               fancybox=True, ncol=5)
# else:
#     ax[1,2].get_legend().remove()
#     box = ax[1,1].get_position()
#     ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), #on static:(0.5, 1)
#               fancybox=True, ncol=5)

# box = ax[0,2].get_position()
# ax[0,2].legend(
#           fancybox=True, ncol=2,loc='upper center', bbox_to_anchor=(0.5, 0.35)) #,, bbox_to_anchor=(1, 0.2)

# box = ax[0,1].get_position()
# ax[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
#           fancybox=True, ncol=2)

# box = ax[0,0].get_position()
# ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
#           fancybox=True, ncol=2)

ax[0,0].get_legend().remove()
ax[0,1].get_legend().remove()
ax[0,2].get_legend().remove()
ax[1,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[1,2].get_legend().remove()
ax[2,0].get_legend().remove()
ax[2,1].get_legend().remove()
ax[2,2].get_legend().remove()

ax[0,0].set_yticks([0.4,0.6,0.8,1],[])
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,0].set_yticks([0,0.5,1],[])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
ax[2,0].set_yticks([-20,-10,0,10,20],[])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])

ax[1,0].set_xticks([])
ax[1,1].set_xticks([])
ax[1,2].set_xticks([])

# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in difficulty.keys()]
ax[2,0].set_xticks(np.arange(0,len(difficulty.keys())),[]) #,labels=ticks
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in rich_poor.keys()]
ax[2,1].set_xticks(np.arange(0,len(rich_poor.keys())),[])
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in learning.keys()]
ax[2,2].set_xticks(np.arange(0,len(learning.keys())),[])

for l in range(3):
    ax[0,l].set_xticks([0,40,80,120,160],[])

ax[1,0].set_xlabel(None)
ax[1,1].set_xlabel(None)
ax[1,2].set_xlabel(None)
ax[2,0].set_xlabel(None)
ax[2,1].set_xlabel(None)
ax[2,2].set_xlabel(None)

ax[0,0].set_ylabel(None)
ax[1,0].set_ylabel(None)
ax[2,0].set_ylabel(None)
ax[1,1].set_ylabel(None)
ax[1,2].set_ylabel(None)
ax[2,1].set_ylabel(None)
ax[2,2].set_ylabel(None)


sns.despine(offset=10,trim=True)

sns.despine(ax=ax[0,1],left=True)
sns.despine(ax=ax[0,2],left=True)
sns.despine(ax=ax[1,0],bottom=True)
sns.despine(ax=ax[1,1],left=True,bottom=True)
sns.despine(ax=ax[1,2],left=True,bottom=True)
sns.despine(ax=ax[2,1],left=True)
sns.despine(ax=ax[2,2],left=True)

# fig.savefig(f'./figures/{init}/volatility_plot.png',dpi=300,bbox_inches='tight')

# fig.savefig(f'./figures/v2/{init}_volatility_strip_blank.pdf',dpi=300,bbox_inches='tight')
# fig.savefig(f'./figures/v2/{init}_volatility_strip_blank.svg',dpi=300,bbox_inches='tight')

#%% plot of the differences
# fig,ax=plt.subplots(1,3,figsize=(12,10))
# plt.subplots_adjust(hspace=0.4,wspace=.2)

# difference_box(reversal,ax[0])
# difference_box(gaussian,ax[1])
# difference_box(rdwalk,ax[2])

# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in reversal.keys()]
# ax[0].set_xticks(ticks=np.arange(0,len(reversal.keys())),labels=ticks)
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in gaussian.keys()]
# ax[1].set_xticks(ticks=np.arange(0,len(gaussian.keys())),labels=ticks)
# ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in rdwalk.keys()]
# ax[2].set_xticks(ticks=np.arange(0,len(rdwalk.keys())),labels=ticks)

# ax[0].set_title('Fixed reversal')
# ax[1].set_title('Gaussian')
# ax[2].set_title('Uniform')

# ax[0].set_ylabel(r'$\alpha⁺-\alpha⁻$')
# ax[1].set_ylabel(None)
# ax[2].set_ylabel(None)

# ax[0].set_xlabel(None)
# ax[1].set_xlabel(None)
# ax[2].set_xlabel(None)

# ax[1].set_yticks([])
# ax[2].set_yticks([])

# fig.savefig(f'./figures/{init}/volatility_alpha_diff.png',dpi=300,bbox_inches='tight')


#%% maxiplot
versions=['stable','volatile']
for v in versions:
    maxi=pd.read_pickle(f'../results/reboot/Maxi_results_{init}_{v}.pkl')
    key=0#'uniform'#(0,0,0,0,0)
    fig,ax=plt.subplots(2,2,figsize=(12,10))
    plt.subplots_adjust(hspace=0.4,wspace=.4)
    
    learningCurves(maxi,ax[0,0])#k=key
    fitness(maxi,key,ax[0,1])
    alpha_gen(maxi,key,ax[1,0])
    beta_gen(maxi,key,ax[1,1])
    ax[0,0].set_xlim(0,1500)
    # ax[1,0].set_xlabel(None)
    # ax[1,1].set_xlabel(None)
    ax[0,0].get_legend().remove()
    ax[0,1].get_legend().remove()
    
    ax[1,0].set_xticks([0,0.5,1],[]) #xaxis.set_major_formatter(FormatStrFormatter('%d'))0,0.5,1
    ax[1,1].set_xticks([0,10,20],[])
    ax[0,0].set_xticks([0,500,1000,1500],[])
    ax[0,1].set_xticks([0,100,200],[])
        # for s in range(3):
        #     sns.despine(ax=ax[s,l],left=True)
        
    ax[1,0].set_yticks([0,0.5,1],[])
    ax[1,1].set_yticks([-20,0,20],[])
    ax[0,1].set_yticks([0.4,0.6,0.8,1],[])
    if v=='volatile':
        ax[0,0].set_yticks([0,0.5,1],[])
        ax[0,0].set_ylim(0,1)
    else:
        ax[0,0].set_yticks([0.5,0.75,1],[])
        ax[0,0].set_ylim(0.5,1)
    ax[1,0].set_xlabel(None)
    ax[1,1].set_xlabel(None)
    # ax[1,2].set_xlabel(None)
    # ax[2,0].set_xlabel(None)
    # ax[2,1].set_xlabel(None)
    # ax[2,2].set_xlabel(None)
    
    sns.despine(offset=10,trim=True)
    
    # fig.savefig('./figures/uniform/maxi_volatile.png',dpi=300,bbox_inches='tight')
    
    # fig.savefig(f'./figures/v2/Maxi_{init}_{v}_blank.pdf',dpi=300,bbox_inches='tight')
    # fig.savefig(f'./figures/v2/Maxi_{init}_{v}_blank.svg',dpi=300,bbox_inches='tight')
    
#%% ablation plots for maxi experiment 
versions=['stable','volatile'] #'uniform',
inits=['static']
for init in inits:
    for v in versions:
        maxi=pd.read_pickle(f'../results/reboot/Maxi_inversion_{v}_{init}.pkl')
        no_abl_max={}
        no_abl_max[0]=maxi.pop(0)
        for gen in maxi[2].keys():
            ix=1
            fig,ax=plt.subplots(3,4,figsize=(13,10))
            plt.subplots_adjust(hspace=0.4,wspace=.2)
            fitness(no_abl_max,0,ax[0,0])
            # x,y=fit_density(no_abl, 0, ax[1,0])
            alpha_gen(no_abl_max, 0, ax[1,0])
            beta_gen(no_abl_max,0,ax[2,0])
            for ab in maxi.keys():
                fitness(maxi,ab,ax[0,ix],gen=gen)
                # x,y=fit_density(ablation, ab, ax[1,ix],gen=gen)
                alpha_gen(maxi, ab, ax[1,ix],gen=gen)
                beta_gen(maxi,ab,ax[2,ix],gen=gen)
                ix+=1
            
            ax[0,0].get_legend().remove()
            ax[0,1].get_legend().remove()
            ax[0,2].get_legend().remove()
            ax[0,3].get_legend().remove()
            # ax[3,0].get_legend().remove()
            # ax[3,1].get_legend().remove()
            # ax[3,2].get_legend().remove()
            
            # box = ax[0,3].get_position()
            # ax[0,3].legend(loc='upper center', bbox_to_anchor=(1.5, 0.8),
            #           fancybox=True, ncol=1)
        
            # box = ax[3,3].get_position()
            # ax[3,3].legend(loc='upper center', bbox_to_anchor=(1.5, 1.3),
            #           fancybox=True, ncol=1)
            
            for l in range(4):
                ax[1,l].set_xticks([0,0.5,1],[]) #xaxis.set_major_formatter(FormatStrFormatter('%d'))0,0.5,1
                ax[2,l].set_xticks([0,10,20],[])
                ax[0,l].set_xticks([0,100,200],[])
                # for s in range(3):
                #     sns.despine(ax=ax[s,l],left=True)
                
            ax[1,0].set_yticks([0,0.5,1],[])
            ax[2,0].set_yticks([-20,0,20],[])
            # if v=='stable':
            for k in range(4):
                ax[0,k].set_ylim(0.4,1)
                ax[0,k].set_yticks([0.4,0.6,0.8,1],[])
            # else:
            #     for k in range(4):
            #         ax[0,k].set_ylim(0,1)
            #         ax[0,k].set_yticks([0,0.5,1],[])
        
                    
            # x+=1
            # ax[0,1].set_yticks([])
            # ax[0,2].set_yticks([])
            # ax[0,3].set_yticks([])
            # ax[1,1].set_yticks([])
            # ax[1,2].set_yticks([])
            # ax[1,3].set_yticks([])
            # ax[2,1].set_yticks([])
            # ax[2,2].set_yticks([])
            # ax[2,3].set_yticks([])
            
            # ax[1,0].set_xticks([])
            # ax[1,1].set_xticks([])
            # ax[1,2].set_xticks([])
            # ax[1,0].set_yticks(ticks=np.arange(0,50,10),labels=np.around(y[0:50:10],1), fontsize=16)
            # for ixy in range(4):
            #     ax[1,ixy].set_xticks(ticks=[0,50,100,150,200],labels=[0,50,100,150,200])
            
            # ax[0,0].set_title('No ablation')
            # ax[0,1].set_title(r'$\phi=0$')
            # ax[0,2].set_title(r'$\alpha⁺=\alpha⁻$')
            # ax[0,3].set_title('Unbiased')
            
            # ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in reversal.keys()]
            # ax[2,0].set_xticks(ticks=np.arange(0,len(reversal.keys())),labels=ticks)
            # ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in gaussian.keys()]
            # ax[2,1].set_xticks(ticks=np.arange(0,len(gaussian.keys())),labels=ticks)
            # ticks=[f'{[k[i] for i in range(len(k))] if type(k)!=str else k}' for k in rdwalk.keys()]
            # ax[2,2].set_xticks(ticks=np.arange(0,len(rdwalk.keys())),labels=ticks)
            
            ax[1,0].set_xlabel(None)
            ax[1,1].set_xlabel(None)
            ax[1,2].set_xlabel(None)
            ax[2,0].set_xlabel(None)
            ax[2,1].set_xlabel(None)
            ax[2,2].set_xlabel(None)
            
            # ax[0,0].set_ylabel('Accuracy')
            # ax[1,0].set_ylabel('Accuracy')
            # ax[2,0].set_ylabel(r'$\alpha⁺$')
            # ax[3,0].set_ylabel(r'$\phi$')
            
            
            # fig.delaxes(fig.axes[16])
            # fig.delaxes(fig.axes[16])
            # fig.delaxes(fig.axes[16])
            # fig.delaxes(fig.axes[17])
            # fig.delaxes(fig.axes[18])
        
            sns.despine(offset=10,trim=True)
            
            for ixx in range(3):
                for ixy in range(1,4):
                    ax[ixx,ixy].set_yticks([])
                    sns.despine(ax=ax[ixx,ixy],left=True)
                    
            # fig.savefig(f'./figures/v2/Maxi_inversion_{init}_{v}_{gen}_blank.pdf',dpi=300,bbox_inches='tight')
            # fig.savefig(f'./figures/v2/Maxi_inversion_{init}_{v}_{gen}_blank.svg',dpi=300,bbox_inches='tight')
        
