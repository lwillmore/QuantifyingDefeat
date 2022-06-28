import scipy
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import pandas as pd
from scipy.stats import levene, ttest_ind, ttest_rel
from sklearn.linear_model import LinearRegression

ggp=np.array([[0.133     , 0.133     , 0.133     , 1.        ],
       [0.22953434, 0.57685998, 0.42976558, 1.        ],
       [0.57609486, 0.42953354, 0.90002989, 1.        ]])

def abline(slope, intercept,ax):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--',color='k',alpha=0.5)

    
def make_behavior_plots(dat,height=1,aspect=3,offset=0.2,titles='{row_name}',ylabel='',save_names='',
                       pal2=[(ggp[1][0],ggp[1][1],ggp[1][2]),
       (ggp[2][0],ggp[2][1],ggp[2][2])],
                       order=None,norm=1,err_style='band',single_test=False,sharey=False):

    if order==None:
        order=np.unique(dat.c)
    
    sns.set_context("paper")
    sns.set_style("ticks",{
        'legend.frameon':False
    })
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    # barplots for each mouse
    plt.figure(figsize=(10,10))
    if titles=='{row_name}':
        titles_tmp='{col_name}'
    else:
        titles_tmp=titles
    g=sns.relplot(data=dat,x='d',y='t',hue='rb',col='c',kind='line',legend=False,
               height=height, aspect=1/aspect,ci=68, palette=pal2,col_order=order,
                  err_style=err_style,
               facet_kws={'sharey': sharey, 'sharex': True}).set_titles(titles_tmp).set_ylabels('')
    g.fig.subplots_adjust(hspace=offset)
    for a,b in zip(g.axes[0],order):
        a.set_xlabel('Defeat Day')
        a.set_xticks([2,4,6,8,10])

        print(len(dat.query('c=="%s"'%b)))
        ps=[]
        for d in range(1,11):
            r=dat.query('c=="%s" and d==%d and rb'%(b,d)).t
            s=dat.query('c=="%s" and d==%d and not rb'%(b,d)).t
            print(len(r),len(s))
            t,p=scipy.stats.ttest_ind(r,s)

            print(b,d,t,p)
            ps.append(p)
#             if p<0.05:
#                 ymin,ymax=a.get_ylim()
#                 yspan=ymax-ymin
#                 a.text(d,ymax-0.04*yspan,'*')
        
        stars,ps2,_,_ = multipletests(ps,alpha=0.05,method='fdr_bh')
        print (b,stars)
        
        md = smf.gee("t ~ rb*d", data=dat.query('c=="%s"'%b), groups=dat.query('c=="%s"'%b)["m"])
        mdf = md.fit()
        print(mdf.summary())
        md = smf.gee("t ~ si*d", data=dat.query('c=="%s"'%b), groups=dat.query('c=="%s"'%b)["m"])
        mdf = md.fit()
        print(mdf.summary())
        
    g.fig.text(0.0, 0.5, ylabel, 
           va='center', 
           rotation='vertical')
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.23)
    if len(save_names):
        plt.savefig(save_names+'_over_time.png',dpi=300,transparent=True,format='pdf')
    plt.show()
    
    FSIZE=g.fig.get_size_inches()
    
    
    # scatter SI ON HORIZONTAL per mouse SI vs 
    fig,axs=plt.subplots(ncols=len(order),nrows=1,figsize=FSIZE,
                        sharex=False,sharey=sharey)
    meandicts=[]
    for ax,rowtitle,title in zip(axs,order,titles):
        if 'sex' in dat.columns:
            meandict={'si':[],'t':[],'rb':[],'mins':[],'maxs':[],'m':[],'sex':[]}
            for (m,sex),g in dat.query('c=="%s"'%rowtitle).groupby(['m','sex']):
                meandict['si'].append(g.si.values[0])
                meandict['t'].append(np.median(g.t.values))
                meandict['rb'].append(g.rb.values[0])
                meandict['mins'].append(np.percentile(g.t.values,30))
                meandict['maxs'].append(np.percentile(g.t.values,70))
                meandict['m'].append(m)
                if 'sex' in g.columns:
                    meandict['sex'].append(g.sex.values[0])  
        else:
            meandict={'si':[],'t':[],'rb':[],'mins':[],'maxs':[],'m':[]}
            for m,g in dat.query('c=="%s"'%rowtitle).groupby('m'):
                meandict['si'].append(g.si.values[0])
                meandict['t'].append(np.median(g.t.values))
                meandict['rb'].append(g.rb.values[0])
                meandict['mins'].append(np.percentile(g.t.values,30))
                meandict['maxs'].append(np.percentile(g.t.values,70))
                meandict['m'].append(m)

        meandictdf=pd.DataFrame(meandict)
        meandictdf['c']=rowtitle
        meandicts.append(meandictdf)
        
        ax.vlines(x=meandictdf.query('rb==1').si,
                  ymin=meandictdf.query('rb==1').mins,
                  ymax=meandictdf.query('rb==1').maxs,color=pal2[1],alpha=0.5)
        ax.vlines(x=meandictdf.query('rb==0').si,
                  ymin=meandictdf.query('rb==0').mins,
                  ymax=meandictdf.query('rb==0').maxs,color=pal2[0],alpha=0.5)
        
        
        if 'sex' in meandictdf.columns:
            sns.scatterplot(data=meandictdf.query('sex=="male"'),y='t',x='si',hue='rb',#err_types='bars'
                          legend=False,s=20,ci=68, palette=pal2,ax=ax,alpha=0.8,zorder=500)
            sns.scatterplot(data=meandictdf.query('sex=="male"'),y='t',x='si',hue='rb',#err_types='bars'
                          legend=False,s=20,ci=68, palette=pal2,ax=ax,alpha=0.8,zorder=500)
            sns.scatterplot(data=meandictdf.query('sex=="female"'),y='t',x='si',hue='rb',#err_types='bars'
                          legend=False,s=20,ci=68, palette=pal2,ax=ax,alpha=0.8,zorder=500,marker='^')
            sns.scatterplot(data=meandictdf.query('sex=="female"'),y='t',x='si',hue='rb',#err_types='bars'
                          legend=False,s=20,ci=68, palette=pal2,ax=ax,alpha=0.8,zorder=500,marker='^')
        
        else:
            sns.scatterplot(data=meandictdf,y='t',x='si',hue='rb',#err_types='bars'
                      legend=False,s=20,ci=68, palette=pal2,ax=ax,alpha=0.8,zorder=500)
        
          
        if titles=='{row_name}':
            ax.set_title(rowtitle)
        else:
            ax.set_title(title)
        ax.set_ylabel('')
    
        a=ax
        test_df=dat.query('c == "%s"'%rowtitle)
        test_df=test_df[test_df.notnull().values.sum(axis=1)==test_df.shape[1]]
        if single_test:
            test_df=test_df[['si','t']].drop_duplicates()
            pval=scipy.stats.pearsonr(test_df['si'],test_df['t'])[1]
            lr = LinearRegression()
            lr.fit(test_df['si'].values.reshape(-1, 1),test_df['t'].values,)

            abline(lr.coef_, lr.intercept_,a)
            y_vals = np.array(a.get_ylim())
            x_vals = np.array(a.get_xlim())
            if pval<0.001:
                ax.set_title(rowtitle+'***'%pval)
            elif pval<0.01:
                ax.set_title(rowtitle+'**')
            elif pval<0.05:
                ax.set_title(rowtitle+'*'%pval)
            print(scipy.stats.pearsonr(test_df['si'],test_df['t']))
                
        else:
            md = smf.gee(formula="t ~ si", data=test_df,
                        groups=test_df['m'])
            res = md.fit()
            print(res.summary())
            pval=res.pvalues[1]
            abline(res.params[1],res.params[0],a)
            y_vals = np.array(a.get_ylim())
            x_vals = np.array(a.get_xlim())
            if pval<0.001:
                ax.set_title(rowtitle+' ***'%pval)
            elif pval<0.01:
                ax.set_title(rowtitle+' **')
            elif pval<0.05:
                ax.set_title(rowtitle+' *'%pval)
        a.set_xlabel(r'S $\leftarrow$ SI Time (%) $\rightarrow$ R')
    
    fig.subplots_adjust(hspace=offset)
    fig.text(0.0, 0.5, ylabel, 
           va='center', 
           rotation='vertical')
    sns.despine()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.23)
    if len(save_names):
        plt.savefig(save_names+'_scatter_six.png',dpi=300,transparent=True,format='pdf',bbox_inches='tight')
    plt.show()
    


    meandictsall=pd.concat(meandicts)

    return meandictsall