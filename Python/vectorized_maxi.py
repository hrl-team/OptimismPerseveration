#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:19:55 2024
Vectorized maxi experiment
Pick which tasks to run, combine them as you please
@author: isabelle
"""
import numpy as np
from math import floor
from joblib import Parallel, delayed
import argparse
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats


#%% utility functions
def Bandit_Task(a, rev, ss, chancemin,chancemax):
    nsub = len(a)
    rand = np.random.uniform(size = nsub)

    if rev == 0:
        r = (((chancemin > rand) - .5) * 2) *(1 - a)  + (((chancemax > rand) - .5) * 2) * a #

    elif rev==1:
        r = (
            (((chancemin > rand) - .5) * 2) * (1 - a) + (((chancemax > rand) - .5) * 2) * a
        ) * (ss%2) + (
            (((chancemin > rand) - .5) * 2) * a + (((chancemax > rand) - .5) * 2) * (1 - a)
        ) * (1 - ss%2)
    return r


def constrain_matrix(A, lower,upper):
    if np.isscalar(lower):
        lower = np.repeat(lower, A.shape[1],axis=0)
    if np.isscalar(upper):
        upper = np.repeat(upper, A.shape[1],axis=0)
    for i in range (A.shape[1]):
        A[A[:,i]<lower[i],i] = lower[i]
        A[A[:,i]>upper[i],i] = upper[i]
    return A

def Random_Walk(repetitions,ntrial):
    states=np.zeros((repetitions*ntrial,1),dtype=int)
    stateval=1
    for i in range(repetitions*ntrial):
        states[i]=stateval
        if np.random.uniform()<=1/ntrial:
            stateval=stateval+1
    return states

def Gaussian_Random(repetitions,ntrial):
    states=np.zeros((repetitions*ntrial,1),dtype=int)
    randvol=0
    for  i in range(repetitions):
        randvoltemp=floor(randvol)
        randvol=np.random.normal(loc=ntrial*i,scale=ntrial/4)
        while randvol>repetitions*ntrial-repetitions+i or randvol<=randvoltemp:
            randvol=np.random.normal(loc=ntrial*i,scale=ntrial/4)
        randvol=floor(randvol)
        for k in range(randvoltemp,randvol):
            states[k]=i
        for k in range(randvol,repetitions*ntrial):
            states[k]=repetitions
    return states

#%% simulation class
class Simulation():
    #here we should list all the parameters
    def __init__(self,nsub,gens,ntrial=20,repetitions=8,task=np.arange(15),time=0,
                 gen_abl=200,ablation=0,params=[0,0,0,0,0],mut_rate=5,replac=50): 
        self.ablation=ablation
        self.time=time
        self.nsub=nsub
        self.gens=gens
        self.ntrial=ntrial
        self.repetitions=repetitions
        self.replac=replac
        self.gen_ablation=gen_abl
        self.params=params
        self.mut_rate=mut_rate
        self.task=task
        
    def get_params(self,n):
        '''The tasks are numbered 0-15. Get the corresponding set of params'''
        #get the kth set of parameters
        k=self.task[n]
        if k<6 or k==15:
            rev=0
        else:
            rev=1
        if k>=3:
            chancemin=0.25
            chancemax=0.75
        elif k==0:
            chancemin,chancemax=0.45,0.55
        elif k==1:
            chancemin,chancemax=0.45,0.95
        elif k==2:
            chancemin,chancemax=0.05,0.95
        if k<9 or k==15:
            distrib=0
        elif k<12:
            distrib=1
        else:
            distrib=2
        if (k<4 or k==7) or k==10 or k==13 or k==15:
            ntrial=self.ntrial
            repetitions=self.repetitions
        elif k==4 or k==6 or k==9 or k==12:
            ntrial,repetitions=80,2
        elif k==5 or k==8 or k==11 or k==14:
            ntrial,repetitions=5,32
        if k==15: #forgot to include that one
            chancemin,chancemax=0.05,0.55
        return ntrial,repetitions,chancemin,chancemax,distrib,rev
            
    def generate_states(self):
        ix=0
        for n in range(len(self.task)):
            ntrial,repetitions,chancemin,chancemax,distrib,rev=self.get_params(n)
            if distrib==0:
                states=[]
                for i in range(repetitions):
                    states.extend(np.repeat([i],ntrial))
                states=np.array(states)
            elif distrib==1:
                states=np.squeeze(Gaussian_Random(repetitions, ntrial))
            elif distrib==2:
                states=np.squeeze(Random_Walk(repetitions, ntrial))
            if ix==0:
                s=states
            else:
                states+=max(s)+1
                s=np.concatenate((s,states))
            ix+=1
        return s
        


    
    def EvolvingHysteresis_function_within_mutation(self,uniform=True,mutation=False,shuffle=False,inversion=False):
        lenstate=int(self.repetitions*self.ntrial*len(self.task))
        #generate initial population
        lower=[0,0,0,0,-20] #initial parameters?
        upper=[20,1,1,1,20]
        params=np.zeros((self.nsub,5))
        
        if uniform:
            params=np.random.uniform(size=params.shape)
            params[:,0]=params[:,0]*20
            params[:,4]=(params[:,4]-0.5)*40
        else:
            params[:,:]=self.params
            
        #pre allocate space
        variability=np.zeros((5,self.gens))
        pgen=np.zeros((5,self.gens))
        sgen=np.zeros((5,self.gens))
        crit=np.zeros((4,self.gens))
        # full_params=np.zeros((5,self.nsub,self.gens))
        
        Qval=np.zeros((self.nsub,lenstate,2))
        Cval=np.zeros((self.nsub,lenstate,2))
        Pro=np.zeros((self.nsub,self.gens))
        Act=np.zeros((self.nsub,self.gens))
        Cho=np.zeros((self.nsub,lenstate))
        
        a=np.zeros((self.nsub,lenstate),dtype=int)
        r=np.zeros((self.nsub,lenstate))
        
        acc=np.zeros((self.nsub,lenstate))
        Pc=np.ones((self.nsub,lenstate))*0.5
        

        states=np.zeros((lenstate,self.nsub))
        if shuffle:
            full_params=np.zeros((5,self.nsub,self.gens))
        #implement the evolution step
        for g in range(self.gens):
            for col in range(5):
                variability[col,g]=len(np.unique(params[:,col],return_counts=True)[1]) 
            pgen[:,g]=np.mean(params,axis=0)
            sgen[:,g]=np.std(params,axis=0)
            #define the status of ablation depending on the generation
            if g<self.gen_ablation:
                ablation=0
            else:
                ablation=self.ablation
                
            for agent in range(self.nsub):
                states[:,agent]=self.generate_states() #TODO: vectorize that?
                
            
            
            #%%this is the simulation function
            beta, lr1, lr2, tau, phi = params.T
            ix=0
            for n in range(len(self.task)):
                
                ntrial,repetitions,chancemin,chancemax,distrib,rev=self.get_params(n)
                
                Q = np.zeros((self.nsub, lenstate, 2)) 
                C = np.zeros((self.nsub, lenstate, 2))#TODO: this is overkill: better way to dimension it?
                
                init=ix*(self.ntrial*self.repetitions)
                s=states[init:init+(ntrial*repetitions),:]
                if rev==1:
                    s2=s
                    s=np.ones((len(s),self.nsub),dtype=int)
                else:
                    s2=np.zeros((len(s),self.nsub)) #,dtype=int
                    s=s.astype(int)
                #create an array for ablation points, to know when everything is cut
                #
                ab_array=np.zeros((1,ntrial))
                ab_array[0,self.time::]=ablation
                ab_array=np.tile(ab_array,repetitions)[0]
                # print(ab_array)
                for k in range(len(s)): #
                    
                    #set parameters according to the ablation condition at given trial
                    i=k+init
                    abl=ab_array[k]
                    # print(abl)
                    if abl>=3:
                        if inversion:
                            if k==0 and g==self.gen_ablation:
                                # print(lr1[0],lr2[0])
                                lr1, lr2=lr2,lr1
                                # print(lr1[0],lr2[0])
                                params[:,[1,2]]=params[:,[2,1]]
                        else:
                            lr1=(lr1+lr2)/2
                            lr2=lr1
                            params[:,1]=(params[:,1]+params[:,2])/2
                            params[:,2]=params[:,1]
                    if abl==1 or abl==4:
                        tau=0
                        params[:,3]=0
                    if abl==2 or abl==4:
                        if inversion:
                            if k==0 and g==self.gen_ablation:
                                # print(phi)
                                phi=-phi
                                # print(phi)
                                params[:,4]=-params[:,4]
                        else:
                            phi=0
                            params[:,4]=0
                    if abl==0:
                        lr1=params[:,1]
                        lr2=params[:,2]
                        tau=params[:,3]
                        phi=params[:,4]
                    
                    # idx_all = tuple([range(self.nsub), s[:, k], 0])
                    idx1 = tuple([range(self.nsub), s[k,:], 0]) #.astype(int)
                    idx2 = tuple([range(self.nsub), s[k,:], 1])
                    idxval1 = tuple([range(self.nsub), i, 0])
                    idxval2 = tuple([range(self.nsub), i, 1])
                    
                    Qval[idxval1]=Q[idx1]
                    Qval[idxval2]=Q[idx2]
                    Cval[idxval1]=C[idx1]
                    Cval[idxval2]=C[idx2]
                    
                    Pc[:,i]=(1/(1+ np.exp(-beta*(Q[idx2]-Q[idx1]) 
                                        - phi*(C[idx2]-C[idx1])))) 
                    # print(Pc[:,i])
                    
                    a[:, i] = (Pc[:, i] > np.random.uniform(size = self.nsub)).astype(int) #+ 0
                    r[:, i] = Bandit_Task(a[:,i], rev, s2[k,:], chancemin,chancemax) 
                    idx = tuple([range(self.nsub), s[k,:], a[:, i]]) #.astype(int)
                    idx_opp = tuple([range(self.nsub), s[k,:], 1-a[:, i]])
                    
                    PEc = r[:, i] - Q[idx]
                    #print(lr1)
                    Q[idx] = Q[idx] + lr1 * PEc *(PEc>0)+lr2*PEc*(PEc<0)
                    C[idx] = C[idx] + tau * (1-C[idx])
                    C[idx_opp] = C[idx_opp] + tau * (-C[idx_opp])
                    
                    
                acc[:,init:init+(ntrial*repetitions)]=a[:,init:init+(ntrial*repetitions)]
                
                if rev==1:
                    sub_acc=acc[:,init:init+(ntrial*repetitions)]
                    indices=(s2+init)%2==0 #[(k+init)%2==0 for k in np.nditer(s2)]
                    sub_acc[indices.T]=1-sub_acc[indices.T]
                    acc[:,init:init+(ntrial*repetitions)]=sub_acc
                ix+=1
                
            Pro[:,g]=np.mean(Pc,axis=1)
            Act[:,g]=np.mean(acc,axis=1)
            Cho=acc #correct/incorrect. To have whether they picked the first correct (in the case of a reversal), change to Cho=a
            

          
            #selection process
            index=np.argsort(Act[:,g])[::-1]#rank subjects in descending order of fitness
            crit[0,g]=np.mean(Act[index[0:self.nsub-self.replac],g])
            crit[1,g]=np.mean(Act[index[self.nsub-self.replac:self.nsub],g]) 
            crit[2,g]=np.mean(Act[:,g])
            crit[3,g]=np.mean(Act[index[0:self.replac],g])
            
            
            if not uniform and g<200:
                
                if not mutation:
                    
                    mutated=params[index[0:self.replac],:]+np.concatenate(
                        (np.random.normal(size=[self.replac,1]),
                         np.random.normal(scale=0.05,size=[self.replac,1]),
                         np.random.normal(scale=0.05,size=[self.replac,1]),
                         np.random.normal(scale=0.05,size=[self.replac,1]),
                         np.random.normal(scale=2,size=[self.replac,1])),axis=1)
                    
                    mutated_c=constrain_matrix(mutated,lower,upper)
                    paramsTemp=np.concatenate((params[index[0:self.nsub-self.replac],:],mutated_c))
                    
                else:
                    mutationRate=floor(self.nsub*self.mut_rate/100)
                    randomIndices=np.random.permutation(self.nsub)[:mutationRate]
                    randomNumbers=np.arange(0,self.nsub)
                    selectedNumbers=randomNumbers[randomIndices]
                    mutated=params[index[selectedNumbers],:]+np.concatenate(
                        (np.random.normal(size=[mutationRate,1]),
                         np.random.normal(scale=0.05,size=[mutationRate,1]),
                         np.random.normal(scale=0.05,size=[mutationRate,1]),
                         np.random.normal(scale=0.05,size=[mutationRate,1]),
                         np.random.normal(scale=2,size=[mutationRate,1])),axis=1)
                    mutated_c=constrain_matrix(mutated, lower, upper)
                    paramsTemp=np.concatenate((params[index[0:self.nsub-self.replac],:],mutated_c))
            else:
                paramsTemp=np.concatenate((params[index[0:self.nsub-self.replac],:],params[index[0:self.replac],:]))
            
            if shuffle:
                full_params[:,:,g]=params.T
                if g==100:
                    rng = np.random.default_rng()
                    y = rng.permuted(paramsTemp, axis=0,out=paramsTemp)
                
            params=paramsTemp
        if not shuffle:
            return Qval,Cval,Pro,Act,Cho,crit,pgen,sgen,variability,params#,full_params
        else:
            return Qval,Cval,Pro,Act,Cho,crit,pgen,sgen,variability,params,full_params
    
def treat_keys(cur_dict,reboots=100,gens=200,shuffle=False):
    ntrials=cur_dict[0]['Qval'].shape[1]
    nagents=cur_dict[0]['Qval'].shape[0]
    # print(ntrials)
    gen_dict={}
    
    gen_dict['Qval']=np.zeros((reboots,ntrials,2))
    gen_dict['Cval']=np.zeros((reboots,ntrials,2))
    gen_dict['Pro']=np.zeros((reboots,gens))
    gen_dict['Act']=np.zeros((reboots,gens))
    gen_dict['Cho']=np.zeros((reboots,ntrials))
    gen_dict['crit']=np.zeros((4,gens))
    gen_dict['pgen']=np.zeros((5,gens))
    gen_dict['sgen']=np.zeros((5,gens,reboots))
    gen_dict['variability']=np.zeros((5,gens))
    gen_dict['params']=np.zeros((reboots,5))
    if shuffle:
        gen_dict['full_params']=np.zeros((5,nagents,gens,reboots))
    
    for r in range (reboots):
        # print(np.mean(cur_dict[r]['Qval'],axis=0).shape)
        gen_dict['crit']+=cur_dict[r]['crit']/reboots
        gen_dict['pgen']+=cur_dict[r]['pgen']/reboots
        gen_dict['sgen'][:,:,r]=cur_dict[r]['pgen']
        gen_dict['variability']+=cur_dict[r]['variability']/reboots
        gen_dict['Qval'][r,:,:]=np.mean(cur_dict[r]['Qval'],axis=0)
        gen_dict['Cval'][r,:,:]=np.mean(cur_dict[r]['Cval'],axis=0)
        gen_dict['Pro'][r,:]=np.mean(cur_dict[r]['Pro'],axis=0)
        gen_dict['Act'][r,:]=np.mean(cur_dict[r]['Act'],axis=0)
        gen_dict['Cho'][r,:]=np.mean(cur_dict[r]['Cho'],axis=0)
        gen_dict['params'][r,:]=np.mean(cur_dict[r]['params'],axis=0)
        if shuffle:
            gen_dict['full_params'][:,:,:,r]=cur_dict[r]['full_params']
    gen_dict['sgen']=np.squeeze(stats.sem(gen_dict['sgen'],axis=2))
    
    return gen_dict

def single_run(nsub,gens,uniform,mutation,shuffle,inversion=False,ab=0,gen=200,scenarios='volatile'):
    if scenarios=='stable':
        task=[0,1,2,3,4,5,15]#np.arange(6)
    elif scenarios=='volatile':
        task=np.arange(6,15)
    elif type(scenarios)==list:
        task=scenarios
    elif type(scenarios)==int:
        task=[scenarios]
    my_simulation=Simulation(nsub,gens,ablation=ab,gen_abl=gen,task=task)
    if shuffle:
        Qval,Cval,Pro,Act,Cho,crit,pgen,sgen,variability,params,full_params=my_simulation.EvolvingHysteresis_function_within_mutation(uniform=uniform,mutation=mutation,shuffle=shuffle,inversion=inversion)
        sub_dict={"Qval":Qval,"Cval":Cval,"Pro":Pro,"Act":Act,"Cho":Cho,
                  "crit":crit,"pgen":pgen,"sgen":sgen,"variability":variability,
                  "params":params,"full_params":full_params}
    else:
        Qval,Cval,Pro,Act,Cho,crit,pgen,sgen,variability,params=my_simulation.EvolvingHysteresis_function_within_mutation(uniform=uniform,mutation=mutation,shuffle=shuffle,inversion=inversion)
        sub_dict={"Qval":Qval,"Cval":Cval,"Pro":Pro,"Act":Act,"Cho":Cho,
                  "crit":crit,"pgen":pgen,"sgen":sgen,"variability":variability,
                  "params":params}
    return sub_dict

def sim_function(nsub=1000,gens=200,reboot=100,mutation=False,uniform=True,shuffle=False,scenarios='volatile',inversion=False):
    rr=Parallel(n_jobs=-1,backend="multiprocessing")(delayed(single_run)(nsub,gens,uniform,mutation,shuffle,scenarios=scenarios,inversion=inversion) for reb in range(reboot))
    reboot_dict={k:rr[k] for k in range(reboot)}
    return reboot_dict

def ablation_function(ablation=[0,1,2,3,4],nsub=1000,gens=200,ntrial=20,repetitions=8,replac=50,gen_abl=[0,10,100],p=[0,0,0,0,0],reboot=100,mutation=False,uniform=True,shuffle=False,scenarios='volatile',inversion=False):
    abl_dict={}
    for ab in ablation:
        abl_dict[ab]={}
        if ab==0:
            print('Running {}'.format(ab))
            rr=Parallel(n_jobs=-1,backend="multiprocessing")(delayed(single_run)(nsub,gens,uniform,mutation,shuffle,scenarios=scenarios, inversion=inversion) for reb in range(reboot))
            reboot_dict={k:rr[k] for k in range(reboot)}
            abl_dict[ab]=treat_keys(reboot_dict,reboots=reboot,gens=gens,shuffle=shuffle)
        else:
            for gen in gen_abl:
                print('Running {}, generation {}'.format(ab,gen))
                rr=Parallel(n_jobs=-1,backend="multiprocessing")(delayed(single_run)(nsub,gens,uniform,mutation,shuffle,ab=ab,gen=gen,scenarios=scenarios, inversion=inversion) for reb in range(reboot))
                reboot_dict={k:rr[k] for k in range(reboot)}
                abl_dict[ab][gen]=treat_keys(reboot_dict,reboots=reboot,gens=gens,shuffle=shuffle)
    return abl_dict
            
def factor(ablation=0,nsub=1000,gens=200,ntrial=20,repetitions=8,replac=50,gen_abl=200,p=[0,0,0,0,0],reboot=100,mutation=False,uniform=True,shuffle=False,inversion=False,cases='difficulty'):
    difficulty_dict={}
    if cases=='difficulty':
        scenes=[2,3,0]
    elif cases=='richness':
        scenes=[15,3,1]
    elif cases=='lp':
        scenes=[4,3,5]
    elif cases=='fixed':
        scenes=[6,7,8]
    elif cases=='gaussian':
        scenes=[9,10,11]
    elif cases=='rdwalk':
        scenes=[12,13,14]
    else:
        scenes=np.arange(16)
    for scenarios in scenes:
        print(f'Processing scenario {scenarios}')
        rr=Parallel(n_jobs=-1,backend="multiprocessing")(delayed(single_run)(nsub,gens,uniform,mutation,shuffle,ab=ablation,gen=gen_abl,scenarios=scenarios,inversion=inversion) for reb in range(reboot))
        reboot_dict={k:rr[k] for k in range(reboot)}
        difficulty_dict[scenarios]=treat_keys(reboot_dict,reboots=reboot,gens=gens)
    return difficulty_dict
    
#%% do the switch thing
def factor_selection(uniform,ablation,scenarios):
    # result_dict=sim_function(uniform=uniform,scenarios=scenarios)
    result_dict=ablation_function(uniform=uniform,ablation=ablation,scenarios=scenarios)
    return result_dict

# my_simulation=Simulation(1000,200,ablation=0,gen_abl=200,task=np.arange(3))
# Qval,Cval,Pro,Act,Cho,crit,pgen,sgen,variability,params=my_simulation.EvolvingHysteresis_function_within_mutation(uniform=True,mutation=False,shuffle=False)
# plt.plot(np.mean(Pro,axis=0))
# plt.plot(np.mean(Cho,axis=0))
# rd=sim_function(nsub=100,gens=10,reboot=1,scenarios='volatile')

# res=single_run(1000,200,True,False,False,scenarios=[3],ab=3,gen=100,inversion=True)

#%% ablation experiments, uniform distribution
res=ablation_function(ablation=[0,2,3,4], gen_abl=[100], scenarios='volatile',inversion=True) #,gens=10,reboot=1,nsub=100
filename='../results/Maxi_inversion_volatile_uniform.pkl'
with open(filename, 'wb') as fp:
        pickle.dump(res,fp)


res=ablation_function(ablation=[0,2,3,4], gen_abl=[100], scenarios='stable',inversion=True) #,gens=100,nsub=100,reboot=1
filename='../results/Maxi_inversion_stable_uniform.pkl'
with open(filename, 'wb') as fp:
        pickle.dump(res,fp)
        

       

# # set random seed
# np.random.seed(42)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Where reboots run in parallel for the simulations')
#     parser.add_argument('-u','--uniform',type=int,required=False,default=1)
#     parser.add_argument('-a','--ablation',type=list,required=False,default=[0])
#     parser.add_argument('-t','--task',type=str,required=False,default='volatile')
#     args = parser.parse_args()
    
#     suffix='uniform'*args.uniform+'static'*(not args.uniform)+'_'+str(args.ablation)+'_'+args.task
#     filename="./results/Maxi_results_"+suffix+".pkl"
#     with open(filename, 'wb') as fp:
#         res=factor_selection(args.uniform,args.ablation,args.task)
        # pickle.dump(res,fp)
        
