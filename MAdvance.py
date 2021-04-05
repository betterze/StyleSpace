


from manipulate import Manipulator
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def lp2istr(x):
    return str(x[0])+'_'+str(x[1])

class MAdvance(Manipulator):
    
    def __init__(self,dataset_name):
        super().__init__(dataset_name)
        self.positive_bank=1000
        self.num_pos=10 #example
        self.num_m=10 #number of output
        self.threshold1=0.5 #pass this ratio
        self.threshold2=0.25 #gap between first and second
        self.w=np.load('./npy/'+dataset_name+'/W.npy')
        
        self.code_mean2=np.concatenate(self.code_mean)
        self.code_std2=np.concatenate(self.code_std)
        
        fmaps=[512, 512, 512, 512, 512, 256, 128,  64, 32]
        self.fmaps=np.repeat(fmaps,3)
        
        try:
            self.LoadSemantic()
        except FileNotFoundError:
            print('semantic_top_32 not exist')
        
        try:
            self.results=pd.read_csv(self.img_path+'attribute')
        except FileNotFoundError:
            print('attribute not exist')
    
    def LoadSemantic(self):
        name='semantic_top_32'
        with open(self.img_path+name, 'rb') as handle:
            all_semantic_top = pickle.load(handle)

            
        self.all_semantic_top2=np.concatenate(all_semantic_top)
        self.num_semantic=self.all_semantic_top2.shape[1] #ignore low frequency area, bed 10
        
        tmp=pd.read_csv(self.img_path+'label')
        self.label=tmp['names']
    
    def RemovePG(self,l_p,findex=None): #l_p or indexs2
        for i in self.pindexs:
            select=l_p[:,0]==i
            l_p=l_p[~select]
            if not findex is None:
                findex=findex[~select]
        
        if findex is None:
            return l_p
        else:
            return l_p,findex
    
    def GetRank(self,target_index):
        top_sum=self.all_semantic_top2[:,target_index].sum(axis=1)
        
        tmp=list(np.arange(self.num_semantic))
        for i in target_index:
            tmp.remove(i)
        tmp=self.all_semantic_top2[:,tmp] #all the rest semantic 
        second_max=tmp.max(axis=1)
        
        select1=top_sum>self.threshold1
        select2=top_sum-second_max>self.threshold2
        
        select=np.logical_and(select1,select2)
        findex=np.arange(len(select))[select]
        l_p=self.GetLCIndex(findex)
        
        index2=np.zeros([len(l_p),3])
        index2[:,2]=top_sum[findex]
        index2[:,(0,1)]=l_p
        
        select_index=np.argsort(index2[:,2])[::-1]
        index2=index2[select_index]
        findex=findex[select_index]

        index2,findex2=self.RemovePG(index2,findex)
        return index2,findex2
    
    def AllCheck(self,positive=True):
        
        tmp_save=self.num_pos
        self.num_pos=self.positive_bank
        
        positive_train,_=self.SimulateInput(positive)
        index2,_=self.GetComponent(positive_train)
        
        self.num_pos=tmp_save
        lp_sort=pd.DataFrame(index2[:,-1])
        lp_sort.index=list(map(lp2istr, index2[:,:-1].astype(int)))
        
        return index2,lp_sort
    
    def SimulateInput(self,positive=True):
        print('bname: '+str(self.bname))
        tmp_indexs=self.results[self.bname].argsort()
        if positive:
            tmp=tmp_indexs[:self.positive_bank]
        else:
            tmp=tmp_indexs[-self.positive_bank:]
        positive_indexs=np.random.choice(tmp,size=self.num_pos,replace=False)
        
        
        tmp=self.w[positive_indexs] #only use 50 images
        tmp=tmp[:,None,:]
        w_plus=np.tile(tmp,(1,self.Gs.components.synthesis.input_shape[1],1))
        tmp_dlatents=M.W2S(w_plus)
        
        positive_train=[tmp for tmp in tmp_dlatents]
        return positive_train,positive_indexs
    
    def GetComponent(self,positive_train): #sort s2n, remove pg, 
        
        feature_s2n=self.S2N(positive_train)
        
        feature_index=feature_s2n.argsort()
        findex=feature_index[::-1] #index in concatenate form 
        
        l_p=self.GetLCIndex(findex)
        
        index2=np.zeros([len(l_p),3])
        index2[:,2]=feature_s2n[findex]
        index2[:,(0,1)]=l_p
        
        index2,findex2=self.RemovePG(index2,findex)
        return index2,findex
    
    def S2N(self,positive_train):
        positive_train2=np.concatenate(positive_train,axis=1)
        normalize_positive=(positive_train2-self.code_mean2)/self.code_std2
        
        feature_mean=np.abs(normalize_positive.mean(axis=0))
        feature_std=normalize_positive.std(axis=0)
        
        feature_s2n=feature_mean/feature_std
        return feature_s2n
    
    def GetLCIndex(self,findex):
        l_p=[]
        cfmaps=np.cumsum(self.fmaps)
        for i in range(len(findex)):
            tmp_index=findex[i]
            tmp=tmp_index-cfmaps
            tmp=tmp[tmp>0]
            lindex=len(tmp)
            if lindex==0:
                cindex=tmp_index
            else:
                cindex=tmp[-1]
            
            if cindex ==self.fmaps[lindex]:
                cindex=0
                lindex+=1
            l_p.append([lindex,cindex])
        l_p=np.array(l_p)
        return l_p
    
    #%%
if __name__ == "__main__":
    dataset_name='ffhq'
    M=MAdvance(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    #%%
    
    M.bname='13-blond-hair'  #01-smiling, 37-wearing-lipstick,13-blond-hair
#    lp_sort=M.ConsistenceCheck(num_run=1000)
    
    lp_candidate,lp_sort= M.AllCheck()
    plt.figure()
    plt.title(M.bname)
    plt.plot(lp_sort[:10],'*')
    plt.ylabel('signal2noise')
    plt.xlabel('(layer_index, channel_index)')
    #%%
    
    M.alpha=[-20,-10,-5,0,5,10,20]
    M.img_index=0
    M.num_images=20
    start=0

    for i in range(10):
        print(i)
        tmp=lp_sort.index[start+i]
        lindex,bname=np.array(tmp.split('_')) 
        lindex,bname=int(lindex),int(bname)
        
        M.manipulate_layers=[lindex]
        codes,out=M.EditOneC(bname) 
        tmp=str(M.manipulate_layers)+'_'+str(bname)
        M.Vis(tmp,'c',out)
    #%%
    
    num_view=5
    target_index=(10,)
    lp_candidate,_=M.GetRank(target_index)
    print(lp_candidate.shape)
    #%%
    
    M.alpha=[-20,-10,-5,0,5,10,20]
    M.img_index=0
    M.num_images=20
    start=0
    
    for i in range(num_view):
        
        lindex,bname,_=lp_candidate[start+i].astype(int)
        lindex=int(lindex)
        M.manipulate_layers=[lindex]
        codes,out=M.EditOneC(bname)
        tmp=str(M.manipulate_layers)+'_'+str(bname)
        M.Vis(tmp,'c',out)
    #%%
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    