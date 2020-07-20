import pandas as pd
import string 


data1=pd.read_csv('MADAR-Corpus-6-train.tsv',sep='\t')
data1.columns = ['text','country_label']

data2=pd.read_csv('MADAR-Corpus-26-train.tsv',sep='\t')
data2.columns = ['text','country_label']

dataset=data1.append(data2,ignore_index=True)

dataset.drop(dataset[dataset['country_label']=='MSA'].index)

for indice ,data2 in dataset.iterrows():
    if data2[1]=='BEI':
        data2[1]='Lebanon'
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='ALE':
        data2[1]='Syria'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='ALG':
        data2[1]='Algeria'     
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='ALX':
        data2[1]='Egypt'
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='AMM':
        data2[1]='Jordan'
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='ASW':
        data2[1]='Egypt'    

for indice ,data2 in dataset.iterrows():
    if data2[1]=='BAG':
        data2[1]='Iraq'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='BAS':
        data2[1]='Iraq'  
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='BEN':
        data2[1]='Libya'       
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='CAI':
        data2[1]='Egypt'
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='DAM':
        data2[1]='Syria'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='DOH':
        data2[1]='Qatar'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='FES':
        data2[1]='Morocco'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='JED':
        data2[1]='Saudi_Arabia'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='JER':
        data2[1]='Palestine'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='KHA':
        data2[1]='Sudan'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='MOS':
        data2[1]='Iraq'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='MUS':
        data2[1]='Oman'   
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='RAB':
        data2[1]='Morocco'

for indice ,data2 in dataset.iterrows():
    if data2[1]=='RIY':
        data2[1]='Saudi_Arabia'
    
for indice ,data2 in dataset.iterrows():
    if data2[1]=='SAL':
        data2[1]='Jordan'
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='SAN':
        data2[1]='Yemen'    
        
for indice ,data2 in dataset.iterrows():
    if data2[1]=='SFX':
        data2[1]='Tunisia' 

for indice ,data2 in dataset.iterrows():
    if data2[1]=='TRI':
        data2[1]='Libya' 
    
for indice ,data2 in dataset.iterrows():
    if data2[1]=='TUN':
        data2[1]='Tunisia' 

dataset.to_csv(r'C:\Users\allou\Desktop\mm\madar\mdar_annoté.csv', index = False)