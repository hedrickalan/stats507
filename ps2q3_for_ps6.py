#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from collections import defaultdict
import time
from tabulate import tabulate
print("Starting question 3")
print("starting a")
#creating dataframes for each dataset
#now includes gender
data2011_2012 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2011-2012/DEMO_G.XPT')
data2013_2014 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2013-2014/DEMO_H.XPT')
data2015_2016 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2015-2016/DEMO_I.XPT')
data2017_2018 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DEMO_J.XPT')
df1 = pd.DataFrame({"sequence_num": data2011_2012["SEQN"].astype(int),\
                    "age": data2011_2012["RIDAGEYR"].astype(int), \
                   "race_and_eth": data2011_2012["RIDRETH3"].astype("category"), \
                    "edu": data2011_2012["DMDEDUC2"].astype("category"), \
                   "marital_st": data2011_2012["DMDMARTL"].astype("category"),\
                    "interview": data2011_2012["RIDSTATR"].astype("category"), \
                   "psu_var": data2011_2012["SDMVPSU"].astype(int),\
                    "str_var": data2011_2012["SDMVSTRA"].astype(int), \
                   "mec_exm_weight": data2011_2012["WTMEC2YR"], "interv_weight": data2011_2012["WTINT2YR"], \
                   "gender" : data2011_2012["RIAGENDR"]})
df1["year"] = "2011-2012"

df2 = pd.DataFrame({"sequence_num": data2013_2014["SEQN"].astype(int),\
                    "age": data2013_2014["RIDAGEYR"].astype(int), \
                   "race_and_eth": data2013_2014["RIDRETH3"].astype("category"),\
                    "edu": data2013_2014["DMDEDUC2"].astype("category"), \
                   "marital_st": data2013_2014["DMDMARTL"].astype("category"),\
                    "interview": data2013_2014["RIDSTATR"].astype("category"), \
                   "psu_var": data2013_2014["SDMVPSU"].astype(int),\
                    "str_var": data2013_2014["SDMVSTRA"].astype(int), \
                   "mec_exm_weight": data2013_2014["WTMEC2YR"], "interv_weight": data2013_2014["WTINT2YR"], \
                   "gender" : data2013_2014["RIAGENDR"]})
df2["year"] = "2013-2014"

df3 = pd.DataFrame({"sequence_num": data2015_2016["SEQN"].astype(int),\
                    "age": data2015_2016["RIDAGEYR"].astype(int), \
                   "race_and_eth": data2015_2016["RIDRETH3"].astype("category"), \
                    "edu": data2015_2016["DMDEDUC2"].astype("category"), \
                   "marital_st": data2015_2016["DMDMARTL"].astype("category"),\
                    "interview": data2015_2016["RIDSTATR"].astype("category"), \
                   "psu_var": data2015_2016["SDMVPSU"].astype(int),\
                    "str_var": data2015_2016["SDMVSTRA"].astype(int), \
                   "mec_exm_weight": data2015_2016["WTMEC2YR"], "interv_weight": data2015_2016["WTINT2YR"], \
                   "gender" : data2015_2016["RIAGENDR"]})
df3["year"] = "2015-2016"

df4 = pd.DataFrame({"sequence_num": data2017_2018["SEQN"].astype(int),\
                    "age": data2017_2018["RIDAGEYR"].astype(int), \
                   "race_and_eth": data2017_2018["RIDRETH3"].astype("category"),\
                    "edu": data2017_2018["DMDEDUC2"].astype("category"), \
                   "marital_st": data2017_2018["DMDMARTL"].astype("category"),\
                    "interview": data2017_2018["RIDSTATR"].astype("category"), \
                   "psu_var": data2017_2018["SDMVPSU"].astype(int),\
                    "str_var": data2017_2018["SDMVSTRA"].astype(int), \
                   "mec_exm_weight": data2017_2018["WTMEC2YR"], "interv_weight": data2017_2018["WTINT2YR"], \
                   "gender" : data2017_2018["RIAGENDR"]})
df4["year"] = "2017-2018"

frames = [df1, df2, df3, df4]
result = pd.concat(frames)
result.head()

result.to_pickle("df.pkl")
#find the path and save with the file extension, not self


print("starting b")

dataDental2011_2012 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2011-2012/OHXDEN_G.XPT')
dataDental2013_2014 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2013-2014/OHXDEN_H.XPT')
dataDental2015_2016 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2015-2016/OHXDEN_I.XPT')
dataDental2017_2018 = pd.read_sas('https://wwwn.cdc.gov/nchs/nhanes/2017-2018/OHXDEN_J.XPT')

#SEQN, OHDDESTS, tooth counts (OHXxxTC) 01-32, and coronal cavities (OHXxxCTC) 02-31

dfDental1 = pd.DataFrame({"sequence_num": dataDental2011_2012["SEQN"].astype(int),                    "den. status code": dataDental2011_2012["OHDDESTS"],                    "tooth_count_01": dataDental2011_2012["OHX01TC"],                     "tooth_count_02": dataDental2011_2012["OHX02TC"],                     "tooth_count_03": dataDental2011_2012["OHX03TC"],                      "tooth_count_04": dataDental2011_2012["OHX04TC"],                           "tooth_count_05": dataDental2011_2012["OHX05TC"],                           "tooth_count_06": dataDental2011_2012["OHX06TC"],                           "tooth_count_07": dataDental2011_2012["OHX07TC"],                           "tooth_count_08": dataDental2011_2012["OHX08TC"],                           "tooth_count_09": dataDental2011_2012["OHX09TC"],                           "tooth_count_10": dataDental2011_2012["OHX10TC"],                           "tooth_count_11": dataDental2011_2012["OHX11TC"],                           "tooth_count_12": dataDental2011_2012["OHX12TC"],                           "tooth_count_13": dataDental2011_2012["OHX13TC"],                           "tooth_count_14": dataDental2011_2012["OHX14TC"],                           "tooth_count_15": dataDental2011_2012["OHX15TC"],                           "tooth_count_16": dataDental2011_2012["OHX16TC"],                           "tooth_count_17": dataDental2011_2012["OHX17TC"],                           "tooth_count_18": dataDental2011_2012["OHX18TC"],                           "tooth_count_19": dataDental2011_2012["OHX19TC"],                           "tooth_count_20": dataDental2011_2012["OHX20TC"],                           "tooth_count_21": dataDental2011_2012["OHX21TC"],                           "tooth_count_22": dataDental2011_2012["OHX22TC"],                           "tooth_count_23": dataDental2011_2012["OHX23TC"],                           "tooth_count_24": dataDental2011_2012["OHX24TC"],                           "tooth_count_25": dataDental2011_2012["OHX25TC"],                           "tooth_count_26": dataDental2011_2012["OHX26TC"],                           "tooth_count_27": dataDental2011_2012["OHX27TC"],                           "tooth_count_28": dataDental2011_2012["OHX28TC"],                           "tooth_count_29": dataDental2011_2012["OHX29TC"],                           "tooth_count_30": dataDental2011_2012["OHX30TC"],                           "tooth_count_31": dataDental2011_2012["OHX31TC"],                           "tooth_count_32": dataDental2011_2012["OHX32TC"],                           "coronal cavities 02": dataDental2011_2012["OHX02CTC"],                           "coronal cavities 03": dataDental2011_2012["OHX03CTC"],                           "coronal cavities 04": dataDental2011_2012["OHX04CTC"],                           "coronal cavities 05": dataDental2011_2012["OHX05CTC"],                           "coronal cavities 06": dataDental2011_2012["OHX06CTC"],                           "coronal cavities 07": dataDental2011_2012["OHX07CTC"],                           "coronal cavities 08": dataDental2011_2012["OHX08CTC"],                           "coronal cavities 09": dataDental2011_2012["OHX09CTC"],                           "coronal cavities 10": dataDental2011_2012["OHX10CTC"],                           "coronal cavities 11": dataDental2011_2012["OHX11CTC"],                           "coronal cavities 12": dataDental2011_2012["OHX12CTC"],                           "coronal cavities 13": dataDental2011_2012["OHX13CTC"],                           "coronal cavities 14": dataDental2011_2012["OHX14CTC"],                           "coronal cavities 15": dataDental2011_2012["OHX15CTC"],                           "coronal cavities 18": dataDental2011_2012["OHX18CTC"],                           "coronal cavities 19": dataDental2011_2012["OHX19CTC"],                           "coronal cavities 20": dataDental2011_2012["OHX20CTC"],                           "coronal cavities 21": dataDental2011_2012["OHX21CTC"],                           "coronal cavities 22": dataDental2011_2012["OHX22CTC"],                           "coronal cavities 23": dataDental2011_2012["OHX23CTC"],                           "coronal cavities 24": dataDental2011_2012["OHX24CTC"],                           "coronal cavities 25": dataDental2011_2012["OHX25CTC"],                           "coronal cavities 26": dataDental2011_2012["OHX26CTC"],                           "coronal cavities 27": dataDental2011_2012["OHX27CTC"],                           "coronal cavities 28": dataDental2011_2012["OHX28CTC"],                           "coronal cavities 29": dataDental2011_2012["OHX29CTC"],                           "coronal cavities 30": dataDental2011_2012["OHX30CTC"],                           "coronal cavities 31": dataDental2011_2012["OHX31CTC"]})
dfDental1["year"] = "2011-2012"

dfDental2 = pd.DataFrame({"sequence_num": dataDental2013_2014["SEQN"].astype(int),                    "den. status code": dataDental2013_2014["OHDDESTS"],                    "tooth_count_01": dataDental2013_2014["OHX01TC"],                     "tooth_count_02": dataDental2013_2014["OHX02TC"],                     "tooth_count_03": dataDental2013_2014["OHX03TC"],                      "tooth_count_04": dataDental2013_2014["OHX04TC"],                           "tooth_count_05": dataDental2013_2014["OHX05TC"],                           "tooth_count_06": dataDental2013_2014["OHX06TC"],                           "tooth_count_07": dataDental2013_2014["OHX07TC"],                           "tooth_count_08": dataDental2013_2014["OHX08TC"],                           "tooth_count_09": dataDental2013_2014["OHX09TC"],                           "tooth_count_10": dataDental2013_2014["OHX10TC"],                           "tooth_count_11": dataDental2013_2014["OHX11TC"],                           "tooth_count_12": dataDental2013_2014["OHX12TC"],                           "tooth_count_13": dataDental2013_2014["OHX13TC"],                           "tooth_count_14": dataDental2013_2014["OHX14TC"],                           "tooth_count_15": dataDental2013_2014["OHX15TC"],                           "tooth_count_16": dataDental2013_2014["OHX16TC"],                           "tooth_count_17": dataDental2013_2014["OHX17TC"],                           "tooth_count_18": dataDental2013_2014["OHX18TC"],                           "tooth_count_19": dataDental2013_2014["OHX19TC"],                           "tooth_count_20": dataDental2013_2014["OHX20TC"],                           "tooth_count_21": dataDental2013_2014["OHX21TC"],                           "tooth_count_22": dataDental2013_2014["OHX22TC"],                           "tooth_count_23": dataDental2013_2014["OHX23TC"],                           "tooth_count_24": dataDental2013_2014["OHX24TC"],                           "tooth_count_25": dataDental2013_2014["OHX25TC"],                           "tooth_count_26": dataDental2013_2014["OHX26TC"],                           "tooth_count_27": dataDental2013_2014["OHX27TC"],                           "tooth_count_28": dataDental2013_2014["OHX28TC"],                           "tooth_count_29": dataDental2013_2014["OHX29TC"],                           "tooth_count_30": dataDental2013_2014["OHX30TC"],                           "tooth_count_31": dataDental2013_2014["OHX31TC"],                           "tooth_count_32": dataDental2013_2014["OHX32TC"],                           "coronal cavities 02": dataDental2013_2014["OHX02CTC"],                           "coronal cavities 03": dataDental2013_2014["OHX03CTC"],                           "coronal cavities 04": dataDental2013_2014["OHX04CTC"],                           "coronal cavities 05": dataDental2013_2014["OHX05CTC"],                           "coronal cavities 06": dataDental2013_2014["OHX06CTC"],                           "coronal cavities 07": dataDental2013_2014["OHX07CTC"],                           "coronal cavities 08": dataDental2013_2014["OHX08CTC"],                           "coronal cavities 09": dataDental2013_2014["OHX09CTC"],                           "coronal cavities 10": dataDental2013_2014["OHX10CTC"],                           "coronal cavities 11": dataDental2013_2014["OHX11CTC"],                           "coronal cavities 12": dataDental2013_2014["OHX12CTC"],                           "coronal cavities 13": dataDental2013_2014["OHX13CTC"],                           "coronal cavities 14": dataDental2013_2014["OHX14CTC"],                           "coronal cavities 15": dataDental2013_2014["OHX15CTC"],                           "coronal cavities 18": dataDental2013_2014["OHX18CTC"],                           "coronal cavities 19": dataDental2013_2014["OHX19CTC"],                           "coronal cavities 20": dataDental2013_2014["OHX20CTC"],                           "coronal cavities 21": dataDental2013_2014["OHX21CTC"],                           "coronal cavities 22": dataDental2013_2014["OHX22CTC"],                           "coronal cavities 23": dataDental2013_2014["OHX23CTC"],                           "coronal cavities 24": dataDental2013_2014["OHX24CTC"],                           "coronal cavities 25": dataDental2013_2014["OHX25CTC"],                           "coronal cavities 26": dataDental2013_2014["OHX26CTC"],                           "coronal cavities 27": dataDental2013_2014["OHX27CTC"],                           "coronal cavities 28": dataDental2013_2014["OHX28CTC"],                           "coronal cavities 29": dataDental2013_2014["OHX29CTC"],                           "coronal cavities 30": dataDental2013_2014["OHX30CTC"],                           "coronal cavities 31": dataDental2013_2014["OHX31CTC"]})
dfDental2["year"] = "2013-2014"

dfDental3 = pd.DataFrame({"sequence_num": dataDental2015_2016["SEQN"].astype(int),                    "den. status code": dataDental2015_2016["OHDDESTS"],                    "tooth_count_01": dataDental2015_2016["OHX01TC"],                     "tooth_count_02": dataDental2015_2016["OHX02TC"],                     "tooth_count_03": dataDental2015_2016["OHX03TC"],                      "tooth_count_04": dataDental2015_2016["OHX04TC"],                           "tooth_count_05": dataDental2015_2016["OHX05TC"],                           "tooth_count_06": dataDental2015_2016["OHX06TC"],                           "tooth_count_07": dataDental2015_2016["OHX07TC"],                           "tooth_count_08": dataDental2015_2016["OHX08TC"],                           "tooth_count_09": dataDental2015_2016["OHX09TC"],                           "tooth_count_10": dataDental2015_2016["OHX10TC"],                           "tooth_count_11": dataDental2015_2016["OHX11TC"],                           "tooth_count_12": dataDental2015_2016["OHX12TC"],                           "tooth_count_13": dataDental2015_2016["OHX13TC"],                           "tooth_count_14": dataDental2015_2016["OHX14TC"],                           "tooth_count_15": dataDental2015_2016["OHX15TC"],                           "tooth_count_16": dataDental2015_2016["OHX16TC"],                           "tooth_count_17": dataDental2015_2016["OHX17TC"],                           "tooth_count_18": dataDental2015_2016["OHX18TC"],                           "tooth_count_19": dataDental2015_2016["OHX19TC"],                           "tooth_count_20": dataDental2015_2016["OHX20TC"],                           "tooth_count_21": dataDental2015_2016["OHX21TC"],                           "tooth_count_22": dataDental2015_2016["OHX22TC"],                           "tooth_count_23": dataDental2015_2016["OHX23TC"],                           "tooth_count_24": dataDental2015_2016["OHX24TC"],                           "tooth_count_25": dataDental2015_2016["OHX25TC"],                           "tooth_count_26": dataDental2015_2016["OHX26TC"],                           "tooth_count_27": dataDental2015_2016["OHX27TC"],                           "tooth_count_28": dataDental2015_2016["OHX28TC"],                           "tooth_count_29": dataDental2015_2016["OHX29TC"],                           "tooth_count_30": dataDental2015_2016["OHX30TC"],                           "tooth_count_31": dataDental2015_2016["OHX31TC"],                           "tooth_count_32": dataDental2015_2016["OHX32TC"],                           "coronal cavities 02": dataDental2015_2016["OHX02CTC"],                           "coronal cavities 03": dataDental2015_2016["OHX03CTC"],                           "coronal cavities 04": dataDental2015_2016["OHX04CTC"],                           "coronal cavities 05": dataDental2015_2016["OHX05CTC"],                           "coronal cavities 06": dataDental2015_2016["OHX06CTC"],                           "coronal cavities 07": dataDental2015_2016["OHX07CTC"],                           "coronal cavities 08": dataDental2015_2016["OHX08CTC"],                           "coronal cavities 09": dataDental2015_2016["OHX09CTC"],                           "coronal cavities 10": dataDental2015_2016["OHX10CTC"],                           "coronal cavities 11": dataDental2015_2016["OHX11CTC"],                           "coronal cavities 12": dataDental2015_2016["OHX12CTC"],                           "coronal cavities 13": dataDental2015_2016["OHX13CTC"],                           "coronal cavities 14": dataDental2015_2016["OHX14CTC"],                           "coronal cavities 15": dataDental2015_2016["OHX15CTC"],                           "coronal cavities 18": dataDental2015_2016["OHX18CTC"],                           "coronal cavities 19": dataDental2015_2016["OHX19CTC"],                           "coronal cavities 20": dataDental2015_2016["OHX20CTC"],                           "coronal cavities 21": dataDental2015_2016["OHX21CTC"],                           "coronal cavities 22": dataDental2015_2016["OHX22CTC"],                           "coronal cavities 23": dataDental2015_2016["OHX23CTC"],                           "coronal cavities 24": dataDental2015_2016["OHX24CTC"],                           "coronal cavities 25": dataDental2015_2016["OHX25CTC"],                           "coronal cavities 26": dataDental2015_2016["OHX26CTC"],                           "coronal cavities 27": dataDental2015_2016["OHX27CTC"],                           "coronal cavities 28": dataDental2015_2016["OHX28CTC"],                           "coronal cavities 29": dataDental2015_2016["OHX29CTC"],                           "coronal cavities 30": dataDental2015_2016["OHX30CTC"],                           "coronal cavities 31": dataDental2015_2016["OHX31CTC"]})
dfDental3["year"] = "2015-2016"

dfDental4 = pd.DataFrame({"sequence_num": dataDental2017_2018["SEQN"].astype(int),                    "den. status code": dataDental2017_2018["OHDDESTS"],                    "tooth_count_01": dataDental2017_2018["OHX01TC"],                     "tooth_count_02": dataDental2017_2018["OHX02TC"],                     "tooth_count_03": dataDental2017_2018["OHX03TC"],                      "tooth_count_04": dataDental2017_2018["OHX04TC"],                           "tooth_count_05": dataDental2017_2018["OHX05TC"],                           "tooth_count_06": dataDental2017_2018["OHX06TC"],                           "tooth_count_07": dataDental2017_2018["OHX07TC"],                           "tooth_count_08": dataDental2017_2018["OHX08TC"],                           "tooth_count_09": dataDental2017_2018["OHX09TC"],                           "tooth_count_10": dataDental2017_2018["OHX10TC"],                           "tooth_count_11": dataDental2017_2018["OHX11TC"],                           "tooth_count_12": dataDental2017_2018["OHX12TC"],                           "tooth_count_13": dataDental2017_2018["OHX13TC"],                           "tooth_count_14": dataDental2017_2018["OHX14TC"],                           "tooth_count_15": dataDental2017_2018["OHX15TC"],                           "tooth_count_16": dataDental2017_2018["OHX16TC"],                           "tooth_count_17": dataDental2017_2018["OHX17TC"],                           "tooth_count_18": dataDental2017_2018["OHX18TC"],                           "tooth_count_19": dataDental2017_2018["OHX19TC"],                           "tooth_count_20": dataDental2017_2018["OHX20TC"],                           "tooth_count_21": dataDental2017_2018["OHX21TC"],                           "tooth_count_22": dataDental2017_2018["OHX22TC"],                           "tooth_count_23": dataDental2017_2018["OHX23TC"],                           "tooth_count_24": dataDental2017_2018["OHX24TC"],                           "tooth_count_25": dataDental2017_2018["OHX25TC"],                           "tooth_count_26": dataDental2017_2018["OHX26TC"],                           "tooth_count_27": dataDental2017_2018["OHX27TC"],                           "tooth_count_28": dataDental2017_2018["OHX28TC"],                           "tooth_count_29": dataDental2017_2018["OHX29TC"],                           "tooth_count_30": dataDental2017_2018["OHX30TC"],                           "tooth_count_31": dataDental2017_2018["OHX31TC"],                           "tooth_count_32": dataDental2017_2018["OHX32TC"],                           "coronal cavities 02": dataDental2017_2018["OHX02CTC"],                           "coronal cavities 03": dataDental2017_2018["OHX03CTC"],                           "coronal cavities 04": dataDental2017_2018["OHX04CTC"],                           "coronal cavities 05": dataDental2017_2018["OHX05CTC"],                           "coronal cavities 06": dataDental2017_2018["OHX06CTC"],                           "coronal cavities 07": dataDental2017_2018["OHX07CTC"],                           "coronal cavities 08": dataDental2017_2018["OHX08CTC"],                           "coronal cavities 09": dataDental2017_2018["OHX09CTC"],                           "coronal cavities 10": dataDental2017_2018["OHX10CTC"],                           "coronal cavities 11": dataDental2017_2018["OHX11CTC"],                           "coronal cavities 12": dataDental2017_2018["OHX12CTC"],                           "coronal cavities 13": dataDental2017_2018["OHX13CTC"],                           "coronal cavities 14": dataDental2017_2018["OHX14CTC"],                           "coronal cavities 15": dataDental2017_2018["OHX15CTC"],                           "coronal cavities 18": dataDental2017_2018["OHX18CTC"],                           "coronal cavities 19": dataDental2017_2018["OHX19CTC"],                           "coronal cavities 20": dataDental2017_2018["OHX20CTC"],                           "coronal cavities 21": dataDental2017_2018["OHX21CTC"],                           "coronal cavities 22": dataDental2017_2018["OHX22CTC"],                           "coronal cavities 23": dataDental2017_2018["OHX23CTC"],                           "coronal cavities 24": dataDental2017_2018["OHX24CTC"],                           "coronal cavities 25": dataDental2017_2018["OHX25CTC"],                           "coronal cavities 26": dataDental2017_2018["OHX26CTC"],                           "coronal cavities 27": dataDental2017_2018["OHX27CTC"],                           "coronal cavities 28": dataDental2017_2018["OHX28CTC"],                           "coronal cavities 29": dataDental2017_2018["OHX29CTC"],                           "coronal cavities 30": dataDental2017_2018["OHX30CTC"],                           "coronal cavities 31": dataDental2017_2018["OHX31CTC"]})
dfDental4["year"] = "2017-2018"
#print(dfDental1)

#combining dataset
framesDental = [dfDental1, dfDental2, dfDental3, dfDental4]
resultDental = pd.concat(framesDental)
print(resultDental)
resultDental.to_pickle("dfDentalNew.pkl")

print("starting c")
print("There are a total of 39156 cases in the dataframe for part a.")
print("There are a total of 35909 cases in the dataframe for part b.")


# In[ ]:




