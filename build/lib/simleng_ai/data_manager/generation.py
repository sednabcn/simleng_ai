import os
from data_abstract import DATA
from ini.init import Init
from resources.sets import reduce_value_list_double_to_single_key
from resources.io import find_full_path,file_reader
from resources.manipulation_data import data_dummy_binary_classification
from resources.pandas import mapping_zero_one
from resources.db import MyDB

#file_input='simlengin.txt'

MACROS=['Simlengin.txt','DATA_PROJECT','FILES','LISTFILES','FEATURES','TARGET','STRATEGIES','PARAMS']


class Data_Generation(DATA,Init):
       
       def __init__(self,file_input,score):
            self.file_input=file_input
            self.score=score
            super().__init__(file_input,score)
            self.title=None 
            self.dataset={}
            self.files={}
            self.listfiles={}
            self.features={}
            self.target={}
            self.strategies={}
            self.params={}
            
            self.data_dummy_train={}
            self.data_dummy_test={}

            
            MACROSIN=Init(file_input,score).get_macros()
            #print(list(MACROSIN.items()))
            
            input={}   
            for key,value in MACROSIN.items():  
                          input.update(reduce_value_list_double_to_single_key({key:value}))
                        
                          if key=="DATA_PROJECT":
                                self.dataset.update(input)
                                input={}
                          elif key=="FILES":
                                self.files.update(input)
                                input={}
                          elif key=="LISTFILES":
                                self.listfiles.update(input)
                                input={}
                          elif key=="FEATURES":
                                self.features.update(input)
                                input={}
                          elif key=="TARGET":
                                self.target.update(input)
                                if not self.target['SCORE'].isnumeric():
                                       self.target['SCORE']=self.score
                                input={}
                          elif key=="STRATEGIES":
                                self.strategies.update(input)
                                input={}
                          elif key=="PARAMS":
                                self.params.update(input) 
                          else:
                                self.title=key
                                
       def print_input_info(self):
                 #programming try to empty results  
                 return print(self.title,self.dataset ,self.files,self.listfiles,\
                              self.features,self.target,self.strategies,self.params)
                                         
       def get_input_info(self):
                 #programming try to empty results  
                 return self.title,self.dataset ,self.files,self.listfiles,\
          self.features,self.target,self.strategies,self.params

       """
       Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0] on linux
       Type "help", "copyright", "credits" or "license" for more information.
       >>> from data_manager.generation import Data_Generation
       >>> Data_Generation().get_input_info()
       Simlengin.txt {'DATASET': 'pimas', 'TYPE': 'numeric', 'STRUCT': 'table', 'SYNTHETIC': 'False', 'DATASOURCE': 'table', 'IMBALANCE': 'False', 'DUMMY': 'True'} {'NUMBER': '1', 'FORMAT': 'csv', 'reader': 'pandas', 'mode': 'read', 'index_col': '0', 'header': 'None', 'sep': '\\n'} {'filename': 'pima-indians-diabetes.csv'} {} {'GOAL': 'CLASSI', 'NCLASS': '2', 'METHOD': 'SUPERV', 'SOLVER': 'stats', 'SCORE': '---', 'SPLITDATA': '0.25'}
       """
       def get_update_parameters_to_read_files(self):
                  pass   
                
       def data_generation_from_table(self):
           """load data_from_file"""
           from sklearn.model_selection import train_test_split
           from resources.pandas import col_data
           from resources.manipulation_data import isdummy,isimbalance
           import pandas as pd
           
           type_file_pd=['csv','xls','xlsx','json','sql',\
                         'html','pickle','tsv','excel']
           type_file_reader=['tar','tar.gz','pdf','HDF5','docx','mp3','mp4']

           type_file_open=["txt"]

           imbalance=self.dataset["IMBALANCE"]
           
           nfiles=self.files['NUMBER']

           type_file=self.files['FORMAT']

          
           reader=self.files["READER"]

           mode=self.files["MODE"]

           index_col=self.files['INDEX_COL']

           header=self.files["HEADER"]

           sep=self.files["SEP"]

           pars=[mode,index_col,header,sep]

     
           name=self.listfiles['FILENAME']

           test_size=float(self.target['SPLITDATA'])
           
           train_size= 1.0 - test_size
           
           self.full_path=find_full_path(name)
           
           self.data=file_reader(self.full_path,type_file,*pars)

           #update datasets_store

           dataset=self.dataset['DATASET']
           source=self.dataset['DATASOURCE']
           
           MyDB(dataset,'datasets',kind=source).datasets_store()
           
           # columns_data_to selected
           last=col_data(self.data,-1)

           nolast=col_data(self.data,range(self.data.shape[1]-1))

           
           if not(imbalance):
                  stratify=None
           else:
                  stratify=last

           #print("pimas",'\n',self.data)
           
           #print("Y_data",'\n',last,'\n',"X_data",'\n',nolast)
           
           self.X_train, self.X_test, self.y_train, self.y_test= \
           train_test_split(nolast,last, test_size=test_size, random_state=1,\
                       stratify=stratify)

           #Older implementations of data_base COMPATIBILITY
           train_size=self.X_train.shape[0]
           df=pd.DataFrame()
           df=self.data.iloc[:train_size,:]
           de=pd.DataFrame()
           de=self.data.iloc[train_size:,:]
           self.data_train={}
           self.data_train["train"] = [self.X_train.columns,self.X_train,self.y_train,df]
           self.data_test={}
           self.data_test["test"]=[self.X_test.columns,self.X_test,self.y_test,de]
           #=====================END COMPATIBILITY============================== 
           if not(imbalance):
                  stratify=None
           else:
                  stratify=self.y_train
           
           self.X_train_val, self.X_test_val, self.y_train_val, self.y_test_val= \
           train_test_split(self.X_train, self.y_train, test_size=0.25*test_size, random_state=1,\
                            stratify=stratify) # 0.25 x 0.8 = 0.2
           
           #print (self.X_train.columns)
           
           return self.data_train,self.data_test, self.data,self.X_train, self.X_test, self.y_train, self.y_test, \
                  self.X_train_val, self.X_test_val, self.y_train_val, self.y_test_val 

       def data_generation_binary_classification(self):
          """Generation of dummy variables to Binary Classification Task
             for a balanced dataset
          """
          from resources.pandas import mapping_zero_one
          from resources.manipulation_data import isdummy
          
          data_dummy_train={}
          data_dummy_test={}

          # Dummy variables to categorical variable
          if  isdummy(self.y_train):
                
                V_train=self.y_train
                V_test=self.y_test
          else:
                V_train=data_dummy_binary_classification(self.y_train,"No",0,1)
                V_test=data_dummy_binary_classification(self.y_test,"No",0,1)
           
          # Mapping 'Training and Testing data to [0,1]".X->U
          if isdummy(self.X_train):
                 U_train=X_train
                 U_test=X_test
          else:
                 U_train=mapping_zero_one(self.X_train)
                 U_test=mapping_zero_one(self.X_test)

          self.data_dummy_train['train']=[U_train,V_train]
          self.data_dummy_test['test']=[U_test,V_test]


          return self.data_dummy_train,self.data_dummy_test
          
               
       def data_generation_target(self):
                target=self.target['GOAL']
                nclass=int(self.target['NCLASS'])
                
                if target=='CLASSIFICATION':
                       if nclass==2:
                          return self.data_generation_binary_classification()                             
                       elif nclass >2:
                          pass
                       else:
                            pass             
                elif target=='REGRESSION':
                       pass

                else:
                       pass
       def data_generation_functions(self):
                 source=self.dataset['DATASOURCE'] 
                 if source=='table':
                       return self.data_generation_from_table()
                 elif source=='cloud':
                       return data_generation_from_cloud()
                 elif source=='web':
                       return data_generation_from_web()
                 elif source=='multiple_sources':
                       return data_generation_from_multiple_sources()
                 elif source=='streaming':
                       return data_generation_from_streaming_data()
                 elif source=='geophysics':
                       return data_generation_from_geophysics()
                 elif source=='synthetic':
                       return data_generation_syntehtic_data()
                 elif source=='time_series':
                       return data_generation_from_time_series()
                 else:
                       pass

"""
EXAMPLES:
Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from data_manager.generation import Data_Generation
>>> Data_Generation().data_generation_from_table()
pimas 
      npreg  glu  bp  skin  ins   bmi    ped  age  type
0        6  148  72    35    0  33.6  0.627   50     1
1        1   85  66    29    0  26.6  0.351   31     0
2        8  183  64     0    0  23.3  0.672   32     1
3        1   89  66    23   94  28.1  0.167   21     0
4        0  137  40    35  168  43.1  2.288   33     1
..     ...  ...  ..   ...  ...   ...    ...  ...   ...
763     10  101  76    48  180  32.9  0.171   63     0
764      2  122  70    27    0  36.8  0.340   27     0
765      5  121  72    23  112  26.2  0.245   30     0
766      1  126  60     0    0  30.1  0.349   47     1
767      1   93  70    31    0  30.4  0.315   23     0

[768 rows x 9 columns]
y_data 
 0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: type, Length: 768, dtype: int64 
X_data 
      npreg  glu  bp  skin  ins   bmi    ped  age
0        6  148  72    35    0  33.6  0.627   50
1        1   85  66    29    0  26.6  0.351   31
2        8  183  64     0    0  23.3  0.672   32
3        1   89  66    23   94  28.1  0.167   21
4        0  137  40    35  168  43.1  2.288   33
..     ...  ...  ..   ...  ...   ...    ...  ...
763     10  101  76    48  180  32.9  0.171   63
764      2  122  70    27    0  36.8  0.340   27
765      5  121  72    23  112  26.2  0.245   30
766      1  126  60     0    0  30.1  0.349   47
767      1   93  70    31    0  30.4  0.315   23
"""
"""
EXAMPLE:data_generarion_binary_classification
Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from data_manager.generation import Data_Generation
>>> Data_Generation().data_generation_binary_classification()
({'train': [        npreg       glu        bp      skin       ins       bmi       ped       age
678  0.176471  0.608040  0.472727  0.000000  0.000000  0.536513  0.018408  0.066667
53   0.470588  0.884422  0.818182  0.343434  0.441176  0.502235  0.163955  0.616667
665  0.058824  0.562814  0.727273  0.454545  0.194118  0.518629  0.056935  0.050000
219  0.294118  0.562814  0.600000  0.000000  0.000000  0.563338  0.075771  0.333333
720  0.235294  0.417085  0.781818  0.191919  0.000000  0.436662  0.099743  0.216667
..        ...       ...       ...       ...       ...       ...       ...       ...
88   0.882353  0.683417  0.636364  0.323232  0.161765  0.552906  0.029538  0.366667
137  0.000000  0.467337  0.545455  0.252525  0.135294  0.427720  0.191781  0.016667
645  0.117647  0.788945  0.672727  0.353535  0.647059  0.587183  0.021404  0.150000
308  0.000000  0.643216  0.618182  0.191919  0.264706  0.454545  0.559503  0.066667
332  0.058824  0.904523  0.000000  0.000000  0.000000  0.645306  0.084760  0.333333

[614 rows x 8 columns], 678    1
53     1
665    0
219    1
720    0
      ..
88     1
137    0
645    0
308    1
332    1
Name: type, Length: 614, dtype: int64]}, {'test': [        npreg       glu        bp      skin       ins       bmi       ped       age
488  0.307692  0.502538  0.590164  0.283333  0.000000  0.465455  0.095957  0.137255
413  0.076923  0.725888  0.606557  0.366667  0.072104  0.476364  0.079076  0.000000
112  0.076923  0.451777  0.622951  0.566667  0.043735  0.567273  0.050644  0.039216
222  0.538462  0.604061  0.000000  0.000000  0.000000  0.458182  0.058196  0.313725
711  0.384615  0.639594  0.639344  0.450000  0.026005  0.538182  0.160373  0.372549
..        ...       ...       ...       ...       ...       ...       ...       ...
139  0.384615  0.532995  0.590164  0.483333  0.384161  0.670909  0.035984  0.137255
178  0.384615  0.725888  0.639344  0.000000  0.000000  0.818182  0.049756  0.509804
654  0.076923  0.538071  0.573770  0.466667  0.159574  0.621818  0.028432  0.019608
110  0.230769  0.868020  0.590164  0.550000  0.159574  0.605455  0.053754  0.058824
102  0.000000  0.634518  0.786885  0.000000  0.000000  0.409091  0.081741  0.000000

[154 rows x 8 columns], 488    0
413    0
112    0
222    0
711    0
      ..
139    0
178    0
654    0
110    1
102    0
Name: type, Length: 154, dtype: int64]})
>>> 
"""
