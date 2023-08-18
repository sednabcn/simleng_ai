"""
#==["data_unpack_kwargs","data_list_unpack_dict","data_extra_dict_to_list",
"data_list_to_matrix","data_list_to_transform_to_integer",
"add_data_list_of_several_lists","add_index_to_list_of_indexes",
"get_index_of_list_from_list","x_add_list","list_to_list_of_dict",
"list_dict_double_keys_from_dict","reduce_double_to_single_key", 
"reduce_value_list_double_to_single_key",
"reduce_multiples_lists_double_to_single_key", 
"get_index_a_x_b","array_sort",
"array_sort_y_on_x","list_plust_list","list_minus_list","list_inter_list",
"order_mode"]
"""
def order_mode(data,tsearch):
     from collections import OrderedDict
     order=OrderedDict()
     
     list_base=data
     list_search=tsearch
     if not isinstance(data,list):
         list_base=list(data)
     if not isinstance(tsearch,list):
         list_search=[]
         for item in tsearch:
             list_item=item
             if not(isinstance(item,list)):
                 list_item=list(item)
             list_search.append(list_item)
        
     for item in list_base:
         order[item]=list_base.index(item)
     
     f_order=lambda x:order[x]
    
     order_search=list(map(f_order,list_search))
 
     if sorted(order_search)!=order_search:
         return False
     else:
         return True
def list_plus_list(x,y):
        from resources.sets import list_minus_list
        ladd=[]
        for name in x :
            ladd.append(name)
        ladd+=list_minus_list(y,ladd)
        return ladd

def list_minus_list(x,y):
        ldiff=[]
        s=set(y)
        ldiff=[ name for name in x if name not in s]
        return ldiff

def list_inter_list(x,y):
        linter=[value for value in x if value in y]
        return linter

def get_index_a_x_b(a,b,x):
        from resources.sets import list_minus_list
        import numpy as np
        eps =1e-10
        ind_b=np.array(np.where(x < b-eps))[0].tolist()
        ind_a=np.array(np.where(x < a-eps))[0].tolist()
        ind=list_minus_list(ind_b,ind_a)
        return ind
def array_sort(x,axis):
        import numpy as np
        x=np.array(x)
        xx=np.empty(x.shape)
        ind=np.argsort(x,axis=axis)
        for ii,n in zip(range(len(ind)),ind):
            xx[ii]=x[n]
        return xx
def array_sort_y_on_x(x,y,axis):
        import numpy as np
        x=np.array(x)
        y=np.array(y)
        xx=np.empty(x.shape)
        yy=np.empty(y.shape)
        ind=np.argsort(x,axis=axis)
        #print(x.shape)
        #print(y.shape)
        for ii,n in zip(range(len(ind)),ind):
            xx[ii]=x[n]
            yy[ii]=y[n]

        return xx,yy

def get_index_of_list_from_list(y,x):
        """Given list x get indexes of elements of y"""
        index=[]
        for ii in y:
            index.append(x.index(ii))
        return index


def add_index_to_list_of_indexes(x,index,kind):
        """Add an index to list of indexes"""
        """
        |x-list of indexes
        | index-to add
        | kind (bool) True:str
        |              False:alphanumeric
        """
        import pandas as pd
        pd_index_list=pd.Index.tolist(x)
        if kind==True:
            index=str(index)
        pd_index_list.append(index)
        pd_index=[str(name) for name in pd_index_list]
        return pd.Index(pd_index)

   
def data_unpack_kwargs(**kwargs):
        # data is a list of keys and values
        # kind='list' to list
        #kind='dict' to dict
        keyss=kwargs.keys()
        data={}
        for name in keyss:
            data[name]=kwargs[name]
        return data
    
def data_list_unpack_dict(dict):
            data=[]
            for name in dict.keys():
                data_list=dict[name]
                if len(data_list)==1:
                    data_list=list(dict[name])[0]
                data.extend(data_list)
            return data
        
def data_extract_dict_to_list(kind,**kwargs):
        from resources.sets import data_unpack_kwargs,data_list_unpack_dict
        data=data_unpack_kwargs(**kwargs)
        data_dict={}
        for k,name in enumerate(kwargs.keys()):
            if name==kind:
                data_dict[kind]=data[kind]
                data_list=data_list_unpack_dict(data_dict)
                return data_list
    
def data_list_to_matrix(x,shape):
        import numpy as np
        Lx=len(x)
        Ls=len(shape)
        if Ls==2: [Nx,Ny]=shape[:]
        if Ls==3: [Nx,Ny,Nz]=shape[:]
        if Ls==2:
            matrix=np.empty((Nx,Ny))
            matrix=np.reshape(x,(Nx,Ny))
        assert Nx,Ny==matrix.shape
        return matrix
    
    
def data_list_to_transform_to_integer(x):
        """get an integers list """
        xx=[]
        for ii in x:
            xx.append(int(ii))
        return xx
    
    
def add_data_list_of_several_lists(nn,x):
        LEN=0
        for ii in range(nn):
            LEN+=len(x[ii])
        return LEN,print(LEN)
 

def x_add_list(list,x):
        """Add a real number to the list"""
        if len(list)==0:
            return [x]
        else:
            return [x+u for u in list]

def reduce_multiples_lists_double_to_single_key(dict0,nameslist):
        if len(namelist)!=list(dict0.values):
            return print("Rise error input lists must have the same length")
        for key,value,name in zip(dict0,namelist):
            name=reduce_one_list_double_to_single_key(value)
            
def reduce_value_list_double_to_single_key(dict0):
        data_dict={}
        for  key1,dict1 in dict0.items():
            if isinstance(dict1,list) and len(dict1)>0:
                for dict2 in dict1:
                    if  isinstance(dict2,dict):
                        data_dict.update(dict2)
            elif isinstance(dict1,dict):
                data_dict.update(reduce_double_to_single_key(dict))
            elif isinstance(dict1,list) and len(dict1)==0:
                return {}
            else:
                pass
        return data_dict
    
                    
                
def reduce_double_to_single_key(dict0):
        data_dict={}
        for key,dict1 in dict0.items():
            if isinstance(dict1,dict):
                data_dict.update(dict1)
        return data_dict
    
def list_dict_double_keys_from_dict(dict0):
        """dict.values are lists of dicts"""
        data_list=[]
        for key1,dict1 in dict0.items():
            if len(dict1)>0:
                for dict2 in dict1:
                    if isinstance(dict2,dict):
                        for key2,val in dict2.items():
                            data_list.append({(key1,key2):val})
                    else:
                        pass
        return data_list 

def list_to_list_of_dict(object):
        P=lambda x:x[0] if len(x)==1 else list(x)
        list0=[]
        for u in zip(*object):
            list0.append({u[0]:P((u[1:]))})
        return list0
