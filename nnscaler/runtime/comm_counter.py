from datetime import datetime  
import os
import json
import torch
class Comm_counter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Comm_counter, cls).__new__(cls, *args, **kwargs)
            cls._instance.rank=-1
            cls._instance.iteration = 0  # 初始化属性
            cls._instance.reduce_scatter_tensor_total_size_MB=0
            cls._instance.reduce_scatter_tensor_shape=[]
            cls._instance.reduce_scatter_time=0
            cls._instance.reduce_scatter_count=0
            cls._instance.reduce_scatter_nranks=0
            cls._instance.allreduce_tensor_total_size_MB=0
            cls._instance.allreduce_tensor_shape=None
            cls._instance.allreduce_time=0
            cls._instance.allgather_tensor_total_size_MB=0
            cls._instance.allgather_tensor_shape=[]
            cls._instance.allgather_time=0
            now = datetime.now()
            date_str=now.strftime("%m-%d")
            time_str=now.strftime("%H-%M")
            cls._instance.date_str=date_str
            cls._instance.time_str=time_str
            cls._instance.folder_path = f"/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-new/examples/logs/comm_time/{date_str}/{time_str}/"
            os.makedirs(cls._instance.folder_path, exist_ok=True)
        return cls._instance

    def set_iteration(self, iteration):
        self.iteration =iteration
        self.rank=torch.distributed.get_rank()
    
    def add_reduce_scatter_tensor_total_size_MB(self,size_MB):
        self.reduce_scatter_tensor_total_size_MB+=size_MB
    
    def set_reduce_scatter_tensor_shape(self,shape):
        if shape not in self.reduce_scatter_tensor_shape:
            self.reduce_scatter_tensor_shape.append(shape)

    def add_reduce_scatter_time(self,time):
        self.reduce_scatter_time+=time
        self.reduce_scatter_count+=1
    
    def set_reduce_scatter_nranks(self,nranks):
        self.reduce_scatter_nranks=nranks
    
    def add_allreduce_tensor_total_size_MB(self,size_MB):
        self.allreduce_tensor_total_size_MB+=size_MB
    
    def set_allreduce_tensor_shape(self,shape):
        if not self.allreduce_tensor_shape:
            self.allreduce_tensor_shape = shape  
    
    def add_allreduce_time(self,time):
        self.allreduce_time+=time
    
    def add_allgather_tensor_total_size_MB(self,size_MB):
        self.allgather_tensor_total_size_MB+=size_MB
    
    def set_allgather_tensor_shape(self,shape):
        if shape not in self.allgather_tensor_shape:
            self.allgather_tensor_shape.append(shape)
    
    def add_allgather_time(self,time):
        self.allgather_time+=time
    

    
    def dump(self):
        json_file_path = f"{self.folder_path}/comm_info_{self.rank}.json"
        if not os.path.exists(json_file_path):
            # 初始化一个空的 JSON 文件
            with open(json_file_path, "a") as f:
                data = {
                    "Framework":"nnscaler",
                    "iteration":self.iteration,
                    "reduce_scatter_tensor_total_size_MB":self.reduce_scatter_tensor_total_size_MB,
                    "reduce_scatter_tensor_shape":self.reduce_scatter_tensor_shape,
                    "reduce_scatter_time":self.reduce_scatter_time,
                    "reduce_scatter_nranks":self.reduce_scatter_nranks,
                    "reduce_scatter_count":self.reduce_scatter_count,
                    "allreduce_tensor_total_size_MB":self.allreduce_tensor_total_size_MB,
                    "allreduce_tensor_shape":self.allreduce_tensor_shape,
                    "allreduce_time":self.allreduce_time,
                    "allgather_tensor_total_size_MB":self.allgather_tensor_total_size_MB,
                    "allgather_tensor_shape":self.allgather_tensor_shape,
                    "allgather_time":self.allgather_time
                    }
                json.dump(data, f, indent=4)
