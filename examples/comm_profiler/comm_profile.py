import argparse
import json
import torch
from pathlib import Path
import os
from typing import Tuple, List, Dict

import torch.distributed
import torch.profiler

import nnscaler
from nnscaler.runtime.adapter.collectives import all_gather, all_reduce, all_to_all, reduce_scatter, move
from nnscaler.profiler import CudaTimer
from nnscaler.runtime.device import DeviceGroup
from nnscaler.autodist.util import get_node_arch, get_default_profile_path

class CommProfiler:

    def __init__(self,
                 nranks: int,
                 warmup_times: int = 10,
                 profile_times: int = 10) -> None:
        self.nranks = nranks
        self.warmup_times = warmup_times
        self.profile_times = profile_times
        self.ranks = tuple(range(self.nranks))

    def collect_profile_info(self,
                             primitive: str) -> Tuple[List[float], List[float]]:

        b_size = 1
        sequence_len = 4096
        element_size=4
        sizes_in_mb=[0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024,2048]
        
        model_dim_list = [
            int(mem * 1024 * 1024 //element_size // b_size // sequence_len)
            for mem in sizes_in_mb
        ]
        times_in_s = []


        for cur_sz, d_size in zip(sizes_in_mb, model_dim_list):
            assert d_size % self.nranks == 0
            if primitive in ['all gather', 'all to all']:
                d_size = d_size // self.nranks
            tensor = torch.rand([b_size, sequence_len, d_size],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
            if primitive == 'all gather':
                func = all_gather
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': self.ranks}
            elif primitive == 'all reduce':
                func = all_reduce
                kwargs = {'tensor': tensor, 'ranks': self.ranks}
            elif primitive == 'reduce scatter':
                func = reduce_scatter
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': self.ranks}
            elif primitive == 'all to all':
                func = all_to_all
                kwargs = {
                    'tensor': tensor,
                    'idim': 0,
                    'odim': 2,
                    'ranks': self.ranks
                }
            elif primitive == 'move':
                func = move
                kwargs = {
                    'tensor': tensor,
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'src': 0,
                    'dst': 1
                }
            else:
                raise ValueError('Unknown primitive: {}'.format(primitive))
            
            if primitive == 'move' and (torch.distributed.get_rank() != 0 and torch.distributed.get_rank()!=1):
                return sizes_in_mb, times_in_s
            else:
                if torch.distributed.get_rank() == 0: print(f'{d_size}_1')
                for _ in range(self.warmup_times):
                    # if torch.distributed.get_rank() == 0: print(f'{_}')
                    func(**kwargs)
                CudaTimer().clear()
                # if torch.distributed.get_rank() == 0: print(f'{d_size}_2')
                for _ in range(self.profile_times):
                    otensor = func(**kwargs)
                # if torch.distributed.get_rank() == 0: print(f'{d_size}_3')
                cur_t = CudaTimer().instance.field_data[primitive] / self.profile_times
                times_in_s.append(cur_t)
        return sizes_in_mb, times_in_s

    def collect_internode_profile_info(self,
                             primitive: str,ranks=[]) -> Tuple[List[float], List[float]]:
        rank = torch.distributed.get_rank()
        if rank not in ranks:
            return None
        b_size = 16
        sequence_len = 16
        element_size=4
        sizes_in_mb=[0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024,2048, 4096, 8192]
        model_dim_list = [
            int(mem * 1024 * 1024 //element_size // b_size // sequence_len)
            for mem in sizes_in_mb
        ]
        times_in_s = []

        for cur_sz, d_size in zip(sizes_in_mb, model_dim_list):
            assert d_size % len(ranks) == 0
            if primitive in ['all gather', 'all to all']:
                d_size = d_size // len(ranks)
            tensor = torch.rand([b_size, sequence_len, d_size],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
            if primitive == 'all gather':
                func = all_gather
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': ranks}
            elif primitive == 'all reduce':
                func = all_reduce
                kwargs = {'tensor': tensor, 'ranks': ranks}
            elif primitive == 'reduce scatter':
                func = reduce_scatter
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': ranks}
            elif primitive == 'all to all':
                func = all_to_all
                kwargs = {
                    'tensor': tensor,
                    'idim': 0,
                    'odim': 2,
                    'ranks': ranks
                }
            elif primitive == 'move':
                func = move
                local_world_size = int(os.environ["LOCAL_WORLD_SIZE"]) 
                kwargs = {
                    'tensor': tensor,
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'src': 0,
                    'dst': local_world_size
                }
            else:
                raise ValueError('Unknown primitive: {}'.format(primitive))
            
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"]) 
            if primitive == 'move' and (torch.distributed.get_rank() != 0 and torch.distributed.get_rank()!= local_world_size):
                 return sizes_in_mb, times_in_s
            else:
                if torch.distributed.get_rank() == 0: print(f'{d_size}_1')
                for _ in range(self.warmup_times):
                    # if torch.distributed.get_rank() == 0: print(f'{_}')
                    func(**kwargs)
                CudaTimer().clear()
                # if torch.distributed.get_rank() == 0: print(f'{d_size}_2')
                for _ in range(self.profile_times):
                    otensor = func(**kwargs)
                # if torch.distributed.get_rank() == 0: print(f'{d_size}_3')
                cur_t = CudaTimer().instance.field_data[primitive] / self.profile_times
                times_in_s.append(cur_t)
        return sizes_in_mb, times_in_s

    def profile(self) -> Dict[str, Tuple[List[float], List[float]]]:
        profile_info = {}
        for primitive in [
                'all gather', 'all reduce', 'reduce scatter', 'all to all', 'move'
        ]:
            print(f'{primitive=}')
            profile_info[primitive] = self.collect_profile_info(
                primitive=primitive)
            torch.distributed.barrier()
        return profile_info

    def profile_inter_node(self) -> Dict[str, Tuple[List[float], List[float]]]:
        profile_info = {}
        for primitive in [
                'all gather', 'all reduce', 'reduce scatter', 'all to all', 'move'
        ]:
        # for primitive in [
        #         'all reduce'
        # ]:
            print(f'{primitive=}')
            profile_info[primitive] = self.collect_internode_profile_info(
                primitive=primitive,ranks=self.ranks)
            torch.distributed.barrier()
        return profile_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Profile runtime communication cost')
    parser.add_argument('--comm_profile_dir',
                        type=str,
                        default=get_default_profile_path() / get_node_arch() / 'comm',
                        help='autodist comm profile folder')
    parser.add_argument('--group_id',
                        type=int,
                        default=0,
                        help='group_id')
    args = parser.parse_args()

    nnscaler.init()
    

    CudaTimer(enable=True, predefined=True)
    world_size = DeviceGroup().world_size
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"]) 

    comm_profiler = CommProfiler(nranks=world_size)
    print(f'CommProfiler initialized with {world_size=}')
    if world_size == local_world_size:
        # profile intra node info
        profile_info = comm_profiler.profile()
        print(f'Profile done')

        if torch.distributed.get_rank() == 0:
            dir_path = Path(args.comm_profile_dir)
            print(f'{world_size=}, {dir_path=}')
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            file_name = dir_path / f'intra_{world_size}.json'
            with open(file_name, 'w') as f:
                json.dump(profile_info, f, indent=2)

    else: # profile inter node info
        profile_inter_info = comm_profiler.profile_inter_node()
        nnodes = world_size // local_world_size
        if torch.distributed.get_rank() == 0:
            dir_path = Path(args.comm_profile_dir)
            file_name = dir_path / f'inter_({nnodes}, {local_world_size}).json'
            with open(file_name, 'w') as f:
                json.dump(profile_inter_info, f, indent=2)
