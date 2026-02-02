#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Optional, Dict, List, Any
import time
import logging
import json
from pathlib import Path

import torch
from nnscaler.utils import print_each_rank

_logger = logging.getLogger(__name__)


class CudaTimer:
    r"""
    Singleton Cuda Timer

    Note that frequently using timer may decrease the performance.

    The runtime predefines the timer on each communication primitive.
    By default, the timer on communications are disabled for higher performance.
    For users who want to analyze communication overhead, turn on the timer
    by using `CudaTimer(enable=True, predefined=True)`.

    There are two switches to allow user to control the timer behaviour

    * enable:
        the overall controller to turn on/off the all profiling.
    * predefined:
        the controller to turn on/off the predefined timer (mostly are communications)
    """
    class __CudaTimer:

        def __init__(self, enable = True, predefined = False):
            self.start_t = None
            self.stop_t = None
            self.field = dict()
            self.field_data = dict()
            self.enabled = enable
            self.predefined = predefined
            self.count=0
            
            # 新增：通信日志记录
            self.comm_logs = []
            self.comm_log_file = None
            self.log_comm_details = False  # 是否记录每次通信的详细信息
    
    instance = None

    def __init__(self, enable: Optional[bool] = None, predefined: Optional[bool] = None):
        # not have instance
        if not self.instance:
            enable = enable if enable is not None else True
            predefined = predefined if predefined is not None else False
            CudaTimer.instance = CudaTimer.__CudaTimer(enable, predefined)
        # have instance
        else:
            if enable is not None:
                self.instance.enabled = enable
            if predefined is not None:
                self.instance.predefined = predefined
    
    def enable_comm_logging(self, log_file: Optional[str] = None):
        """
        启用通信操作详细日志记录
        
        @param log_file str: 日志文件路径，如果为 None 则只保存在内存中
        """
        self.instance.log_comm_details = True
        if log_file:
            self.instance.comm_log_file = Path(log_file)
    
    def start(self, field_name='default', predefined: bool = False, stream: Optional[torch.cuda.Stream] = None, 
              comm_ranks: Optional[List[int]] = None, comm_size: Optional[int] = None):
        """
        Start recording time on the the field

        Note `start` and `stop` on the same field can be called nestly

        @param field_name str: 字段名，通常为通信操作类型（如 'all_reduce', 'all_gather' 等）
        @param predefined bool: whether the field is a predefined field
        @param stream Optional[torch.cuda.Stream]: CUDA stream
        @param comm_ranks Optional[List[int]]: 参与通信的 ranks
        @param comm_size Optional[int]: 通信数据大小（字节数）

        @return None
        """
        if (not self.instance.enabled) or (predefined and not self.instance.predefined):
            return
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        # torch.cuda.default_stream().synchronize()
        start_time = time.time()
        if field_name not in self.instance.field:
            self.instance.field[field_name] = list()
            self.instance.field_data[field_name] = 0
        self.instance.field[field_name].append(start_time)
        
        # 如果启用了通信日志记录，保存额外的信息
        if self.instance.log_comm_details and predefined:
            self.instance.field[field_name].append({
                'ranks': comm_ranks,
                'size': comm_size,
                'start_time': start_time
            })
    
    def stop(self, field_name='default', predefined: bool = False, stream: Optional[torch.cuda.Stream] = None) -> float:
        """
        Record the time span from last `start` on the same field_name to now

        @param field_name str
        @param predefined bool: whether the field is a predefined field
        @param stream Optional[torch.cuda.Stream]: CUDA stream

        @return span float: time span in seconds
        """
        if (not self.instance.enabled) or (predefined and not self.instance.predefined):
            return
        if field_name not in self.instance.field:
            raise RuntimeError("Missing start on the field")
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        # torch.cuda.default_stream().synchronize()
        stop_time = time.time()
        
        # 检查是否有通信日志信息
        comm_info = None
        if self.instance.log_comm_details and predefined:
            comm_info = self.instance.field[field_name].pop(-1)
            start_time = comm_info['start_time']
        else:
            start_time = self.instance.field[field_name].pop(-1)
        
        span = stop_time - start_time # in seconds
        self.instance.field_data[field_name] += span
        
        # 记录通信日志
        if self.instance.log_comm_details and predefined and comm_info:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            log_entry = {
                'timestamp': start_time,
                'rank': rank,
                'op_type': field_name,
                'tensor_size_bytes': comm_info.get('size', 0),
                'ranks': comm_info.get('ranks', []),
                'latency_ms': span * 1000
            }
            self.instance.comm_logs.append(log_entry)
        
        return span

    def duration(self, times: int, field_name: str = 'default') -> float:
        """
        Get the total span (wall clock) of a field name. The span is divided by times.

        @param times int: division factor
        @param field_name str: the field name

        @return span float: wall clock in milliseconds.
        """
        if field_name not in self.instance.field:
            _logger.warning(f"CudaTimer: {field_name} doesn't record.")
            return 0.0
        if len(self.instance.field[field_name]) != 0:
            raise RuntimeError(f"timer for field {field_name} not stopped")
        return self.instance.field_data[field_name] / times * 1000  # in ms

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def clear(self):
        CudaTimer.instance = CudaTimer.__CudaTimer(
            enable=self.enabled, predefined=self.predefined
        )

    def print_all(self, times: int, rank_only: Optional[int] = None):
        """
        Print the total span of each recorded field divided by `times`

        Note this should be called by each process

        @param times int: division factor
        @param rank_only Optional[int]: select only one rank for print

        @return None
        """
        msg = list()
        names = list(self.instance.field_data.keys())
        names.sort()
        for field_name in names:
            span = self.duration(times, field_name)
            msg.append('{} : {:.2f} ms'.format(field_name, span))
        msg = ' | '.join(msg)
        print_each_rank(msg, rank_only)
    
    def save_comm_logs(self):
        """
        保存通信日志到文件
        
        @return None
        """
        if not self.instance.log_comm_details or not self.instance.comm_logs:
            return
        
        if self.instance.comm_log_file:
            # 按时间戳排序
            sorted_logs = sorted(self.instance.comm_logs, key=lambda x: x['timestamp'])
            with open(self.instance.comm_log_file, 'w') as f:
                json.dump(sorted_logs, f, indent=2)
            _logger.info(f"Communication logs saved to {self.instance.comm_log_file}")
    
    def get_comm_logs(self) -> List[Dict[str, Any]]:
        """
        获取通信日志
        
        @return List[Dict]: 通信日志列表
        """
        return self.instance.comm_logs.copy()

    def warmup(self, seconds=1.0):
        """
        Warm up GPU for `span` seconds.
        """
        print('> warming up for 1 second')
        data1 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        data2 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        # warm up 1s
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < seconds:
            _ = torch.matmul(data1, data2)
            # if torch.distributed.is_initialized():
            #     torch.distributed.all_reduce(out)
            torch.cuda.synchronize()
