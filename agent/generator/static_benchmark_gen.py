from __future__ import annotations
from os import times
import random
import math
from typing import List, Optional, Dict, Tuple
from .base import BaseDemandGenerator, Demand
import numpy as np
import os
import pandas as pd

def load_saved_dataset(customers_csv_path: str):
    """
    读取保存的CSV文件并重建DataFrame结构
    
    Args:
        customers_csv_path: 客户数据CSV文件路径
        
    Returns:
        重建的DataFrame（包含attrs属性）
    """
    # 读取客户数据
    df = pd.read_csv(customers_csv_path)
    
    # 尝试读取相应的车辆信息文件
    base_path = os.path.dirname(customers_csv_path)
    base_name = os.path.basename(customers_csv_path).replace('_customers.csv', '')
    
    vehicle_csv_path = os.path.join(base_path, f"{base_name}_vehicle.csv")
    
    if os.path.exists(vehicle_csv_path):
        vehicle_df = pd.read_csv(vehicle_csv_path)
        if not vehicle_df.empty:
            vehicle_info = vehicle_df.iloc[0].to_dict()
            df.attrs['vehicle_info'] = vehicle_info
    
    return df

class StaticBenchmarkGenerator(BaseDemandGenerator):
    def __init__(self, width: int, height: int, **params) -> None:
        super().__init__(width, height, **params)
        self.max_time=params.get("max_time")
        if self.max_time is None:
            raise ValueError("max_time parameter is required for StaticBenchmarkGenerator")
        self.dataframe = params.get("instance_data")
        if self.dataframe is None:
            print("instance_data is None!")
        self.static_demands: List[Demand] = []
        self._prepare_demands()

    def _prepare_demands(self) -> None:
        for _, row in self.dataframe.iterrows():
            demand = Demand(
                x=int(row['xcoord']),
                y=int(row['ycoord']),
                t=0,
                c=int(row['demand']),
                end_t=self.max_time,
                service_time=int(row.get('service_time'))
            )
            self.static_demands.append(demand)

    def reset(self, seed: Optional[int] = None) -> None:
        seed = seed if seed is not None else self.params.get("rng_seed")
        super().reset(seed)

    def sample(self, t: int) -> List[Demand]:
        if t>0:
            return []
        return self.static_demands