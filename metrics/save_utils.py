import pandas as pd
import numpy as np
import os

class MetricsManager:
    """
    管理多个序列的多种指标的类
    """
    
    def __init__(self, metric_names, sequence_names=None):
        """
        初始化MetricsManager，允许序列列表为空
        
        Args:
            metric_names (list): 指标名列表
            sequence_names (list, optional): 序列名列表，默认为空
        """
        self.sequence_names = [] if sequence_names is None else sequence_names
        self.metric_names = metric_names
        
        # 初始化一个空的DataFrame来存储所有指标
        self.metrics_df = pd.DataFrame(
            columns=metric_names,
            dtype=float
        )
        
    def update_metrics(self, metrics_dict):
        """
        更新指标值，如果序列不存在则添加新序列
        
        Args:
            metrics_dict (dict): 包含序列名和指标值的字典
                               格式: {'seq_name': 'sequence_1', 'metric1': value1, 'metric2': value2, ...}
        """
        seq_name = metrics_dict.get('seq_name')
        if seq_name is None:
            print("错误: 'seq_name'键不存在于提供的字典中。")
            return
            
        # 如果是新序列，添加到序列列表和DataFrame中
        if seq_name not in self.sequence_names:
            self.sequence_names.append(seq_name)
            # 为新序列创建一个新行，填充NaN
            self.metrics_df.loc[seq_name] = [np.nan] * len(self.metric_names)
            
        # 更新该序列的所有提供的指标
        for metric in self.metric_names:
            if metric in metrics_dict:
                self.metrics_df.at[seq_name, metric] = metrics_dict[metric]
    
    def calculate_averages(self):
        """
        计算每个指标的平均值
        
        Returns:
            dict: 包含每个指标平均值的字典
        """
        averages = {}
        for metric in self.metric_names:
            # 忽略NaN值计算平均值
            averages[metric] = self.metrics_df[metric].mean(skipna=True)
        return averages
    
    def export_to_csv(self, filepath):
        """
        导出指标到CSV文件
        
        Args:
            filepath (str): CSV文件路径
        """
        # 检查是否有数据
        if len(self.sequence_names) == 0:
            print("警告：没有数据可导出")
            return
            
        # 计算平均值
        averages = self.calculate_averages()
        
        # 创建一个新的DataFrame来包括平均值行
        export_df = self.metrics_df.copy()
        
        # 添加平均值行
        export_df.loc['Average'] = pd.Series(averages)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 导出到CSV
        export_df.to_csv(filepath, float_format='%.5f')
        print(f"metrics export {filepath}")
