"""超参数搜索器"""

import itertools
import random
from PyQt5.QtCore import QObject, pyqtSignal


class HyperparameterSearch(QObject):
    """超参数搜索器"""

    search_started = pyqtSignal(int)  # 总搜索次数
    search_progress = pyqtSignal(int, dict, float)  # 当前次数, 参数, 得分
    search_completed = pyqtSignal(dict, list)  # 最优参数, 所有结果

    def __init__(self, model_type, model_class, search_space, strategy='grid'):
        super().__init__()
        self.model_type = model_type
        self.model_class = model_class
        self.search_space = search_space
        self.strategy = strategy
        self.results = []

    def generate_configs(self, max_trials=None):
        """生成参数配置列表"""
        if self.strategy == 'grid':
            return self._grid_search_configs()
        elif self.strategy == 'random':
            return self._random_search_configs(max_trials or 20)
        else:
            raise ValueError(f"不支持的搜索策略: {self.strategy}")

    def _grid_search_configs(self):
        """网格搜索：遍历所有组合"""
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]

        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)

        return configs

    def _random_search_configs(self, max_trials):
        """随机搜索：随机采样N组"""
        configs = []
        keys = list(self.search_space.keys())

        for _ in range(max_trials):
            config = {}
            for key in keys:
                config[key] = random.choice(self.search_space[key])
            configs.append(config)

        return configs

    def get_best_config(self):
        """获取最优配置"""
        if not self.results:
            return None

        best = max(self.results, key=lambda x: x['score'])
        return best['config']
