"""
统一的坐标转换工具类
用于在像素坐标和地理坐标之间进行转换
"""

class CoordinateConverter:
    """坐标转换器，确保所有模块使用一致的转换规则"""

    def __init__(self, metadata):
        """
        从 H5 metadata 初始化转换器

        Args:
            metadata: 包含 x_min, x_max, y_min, y_max, nx, ny 的字典
        """
        self.x_min = float(metadata['x_min'])
        self.x_max = float(metadata['x_max'])
        self.y_min = float(metadata['y_min'])
        self.y_max = float(metadata['y_max'])
        self.nx = int(metadata['nx'])
        self.ny = int(metadata['ny'])

    def pixel_to_geo(self, pixel_x, pixel_y):
        """
        像素坐标 -> 地理坐标

        Args:
            pixel_x: 像素 X 坐标
            pixel_y: 像素 Y 坐标

        Returns:
            (geo_x, geo_y): 地理坐标元组
        """
        geo_x = self.x_min + (float(pixel_x) / max(self.nx - 1, 1)) * (self.x_max - self.x_min)
        geo_y = self.y_max - (float(pixel_y) / max(self.ny - 1, 1)) * (self.y_max - self.y_min)
        return geo_x, geo_y

    def geo_to_pixel(self, geo_x, geo_y):
        """
        地理坐标 -> 像素坐标

        Args:
            geo_x: 地理 X 坐标
            geo_y: 地理 Y 坐标

        Returns:
            (pixel_x, pixel_y): 像素坐标元组
        """
        pixel_x = int(round((float(geo_x) - self.x_min) / (self.x_max - self.x_min) * max(self.nx - 1, 1)))
        pixel_y = int(round((self.y_max - float(geo_y)) / (self.y_max - self.y_min) * max(self.ny - 1, 1)))
        pixel_x = max(0, min(self.nx - 1, pixel_x))
        pixel_y = max(0, min(self.ny - 1, pixel_y))
        return pixel_x, pixel_y
