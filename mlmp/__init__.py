import os.path
from mlmp.predict import MLMouse, DevMLMouse
from importlib.resources import files as resource_path


__all__ = ['mouse', 'TrainBySelf']


class mouse:
    def __init__(self, model_path: str = None, dll: bool = True):
        if model_path is None:
            dll_resource = resource_path('mlmp').joinpath('mouse.dll')
            model_path = str(dll_resource)
        self.mlmouse = MLMouse(model_path, dll)

    def mouse_output(self, start, end, absulute=False) -> list:
        return self.mlmouse.mouse_output(start, end, absulute)

class dev_mouse:
    def __init__(self, model_path: str = None, dll: bool = True):
        if model_path is None:
            dll_resource = resource_path('mlmp').joinpath('mouse.dll')
            model_path = str(dll_resource)
        self.mlmouse = DevMLMouse(model_path, dll)

    def mouse_output(self, start, end, absulute=False) -> list:
        return self.mlmouse.mouse_output(start, end, absulute)

class TrainBySelf:
    def __init__(self, dev_features: bool = False):
        self.dev_features = dev_features

    @staticmethod
    def collect_data():
        import mlmp.normal_mouse_data
        mlmp.normal_mouse_data.MouseDataCollector()

    @staticmethod
    def train():
        if os.path.exists('mouse_data.csv'):
            pass
        else:
            raise Exception('mouse_data.csv not found')
        if os.path.exists('mouse_data_test.csv'):
            pass
        else:
            raise Exception('mouse_data_test.csv not found')
        import mlmp.train
        mlmp.train.main_prog()

    def time_seq_mouse_data(self):
        import mlmp.time_seq_mouse_data
        if not self.dev_features:
            raise Exception('Dev features not enabled')
        mlmp.time_seq_mouse_data.MouseDataCollector()

    def time_seq_train(self):
        import mlmp.time_seq_train
        if not self.dev_features:
            raise Exception('Dev features not enabled')
        if os.path.exists('mouse_data_time_seq.csv'):
            pass
        else:
            raise Exception('mouse_data_time_seq.csv not found')
        if os.path.exists('mouse_data_time_seq_test.csv'):
            pass
        else:
            raise Exception('mouse_data_time_seq_test.csv not found')
        mlmp.time_seq_train.train_model()