import os.path

from predict import MLMouse
import mlmouse.normal_mouse_data
import mlmouse.train
import mlmouse.time_seq_mouse_data
import mlmouse.time_seq_train

__all__ = ['mouse', 'normal_mouse_data', 'train', 'time_seq_mouse_data', 'time_seq_train']


class mouse:
    def __init__(self, model_path, dll: bool = True):
        self.mlmouse = MLMouse(model_path, dll)

    def mouse_output(self, start, end, absulute=False) -> list:
        return self.mlmouse.mouse_output(start, end, absulute)


class TrainBySelf:
    def __init__(self, dev_features: bool = False):
        self.dev_features = dev_features

    def collect_data(self):
        normal_mouse_data.main_prog()

    def train(self):
        if os.path.exists('mouse_data.csv'):
            pass
        else:
            raise Exception('mouse_data.csv not found')
        if os.path.exists('mouse_data_test.csv'):
            pass
        else:
            raise Exception('mouse_data_test.csv not found')
        mlmouse.train.main_prog()

    def time_seq_mouse_data(self):
        if not self.dev_features:
            raise Exception('Dev features not enabled')
        mlmouse.time_seq_mouse_data.main_prog()

    def time_seq_train(self):
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
        mlmouse.time_seq_train.train_model()