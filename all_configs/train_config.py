from .config import Config

class trainConfig(Config):
    def __init__(self, **args):
        self.epochs:int = None
        self.checkpoint_dir:str = ''
        self.log_dir:str = ''

        self.set_default()
        self.__setattr__(**args)

    def set_default(self):
        self.epochs:int = 16
        self.checkpoint_dir:str = 'checkpoint/'
        self.log_dir:str = ' logs/'



