import json

class Config():
    def __init__(self, **args):
        self.version:int = None 

        self.dir:str = ''
        self.image_size: tuple[int,int] = ()
        self.batch_size: int = None

        self.model:str = ''
        self.optimizer = None
        self.learning_rate: float = 0.0
        self.loss:str = ''

        self.set_args(**args)
        self.set_default()

    def set_default(self):
        self.version:int = 1

        self.dir:str = ''
        self.image_size: tuple[int,int] = (224,224)

        self.model = 'resnet'
        self.batch_size: int = 32
        self.optimizer = None
        self.learning_rate: float = 0.0
        self.loss = 'cross_entropy'

    def set_args(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, filename):
        filename = f'config_{self.version}'
        with open(f'{filename}.json', 'w') as f:
            json.dump(self.__dict__ , f, indent = 4)