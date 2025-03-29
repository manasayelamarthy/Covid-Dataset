import json

class train_Config():
    def __init__(self, **args):
        self.version:int = None 
        self.mode:str = ''

        self.datapath:str = ''
        self.image_size: tuple[int,int] = ()
        self.batch_size: int = None

        self.model:str = ''
        self.num_classes:int = None
        self.optimizer = None
        self.metrics:str = None
        self.learning_rate: float = 0.0
        self.loss:str = ''
        self.epochs:int = None
        self.checkpoint_dir:str = ''
        self.log_dir:str = ''

        self.set_args(**args)
        self.set_default()

    def set_default(self):
        self.version:int = 1
        self.mode:str = 'train'

        self.datapath:str = ''
        self.image_size: tuple[int,int] = (224,224)
        self.batch_size: int = 32

        self.model = 'cnn_model'
        self.num_classes:int =3
        self.optimizer = 'adam'
        self.metrics = 'all_metrics'
        self.learning_rate: float = 0.001
        self.loss = 'cross_entropy'
        self.epochs:int = 16
        self.checkpoint_dir:str = 'checkpoint/'
        self.log_dir:str = ' logs/'



    def set_args(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, filename):
        filename = f'config_{self.version}'
        with open(f'{filename}.json', 'w') as f:
            json.dump(self.__dict__ , f, indent = 4)