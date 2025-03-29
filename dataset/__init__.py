from .covid_dataset import CovidDataset
from config import train_Config
from torch.utils.data import DataLoader

def get_dataloaders(config = train_Config()):
    if config.mode =='train':
        train_dict = config.__dict__.copy()
        train_dict['data_dir'] = train_dict['data_dir'] + 'train'
        
        train_dataset = CovidDataset(**train_dict)

        train_dataloader = DataLoader(train_dataset,
                            batch_size = config.batch_size,
                            shuffle = True)
        
        val_dict = config.__dict__.copy()
        val_dict['data_dir'] = val_dict['data_dir'] + 'test'
        val_dataset =  CovidDataset(**val_dict)

        val_dataloader = DataLoader (
            val_dataset,
            batch_size = config.batch_size,
            shuffle = False            
        )
    else:
        data_config = config.__dict__.copy()
        data_config['data_dir'] = data_config['data_dir'] + 'val'
        val_dataset =  CovidDataset(**data_config)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size = config.batch_size,
            shuffle = False
            
        )
    return train_dataloader, val_dataloader

    


    






