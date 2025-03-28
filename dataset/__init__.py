from .covid_dataset import CovidDataset
from all_configs.config import train_Config

def get_dataloaders(config = train_Config()):
    if config.mode =='train':
        data_config = config.__dict__

        train_dataset = CovidDataset(**data_config)

        train_dataloader = (train_dataset,
                            batch_size = config.batch_size,
                            shuffle = True)
        
        val_dataset =  CovidDataset(**data_config)

        val_dataloader = (
            val_dataset,
            batch_size = config.batch_size,
            shuffle = False            
        )
    else:
        val_dataset =  CovidDataset(**data_config)

        val_dataloader = (
            val_dataset,
            batch_size = config.batch_size,
            shuffle = False
            
        )
    return train_dataloader, val_dataloader

    


    






