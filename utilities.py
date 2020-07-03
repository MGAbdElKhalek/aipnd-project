
import torch
from torchvision import datasets, transforms
from PIL import Image

from datetime import datetime

from model_handling import creat_model

def data_preprocessing(data_dir):
    ''' Prepare images for training
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'    
    
    # Define your transforms for the training, validation, and testing sets
    transforms_train = transforms.Compose([transforms.RandomResizedCrop(size=224,
                                                                        scale=(0.08, 1.0),
                                                                        ratio=(0.75, 1.3333333333333333), 
                                                                        interpolation=2),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])
    
    transforms_valid = transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), 
                                                                (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform = transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform = transforms_valid)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
    dataloader_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size = 64, shuffle = False)
    
    return dataloader_train, dataloader_valid


def save_model_cp(model, arch, input_size, output_size, hidden_layers_size, dir_cp):
    
    #Save the checkpoint 
    checkpoint = {'model_arch':arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers_size,
                  'state_dict': model.state_dict()}
    
    now = datetime.now()
    cp_path = dir_cp + '/'+ str(now)

    torch.save(checkpoint, cp_path)
    return None

def load_checkpoint(filepath):
    ''' Load pretrained NN.
       
        Arguments
        ---------
        filepath: Path to checkpoint file
    '''
    checkpoint = torch.load(filepath)
    
    # Create the network, adjust classifier define the criterion and optimizer
    model_restored = creat_model(checkpoint['model_arch'],
                                 checkpoint['hidden_layers'],
                                 checkpoint['input_size'], 
                                 checkpoint['output_size'])

    model_restored.load_state_dict(checkpoint['state_dict'])
    
    return model_restored

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open PIL image 
    pil_image = Image.open(image)
    
    # Creat Transform for the image
    transforms_image = transforms.Compose([transforms.Resize(225),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # Apply the transform to the PIL image
    Ten_img = transforms_image(pil_image)
    return Ten_img