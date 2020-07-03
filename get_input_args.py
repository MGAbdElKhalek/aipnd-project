import argparse


def get_input_args_train():
    ''' Function to get all the input arguments
    Return:
    parser opject
    '''
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument('path', type = str, default = './flowers',
                   help='path to the images folder')
    parser.add_argument('--save_dir', type = str, default = './cp', 
                    help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg13', 
                    help = 'architecture for pretrained model')
    parser.add_argument('--learning_rate', type = float, default = '0.001', 
                    help = 'Training learn rate')
    parser.add_argument('--hidden_units', nargs="*", type = int, default = [4096 , 1024], 
                    help = 'Hidden layers list for classifier')
    parser.add_argument('--epochs', type = int, default = '20', 
                    help = 'Number of epoches in the training phase')
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Flage for using GPU')
    return parser.parse_args()


def get_input_args_predict():
    ''' Function to get all the input arguments
    Return:
    parser opject
    '''
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 5 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument('path', type = str,
                   help='path to the an image')
    parser.add_argument('checkpoint', type = str,
                   help='path to the model checkpoint')
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'Set the number of probabilities to show')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'mapping of categories to real names json file')
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Flage for using GPU')
    return parser.parse_args()