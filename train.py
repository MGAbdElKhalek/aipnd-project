# Imports python modules
import warnings

from get_input_args import get_input_args_train
from utilities import data_preprocessing, save_model_cp
from model_handling import creat_model, model_training
from gpu_handling import hw_control


# Main program function defined below
def main():
    
    # Get the arguments passed to the python script 
    in_arg = get_input_args_train()
    train_data_loader, valid_data_loader = data_preprocessing(in_arg.path)
    model = creat_model(in_arg.arch, in_arg.hidden_units, 25088, 102)
    
    # Send model to gpu
    model_gpu, device = hw_control(model, in_arg.gpu)

    # Train the model    
    model = model_training(model_gpu, train_data_loader, valid_data_loader, in_arg.gpu, in_arg.learning_rate, in_arg.epochs)
    
    # Save CP of model
    save_model_cp(model, arch, input_size, output_size, hidden_layers_size, in_arg.save_dir)


    
if __name__ == "__main__":
    main()