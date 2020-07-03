# Imports python modules
import json

from get_input_args import get_input_args_predict
from utilities import load_checkpoint
from prediction_proceduers import predict
from gpu_handling import hw_control

# Main program function defined below
def main():
    
    # Get the arguments passed to the python script 
    in_arg = get_input_args_predict()

    # Load check point from pass
    model = load_checkpoint(in_arg.checkpoint)
    
   # Send model to gpu 
    model_gpu, device = hw_control(model, in_arg.gpu)
    
    # Perform the prediction
    probs, classes = predict(in_arg.path, model, in_arg.top_k, in_arg.gpu)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    Class_names = [cat_to_name[str(class_name)] for class_name in classes]
    
    print("Most probable cat.:\t", Class_names)
    print("Probalbilities \t",probs)
    
    return probs, classes
    
if __name__ == "__main__":
    main()