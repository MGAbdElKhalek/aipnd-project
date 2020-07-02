
import torch


def hw_control(data, gpu_flag):
    # Send model to GPU

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if gpu_flag == True:
        if device == "cpu":
            warnings.warn("You requested GPU but we couldn't so it's on cpu")
        else:
            data.to(device)
            print("Successfuly sent to gpu")
    else:
        ("The model is in cpu as requested by default")
    
    return data, device