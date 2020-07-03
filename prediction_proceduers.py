import torch



def predict(image_path, model, topk, gpu_f):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Arguments
        ---------
        image_path: Pathh to image
        model: the pretrained model
        topk: integer, number of class probabilities to be displayed
    '''
    
    # Prepare image
    Tensor_image = process_image(image_path)
    Tensor_image, device = hw_control(Tensor_image, gpu_f)
    Tensor_image.unsqueeze_(0)
    Tensor_image = Tensor_image.float()
    
    with torch.no_grad():
    # Put the model in evaluation mode
        model.eval()
    # Forward propagation through the model
        output = model.forward(Tensor_image)
        ps = torch.exp(output)
    print(ps)
    print(ps.max())
    top_k_prob = ps.topk(topk)
    #Make sure the module is back to training mode
    model.train()
    
    # Return the results as numpy arrays
    probablities = top_k_prob[0][0].cpu().numpy()
    classes = top_k_prob[1][0].cpu().numpy()
    return probablities, classes