
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import models
from gpu_handling import hw_control

def creat_classifier(input_size, output_size, hidden_layers):
    ''' creat classifier that can be part of model.
       
        Arguments
        ---------
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
    '''
    classifier_temp = nn.Sequential()
    classifier_temp.add_module('fc0', nn.Linear(input_size, hidden_layers[0]))
    classifier_temp.add_module('relu0', nn.ReLU(inplace = True))
    classifier_temp.add_module('D0', nn.Dropout(p=0.5))
    
    for index, layer in enumerate(hidden_layers):
        if index == 0:
            continue
        classifier_temp.add_module('fc'+str(index), nn.Linear(hidden_layers[index-1],layer))
        classifier_temp.add_module('relu'+str(index), nn.ReLU(inplace = True))
        classifier_temp.add_module('D'+str(index), nn.Dropout(p=0.5))
    
    classifier_temp.add_module('output', nn.Linear(hidden_layers[-1], output_size))
    classifier_temp.add_module('output_activation', nn.LogSoftmax(dim=1))    
    
    return classifier_temp
    

def creat_model(pre_model_arch, hidden_layers_size, input_size, output_size):

    # Create the network, adjust classifier define the criterion and optimizer
    if pre_model_arch == "vgg13":
        model_flower_imgs = models.vgg16(pretrained=True)
    elif pre_model_arch == "vgg16":
        model_flower_imgs = models.vgg16(pretrained=True)
    else:
        assert 0, "Architecture isn't supported"

    
    # Freeze parameters so we don't backprop through them
    for param in model_flower_imgs.parameters():
        param.requires_grad = False
    
    # Update the classifier by desired conf.
    model_flower_imgs.classifier = creat_classifier(input_size, output_size, hidden_layers_size)
    
    return model_flower_imgs

def evaluation(model, data_loader, criterion, gpu_f):
    ''' Test the performence of NN against givien data set either validation of test.
       
        Arguments
        ---------
        model: the model to be tested
        data_loader: data loader for the data set to be used in testing
        criterion: criterion to measure the performence upon
    '''
    model.eval()
    loss = 0
    accuracy = 0

    for image, label in data_loader:
        
        image, device = hw_control(image, gpu_f)
        label, device = hw_control(label, gpu_f)
        
        output = model.forward(image)
        loss += criterion(output,label).item()
        ps = torch.exp(output)
        equality = (label.data == ps.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()
    
    model.train()
    return loss, accuracy


def model_training(model, dataloader_train, dataloader_valid, gpu_f, learn_r, num_epoches):
    # Define the optimizer to be used in training and number of epoches to run
    optmizer = optim.Adam(model.classifier.parameters(), lr=learn_r)
    # Define the criterion that will be used in training and evaluation
    criterion = nn.NLLLoss()

    # Training of model
    for e in range(num_epoches):

        for images , labels in iter(dataloader_train):
            optmizer.zero_grad()
            images, device = hw_control(images, gpu_f)
            labels, device = hw_control(labels, gpu_f)

            output = model.forward(images)
            loss = criterion(output , labels)
            loss.backward()
            optmizer.step()
    
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            Valid_loss, accuracy = evaluation(model, dataloader_valid, criterion)
            print('Epoch {}/{}: Validation loss= {} \n Accuracy= {}'.format(e,num_epoches, Valid_loss, accuracy))
    
    return model