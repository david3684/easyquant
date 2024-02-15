import torch
import torch.nn as nn
import copy
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import argparse, json
import quant, quantizer, calibration
from recon import reconstruct
import torch.onnx
# from fvcore.nn import FlopCountAnalysis


def save_results_to_file(filename, args, results):
    with open(filename, 'a') as file:  
        file.write("\n\n--- New Execution ---\n")
        file.write("Terminal Parameters:\n")
        file.write(json.dumps(vars(args), indent=4))
        file.write("\n\nEvaluation Results:\n")
        for result in results:
            file.write(f"{result['stage']}: Val loss {result['loss']:.3f}, Val accuracy {result['accuracy']:.1f}%\n")
            
def print_memory_usage(description, device='cuda'):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  
    print(f"{description}: {allocated:.2f} MB")
    
def count_flop(model):
    dummy_input = torch.randn(1,3,224,224).cuda()
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"Total FLOPs: {flops.total()}")
    

def infer_and_measure_memory_usage(model, loader, device='cuda'):
    torch.cuda.reset_peak_memory_stats(device)
    print_memory_usage("Before inference")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            break  

    print_memory_usage("After inference")
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  
    print(f"Peak memory usage during inference: {peak_memory:.2f} MB")

def process_epoch(model, criterion, loader, optimizer=None, trainmode=True):
    if trainmode:
        model.train()
    else:
        model.eval()
    
    closs = 0
    correct = 0
    total = 0
    with tqdm(loader, unit='batch') as tepoch:
        for images, labels in tepoch:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            if trainmode:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            closs += loss.item() * images.size(0)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  

            tepoch.set_postfix(loss=(closs / total), acc_pct=(correct / total * 100))

    return (closs / total), (correct / total)

def export_quantized_model(qmodel):
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    torch.onnx.export(qmodel,         
                    dummy_input,    
                    "./quantized_model.onnx",  
                    export_params=True,      
                    opset_version=11,        
                    do_constant_folding=True, 
                    input_names=['input'],   
                    output_names=['output'], 
                    dynamic_axes={'input': {0: 'batch_size'}, 
                                    'output': {0: 'batch_size'}})

    print("Quantized model has been exported to quantized_model.onnx")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easy Quantization Package')

    # weight quantization bitwidth
    parser.add_argument('--w_n_bits', type=int, default='8',
                        help='weight quantization bitwidth')
    # activation quantization bitwidth
    parser.add_argument('--a_n_bits', type=int, default='None',
                        help='activation quantization bitwidth')
    
    #weight quantization parameter init method
    parser.add_argument('--init_method', type=str, default='minmax',
                        choices=['minmax', 'mse'],
                        help='Weight quantization parameter init method')
    # dataset path
    parser.add_argument('--data_path', type=str, default='../datasets/imagenet2012/val',
                        help='Validation dataset path')

    # calibration dataset path
    parser.add_argument('--cali_path', type=str, default=None,
                        help='Calibration dataset path. If none, sample from validation set')
    # calibration set size
    parser.add_argument('--cali_size', type=int, default=1024,
                        help='Number of samples to use for calibration.')
    
    # quantizer 
    parser.add_argument('--adaround', action='store_true',
                        help='Apply adaptive rounding in reconstruction')

    args = parser.parse_args()
    
    if args.cali_path is None:
        args.cali_path = args.data_path
        
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    

    val_data = datasets.ImageFolder(root='../datasets/imagenet2012/val', transform=transforms)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
    cali_loader = calibration.sample_calibration_set(dataset_path=args.cali_path, calibration_size=args.cali_size, transform=transforms)
    
    results = []
    model = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True)
    model.cuda()
    model.eval()
    vloss, vacc = process_epoch(model, criterion, cali_loader, trainmode = False)
    print('Baseline : Val loss {:.3f} Val accuracy {:.1f}%'.format(vloss,vacc*100))
    results.append({'stage': 'Baseline', 'loss': vloss, 'accuracy': vacc*100})
    
    model_copy = copy.deepcopy(model)
    
    qmodel = quant.QModel(model_copy, w_n_bits=args.w_n_bits, a_n_bits=args.a_n_bits, init_method=args.init_method)
    qmodel.cuda()
    qmodel.eval()
    
    # count_flop(model)
    # count_flop(qmodel)
    vloss, vacc = process_epoch(qmodel, criterion, cali_loader, trainmode = False)
    print('Accuracy Before Reconstruction : Val loss {:.3f} Val accuracy {:.1f}%'.format(vloss,vacc*100))
    results.append({'stage': 'Before Reconstruction', 'loss': vloss, 'accuracy': vacc*100})
    

    reconstruct(qmodel, model, cali_loader, adaround=args.adaround)
    if args.a_n_bits is not None:
        reconstruct(qmodel, model, cali_loader, adaround=False, recon_act=True)
    #export_quantized_model(qmodel)
    
    vloss, vacc = process_epoch(qmodel, criterion, cali_loader, trainmode = False)
    print('Accuracy After Reconstruction : Val loss {:.3f} Val accuracy {:.1f}%'.format(vloss,vacc*100))
    results.append({'stage': 'After Reconstruction', 'loss': vloss, 'accuracy': vacc*100})
    
    save_results_to_file("evaluation_results.txt", args, results)
    
    vloss, vacc = process_epoch(qmodel, criterion, val_loader, trainmode = False)
    print('Accuracy After Reconstruction : Val loss {:.3f} Val accuracy {:.1f}%'.format(vloss,vacc*100))