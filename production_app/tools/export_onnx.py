import torch
import onnx
import onnxruntime
import numpy as np
from core_model import DCNN_BiLSTM_DAM

def export_to_onnx():
    print("🚀 Initializing ONNX Export Engine...")
    device = torch.device('cpu') 
    model = DCNN_BiLSTM_DAM(num_classes=7)
    weights_path = "./models/best_model_dcnn_dam.pth"
    
    # Load PyTorch weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Create dummy input that matches OpenCV image structure [Batch=1, Channel=1, H=64, W=64]
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    
    onnx_path = "./models/dcnn_dam_opt.onnx"
    
    # Export explicitly locking input and output nodes
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ Fast-Inference ONNX Matrix compiled successfully at: {onnx_path}")
    
    # Verify ONNX Graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Directed Acyclic Graph Check: PASSED.")

if __name__ == "__main__":
    export_to_onnx()
