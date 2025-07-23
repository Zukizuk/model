import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Simple test to check if your engine loads
def test_engine(engine_path):
    print(f"Testing engine: {engine_path}")
    
    # Load engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("❌ Failed to load engine")
        return False
    
    print("✅ Engine loaded successfully!")
    
    # Print engine info
    context = engine.create_execution_context()
    print(f"Number of bindings: {engine.num_bindings}")
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = context.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)
        print(f"  {name}: {shape} {dtype} {'(input)' if is_input else '(output)'}")
    
    print("🎉 Engine test passed!")
    return True

# Test your engine
if __name__ == "__main__":
    engine_path = "weights.engine"  # Change this to your engine path
    test_engine(engine_path)