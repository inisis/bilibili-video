import timeit
import numpy as np
import torch.backends.cudnn as cudnn

# Config parsers and report generations
import argparse
import yaml
import os
import pandas as pd

# Importing supported Backends
import torch
import trtorch
import tensorrt as trt

WARMUP_ITER = 10
results = []

# YAML Parser class for parsing the run configurations
class ConfigParser:
    def __init__(self, config_file):
        self.parser = None
        self.config = config_file
        self.params = None
    
    # Reads and loads the yaml file
    def read_config(self):
        with open(self.config, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return self.params

    # Retrieves the value from the configuration else uses default values
    def get(self, key, default_value=None):
        if not key in self.params:
            if not default_value:
                raise ValueError('Key {} is not present and default_value is not configured. Please run it with default value', key)
            self.params[key] = default_value
        return self.params[key]

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

def precision_to_dtype(pr):
    if pr == 'fp32':
        return torch.float
    elif pr == 'fp16' or pr == 'half':
        return torch.half
    else:
        return torch.int8

def precision_to_torch_dtype(pr):
    if pr == 'fp32':
        return trt.float32
    elif pr == 'fp16' or pr == 'half':
        return trt.float16
    else:
        return trt.int8

def precision_to_builder_dtype(pr):
    if pr == 'fp32':
        return trt.BuilderFlag.TF32 
    elif pr == 'fp16' or pr == 'half':
        return trt.BuilderFlag.FP16
    else:
        return trt.BuilderFlag.INT8

def load_model(params):
    model = None
    is_trt_engine = False
    # Load torch model traced/scripted
    model_file = params.get('model').get('filename')

    if model_file.endswith('.jit.pt'):
        model = torch.jit.load(model_file).cuda()
    else:
        is_trt_engine = True
        # Read the TensorRT engine file
        with open(model_file, 'rb') as fin:
            model = fin.read()
    return model, is_trt_engine

# Runs inference using Torch backend
def run_torch(model, inp_tensor_shape, params, precision):
    print("Running Torch for precision: ", precision)
    iters = params.get('iterations', 1000)

    input_tensors = (torch.randint(0, 2, inp_tensor_shape, dtype=precision_to_dtype(precision)).cuda())
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, iters+1):
            start_time = timeit.default_timer()
            features = model(input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
            if i%1000==0:
                print('Iteration %d/%d, ave batch time %.6f ms'%(i, iters, np.mean(timings)))
    
    printStats("Torch", timings, precision, inp_tensor_shape[0])

# Runs inference using Torch-TensorRT backend
def run_torch_tensorrt(model, inp_tensor_shape, params, precision):
    print("Running Torch-TensorRT")
    
    input_tensors = (torch.randint(0, 2, inp_tensor_shape, dtype=precision_to_dtype(precision)).cuda())
    # Compiling Torch-TensorRT model
    compile_settings = {
       "inputs": [trtorch.Input(inp_tensor_shape)],
       "enabled_precisions": {precision_to_dtype(precision)} 
    }

    if precision == 'int8':
        compile_settings.update({"calib": params.get('calibration_cache')})

    model = trtorch.compile(model, compile_settings)
 
    iters = params.get('iterations', 1000)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, iters+1):
            start_time = timeit.default_timer()
            features = model(input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
            if i%1000==0:
                print('Iteration %d/%d, ave batch time %.6f ms'%(i, iters, np.mean(timings)))
    
    printStats("Torch-TensorRT", timings, precision, inp_tensor_shape[0])

def run_tensorrt(model, inp_tensor_shape, params, precision, is_trt_engine=False):
    engine = None
    input_tensors = (torch.randint(0, 2, inp_tensor_shape, dtype=precision_to_dtype(precision)).cuda())
    # If the model file is a TensorRT engine then directly deserialize and run inference
    # else convert the torch module to a TensorRT engine first and then run inference

    batch_size = inp_tensor_shape[0]

    if not is_trt_engine:
        # Compiling Torch-TensorRT model
        compile_settings = {
            "inputs": [trtorch.Input(inp_tensor_shape)],
            "enabled_precisions": {precision_to_dtype(precision)},
            "max_batch_size": batch_size
        }

        print("Converting method to TensorRT engine...")
        with torch.no_grad():
            model = trtorch.convert_method_to_trt_engine(model, "forward", compile_settings)
    
    # Deserialize the TensorRT engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(model)
  
    print("Running TensorRT")
    iters = params.get('iterations', 1000)

    # Compiling the bindings
    bindings = engine.num_bindings * [None]

    k = 0
    for idx,_ in enumerate(bindings):
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
        shape = (batch_size,) + tuple(engine.get_binding_shape(idx))
        device = torch_device_from_trt(engine.get_location(idx))
        if not engine.binding_is_input(idx):
            # Output bindings
            output = torch.empty(size=shape, dtype=dtype, device=device)
            bindings[idx] = output.data_ptr()
        else:
            # Input bindings
            bindings[idx] = input_tensors.data_ptr()
            k += 1
    
    timings = []
    with engine.create_execution_context() as context:
        for i in range(WARMUP_ITER):
            context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

        for i in range(1, iters+1):
            start_time = timeit.default_timer()
            context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
            if i%1000==0:
                print('Iteration %d/%d, ave batch time %.6f ms'%(i, iters, np.mean(timings)))
    
    printStats("TensorRT", timings, precision, inp_tensor_shape[0])

def run_tensorrt_naive(model, inp_tensor_shape, params, precision, is_trt_engine=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def GiB(val):
        return val * 1 << 30

    def populate_network(network, precision):
        # Configure the network layers based on the weights provided.
        input_tensor = network.add_input(name="data", dtype=precision_to_torch_dtype(precision), shape=inp_tensor_shape[1:])

        # Set dummy weights for the kernel and bias weights in the conv1 layer. We
        # will refit the engine with the actual weights later.
        conv1_w = np.ones((20,5,5), dtype=np.float32)
        conv1_b = np.zeros(20, dtype=np.float32)

        conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
        conv1.name = "conv_1"
        conv1.stride = (1, 1)

        pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
        pool1.stride = (2, 2)

        conv2_w = np.ones((50,20,5,5), dtype=np.float32)
        conv2_b = np.zeros(50, dtype=np.float32)
        conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
        conv2.stride = (1, 1)

        pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool2.stride = (2, 2)

        fc1_w = np.ones((500,800), dtype=np.float32)
        fc1_b = np.zeros(500, dtype=np.float32)
        fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

        relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

        fc2_w = np.ones((10,500), dtype=np.float32)
        fc2_b = np.zeros(10, dtype=np.float32)
        fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)

        fc2.get_output(0).name = "prob"
        network.mark_output(tensor=fc2.get_output(0))
    
    def build_engine(precision):
        # For more information on TRT basics, refer to the introductory samples.
        builder = trt.Builder(TRT_LOGGER)
        builder.max_batch_size = inp_tensor_shape[0]
        network = builder.create_network()
        config = builder.create_builder_config()
        runtime = trt.Runtime(TRT_LOGGER)

        config.max_workspace_size = GiB(1)
        config.set_flag(precision_to_builder_dtype(precision))
        # Populate the network using weights from the PyTorch model.
        populate_network(network, precision)
        # Build and return an engine.
        plan = builder.build_serialized_network(network, config)
        return runtime.deserialize_cuda_engine(plan)
    
    batch_size = inp_tensor_shape[0]

    engine = build_engine(precision)
    print("Running Naive-TensorRT")
    iters = params.get('iterations', 1000)

    # Compiling the bindings
    bindings = engine.num_bindings * [None]

    k = 0
    input_tensors = (torch.randint(0, 2, inp_tensor_shape, dtype=precision_to_dtype(precision)).cuda())
    for idx,_ in enumerate(bindings):
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
        shape = (batch_size,) + tuple(engine.get_binding_shape(idx))
        device = torch_device_from_trt(engine.get_location(idx))
        if not engine.binding_is_input(idx):
            # Output bindings
            output = torch.empty(size=shape, dtype=dtype, device=device)
            bindings[idx] = output.data_ptr()
        else:
            # Input bindings
            bindings[idx] = input_tensors.data_ptr()
            k += 1
    
    timings = []
    with engine.create_execution_context() as context:
        for i in range(WARMUP_ITER):
            context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

        for i in range(1, iters+1):
            start_time = timeit.default_timer()
            context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
            if i%1000==0:
                print('Iteration %d/%d, ave batch time %.6f ms'%(i, iters, np.mean(timings)))
    
    printStats("Naive-TensorRT", timings, precision, inp_tensor_shape[0])

# Deploys inference run for different backend configurations
def run(model, inp_tensor_shape, params, precision, is_trt_engine = False):
    for backend in params.get('backend'):

        if precision == 'int8':
            if backend == 'all' or backend == 'torch':
                print("int8 precision is not supported for torch runtime in this script yet")
                return False

            if backend == 'all' or backend == 'torch_tensorrt' or params.get('calibration_cache', None) == None:
                print("int8 precision expects calibration cache file for inference")
                return False

        if backend == 'all':
            run_torch(model, inp_tensor_shape, params, precision)
            run_torch_tensorrt(model, inp_tensor_shape, params, precision)
            run_tensorrt(model, inp_tensor_shape, params, precision, is_trt_engine)
            run_tensorrt_naive(model, inp_tensor_shape, params, precision, is_trt_engine)
    
        elif backend == "torch":
            run_torch(model, inp_tensor_shape, params, precision)
    
        elif backend == "torch_tensorrt":
            run_torch_tensorrt(model, inp_tensor_shape, params, precision)
    
        elif backend == "tensorrt":
            run_tensorrt(model, inp_tensor_shape, params, precision, is_trt_engine)

# Generate report
def printStats(backend, timings, precision, batch_size = 1):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ("\n%s =================================\n"
            "batch size=%d, num iterations=%d\n"
            "  Median FPS: %.1f, mean: %.1f\n"
            "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
            ) % (backend,
                batch_size, steps,
                speed_med, speed_mean,
                time_med, time_mean, time_99th, time_std)
    print(msg)
    meas = {
        'Backend' : backend,
        'precision' : precision,
        'Median(FPS)' : speed_med,
        'Mean(FPS)' : speed_mean,
        'Median-Latency(ms)' : time_med,
        'Mean-Latency(ms)' : time_mean,
        '99th_p' : time_99th,
        'std_dev': time_std
    }
    results.append(meas)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Run inference on a model with random input values")
    arg_parser.add_argument("--config", help="Load YAML based configuration file to run the inference. If this is used other params will be ignored")
    args = arg_parser.parse_args()
    
    parser = ConfigParser(args.config)
    # Load YAML params
    params = parser.read_config()
    print("Loading model: ", params.get('model').get('filename'))

    model = None

    # Default device is set to 0. Configurable using yaml config file.
    torch.cuda.set_device(params.get('runtime').get('device', 0))

    # Load the model file from disk. If the loaded file is TensorRT engine then is_trt_engine is returned as True
    model, is_trt_engine = load_model(params)
    cudnn.benchmark = True

    # Create random input tensor of certain size
    torch.manual_seed(12345)

    for precision in params.get('runtime').get('precision', 'fp32'):
        inp_tensor_shape = params.get('input')

        if is_trt_engine:
            print("Warning, TensorRT engine file is configured. Please make sure the precision matches with the TRT engine for reliable results")

        if not is_trt_engine and precision == "fp16" or precision == "half":
            # If model is TensorRT serialized engine then model.half will report failure
            model = model.half()
        
        # Run inference
        status = run(model, inp_tensor_shape, params, precision, is_trt_engine)
        if status == False:
            continue

    # Generate report
    print('Model Summary:')
    summary = pd.DataFrame(results)
    print(summary)

