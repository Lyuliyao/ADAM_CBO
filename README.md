# ADAM_CBO
## This is the code written for A CONSENSUS-BASED GLOBAL OPTIMIZATION METHOD WITH ADAPTIVE MOMENTUM ESTIMATION
This is a matlab code and require gpu for running it.  If you have some question about using gpu in Matlab. You can refer to 

[GPU Computing]: https://www.mathworks.com/help/parallel-computing/gpu-computing.htm

and  "GPU setting of these code" part also gives some advice about how to run it in cpu.

### Rastrigin function

We show the code about how to find the local minima of high dimensional Rastrigin function,
<img src="http://latex.codecogs.com/gif.latex?f(x)=\frac{1}{d} \sum_{i=1}^d\left[(x_i-B)^2-10\cos(2\pi (x_i-B)+10\right]+C" />

This object function is defined in obj_fcn.m.

In CBO file, you can run folder CBO, we compute the success rate of CBO method under different setting.

```matlab
swarming_general_onepara(dim,p_N,p_batch,random,lambda,gama,sigma,simu_N)
```

dim is the dimension method, p_N is the number of the particle, p_batch is the batch of particle update in each stage, random give us three choice to add random noise, including "uniform", "normal", "levy".  lambda, gamma and sigma is the parameter is CBO update relu. simu_N is the number of the simulation. The output is the success rate in this setting.

Some example can be seen in CBO_example.ipynb, but it is a jupyer notebook with kernal matlab rather than python.

In CBO_momentum folder, the name of the .m file specify the random noise and initialization. The paramters of the function are similar with CBO, except adding a decay_rate paramters.s

```matlab
ADAM_normal(dim,p_N,p_batch,simu_N,decay_rate,lambda)
```

### Deep Neural Network

DNN part have 3 example, including appriximate function, frequency principle and solving PDE. The main function is ADAM and you can simply run it to reproduce the result.

### GPU setting of these code

You can use "gpuDevice" in matlab to see whether your matlab support gpu computation. If your matlab support gpu computation, it may output

```matlab
gpuDevice
ans = 
CUDADevice with properties:
               Name: 'Tesla V100S-PCIE-32GB'
              Index: 1
  ComputeCapability: '7.0'
     SupportsDouble: 1
      DriverVersion: 11
     ToolkitVersion: 9
 MaxThreadsPerBlock: 1024
   MaxShmemPerBlock: 49152
 MaxThreadBlockSize: [1024 1024 64]
        MaxGridSize: [2.1475e+09 65535 65535]
          SIMDWidth: 32
        TotalMemory: 3.4090e+10
    AvailableMemory: 3.0040e+10
MultiprocessorCount: 80
       ClockRateKHz: 1597000
        ComputeMode: 'Default'
GPUOverlapsTransfers: 1
KernelExecutionTimeout: 0
   CanMapHostMemory: 1
    DeviceSupported: 1
     DeviceSelected: 1
```


or it may output
```matlab
gpuDevice
Error using gpuDevice (line 26)
No supported GPU device was found on this computer. To learn more about supported GPU devices, see <a
href="matlab:web('http://www.mathworks.com/gpudevice','-browser')">www.mathworks.com/gpudevice</a>.
```

One can delete

```matlab
 'single','gpuArray'
```

command, like replace

```matlab
rand(p_N,N_W,'single','gpuArray')
```

by

```matlab
rand(p_N,N_W)
```

in the code to stop it from using gpu, but runing dnn problem will cause much time.



