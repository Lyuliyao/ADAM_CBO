# ADAM_CBO
## This is the code written for A CONSENSUS-BASED GLOBAL OPTIMIZATION METHOD WITH ADAPTIVE MOMENTUM ESTIMATION
This is a matlab code and require gpu for running it. You can use "gpuDevice" in matlab to see whether your matlab support gpu computation.
If your matlab support gpu computation, it may output
>> gpuDevice
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


or it may output
>> gpuDevice
Error using gpuDevice (line 26)
No supported GPU device was found on this computer. To learn more about supported GPU devices, see <a
href="matlab:web('http://www.mathworks.com/gpudevice','-browser')">www.mathworks.com/gpudevice</a>.

One can delete 'single' and 'gpuArray' command, like replace
rand(p_N,N_W,'single','gpuArray')
by 
rand(p_N,N_W)
in the code to stop it from using gpu, but runing dnn problem will cause much time.



