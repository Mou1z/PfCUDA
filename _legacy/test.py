import cupy as cp

a = cp.array([1,2,3])
a += 1
cp.cuda.runtime.deviceSynchronize()
print("Test passed, device works!")
