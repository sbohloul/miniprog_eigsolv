from ctypes import c_int, addressof
import numpy as np
import _pb11_hello_array

v1 = np.arange(6, dtype=np.float64)
id_v1 = v1.ctypes.data
print("id_v1: ", hex(id_v1))
_pb11_hello_array.array_info(v1)

v2 = np.arange(6, dtype=np.float64).reshape(2, 3)
id_v2 = v2.ctypes.data
print("id_v2: ", hex(id_v2))
_pb11_hello_array.array_info(v2)
