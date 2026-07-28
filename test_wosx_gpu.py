import sys, os
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\python\Release')
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\msvc_19.51_cxx_64_md_release')
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\Release')
sys.path.insert(0, r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\python\Release')

import _wosx
s = _wosx.Solvers
print('WalkOnSpheres 2D float:', 'WalkOnSpheres_float_2d' in dir(s))
print('WalkOnStars 2D float:', 'WalkOnStars_float_2d' in dir(s))
print('GPU WalkOnSpheres:', [x for x in dir(s) if x.startswith('GPUWalkOnSpheres')])
print('GPU WalkOnStars:', [x for x in dir(s) if x.startswith('GPUWalkOnStars')])
print('GPU TaskHandle:', [x for x in dir(_wosx) if 'TaskHandle' in x])
