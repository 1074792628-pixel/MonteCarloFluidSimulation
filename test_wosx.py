import sys, os
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\python\Release')
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\msvc_19.51_cxx_64_md_release')
os.add_dll_directory(r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\Release')
sys.path.insert(0, r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings\python\Release')

import _wosx
print('WoSX loaded!')
core = _wosx.Core
print('Core types:', [x for x in dir(core) if not x.startswith('_')])
print('Solvers:', [x for x in dir(_wosx.Solvers) if not x.startswith('_')])
print('Utils:', [x for x in dir(_wosx.Utils) if not x.startswith('_')])
