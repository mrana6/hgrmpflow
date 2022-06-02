# Differentiable-RMPflow
RMPflow implementation in PyTorch. This package has been tested on python3.5.

## Requirements
1. urdf_parser_py: `pip3 install urdf-parser-py`
2. pytorch (>=1.5.0)
3. numpy (>=1.18.5)
4. matplotlib (>=0.1.9)
5. fastdtw (>=0.3.4)
6. tensorboardX (>=2.0)


## (OPTIONAL) Setup for KDL version of Robot Class Setting up KDL with Python3 alongside ROS kinetic
NOTE: Unsource ROS before doing this

Full ROS installation comes pre-packaged with kdl and its python binding PyKDL. However KDL (1.3.0) for ros-kinetic does not have `ChainToJacDotSolver` function which is otherwise there in the newer version. To use the 1.4.0 alongside ROS kinetic, we will have to install KDL and PyKDL from source,
1. Install SIP (Check the version already installed by `sip -V`. If 4.19 or later is installed, you might wanna install 4.17-4.18 instead)
    1. Download the 4.17 tar file from https://sourceforge.net/projects/pyqt/files/sip/sip-4.17/.  
    2. Extract `tar -zxf sip-4.7.tar.gz`
    3. `cd sip-4.17.tar.gz` and build using `python configure.py`
    4. Install `make`, `sudo make install`
    5. run `sudo ldconfig`
2. Clone KDL and PyKDL (make sure to use this fork!): `git clone https://github.com/MichaelLutter/orocos_kinematics_dynamics`
3. Install KDL C++ library:
    1. Configure: `cd <kdl_dir>/orocos_kdl`.  `mkdir build && cd build`.
    2. `ccmake ..`, press `[c]` to configure and `[g]` to generate
    3. Install: `make` and `sudo make install`
4. Install PyKDL for python3
    1. Configure: `cd <kdl_dir>/python_orocos_kdl`.  `mkdir build && cd build`
    2. `ccmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3 -Dorocos_kdl_DIR=/usr/local/share/orocos_kdl/cmake ..`
    3. IMPORTANT: You might wanna double check `orocos_kdl_path` is set to `/usr/local/share/orocos_kdl/cmake`. You can just do `ccmake ..`.
    4. Install: `make -j4` and `sudo make install`
5. The above steps will install generate `/usr/local/lib/python3/dist-packages/PyKDL.so`. Now we have to force our python scripts to link to this version instead of that in `/opt/ros/kinetic/lib/python2.1/dist-packages/PyKDL.so`:
    1. Run `sudo ldconfig` to make the system find the new version first
    2. To run alongside ROS, prepend the installation path to PYTHONPATH by adding this line after sourcing ROS to your bashrc `export PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH`
    3. You might also want to add to `LD_LIBRARY_PATH`. Add this to bashrc: `export LD_LIBRARY_PATH=/usr/local/lib/python3/dist-packages:$LD_LIBRARY_PATH`
    4. Source bashrc
    5. Run `sudo ldconfig` again

## Debugging
Did you unsource ROS? Check by running `roscore`.
Check this issue: [https://github.com/orocos/orocos_kinematics_dynamics/issues/115]()

## Setting kdl_parser_py in python3 alongside ROS kinetic
NOTE: Unsource ROS before doing this

kdl_parser that comes with ROS kinetic does not support Python3 either. However the newer version of kdl_parser does use python3.
We need to install the newer version from source.

1. Clone repo `git clone https://github.com/ros/kdl_parser.git`.
2. Checkout the noetic release: `cd kdl_parser`, `git checkout tags/1.14.0`.
3. Install kdl_parser: `cd kdl_parser && mkdir build && cd build`, `ccmake ..` (See the debug info below if cmake is old).
4. Install kdl_parser_py: `cd kdl_parser_py`, `sudo setup.py install`
5. Add to PYTHONPATH: `export PYTHONPATH=/usr/local/lib/python2.7/dist-packages/kdl_parser_py-1.14.0-py2.7.egg:$PYTHONPATH`


## Debugging
kdl_parser installation might need upgrading cmake. Follow these steps:

Warning -- Do not do step 2 if you have Robot Operating System (ROS) installed
1. Check your current version with cmake --version
2. Uninstall it with sudo apt remove cmake
3. Visit https://cmake.org/download/ and download the latest binaries
4. In my case cmake-3.6.2-Linux-x86_64.sh is sufficient copy the binary to /opt/
5. `chmod +x /opt/cmake-3.*your_version*.sh` (chmod makes the script executable)
6. `sudo bash /opt/cmake-3.*your_version.sh*` (you'll need to press y twice)
7. The script installs to /opt/cmake-3.*your_version* so in order to get the cmake command, make a symbolic link: 
`sudo ln -s /opt/cmake-3.*your_version*/bin/* /usr/local/bin`
7. `source ~/.bashrc`
8. Test your results with `cmake --version`.

