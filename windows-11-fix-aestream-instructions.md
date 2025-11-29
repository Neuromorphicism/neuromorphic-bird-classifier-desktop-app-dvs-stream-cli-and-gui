
# None of the fixes below worked for me (November 2025)

<br>

## Windows without conda

Open terminal in the NeuroBCDA folder 

Check if you are using python 3.9 or newer by running: `python --v`

Run: `pip install -r requirements.txt`

The playsound package will not work with Python > 3.11 on Windows!

If you have such Python run: `pip install playsound@git+https://github.com/taconi/playsound`

Then again run: `pip install -r requirements-windows.txt` with all the uncommented packages (removed #)

Now aestream will most probably fail…

<br>

## Install miniconda on Windows

So you must install miniconda on Windows in a terminal: https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell

now in the folder run: 

`<PATH-TO-CONDA>\miniconda3\Scripts\activate.bat`

Replace <PATH-TO-CONDA> with the path to your conda installation

Usually: 

```bash
source C:\Users\<YOUR-USERNAME>\miniconda3\Scripts\activate.bat
conda init --all
conda create -n neurobcda python=3.9
conda activate neurobcda
pip install -r requirements-windows.txt
```

<br>

## Anaconda PowerShell Prompt

It is anyway best to simply use Anaconda PowerShell Prompt, change directory to the program folder and run there:

```bash
conda create -n neurobcda python=3.9
conda activate neurobcda
pip install cmake
pip install -r requirements-windows.txt
```

If you have an error with outdated CMAKE then in cmd run: `choco upgrade cmake`

<br>

## CMake Problems

If you are still having problems with aestream and cmake run:

```bash
conda install -c conda-forge gcc
pip install cmake==3.27.9
pip install ninja
pip install flatbuffers
pip cache purge
pip install --no-build-isolation --no-dependencies aestream
```

*Actually lz4 package should not be a dependency here because we are not using aedat4 files from a real DVS.*

<br>

## The main problem of AEStream is lz4

If the problem persists manually clone the lz4 Python bindings repository, then build the package using CMake manually to control CMake options:

```bash
git clone https://github.com/python-lz4/python-lz4.git
cd python-lz4
mkdir build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .
python setup.py install
```

<br>

## Install CMake from source

```bash
git clone https://github.com/aestream/aestream
cd aestream
mkdir build
cd build
cmake -GNinja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_PYTHON=ON -DUSE_CUDA=0 .. 
ninja install
```

Here you would have to make sure that CMake finds the nanobind package installed with pip

<br>

## GCC version below 10

AEStream relies on modern compiler features and requires at least GCC 10 and CMake 3.20. To update GCC to version 10 on Windows the recommended approach is to install the MinGW-w64 toolchain which provides up-to-date GCC builds for Windows. MSBuild might also fail due to exceeding the maximum path length limit (260 characters) on Windows, which means that your folders can not be deeply nested (actually a hilarious fail, who even set that limit).

In general the whole Aestream library was not yet tested on Windows but its dependencies also require POSIX-like headers "unistd.h" so it might never run on pure Windows. MinGW-w64 and clang on Windows both support __attribute__ and unistd.h much better.

For me the problem was with nanobind so maybe there is a missing "find_package(nanobind CONFIG REQUIRED)" in one of the CMakeLists.txt files in the aestream repository.

<br>

## Other possible fixes

Other fixes for Windows might include running this app in docker or WSL2 but those would be Linux envs not Windows anymore: https://github.com/aestream/aestream/issues/94

Finally run: 
```bash
pip install scikit-build
pip install --no-build-isolation aestream==0.5.1
pip install lz4
pip install --no-deps aestream
```

The above problem with aestream cmake build occurs only in versions 0.6.4; 0.6.3; 0.6.2; 0.6.0 and the necessary DAT file manipulation is only present from version 0.5.1 of aestream but the dependencies conflict is still present even in the 0.5.1 version.

Most probably the AEStream library should be replaced with AEDAT (thanks to Rust the horrors of Cmake are gone with magnificent cargo) or Expelliarmus to make NeuroBCDA fully work on Windows.
