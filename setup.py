#!/usr/bin/env python3
"""
Tenzor: High-performance tensor computation and neural network library

This setup.py is a legacy fallback for environments that do not support
PEP 517 / pyproject.toml builds.  The preferred build backend is
scikit-build-core (configured in pyproject.toml).
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel


class CMakeExtension(Extension):
    """CMake extension for building C++ modules"""

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command that uses CMake"""

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build Tenzor. "
                "Install with: pip install cmake or apt-get install cmake"
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DTENZOR_BUILD_PYTHON=ON',
            '-DTENZOR_BUILD_TESTS=OFF',
            '-DTENZOR_BUILD_EXAMPLES=OFF',
            '-DTENZOR_BUILD_BENCHMARKS=OFF',
            '-DTENZOR_BUILD_MODEL_HUB=OFF',
        ]

        # Build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Platform-specific settings
        if sys.platform.startswith('win'):
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            # Use all available cores
            import multiprocessing
            build_args += ['--', f'-j{multiprocessing.cpu_count()}']

        # Run CMake configuration
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )

        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


class PlatformBdistWheel(bdist_wheel):
    """Ensure wheels are tagged as platform-specific (not pure-Python)."""

    def finalize_options(self):
        super().finalize_options()
        # Mark the wheel as platform-specific since it contains C++ extensions
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        # On Linux, replace generic 'linux' with the manylinux tag if applicable
        if plat.startswith('linux'):
            arch = platform.machine()
            plat = f'linux_{arch}'
        return python, abi, plat


# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

setup(
    name='tenzor',
    version='1.0.0',
    author='Tenzor Contributors',
    description='High-performance tensor computation and neural network library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('tenzor.tenzor_core')],
    cmdclass={
        'build_ext': CMakeBuild,
        'bdist_wheel': PlatformBdistWheel,
    },
    zip_safe=False,
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'sphinx-autodoc-typehints>=1.12',
        ],
        'viz': [
            'matplotlib>=3.3',
            'tensorboard>=2.5',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
