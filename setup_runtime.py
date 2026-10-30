"""Build the Metal GGUF runtime. Requires Xcode Command Line Tools and CMake."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor' / 'stable-diffusion.cpp'
REVISION = '77b72a2efed78f55de26892c7e909cee001e8b83'


def main():
    cmake = shutil.which('cmake') or str(Path(sys.executable).parent / 'cmake')
    if not SOURCE.exists():
        subprocess.run(['git', 'clone', 'https://github.com/leejet/stable-diffusion.cpp.git', str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', REVISION], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'submodule', 'update', '--init', 'ggml'], check=True)
    subprocess.run([cmake, '-S', str(SOURCE), '-B', str(SOURCE / 'build'),
                    '-DSD_METAL=ON', '-DSD_WEBP=OFF', '-DSD_WEBM=OFF', '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run([cmake, '--build', str(SOURCE / 'build'), '--config', 'Release',
                    '--target', 'sd-cli', '-j', '4'], check=True)
    print('Runtime ready. Run python app.py')


if __name__ == '__main__':
    main()
