from __future__ import print_function

import distutils.spawn
import importlib
import os.path
import shlex
import subprocess
import sys

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(here, 'labelme', '_version.py')

version = importlib.machinery.SourceFileLoader(
    '_version', version_file
).load_module().__version__
del here

install_requires = [
    'matplotlib>=3.1',
    'numpy>=1.16',
    'Pillow>=2.8',
    'PyYAML>=5.1',
    'qtpy>=1.8',
    'termcolor',
    'jsonpickle>=1.2',
    'scipy>=1.2',
    'configobj',
    'keras',
    'tensorflow>=1.8,<2',
    'sip',
    'pyqt5>=5.12',
    'opencv-python',
    'IPython'
]

if sys.argv[1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/labelme-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()
    try:
        import github2pypi
        return github2pypi.replace_url(
            slug='wkentaro/labelme', content=long_description
        )
    except Exception:
        return long_description


# TODO: update descriptions
setup(
    name='semiauto_annotate',
    version=version,
    packages=find_packages(),
    description='Semi-automatic Image Annotation with Python',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Yifan Zhao, Shucheng Zhong',
    author_email='evanzhao@umich.edu',
    url='',
    install_requires=install_requires,
    license='GPLv3',
    keywords='Image Annotation, Machine Learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    package_data={
        'labelme': ['icons/*', 'config/*.yaml'],
        'pose_estm': ['model/*', 'config'],
        'yolo': ['model_data/*']
    },
    entry_points={
        'console_scripts': [
            'labelme=labelme.main:main',
            'labelme_draw_json=labelme.cli.draw_json:main',
            'labelme_draw_label_png=labelme.cli.draw_label_png:main',
            'labelme_json_to_dataset=labelme.cli.json_to_dataset:main',
            'labelme_on_docker=labelme.cli.on_docker:main',
            'pose_estm=pose_estm.pose_detection:main',
            'yolo=yolo.yolo_video:main'
        ],
    },
)
