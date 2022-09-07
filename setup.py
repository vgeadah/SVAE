from distutils import core

core.setup(
    name="svae",
    version="0.0.1",
    packages=["svae"],
    license="Copyright (c) 2019 Pillow Lab",
    install_requires=["numpy", "scipy", "torch", "torchvision"],
)
