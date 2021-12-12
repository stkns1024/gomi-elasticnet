from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="gmelasticnet",
    version="0.1.0",
    rust_extensions=[
        RustExtension("gmelasticnet.gmelasticnet", binding=Binding.PyO3)
    ],
    packages=["gmelasticnet"],
    zip_safe=False,
)
