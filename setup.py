from setuptools import setup, find_packages

setup(
    name="mlmp",  # 包名
    version="0.1.3",  # 版本号
    author="zaixia108",  # 作者名
    author_email="xvbowen2012@gmail.com",  # 作者邮箱
    description="A Python package for mouse path control and training.",  # 描述
    long_description=open("README.md").read(),  # 长描述（README 文件）
    long_description_content_type="text/markdown",  # 长描述格式
    url="https://github.com/zaixia108/MLMousePath",  # 项目主页
    packages=find_packages(),  # 自动查找包
    package_data={
        "mlmp": ["*.dll"],  # 包含 DLL 文件
    },
    include_package_data=True,  # 包含非 Python 文件
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Python 版本要求
    install_requires=[
        'onnxruntime',
    ],  # 依赖项
)