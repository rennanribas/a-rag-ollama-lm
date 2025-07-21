"""Setup script for AI RAG Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-rag-agent",
    version="1.0.0",
    author="AI RAG Agent",
    author_email="contact@example.com",
    description="Domain-specific RAG agent with incremental crawling and OpenAI integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ai-rag-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-rag=src.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)