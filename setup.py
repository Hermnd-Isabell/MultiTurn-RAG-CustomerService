from setuptools import setup, find_packages

setup(
    name="RAG_project_zhyd",  # Replace with your package name
    version="0.1",  # Version of the package
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "numpy>=1.19.2",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",  # FAISS for CPU, use faiss-gpu if needed
        "elasticsearch>=7.10.0",
        "python-dotenv>=0.19.2",
        "openai>=0.27.0",
        "python-docx>=0.8.10", # Changed from docx to python-docx
        "gradio>=2.2.4",
        "langchain>=0.2.0",  # embed.py imports RetrievalQA (legacy import, kept for compatibility)
        # If you have an additional custom package like "embed", include it here if available on PyPi
        # For example, if 'embed' is not a custom module but you want to include some custom scripts, make sure they are part of your package
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change this if your package uses a different license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # Specify your Python version
)
