"""
Setup configuration for the Inference Engine package.
"""

from setuptools import setup, find_packages

# Read README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='DocIntelligence',
    version='0.0.0',
    description='A package for intelligent document processing, featuring in processing image and tabular context.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zereo AI',
    author_email='zereo@zereo-ai.com',
    url='https://github.com/Zereo0317/DocIntelligence',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        # Core dependencies
        'langchain>=0.1.0',
        'langchain-core>=0.1.0',
        'langchain-anthropic>=0.1.0',
        'langchain-openai>=0.1.0',
        'langgraph>=0.0.10',
        'pydantic>=2.0.0',
        'python-dotenv>=1.0.0',
        
        # LLM APIs
        'anthropic>=0.7.0',
        'openai>=1.0.0',
        'google-generativeai>=0.3.0',
        
        # Database and RAG
        'sqlalchemy>=2.0.0',
        'sentence-transformers>=2.2.0',
        'chromadb>=0.4.0',
        'faiss-cpu>=1.7.0',
        'rapidfuzz>=3.0.0',
        
        # Utilities
        'aiohttp>=3.9.0',
        'requests>=2.31.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'python-json-logger>=2.0.0',
        'tenacity>=8.2.0',
        'xmltodict',
        'cloud-sql-python-connector',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
            'pylint>=2.17.0',
            'pre-commit>=3.3.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
            'sphinx-autodoc-typehints>=1.24.0',
        ],
        'gpu': [
            'faiss-gpu>=1.7.0',
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'accelerate>=0.20.0',
            'safetensors>=0.3.0',
            'einops>=0.6.0',
            'bitsandbytes',
        ],
        'postgres': [
            'psycopg2-binary>=2.9.0',
        ],
        'mysql': [
            'pymysql>=1.0.0',
            'cryptography>=41.0.0',
        ],
        'gcp': [
            'google-cloud-bigquery>=3.0.0',
            'google-cloud-storage>=2.10.0',
            'google-cloud-aiplatform>=1.36.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Framework :: AsyncIO',
        'Framework :: Pytest',
        'Typing :: Typed',
    ],
    keywords=[
        'docs',
        'document-processing',
        'machine-learning',
        'nlp',
        'langgraph',
        'langchain',
        "DocLayout-Analysis",
    ],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
)
