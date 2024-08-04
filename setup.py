from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
with codecs.open(readme_path, encoding="utf-8") as fh:
    long_description = fh.read()
VERSION = '0.1.6'
DESCRIPTION = 'Performing thematic analysis with OpenAI\'s GPT-4 models'
LONG_DESCRIPTION = 'A package uses openAI\'s GPT-4 model to perform thematic analysis using interview transcripts from qualititative studies'

# Setting up
setup(
    name="AutoThemeGenerator",
    version=VERSION,
    author="Charles Alba",
    author_email="alba@wustl.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['textract', 'openai==1.12.0', 'nltk==3.8.1', 'requests==2.31.0','python-docx','docx','tqdm','requests'],
    keywords=['GPT models', 'Thematic analysis', 'OpenAI', 'Qualitiative studies', 'transcripts', 'interviews'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)