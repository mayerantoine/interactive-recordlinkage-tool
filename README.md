# interactive-recordlinkage-tool

Interactive tool to experiment and automatically compare matching quality of different data matching algorithm. This is a proof of concept  developed using Python and Streamlit.

This tool is built on top of [Python Record Linkage Toolkit](https://github.com/J535D165/recordlinkage). 

## Run as a docker container

The easiest way to start is to use docker. Download or clone the repository. Build and test your image.
```
docker build -t interactive-recordlinkage:latest .

```
Run your image as a docker container.
```
docker run --publish 8501:8501 --name interative-recordlinkage interactive-recordlinkage:latest

```

## Run on your computer

You need to have Python 3.6 installed on your machine. Download or clone the repository. Install all the dependencies.
```
$pip install -r requirements.txt
```
Run the streamlit app
```
streamlit run interactive_app.py
```
## Demo

![demo](Media1_interactive_rl.gif)