# Elasticsearch Windows Installation Guide

Follow these steps to install Elasticsearch on Windows:

## 1. Download Elasticsearch
Visit the Elasticsearch download page on the official website ([link](https://www.elastic.co/guide/en/elasticsearch/reference/current/zip-windows.html#zip-windows)) and download the desired version. It's generally recommended to use the latest version.

## 2. Extract the ZIP File
Extract the downloaded ZIP file to a location of your choice. For example, if you extract it to `C:\elasticsearch`, all the Elasticsearch files would be in this location.

## 3. Run Elasticsearch from Command Prompt
Navigate to the extracted folder and find the `bin` directory. Run the `elasticsearch.bat` file in this directory to start Elasticsearch. To run this file, open Command Prompt, navigate to the `bin` directory, and enter `elasticsearch.bat`.

Here is an example command line sequence:

~~~
cd C:\elasticsearch\bin
elasticsearch.bat
~~~

## 4. Run Elasticsearch from Jupyter Notebook
You can also run Elasticsearch from a Jupyter Notebook. Here is an example Python code to start Elasticsearch:

~~~
import os
from subprocess import Popen, PIPE, STDOUT
import time

# Assuming Elasticsearch is installed in 'C:\elasticsearch'
es_server = Popen(['C:/elasticsearch/bin/elasticsearch.bat'], stdout=PIPE, stderr=STDOUT)

# Wait until Elasticsearch has started
time.sleep(30)
~~~

This Python code launches Elasticsearch as a separate process. After this code is executed, Elasticsearch should be running and ready for interactions from subsequent cells in the notebook.
Elasticsearch runs on port 9200 by default, and you can check if Elasticsearch is running properly by accessing `http://localhost:9200` in your browser.
