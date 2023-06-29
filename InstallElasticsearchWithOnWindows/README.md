# Elasticsearch Windows Installation Guide

Follow these steps to install Elasticsearch on Windows:

## 1. Download Elasticsearch
Visit the Elasticsearch download page on the official website ([link](https://www.elastic.co/guide/en/elasticsearch/reference/current/zip-windows.html#zip-windows)) and download the desired version. It's generally recommended to use the latest version.

## 2. Extract the ZIP File
Extract the downloaded ZIP file to a location of your choice. For example, if you extract it to `C:\elasticsearch`, all the Elasticsearch files would be in this location.

## 3. Run Elasticsearch
Navigate to the extracted folder and find the `bin` directory. Run the `elasticsearch.bat` file in this directory to start Elasticsearch. To run this file, open Command Prompt, navigate to the `bin` directory, and enter `elasticsearch.bat`.

Here is an example command line sequence:

```shell
cd C:\elasticsearch\bin
elasticsearch.bat
