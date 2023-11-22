## Setup
(Recommended) activate a conda environment and run:
```shell
$ pip install label-studio
```
Activate the environment above and proceed to the next step.

## Start Labeling Project
A project is usually composed of a labeling config (e.g in `.xml`) and data.
To start a new project, run in a the following in your terminal:
```shell
$ label-studio init <PROJECT_NAME> -l <LABELING_CONFIG (XML)>
```
This command will initialize a new project named `<PROJECT_NAME>`, with the labeling task found in `labeling_config.xml`.
It should ask you to provide an email and password. After you've provided that, you can start the project on your local server:
```shell
$ label-studio start <PROJECT_NAME>
```
Go to http://localhost:8080/ in your browser, and import data, by uploading an example JSON (`data.json`).
Now, the labeling task is set and you begin annotating.

## Exporting Annotations
The easiest way to export your annotation, is through the UI.
1. Go to the project page.
2. Press `Export` in the top right corner.
3. A window will pop -> export in the desired format.