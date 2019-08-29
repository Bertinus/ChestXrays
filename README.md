# Chester: A Web Delivered Locally Computed Chest X-Ray Disease Prediction System

## Joseph Paul Cohen, Paul Bertin, Vincent Frappier

This repository contains the python code used to build and train the models used in our paper 
[Chester: A Web Delivered Locally Computed Chest X-Ray Disease Prediction System](https://arxiv.org/pdf/1901.11210.pdf).

## Environment Setup

The conda environment used for this project is provided with the **spec-file.txt**
To use the spec file to create an identical environment, run :

````
conda create --name myenv --file spec-file.txt
````

## Abstract

Deep learning has shown promise to augment radiologists 
and improve the standard of care globally. Two main issues
 that complicate deploying these systems are patient 
 privacy and scaling to the global population. To deploy
  a system at scale with minimal computational cost while
   preserving privacy we present a web delivered (but 
   locally run) system for diagnosing chest X-Rays. 
   Code is delivered via a URL to a web browser 
   (including cell phones) but the patient data remains
    on the users machine and all processing occurs 
    locally. The system is designed to be used as a 
    reference where a user can process an image to 
    confirm or aid in their diagnosis. The system 
    contains three main components: out-of-distribution 
    detection, disease prediction, and prediction 
    explanation. The system open source and freely 
    available [here](https://mlmed.org/tools/xray).
