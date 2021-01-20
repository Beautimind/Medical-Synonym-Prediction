# Medical-Synonym-Prediction

## Basic settings
The setence pre-processing and pretrained word embedding follows this repo: [https://github.com/minhcp/BNE]

## Usage
directly run the train_*.py to run the training and testing for different model.
All the data are in the '/data' folder, and the trained models will be save to the '/model' folder

## Accuracy
BNE with exact phrases only: 83.3%<br> 
ESIM with exact phrases only: 97.3%<br>
SSE with exact phrases only: 97.8%<br>

BNE with all phrases: 80.4%<br>
ESIM with all phrases: 96.5%<br>
SSE with all phrases: 96.6%<br>
