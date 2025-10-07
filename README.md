How to run the file : 

Extract all of the contents that is available in here
meta.csv, inference.py, dataset.

*advised to adjust the epochs count from (50 - 100) in order to increase the accuracy of the model. (line 435)
*epoch - is one complete pass through the entire training dataset by a learning algorithm


run the file using the command with the format "python3 inference.py --images <--path--> --meta meta.csv --out preds.csv" to the terminal on your IDE


How it works : 
1. it takes the meta.csv data that we have made and scripted carefully.
2. it takes 19 images that we've been given as a training material for the model.
3. Please adjust the epoch (optional but recommended)
4. takes into account lighting, skintone.
5. evaluate the HgB based on the data that is available, and prints out the HgB and the confidence rate to the terminal
6. it will be outputtted into preds.csv as per the command line.
