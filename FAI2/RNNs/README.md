## Steps to run my codes
1. I have 9 .py files in the directory. Namely :

data_processsing.py , 
rnn_module.py, 
lstm_module.py, 
transfromer_module.py, 
evaluation.py, 
evaluation_metics_for_all.py,
result_plot.py, 
train.py, and 
prompt_responses.py

Once all the three models have been implemented, I will do the data data_processsing by combining all the .txt in the raw_data directory into a single .txt file. And tokenize my dataset (train.jsonl and test.jsonl)

Then I will run the train.py file to train the models.
Then I will run the evaluation.py file to evaluate the models.
Then I will run the evaluation_metics_for_all.py file to get the evaluation metrics for all the models.
Then I will run the result_plot.py file to plot the results of the models.
Finally, I will run the prompt_responses.py file to get the prompt responses from the models.


To run the codes, I will use the following command in the terminal:

## python .\prompt_responses.py

You can run each and every .py file in my directory with command : python followed by the filename.