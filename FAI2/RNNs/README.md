## Project Files

```

 |-- data_processsing.py 
 |-- rnn_module.py, 
 |-- lstm_module.py, 
 |-- transfromer_module.py, 
 |-- evaluation.py, 
 |-- evaluation_metics_for_all.py,
 |-- result_plot.py, 
 |-- train.py, and 
 |-- prompt_responses.py
```

## Setup
```
# clone repo
$ git clone https://github.com/Sbell-ppd/Foundational-AI-2.git
$ cd Foundational-AI-2
```

## Usage

```
# Model Implementation, data processing, training, evaluation(metrics), result_plot etc
Once all the three models have been implemented, I will do the data data_processsing by combining all the .txt in the raw_data directory into a single .txt file. And tokenize my dataset (train.jsonl and test.jsonl)
Then run the train.py file to train the models.
Then run the evaluation.py file to evaluate the models.
Then run the evaluation_metics_for_all.py file to get the evaluation metrics for all the models.
Then run the result_plot.py file to plot the results of the models.
Finally, run the prompt_responses.py file to get the prompt responses from the models(N:B Your models are working if they are generating anything other than complete gibberish).

To run the codes, I will use the following command in the terminal:

python .\prompt_responses.py

You can run each and every .py file in my directory with command : python followed by the filename.
```
### Command-line Arguments
- `--epochs` : Numbers of training epochs (30)
- `--batch_size` : Batch size for training (128)
- `--Optimizer` : AdamW
- `--learning rate scheduler` : 0.0005
- `--loss function`: CrossEntropyLoss
```
