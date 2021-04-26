# pytorch_transformer_lm

This is a pytorch implementation of a Transformer-based language model. 
The dependencies are pytorch 1.8.1 and torchtext 0.9.1.

You can train the model by running
```
python main.py --mode=train
```

After training, you can sample from the model by running 
```
python main.py --mode=generate
```
and the samples will be appended to a file ```output/model_name/samples.txt```.

PS: It's actually working! In this git repo, samples from a transformer trained for 27 epochs are provided in ```output/model/samples.txt```. 
After you clone the repo, delete the file to start from scratch, or name your model something different using the ```--model_name``` flag.
