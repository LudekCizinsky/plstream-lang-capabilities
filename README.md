## About project
In this project, we compared language capabilities of [PLstream](https://arxiv.org/pdf/2203.12368v1.pdf), unsupervised ML framework for polarity labelling of massive data streams, to state-of-the-art supervised classification methods such as [Bert](https://arxiv.org/abs/1810.04805). Language capabilities were tested using predefined tests for sentiment classification models which were introduced in [Checklist paper](https://arxiv.org/abs/2005.04118).


## Reproduce results
### Create Virtual Environment
To reproduce our results, it is essential that you use virtual environment. If
you are not familiar with virtual environment, then we provided a small
tutorial below which assumes that you have python (>=3.3) installed. Otherwise,
you can skip this part.

First, navigate to the folder where you want to store your virtual environment.
(usually one has a defined folder outside of the project root for this purpose). Then run the following command (do not forget the change thee variable in square brackets):

```
python3 -m venv [name of venv]
```

Now, you can activate the virtual environment through the command (assuming you are in the
folder where you ran the above command): 

```
source [name of env]/bin/activate
```

You can then deactivate it using:

```
deactivate
```

Note that this one of the ways how you can manage your virtual environments, for more info,
you can read this [article](https://realpython.com/python-virtual-environments-a-primer/).

### Install requirements 

First, make sure your pip is updated:

```
pip install --upgrade pip
```

Then, run the following command from the root of the cloned directory to install all dependencies:
    
```
pip install -r requirements.txt
```

### Get the results
Finally, with all the dependencies installed in your active virtual environment, you can go to [project](src/project) folder and see instructions on how to reproduce results reported in our [report](). In addition, you can also inspect our [pre-project](src/pre-project) folder where you can find information about our work done before the actual project. (e.g. generating test cases with [Checklist framework](https://github.com/marcotcr/checklist)).


## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/LudekCizinsky"><img src="https://github.com/LudekCizinsky.png?size=100" width="100px;" alt=""/><br /><sub><b>Ludek Cizinsky</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/jonas-mika"><img src="https://github.com/jonas-mika.png?size=100" width="100px;" alt=""/><br /><sub><b>Jonas-Mika Senghaas</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/lukyrasocha"><img src="https://github.com/lukyrasocha.png?size=100" width="100px;" alt=""/><br /><sub><b>Lukas Rasocha</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/IbenH"><img src="https://www.facebook.com/photo/?fbid=1368561793173156&set=a.151233234906024" width="100px;" alt=""/><br /><sub><b>Iben Huse</b></sub></a><br /></td>

  </tr>
</table>

