## About 

This repository stores the source code for the
*Second Year Project* (Introduction to Natural
Language Processing and Deep Learning) at the 
IT University of Copenhagen. 

Within the project we put
[PLStream](https://arxiv.org/pdf/2203.12368v1.pdf)
, a novel framework for fast polarity labelling of
massive data streams, to a test by analysing its
linguistic capabilities through [CheckList](https://arxiv.org/abs/2005.04118),
a general-purpose framework facilitating
comprehensive testing of NLP models. The results
are compared to two state-of-the art supervised
models. Our analysis reveals shortcomings in
PLStream’s overall ability to understand and learn
from language - especially regarding not
understanding contextual information and learning
biases from training data

## Reproducing Results

Detailed information of additional installations,
downloads of data and which scripts to run is
located in the README's in the `src/pre-project`
and `src/project` folders. Both guides, however,
assume the same Python virtual environment. Follow
the below steps to set-up the project with the
correct python versions and dependencies in the
right versions. 

*Feel free to use your preferred package manager*
(`conda`/ `venv`/ `pyenv`)

### `venv`: Create Virtual Environment 

First, navigate to the folder where you want to
store your virtual environment. (usually one has
a defined folder outside of the project root for
this purpose). Then run the following command (do
not forget the change thee variable in square
brackets):

```
python3 -m venv [name of venv]
```

Now, you can activate the virtual environment
through the command (assuming you are in the
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


Next, we download the dependencies in the virtual
environment. First, make sure your pip is updated:

```
pip install --upgrade pip
```

Then, run the following command from the root of
the cloned directory to install all dependencies:
    
```
pip install -r requirements.txt
```

You are all set!

### Create Virtual Environment (`conda`)

You can create a virtual environment within in
`conda` (this assumes you have installed either
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/). 
First, create the new virtual environment
including Python 3.7 through 

```bash
conda create --name plstream python=3.7
```

Activate the environment 

```
conda activate plstream
```

Check that the `PYTHONPATH` and `PIP` is correctly
udpated by running 

```bash
which python
which pip
```

The output of both should point to a path
including the virtual environments name `plstream`
in it.

Lastly, install the requirements from
`requirements.txt`

`pip install -r requirements.txt`

You are all set!

### Get the results

Finally, with all the dependencies installed in
your active virtual environment, you can go to
[project](src/project) folder and see instructions
on how to reproduce results reported in our
[report](). In addition, you can also inspect our
[pre-project](src/pre-project) folder where you
can find information about our work done before
the actual project. (e.g. generating test cases
with [Checklist
framework](https://github.com/marcotcr/checklist)).


## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/LudekCizinsky"><img src="https://github.com/LudekCizinsky.png?size=100" width="100px;" alt=""/><br /><sub><b>Ludek Cizinsky</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/jonas-mika"><img src="https://github.com/jonas-mika.png?size=100" width="100px;" alt=""/><br /><sub><b>Jonas-Mika Senghaas</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/lukyrasocha"><img src="https://github.com/lukyrasocha.png?size=100" width="100px;" alt=""/><br /><sub><b>Lukas Rasocha</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/IbenH"><img src="https://scontent-arn2-1.xx.fbcdn.net/v/t31.18172-8/14188125_1368561793173156_2247276646324128922_o.jpg?_nc_cat=105&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=lphH7XkJDNMAX_hNnaF&_nc_ht=scontent-arn2-1.xx&oh=00_AT8UDm-bDNWs0ui4xa9u8TM89T-8L7OwQkjVTvkgQc9TPA&oe=62B38B85" width="100px;" alt=""/><br /><sub><b>Iben Huse</b></sub></a><br /></td>

  </tr>
</table>

