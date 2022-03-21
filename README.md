## Run this project [draft]

Reproduce the results in the few following steps: 

### [optional] Create Virtual Environment

First, navigate to the folder where you are storing your venvs (or just create one) and then use venv to create virtual env 
for this project as follows: 

```
python3 -m venv [name of venv]
```

Now, you can activate the venv through the command (assumeing you are in the
folder where you ran the above command): 

```
source [name of env]/bin/activate
```

Deactivate through:

```
deactivate
```

Note that this one of the ways how you can manage you `venvs`, for more info,
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

### Run the pipeline
#### macOS
Start by allowing execution of the `shell` script:

```
chmod +x src/run.all
```

Then simply execute the shell script as follows:

```
cd src/
./run.all
```

## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/LudekCizinsky"><img src="https://github.com/LudekCizinsky.png?size=100" width="100px;" alt=""/><br /><sub><b>Ludek Cizinsky</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/jonas-mika"><img src="https://github.com/jonas-mika.png?size=100" width="100px;" alt=""/><br /><sub><b>Jonas-Mika Senghaas</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/lukyrasocha"><img src="https://github.com/lukyrasocha.png?size=100" width="100px;" alt=""/><br /><sub><b>Lukas Rasocha</b></sub></a><br /></td>
  </tr>
</table>

