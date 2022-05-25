## About 

The goal of the project phase was to use
modified
[PLstream](https://github.com/HuilinWu2/PLStream/tree/main/PLStream)
framework to produce [predictions](predictions/)
on the [test cases](data/checklist_tests.txt)
proposed by `Ribeiro et al.` in CheckList in their
[paper](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf).

## Reproducing Results

### Installations and Downloads

On a implementation level, this project required
making [PLStream's source
code](https://github.com/HuilinWu2/PLStream/tree/main/PLStream) compatible with the [Checklist Testing API](https://github.com/marcotcr/checklist#tutorials).

PLStream is implemented through Apache Flink.
To reproduce, we need to install the dependencies
needed to run the streaming processing framework
(which is built upon Java).

At this point, it is assumed that the Python VENV
is already resolved and activate (*as specified in
the README at the root of this project*).

#### Java V8 

Check if you have `Java V8` installed:

```bash 
java -version 
```

If not, then you can install it according to this
[tutorial](https://docs.oracle.com/javase/8/docs/technotes/guides/install/mac_jdk.html).
Note that you can use multiple versions of Java on
the same system. You can then set the default one
in your terminal language setting file (e.g.
`.bashrc`).

#### Apache Flink 

To install Apache Flink, please follow the steps in download
[section](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/try-flink/local_installation/#downloading-flink)
in the official solution. (ignore the java section
part) Alternatively, on macOS, assuming you have
`brew` installed, you can just run:

```bash 
brew install apache-flink 
```

#### Redis 

The PLStream implementation depends on dynamically
storing and loading the most-up-to-date `word2vec`
model from the distributed, in-memory key–value
database `redis`. Install `redis` using `brew` on
mac:

```bash
brew install redis
```

#### Python >3.7 
Again, check if you have python installed with at least version 3.7. This should be the case if you have correctly resolved the virtual environment at the root the project.

```bash 
python --version 
```

If not, you can follow the official
[docs](https://www.python.org/downloads/) to
install it.

#### Data

PLStream is going to be trained on the Yelp Review
data set before evaluated on the CheckList test
cases. The training data needs to be downloaded.
Run the following bash script from the current
directory (`src/project`):

```bash 
source download.sh 
```

### Getting Results

Before executing any scrips, fire up redis-server
locally in a separate shell session.


```bash 
redis-server 
```

Now, the `main.py` script can be run. The main
script uses `argparse` to control what actions
should be performed when running the script.

If this is the first time running this project,
activate all flags:

```bash
python main.py -F -T -R
```

This is, what the flags are doing:

```
Control Runflow of main.py

optional arguments:
  -h, --help                      show this help message and exit
  -F, --format-plstream-train     Format PLStream Training Data and write to
                                  data/processed/final.csv
  -T, --train-plstream            Train PLStream
  -R, --reproduce-checklist       Reproduce the Checklist Paper Results
```

Use the `-F` flag, in order to concatenate the
`data/rawtrain.csv` (Yelp Review Dataset) with the
`data/raw/checklist-tests.txt` (CheckList Test)
and save it in `data/processed/final.csv`. This
data set is required for the training and
evaluation of PLStream to work. 

Use the `-T` flag to train PLStream. By default,
the PL100 model (trained on 100.000 review from
the Yelp Review dataset) is being trained. The
script will output the predictions of the PLStream
model into `predictions/plstream_100000` in the
format required by the `checklist` python package.

Use the `-R` flag to create comprehensive
summaries of the checklist performance of every
model that has predictions stored in
`results/checklist_predictions`. The results 
will be saved to `results/checklist_summaries`.


*Feel free to contact the group, if any of the
above steps causes problems*
