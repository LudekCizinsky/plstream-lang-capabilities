## About
The goal of the project phase was to use modified [PLstream](https://github.com/HuilinWu2/PLStream/tree/main/PLStream) framework to produce [predictions](predictions/) on the [test cases](data/checklist_tests.txt) proposed by `Ribeiro et al.` in their [paper](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf).

## Reproduce results

### Additional installations and downloads
In order to run the code, you first need to make sure you have the below
mentioned software installed as well as have downloaded the required data.

#### Java V8
Check if you have `Java v8` installed:

```bash
java -version
```

If not, then you can install it according to this [tutorial](https://docs.oracle.com/javase/8/docs/technotes/guides/install/mac_jdk.html). Note that you can use multiple versions of Java on the same system. You can then set the default one in your terminal language setting file (e.g. `.bashrc`).

#### Apache Flink
To install Apache Flink, please follow the steps in download
[section](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/try-flink/local_installation/#downloading-flink)
in the official solution. (ignore the java section part) Alternatively, on macOS, assuming you have `brew` installed, you can just run:

```bash
brew install apache-flink
```

#### Python >3.7
Again, check if you have python installed with at least version 3.7:

```bash
python --version
```

If not, you can follow the official [docs](https://www.python.org/downloads/) to install it.

#### Downloading data
Assuming you are at the root of repo, first, we need to download yelp review data and
install python dependencies:

```bash
source download.sh
```

### Getting results
Before execution of the scripts, please run `redis` in a separate terminal window:

```bash
redis-server
```


----

















You might need to install it seperately with
```bash
brew install redis
```

#### Get text with predictions
Finally, you can run the code and get reviews with corresponding label:

```bash
cd src
plreview
```
If you get an error with `nltk` run the python interpreter and run:
```
import nltk
nltk.download('stopwords')
```

The output is stored in the folder called `output` present within the `src`
directory. In the output folder, you can find sub-directories which are named in
the form of `YY-mm-hh`. If you then enter corresponding sub-folder you can check
its content by writing:

```
ls -a
```

As you can see, this subfolder includes several files. If you choose the one
created most recently you can check the output by for example using `head`.
As mentioned in the original docs:
>  The outputs' form is "original text" + "label" + "@@@@". With help of a split("@@@@") function we can further reorganize the labelled dataset.

So for example, you can run the following command to get the result in a nice
form to the file called `result.out` (make sure you replace the variable in
square brackets):

```bash
cat [name_of_the_raw_file] | tr "@@@@" "\n" > result.out
```

#### Get accuracy
You can check how accuracy of the model evolves as you input more data by
running the following:

```bash
cd src
placc
```

To check the output, follow the similar steps as in the previous section.

## Todo

Here are the possible improvements:

- [ ] Add better logging including time it takes to execute particular parts
- [ ] Figure out how to better save the results - better naming of the files
- [ ] Figure out how to use all available cores 

## Bug report
If you have encountered any problem, please report an issue.
