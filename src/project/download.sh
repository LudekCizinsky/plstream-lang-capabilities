#/!bin/bash

echo '-- Downloading data';
cd src;
wget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz;
tar zxvf yelp_review_polarity_csv.tgz;
mkdir data
mv yelp_review_polarity_csv/train.csv data/plstreamtrain.csv;
rm yelp_review_polarity_csv.tgz;
rm -r yelp_review_polarity_csv;
cd ..;
echo '> Done!\n';

