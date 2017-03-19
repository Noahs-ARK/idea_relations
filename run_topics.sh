#!/bin/bash

./mallet.sh /home/chenhao/perspectives/data/acl_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/immigration_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/tobacco_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/abortion_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/terrorism_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/same_sex_marriage_processed 50 &
./mallet.sh /home/chenhao/perspectives/data/nips_processed 50 &
