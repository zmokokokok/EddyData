#!/usr/bin/env bash
hdfs dfs -get $PAI_DATA_DIR
tar -I pigz -xf eddy_data.tar
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/eddydlv3net/
mkdir eddydlv3net
python3 EddyDLv3plus.py
tar -cf ./eddydlv3net.tar ./eddydlv3net
hdfs dfs -put -f eddydlv3net.tar $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/eddydlv3net/