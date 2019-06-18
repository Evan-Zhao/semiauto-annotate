#!/bin/bash
rm ~/.labelmerc
python3 setup.py install --user 2>&1 1>/dev/null
labelme
