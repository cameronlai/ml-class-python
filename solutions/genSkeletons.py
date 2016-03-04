#!/usr/bin/python
# This scripts generates the skeleton from the solution files

import os

pattern1 = '# ============= YOUR CODE HERE ============='
pattern2 = '# ==========================================='

for d in os.listdir('.'):
    if os.path.isdir(d):
        for f in os.listdir(d):
            print(f)
            if '.pyc' in f:
                print('Skip pyc file')
                continue
            src = d + '/' + f
            dst = '../skeletons/' + src
            with open(src, 'r') as f:
                read_data = f.read()
            read_data = read_data.splitlines()
            with open(dst, 'w') as f:
                code_flag = False
                for line in read_data:
                    if pattern1 in line:
                        code_flag = True
                    elif pattern2 in line:
                        code_flag = False

                    if (code_flag and '#' in line) or not code_flag:
                        f.write(line + '\n')
