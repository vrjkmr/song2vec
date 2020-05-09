# -*- coding: utf-8 -*-
import os
import errno

def create_dir_if_not_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise