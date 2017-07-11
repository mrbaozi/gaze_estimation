#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, logging


def except_stackoverflow():
    d = ': ' if str(sys.exc_info()[1]).strip() else ''
    msg = '{}{}{}'.format(sys.exc_info()[0].__name__, d, sys.exc_info()[1])
    print(msg)
    msg = msg.replace(' ', '%20').strip()
    print('https://stackoverflow.com/search?q=[python]+{}'.format(msg))
    sys.exit(1)
