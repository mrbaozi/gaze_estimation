#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging
import configparser
import pika


def callback(ch, method, properties, body):
    pass


logging.basicConfig()

config = configparser.ConfigParser()
config.read('./config.ini')

url = config.get('rabbitmq', 'server')
queue = config.get('rabbitmq', 'queue')
no_ack = config.getboolean('rabbitmq', 'no_ack')

connection = pika.BlockingConnection(pika.ConnectionParameters(host=url))
channel = connection.channel()
channel.basic_consume(callback, queue=queue, no_ack=no_ack)
