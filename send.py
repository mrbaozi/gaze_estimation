#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pika
import sys, logging
from modules.util import except_stackoverflow


def main():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('AMQP Receiver')
    logger.setLevel(20)

    credentials = pika.PlainCredentials('augeny', '18413039')
    parameters = pika.ConnectionParameters('energietun.de', 5672, '/', credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue='hello')

    logger.info('Connection open - start typing messages:')

    while True:
        msg = input()
        channel.basic_publish(exchange='', routing_key='hello', body=msg)
        logger.info("Sent message: {0!s}".format(msg))
        if msg == 'quit':
            logger.info('Quitting...')
            break

    connection.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        except_stackoverflow()
