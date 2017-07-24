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

    logger.info('Waiting for messages. To exit press CTRL+C')

    def callback(ch, method, properties, body):
        logger.info('Received message: {0!s}'.format(body.decode('utf8')))
        if body.decode('utf8') == 'quit':
            logger.info('Quitting...')
            channel.stop_consuming()
            connection.close()
            sys.exit(0)

    channel.basic_consume(callback, queue='hello', no_ack=True)
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        except_stackoverflow()
