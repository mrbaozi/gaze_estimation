#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pika
import logging
import json


class PyOSBReceiver():

    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])
        self._logger = logging.getLogger(self.__class__.__name__)
        self._connection = None
        self._channel = None
        self._consumer_tag = None
        self._closing = False

    def connect(self):
        self._logger.info('Connecting to {}:{} as user {}'.format(self.host, self.port, self.user))
        credentials = pika.PlainCredentials(self.user, self.password)
        params = pika.ConnectionParameters(self.host, int(self.port), '/', credentials)
        return pika.SelectConnection(params, self.on_connection_open, stop_ioloop_on_close=False)

    def on_connection_open(self, unused_connection):
        self._logger.info('Connection opened')
        self.add_on_connection_close_callback()
        self.open_channel()

    def add_on_connection_close_callback(self):
        self._logger.info('Adding connection close callback')
        self._connection.add_on_close_callback(self.on_connection_closed)

    def on_connection_closed(self, connection, reply_code, reply_text):
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            self._logger.warning('Connection closed, reopening in 5 seconds: ({}) {}'.format(reply_code, reply_text))
            self._connection.add_timeout(5, self.reconnect)

    def reconnect(self):
        self._connection.ioloop.stop()
        if not self._closing:
            self._connection = self.connect()
            self._connection.ioloop.start()

    def open_channel(self):
        self._logger.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        self._logger.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.exchange)

    def add_on_channel_close_callback(self):
        self._logger.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reply_code, reply_text):
        self._logger.warning('Channel {} was closed: ({}) ()'.format(channel, reply_code, reply_text))
        self._connection.close()

    def setup_exchange(self, exchange_name):
        self._logger.info('Declaring exchange {}'.format(exchange_name))
        self._channel.exchange_declare(self.on_exchange_declareok, exchange_name, self.exchange_type, auto_delete=True)

    def on_exchange_declareok(self, unused_frame):
        self._logger.info('Exchange declared')
        self.setup_queue(self.queue)

    def setup_queue(self, queue_name):
        self._logger.info('Declaring queue {}'.format(queue_name))
        self._channel.queue_declare(self.on_queue_declareok, queue_name)

    def on_queue_declareok(self, method_frame):
        self._logger.info('Binding {} to {} with {}'.format(self.exchange, self.queue, self.routing_key))
        self._channel.queue_bind(self.on_bindok, self.queue,self.exchange, self.routing_key)

    def on_bindok(self, unused_frame):
        self._logger.info('Queue bond')
        self.start_consuming()

    def start_consuming(self):
        self._logger.info('Issuing consumer related RPC commands')
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(self.on_message, self.queue)

    def add_on_cancel_callback(self):
        self._logger.info('Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        self._logger.info('Consumer was cancelled remotely, shutting down: {}'.format(method_frame))
        if self._channel:
            self._channel.close()

    def on_message(self, unused_channel, basic_deliver, properties, body):
        msg = json.dumps(json.loads(body.decode('utf8')), indent=2, sort_keys=True)
        self._logger.info('Received message # {} from {}: {}'.format(basic_deliver.delivery_tag, properties.app_id, msg))
        self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        self._logger.info('Acknowledging message {}'.format(delivery_tag))
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        if self._channel:
            self._logger.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            self._channel.basic_cancel(self.on_cancelok, self._consumer_tag)

    def on_cancelok(self, unused_frame):
        self._logger.info('RabbitMQ acknowledged the cancellation of the consumer')
        self.close_channel()

    def close_channel(self):
        self._logger.info('Closing the channel')
        self._channel.close()

    def run(self):
        self._connection = self.connect()
        self._connection.ioloop.start()

    def stop(self):
        self._logger.info('Stopping')
        self._closing = True
        self.stop_consuming()
        self._connection.ioloop.start()
        self._logger.info('Stopped')

    def close_connection(self):
        self._logger.info('Closing connection')
        self._connection.close()

