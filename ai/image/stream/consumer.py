import logging

from kafka import KafkaConsumer

from caption.stream.producer import produce_caption_classification_message
from image.stream.classify import classify_using_latest


def consume_message():
    consumer = KafkaConsumer('image_classification_request', bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest'  # start from the earliest message
                             )
    for message in consumer:
        classification = classify_using_latest(message.value)
        logging.info(f"We classified this message as {classification}")
        produce_caption_classification_message(classification)
