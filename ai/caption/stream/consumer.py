import logging

from kafka import KafkaConsumer

from caption.stream.classify import classify_caption_using_latest_model
from caption.stream.producer import produce_caption_classification_message


def consume_message():
    consumer = KafkaConsumer('caption_classification_request', bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest'  # start from the earliest message
                             )
    for message in consumer:
        classification = classify_caption_using_latest_model(message.value)
        logging.info(f"We classified this message as {classification}")
        produce_caption_classification_message(classification)
