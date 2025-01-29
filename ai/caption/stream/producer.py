from kafka import KafkaProducer


def produce_caption_classification_message(classifcation: str) -> None:
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    producer.send('caption_classification_response', classifcation)
    producer.flush()
    producer.close()
