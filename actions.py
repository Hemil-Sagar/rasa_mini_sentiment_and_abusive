# # This files contains your custom actions which can be used to run
# # custom Python code.
# #
# # See this guide on how to implement these action:
# # https://rasa.com/docs/rasa/custom-actions


# # This is a simple example for a custom action which utters "Hello World!"

# # from typing import Any, Text, Dict, List
# #
# # from rasa_sdk import Action, Tracker
# # from rasa_sdk.executor import CollectingDispatcher
# #
# #
# # class ActionHelloWorld(Action):
# #
# #     def name(self) -> Text:
# #         return "action_hello_world"
# #
# #     def run(self, dispatcher: CollectingDispatcher,
# #             tracker: Tracker,
# #             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
# #
# #         dispatcher.utter_message(text="Hello World!")
# #
# #         return []
# from typing import Any, Text, Dict, List
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from transformers import pipeline

# class ActionCheckSentimentAndAbuse(Action):
#     def name(self) -> Text:
#         return "action_check_sentiment_and_abuse"

#     def __init__(self):
#         # Initialize models
#         self.sentiment_analyzer = pipeline(
#             "sentiment-analysis",
#             model="distilbert-base-uncased-finetuned-sst-2-english"
#         )
#         self.abuse_detector = pipeline(
#             "text-classification",
#             model="Hate-speech-CNERG/dehatebert-mono-english",
#             top_k=None
#         )

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         user_message = tracker.latest_message.get("text")

#         # Check for abusive content first
#         abuse_results = self.abuse_detector(user_message)[0]
#         for result in abuse_results:
#             if result['label'].lower() in ['hate', 'abusive', 'offensive'] and result['score'] > 0.5:
#                 dispatcher.utter_message(text="⚠️ Please keep the conversation respectful.")
#                 return []

#         # Analyze sentiment if no abuse detected
#         sentiment = self.sentiment_analyzer(user_message)[0]
        
#         if sentiment['label'] == 'POSITIVE':
#             dispatcher.utter_message(text="😊 That's wonderful to hear!")
#         elif sentiment['label'] == 'NEGATIVE':
#             dispatcher.utter_message(text="😔 I'm sorry you're feeling this way. How can I help?")
#         else:
#             dispatcher.utter_message(text="Thank you for sharing your feelings.")

#         return []
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline
import re

class ActionCheckSentimentAndAbuse(Action):
    def name(self) -> Text:
        return "action_check_sentiment_and_abuse"

    def __init__(self):
        # Layer 1: Keyword-based detection
        self.abusive_keywords = [
            'stupid', 'idiot', 'hate', 'hell', 'fuck', 'shit', 
            'shut up', 'dumb', 'worthless', 'bastard', 'retard'
        ]
        
        # Layer 2: FastText model as fallback
        try:
            self.abuse_detector = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-hate-latest"
            )
        except:
            self.abuse_detector = None

        # Layer 3: Sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def contains_abuse(self, text: str) -> bool:
        # Layer 1: Check against known abusive words
        text_lower = text.lower()
        if any(re.search(rf'\b{kw}\b', text_lower) for kw in self.abusive_keywords):
            return True
            
        # Layer 2: Model-based detection if available
        if self.abuse_detector:
            try:
                result = self.abuse_detector(text)[0]
                return result['label'] in ['hate', 'abusive'] and result['score'] > 0.7
            except:
                pass
                
        return False

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        # Abuse detection
        if self.contains_abuse(user_message):
            dispatcher.utter_message(text="⚠️ Please keep conversations respectful.")
            return []

        # Sentiment analysis
        try:
            sentiment = self.sentiment_analyzer(user_message)[0]
            if sentiment['label'] == 'POSITIVE':
                dispatcher.utter_message(text="😊 That's wonderful!")
            else:
                dispatcher.utter_message(text="😔 I'm sorry. How can I help?")
        except:
            dispatcher.utter_message(text="Thanks for sharing your thoughts.")

        return []ri