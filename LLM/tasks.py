import re
import pandas as pd
import numpy as np
from Levenshtein import ratio
from dataclasses import dataclass, fields


def get_addition_question(rng: np.random.Generator, size=10):
    num1_arr = rng.integers(1_000_000, 10_000_000, size=size)
    num2_arr = rng.integers(1_000_000, 10_000_000, size=size)
    answers = num1_arr + num2_arr
    return num1_arr, num2_arr, answers

def get_subtraction_question(rng: np.random.Generator, size=10):
    num1_arr = rng.integers(1_000_000, 10_000_000, size=size)
    num2_arr = rng.integers(1_000_000, 10_000_000, size=size)
    answers = num1_arr - num2_arr
    return num1_arr, num2_arr, answers

def get_multiplication_question(rng: np.random.Generator, size=10):
    num1_arr = rng.integers(1_000, 10_000, size=size)
    num2_arr = rng.integers(1_000, 10_000, size=size)
    answers = num1_arr * num2_arr
    return num1_arr, num2_arr, answers


def get_arithmetic_messages(num1_arr, num2_arr, answers, operation, num_training_examples):
    messages = []
    for num1, num2 in zip(num1_arr, num2_arr):
        message = [
            {
                "role": "user",
                "content": "You are a calculator that gives answer in numbers only"
            },
            {
                "role": "assistant",
                "content": "Okay"
            },
            {
                "role": "user",
                "content": f"{num1} {operation} {num2}"
            },
        ]
        messages.append(message)

    messages_train, messages_test = messages[:num_training_examples], messages[num_training_examples:]
    answers_train, answers_test = answers[:num_training_examples], answers[num_training_examples:]

    return messages_train, answers_train, messages_test, answers_test


def extract_arithmetic_answer(output: str):
    number_pattern = re.compile(r'\[\/INST\][^\[]*?(-?\d[\d,\.]*)')
    number = number_pattern.findall(output)[-1].replace(',', '')
    if '.' in number:
        number = str(int(float(number)))
    return number

def get_arithmetic_accuracy(output: str, real_answer: int):
    try:
        return ratio(extract_arithmetic_answer(output), str(real_answer))
    except:
        return 0
    

languages_and_messages = {
    "eng": ["You will answer my next question in only one word without explanation.", "okay", "What is the language of the following sentence: "],
    "deu": ["Sie werden meine nächste Frage ohne Erklärung in nur einem Wort beantworten.", "okay", "Was ist die Sprache des folgenden Satzes: "],
    "fra": ["Vous répondrez à ma prochaine question en un seul mot sans explication.", "d'accord", "Quelle est la langue de la phrase suivante: "],
    "spa": ["Responderás mi siguiente pregunta en una sola palabra sin explicación.", "bueno", "¿Cuál es el idioma de la siguiente oración: "],
    "ita": ["Risponderai alla mia prossima domanda in una sola parola senza spiegazioni.", "Va bene", "Qual è il linguaggio della seguente frase: "]
}

def get_language_sentence_message(lang: str, languaged_sentences_data: pd.DataFrame, num_sentences=10):
    # controlled random sampling
    sample_sentences = languaged_sentences_data[languaged_sentences_data.Lang == lang].sample(num_sentences).Text
    messages = [[
        {
            "role": "user",
            "content": languages_and_messages[lang][0], 
        },
        {
            "role": "assistant",
            "content": languages_and_messages[lang][1],
        },
        {
            "role": "user",
            "content": languages_and_messages[lang][2] + sample_sentence + "?",
        }
    ] for sample_sentence in sample_sentences]
    return messages

def extract_language_from_output(output: str):
    output = output.split("[/INST]")[-1].strip().lower().split(" ")[0]
    return output

def get_countries_capitals_messages(countries_capitals_data):
    country_capital_messages = [
        [{
            "role": "user",
            "content": "Answer the following question without explanation. What is the capital of Spain? ",
        },
        {
            "role": "assistant",
            "content": "Madrid",
        },
        {
            "role": "user",
            "content": "What is the capital of " + country + "? ",
        }]
        for country in countries_capitals_data["country"].values
    ]
    return country_capital_messages

def extract_capital_from_output(output: str):
    return output.split("[/INST]")[-1].split("\n")[0].strip().lower()

@dataclass
class LesionedModelPerformance:
    addition_accuracy: float
    subtraction_accuracy: float
    multiplication_accuracy: float
    english_accuracy: float
    german_accuracy: float
    french_accuracy: float
    spanish_accuracy: float
    italian_accuracy: float
    country_capital_accuracy: float

    def __repr__(self):
        return (
            f"LesionedModelPerformance(\n"
            f"  addition_accuracy={self.addition_accuracy:.4f},\n"
            f"  subtraction_accuracy={self.subtraction_accuracy:.4f},\n"
            f"  multiplication_accuracy={self.multiplication_accuracy:.4f},\n"
            f"  english_accuracy={self.english_accuracy:.4f},\n"
            f"  german_accuracy={self.german_accuracy:.4f},\n"
            f"  french_accuracy={self.french_accuracy:.4f},\n"
            f"  spanish_accuracy={self.spanish_accuracy:.4f},\n"
            f"  italian_accuracy={self.italian_accuracy:.4f},\n"
            f"  country_capital_accuracy={self.country_capital_accuracy:.4f}\n"
            f")"
        )
    
    def __operate(self, other, op):
        if isinstance(other, LesionedModelPerformance):
            return LesionedModelPerformance(**{
                f.name: op(getattr(self, f.name), getattr(other, f.name))
                for f in fields(self)
            })
        elif isinstance(other, (int, float)):
            return LesionedModelPerformance(**{
                f.name: op(getattr(self, f.name), other)
                for f in fields(self)
            })
        return NotImplemented

    def __add__(self, other):
        return self.__operate(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.__operate(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.__operate(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.__operate(other, lambda x, y: x / y)

