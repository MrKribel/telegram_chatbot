from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import random
import nltk
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier



import warnings
warnings.filterwarnings('ignore')



class Bot:
    def __init__(self):
        self.read_json = json.load(open('data/bot_config.json'))

class Datasets(Bot):
    def __init__(self):
        super().__init__()

    def get_datasets(self):
        self.df1 = pd.DataFrame(self.read_json["intents"])
        self.df2 = pd.DataFrame(self.read_json["failure_phrases"], columns=['failure_phrases'])

        self.dataset = []
        [self.dataset.append([example, ident]) for ident in list(self.df1.columns.values) for example in self.df1[ident].iloc[0]]

        self.X = [example for example, intent in self.dataset]
        self.y_class = [intent for example, intent in self.dataset]

        return self.X, self.y_class, self.df1, self.df2

    def datasets_to_vecor(self, example):

        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))#используется для оценки важности слова
        self.X_vector = self.vectorizer.fit_transform(example)# преобразуем example в ветокра

        return self.X_vector, self.vectorizer

    def machine_learning(self, X_vector, y_class):

        self.clf_proba = KNeighborsClassifier(n_jobs=-1, n_neighbors=4, weights='distance').fit(X_vector, y_class)
        self.clf = LinearSVC().fit(X_vector, y_class)

        return self.clf_proba, self.clf

    def clear_question(self, question):
        question = question.lower().strip()
        alphabet = ' -1234567890йцукенгшщзхъфывапролджэёячсмитьбю'
        question = ''.join(c for c in question if c in alphabet)
        return question

    def dataset_to_generative(self):
        with open('data/dialogues.txt') as f:
            content = f.read()
        f.close()

        dialogues = content.split('\n\n')

        # формируем датасет
        questions = set()
        dataset = {}  # ключ - слово, знаечние - те фразы которые встречаются с этим словом

        for i in dialogues:
            replicas = i.split('\n')[:2]
            if len(replicas) == 2:
                question, answer = replicas
                question, answer = self.clear_question(question[2:]), answer[2:]  # убираем пробел и дефис

                if question and question not in questions:
                    questions.add(question)
                    words = question.split(' ')  # разбиваем слова по пробелу
                    for word in words:
                        if word not in dataset:
                            dataset[word] = []
                        dataset[word].append([question, answer])

        to_popular = set()
        dataset_short = dataset

        for word in dataset:
            if len(dataset[word]) > 40000:
                to_popular.add(word)

        for word in to_popular:
            dataset_short.pop(word)

        return dataset, dataset_short



class Input_processing(Datasets):
    def __init__(self, clf_proba, clf, vectorizer):
        super().__init__()


    # классификация намереней пользователя (сущность intent)
    def get_intent(self,question):

        best_intent = clf.predict(vectorizer.transform([question]))[0]
        index = list(clf_proba.classes_).index(best_intent)

        prediction = clf_proba.predict_proba(vectorizer.transform([question]))[0]

        best_intent_proba = prediction[index]
        print(best_intent_proba)
        if best_intent_proba > 0.4:
            return best_intent
        else:
            return None


    def get_answer_by_intent(self, intent, df1, df2 ):
        #извлечение дополнительных сущностей
        try:
            return random.choice(df1[intent]['responses'])
        except:
            Output_processing().get_failure_phrase(df2)


class Output_processing(Datasets):
    def __init__(self, df2):
        super().__init__()

    def get_generative_answer(self, question_user):
        #умная автоматическая генеарция реплики по контексту (генеративная модель)

        question_user = Datasets().clear_question(question_user)
        words = question_user.split(' ')

        mini_dataset = []
        candidates = []

        for word in words:
            if word in dataset:
                mini_dataset += dataset_short[word]

        for question, answer in mini_dataset:
            if abs(len(question) - len(question_user)) / len(question) < 0.4:
                d = nltk.edit_distance(question, question_user)
                diff = d / len(question)
                if diff < 0.4:
                    candidates.append([question, answer, diff])

                    winner = min(candidates, key=lambda candidate: candidate[2])
                    return winner[1]
        else:
            return Output_processing(df2).get_failure_phrase()


    def get_failure_phrase(self):
        # не смог распознать
        return random.choice(list(df2['failure_phrases']))


class Run(Datasets):
    def __init__(self, clf_proba, clf, dataset, dataset_short, vectorizer, examples, intent, df1, df2):
        super().__init__()

    def run_bot(self, question_user):
        try:
            #заготовленный ответ
            intent_response = Input_processing(clf_proba, clf, vectorizer).get_intent(question_user)
            if intent_response:
                print('заготовленный ответ')
                print(intent_response)
                return Input_processing(examples, intent, vectorizer).get_answer_by_intent(intent_response, df1, df2)


            #применяем генеративную модель
            if intent_response == None:
                answer = Output_processing(df2).get_generative_answer(question_user)
                if answer:
                    print('применяем генеративную модель')
                    return answer

                else:
                    print('не знаю')
                    return Output_processing(df2).get_failure_phrase()


        except KeyboardInterrupt:
            print('\n')
            print(random.choice(df1['bye']['responses']))


    def start(self, update, context):

        update.message.reply_text('Hi!')

    def help_command(self,update, context):

        update.message.reply_text('Help!')

    def echo(self, update, context):

        question_user = update.message.text
        answer = self.run_bot(question_user)

        update.message.reply_text(answer)


    def telegram(self):

        updater = Updater("1121431523:AAFOKdvErEL9LAT2u7520alovq1mwZZFA20", use_context=True)

        dp = updater.dispatcher

        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(CommandHandler("help", self.help_command))
        dp.add_handler(MessageHandler(Filters.text & ~Filters.command, self.echo))

        # Start the Bot
        updater.start_polling()

        updater.idle()




if __name__ == '__main__':

    #формируем датасеты
    examples, intent, df1, df2 = Datasets().get_datasets()

    #векторизуем датасеты и разделим на тестовы и трейновый
    X_vector, vectorizer = Datasets().datasets_to_vecor(examples)

    dataset, dataset_short = Datasets().dataset_to_generative()

    clf_proba, clf = Datasets().machine_learning(X_vector, intent)

    Run(clf_proba, clf, dataset, dataset_short, vectorizer, examples, intent, df1, df2).telegram()





