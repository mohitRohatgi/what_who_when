#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:47:57 2017

@author: m0r00ds
"""

import numpy as np
from trainer import Trainer
from sklearn.decomposition import PCA



def main():
    train_filename = "LabelledData.txt"
    label_idx = {
                 'what': 0,
                 'when': 1,
                 'who' : 2,
                 'affirmation': 3,
                 'unknown': 4
                 }
    idx_to_label = {
                    0: 'what',
                    1: 'when',
                    2: 'who',
                    3: 'affirmation',
                    4: 'unknown'
                    }
    labels, questions = extract_labels_and_questions(train_filename)
    unique_words, index_to_word = extract_unique(questions)
    co_occurence_matrix = extract_coocurrence_matrix(unique_words, questions)
    
    # separating datasets
    rare,what,who, when,affirmation,unknown = separate_cases(questions, labels)
    datasets_idx = get_datasets_idx(rare,what,who, when,affirmation,unknown, 10)
    datasets, labels = get_datasets(co_occurence_matrix, datasets_idx,
                                    labels, label_idx)
    dataset = datasets[0]
    label = labels[0]
    for i in range(1, len(datasets)):
        dataset = np.concatenate((dataset, datasets[i]), axis=0)
        label = np.concatenate((label, labels[i]), axis=0)
    
    # finding test datasets
    test_filename = "test.txt"
    test_questions = extract_questions(test_filename)
    test_dataset = get_test_datasets(test_questions, unique_words)
    trainer = Trainer(dataset, label)
    predicted, scores = trainer.predict(test_dataset)
    print (predicted.shape)
    for i, question in enumerate(test_questions):
        print (question, " ---> ", idx_to_label[int(predicted[i])])
    print (scores)
    outputFile = open('output.txt', 'wt', encoding='latin1')
    for i, question in enumerate(test_questions):
        output = question.rstrip() +  " Type: " + idx_to_label[int(predicted[i])] + "\n"
        outputFile.write(output)
        outputFile.flush()
    outputFile.close()

def get_test_datasets(test_questions, word_to_idx):
    data = np.zeros((len(test_questions), len(word_to_idx)))
#    pca = PCA(n_components=1000)
    for row, question in enumerate(test_questions):
        word_array = question.split(" ")
        ques_length = len(word_array)
        for word in word_array:
            word = word.lower().strip()
            if word in word_to_idx:
                column = word_to_idx[word]
                data[row][column] += 1.0 / ques_length
#    data = pca.fit_transform(data)
    return data

def get_datasets(co_occurence_matrix, datasets_idx, labels, label_idx):
    co_occurence_matrix = np.array(co_occurence_matrix)
    datasets = []
    labels_list = []
#    pca = PCA(n_components=1000)
    for dataset_idx in datasets_idx:
        dataset = co_occurence_matrix[dataset_idx]
#        dataset = pca.fit_transform(dataset)
        datasets.append(dataset)
        label = np.zeros(len(dataset_idx))
        for row, idx in enumerate(dataset_idx):
            word = labels[idx].lower().strip()
            label[row] = label_idx[word]
        labels_list.append(label)
    return datasets, labels_list
    
def get_datasets_idx(rare, what, who, when, affirmation, unknown, n_datasets):
    datasets = []
    cons = rare + when + affirmation
    len_what = int(len(what) / n_datasets)
    len_who = int(len(who) / n_datasets)
    len_unknown = int(len(unknown) / n_datasets)
    for i in range(n_datasets):
        what_indices = what[i * len_what : (i + 1) * len_what]
        who_indices = who[i * len_who : (i + 1) * len_who]
        unknown_indices = unknown[i * len_unknown : (i + 1) * len_unknown]
        dataset = what_indices + who_indices + unknown_indices + cons
        datasets.append(dataset)
    return datasets
        

def separate_cases(questions, labels):
    rare_indices = []
    what_indices = []
    who_indices = []
    when_indices = []
    affirmation_indices = []
    unknown_indices = []
    word_array = []
    for question in questions:
        word_array.append(question.split(" "))
    for index, row in enumerate(word_array):
        first_word = row[0].lower().strip()
        if first_word != labels[index].lower().strip() and first_word in labels:
            rare_indices.append(index)
        elif labels[index] == 'what':
            what_indices.append(index)
        elif labels[index] == 'who':
            who_indices.append(index)
        elif labels[index] == 'when':
            when_indices.append(index)
        elif labels[index] == 'affirmation':
            affirmation_indices.append(index)
        elif labels[index] == 'unknown':
            unknown_indices.append(index)
            
    return [rare_indices, what_indices, who_indices, when_indices,
            affirmation_indices, unknown_indices]
    
def extract_questions(filename):
    currentFile = open(filename, 'rt', encoding='latin1')
    questions = []
    for line in currentFile.readlines():
        questions.append(line)
    currentFile.close()
    return questions
    
def extract_labels_and_questions(filename):
    currentFile = open(filename, 'rt', encoding='latin1')
    labels = []
    questions = []
    for line in currentFile:
        data = line.split(" ,,, ")
        labels.append(data[1].lower().strip())
        questions.append(data[0])
    currentFile.close()
    return labels, questions

def extract_unique(questions):
    word_to_index = {}
    index_to_word = {}
    current_index = 0
    for question in questions:
        words = question.split(" ")
        for word in words:
            word = word.lower().strip()
            if not word in word_to_index:
                word_to_index[word] = current_index
                index_to_word[current_index] = word
                current_index += 1
    return word_to_index, index_to_word

def extract_coocurrence_matrix(unique_words, questions):
    data = np.zeros((len(questions), len(unique_words)))
    for row, question in enumerate(questions):
        ques_array = question.split(" ")
        ques_length = len(ques_array)
        for word in ques_array:
            word = word.lower().strip()
            column = unique_words[word]
            data[row][column] += 1.0 / ques_length
    return data


if __name__ == '__main__':
    main()