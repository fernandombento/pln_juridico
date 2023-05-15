import re
import nltk
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords as sw
nltk.download("stopwords")
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from csv import writer
import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from legalnlp.clean_functions import *
from legalnlp.get_premodel import *
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import textwrap
from tqdm import tqdm
import numpy as np

import pickle
import copy 

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import BertForPreTraining, BertModel, BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction, BertForQuestionAnswering

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda:0')

stopwords = sw.words("portuguese")
stemmer = PorterStemmer()

os.chdir("C:\\Users\\ferna\\OneDrive\\Documents\\Recuperação HD\\Arquivos existentes\\Não particionado\\home\\fernando\\projetos\\MCA\\dissertacao\\artigo")

def tiraAcentos(texto):
    texto.replace('á', 'a')
    texto.replace('à', 'a')
    texto = texto.replace('ã', 'a')
    texto = texto.replace('é', 'e')
    texto = texto.replace('ẽ', 'e')
    texto = texto.replace('í', 'i')
    texto = texto.replace('ó', 'o')
    texto = texto.replace('ú', 'u')
    return texto

def bootstrap(X_test, y_test, model, nn=False, B=250):
    #Criando dicinário para armazenar os resultados
    out={}
    out['accuracy']=[]
    out['macro avg']={}
    out['macro avg']['f1-score']=[]
    out['macro avg']['recall']=[]
    out['macro avg']['precision']=[]
    out['weighted avg']={}
    out['weighted avg']['f1-score']=[]
    out['weighted avg']['recall']=[]
    out['weighted avg']['precision']=[]

    #Aplicando Bootstrap no conjunto de teste
    for b in tqdm(range(B)):
        ind = np.random.choice(range(y_test.shape[0]),y_test.shape[0])
        X_test_boot, y_test_boot = X_test[ind,:], y_test[ind]

        y_pred=model.predict(X_test_boot)
        
        if nn:
            y_pred=np.argmax(y_pred,axis=1)
            report=classification_report(y_test_boot, y_pred, labels=[0, 1, 2], output_dict=True)
        else:
            report=classification_report(y_test_boot, y_pred, labels=[0, 1, 2], output_dict=True)

        out['accuracy'].append(report['accuracy'])
        out['macro avg']['f1-score'].append(report['macro avg']['f1-score'])
        out['macro avg']['recall'].append(report['macro avg']['recall'])
        out['macro avg']['precision'].append(report['macro avg']['precision'])
        out['weighted avg']['f1-score'].append(report['weighted avg']['f1-score'])
        out['weighted avg']['recall'].append(report['weighted avg']['recall'])
        out['weighted avg']['precision'].append(report['weighted avg']['precision'])

    #Preparando a saída
    y_pred=model.predict(X_test)
    
    if nn:
        y_pred=np.argmax(y_pred,axis=1)
        report=classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)
    else:
        report=classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)

    out['accuracy'] = [report['accuracy'], np.std(out['accuracy'])]
    out['macro avg']['f1-score'] = [report['macro avg']['f1-score'], np.std(out['macro avg']['f1-score'])] 
    out['macro avg']['recall'] = [report['macro avg']['recall'], np.std(out['macro avg']['recall'])] 
    out['macro avg']['precision'] = [report['macro avg']['precision'], np.std(out['macro avg']['precision'])] 
    out['weighted avg']['f1-score'] = [report['weighted avg']['f1-score'], np.std(out['weighted avg']['f1-score'])] 
    out['weighted avg']['recall'] = [report['weighted avg']['recall'], np.std(out['weighted avg']['recall'])] 
    out['weighted avg']['precision'] = [report['weighted avg']['precision'], np.std(out['weighted avg']['precision'])]
    
    return out

#transofrmando uma matriz confusao 3x3 em list
def get_list_MT(mc):
    list_MC = [ mc[0][0],
                mc[0][1],
                mc[0][2],
                mc[1][0],
                mc[1][1],
                mc[1][2],
                mc[2][0],
                mc[2][1],
                mc[2][2] ]
    return list_MC

def buscaPalavras(texto):
    todasPalavras = []
    for frase in texto:
        todasPalavras.extend(frase)
    return todasPalavras

def buscaFrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

def buscaPalavrasUnicas(frequencia):
    freq = frequencia.keys()
    return freq

def extratorPalavras(textos):
    doc = set(textos)
    caracteristicas = {}
    for palavras in palavrasUnicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def calculaIndices(pMC):
    indices = {}
    
    #DI = pMC[0,0]+pMC[0,1]+pMC[0,2]
    DI = pMC[0]+pMC[1]+pMC[2]
    #DE = pMC[1,0]+pMC[1,1]+pMC[1,2]
    DE = pMC[3]+pMC[4]+pMC[5]
    #SE = pMC[2,0]+pMC[2,1]+pMC[2,2]
    SE = pMC[6]+pMC[7]+pMC[8]

    total = DI+DE+SE

    VP_DI = pMC[0]
    VP_DE = pMC[4]
    VP_SE = pMC[8]

    FP_DI = pMC[3]+pMC[6]
    FP_DE = pMC[1]+pMC[7]
    FP_SE = pMC[2]+pMC[5]

    FN_DI = pMC[1]+pMC[2]
    FN_DE = pMC[3]+pMC[5]
    FN_SE = pMC[6]+pMC[7]

    precisao_DI = VP_DI/(VP_DI+FP_DI)
    precisao_DE = VP_DE/(VP_DE+FP_DE)
    precisao_SE = VP_SE/(VP_SE+FP_SE)

    recall_DI = VP_DI/(VP_DI+FN_DI)
    recall_DE = VP_DE/(VP_DE+FN_DE)
    recall_SE = VP_SE/(VP_SE+FN_SE)

    f_score_DI = (2*recall_DI*precisao_DI)/(recall_DI+precisao_DI)
    f_score_DE = (2*recall_DE*precisao_DE)/(recall_DE+precisao_DE)
    f_score_SE = (2*recall_SE*precisao_SE)/(recall_SE+precisao_SE)

    precisao = (precisao_DI * DI / total)+(precisao_DE * DE / total)+(precisao_SE * SE / total)
    recall = (recall_DI * DI / total)+(recall_DE * DE / total)+(recall_SE * SE / total)
    f_score = (f_score_DI * DI / total)+(f_score_DE * DE / total)+(f_score_SE * SE / total)

    indices['precisao'] = precisao
    indices['recall'] = recall
    indices['f_score'] = f_score
    
    return indices


def grava_MC(nome, mc):
    with open(nome+'.csv', 'a', newline='') as csv:  
        writer_object = writer(csv)    
        writer_object.writerow(mc)  
        csv.close()

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()


###################################################################################
#Baixando a base principal
path = "C:\\Users\\ferna\\OneDrive\\Documents\\Recuperação HD\\Arquivos existentes\\Não particionado\\home\\fernando\\projetos\\MCA\\dissertacao\\artigo\\openai\\"
base_original = pd.read_excel(path+'publicacoes_508_GPT.xlsx').sample(frac=1)

#Limpando a base_total
i = 1
while i < len(base_original):    
    base_original['publicacao'][i] = base_original['publicacao'][i].lower()
    base_original['publicacao'][i] = tiraAcentos(base_original['publicacao'][i])
    base_original['publicacao'][i] = re.findall(r'\b[A-zÀ-úũ]+\b', base_original['publicacao'][i])
    base_original['publicacao'][i] = [p for p in base_original['publicacao'][i] if p not in stopwordsnltk]
    base_original['publicacao'][i] = [p for p in base_original['publicacao'][i] if len(p) > 2]
    base_original['publicacao'][i] = [str(stemmer.stem(p)) for p in base_original['publicacao'][i]]
    
    i = i + 1


for zeta in range(100):
    print('Rodada '+str(zeta))
    
    #Separando base de teste e treino
    #Treino 355 - 70%
    #Teste 153 - 30%
    base_total = base_original.sample(frac=1)

    base = base_total.head(355)
    base = base.reset_index(drop=True)
    freq = pd.DataFrame(base['classe'].value_counts())

    baseTeste = base_total.tail(153)
    baseTeste = baseTeste.reset_index(drop=True)
    freqTeste = pd.DataFrame(baseTeste['classe'].value_counts())

    palavras = buscaPalavras(base['publicacao'])
    palavrasTeste = buscaPalavras(baseTeste['publicacao'])

    frequencia = buscaFrequencia(palavras)
    frequenciaTeste = buscaFrequencia(palavrasTeste)

    palavrasUnicas = buscaPalavrasUnicas(frequencia)
    palavrasUnicasTeste = buscaPalavrasUnicas(frequenciaTeste)


    #Oganização para os demais algoritmos
    i = 0
    X_train = []
    Y_train = []
    while i < len(base):
        texto = extratorPalavras(base['publicacao'][i])
        X_train.append(list(texto.values()))
        Y_train.append(base['classe'][i])
        i = i + 1

    i = 0
    X_test = []
    Y_test = []
    while i < len(baseTeste):
        texto = extratorPalavras(baseTeste['publicacao'][i])
        X_test.append(list(texto.values()))    
        Y_test.append(baseTeste['classe'][i])
        i = i + 1


    # Extrair as características dos textos para o TF-IDF
    vectorizer = TfidfVectorizer()

    base2 = base
    baseTeste2 = baseTeste

    base2["publicacao"] = base2['publicacao'].apply(lambda x: ' '.join(x))
    baseTeste2["publicacao"] = baseTeste2['publicacao'].apply(lambda x: ' '.join(x))

    train_features = vectorizer.fit_transform(base2['publicacao'])
    test_features = vectorizer.transform(baseTeste2['publicacao'])


    #Aplicando Naive Bayes#######################################
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)

    # Fazer previsões e avaliar o desempenho do modelo
    previsoes = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, previsoes)
    print("Acurácia de Naive Bayes:", accuracy)

    mc = confusion_matrix(Y_test, previsoes)
    list_mc_NB = get_list_MT(mc)

    ac_NB = accuracy
    indices = calculaIndices(list_mc_NB)
    precisao_NB = indices['precisao']
    recall_NB = indices['recall']
    f_score_NB = indices['f_score']

    #Naive Bayes TFIDF###################################
    classifier = MultinomialNB()
    classifier.fit(train_features, Y_train)

    # Fazer previsões e avaliar o desempenho do modelo
    previsoes = classifier.predict(test_features)
    accuracy = accuracy_score(Y_test, previsoes)
    #print("Acurácia de Naive Bayes com TF-IDF:", accuracy)
    mc = confusion_matrix(Y_test, previsoes)
    list_mc_NB_TFIDF = get_list_MT(mc)

    ac_NB_TFIDF = accuracy

    indices = calculaIndices(list_mc_NB_TFIDF)
    precisao_NB_TFIDF = indices['precisao']
    recall_NB_TFIDF = indices['recall']
    f_score_NB_TFIDF = indices['f_score']


    #Árvore de decisão ####################################
    arvore = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore.fit(X_train, Y_train)

    previsoes = arvore.predict(X_test)

    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurácia de árvore de decisão: '+str(accuracy))

    ac_DT = str(accuracy)
    list_mc_DT = get_list_MT(mc)

    indices = calculaIndices(list_mc_DT)
    precisao_DT = indices['precisao']
    recall_DT = indices['recall']
    f_score_DT = indices['f_score']



    #Árvore de decisão com TF-IDF############################
    arvore = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore.fit(train_features, Y_train)

    previsoes = arvore.predict(test_features)

    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurácia de árvore de decisão: '+str(accuracy))

    ac_DT_TFIDF = str(accuracy)

    list_mc_DT_TFIDF = get_list_MT(mc)

    indices = calculaIndices(list_mc_DT_TFIDF)
    precisao_DT_TFIDF = indices['precisao']
    recall_DT_TFIDF = indices['recall']
    f_score_DT_TFIDF = indices['f_score']


    # Random Forest ############################################
    random_forest = RandomForestClassifier(criterion='entropy', random_state=0)
    random_forest.fit(X_train, Y_train)

    previsoes = random_forest.predict(X_test)

    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurácia da random forest: '+str(accuracy))
    ac_RF = str(accuracy)

    list_mc_RF = get_list_MT(mc)

    indices = calculaIndices(list_mc_RF)
    precisao_RF = indices['precisao']
    recall_RF = indices['recall']
    f_score_RF = indices['f_score']


    #Random forest com TF-IDF #################################
    random_forest = RandomForestClassifier(criterion='entropy', random_state=0)
    random_forest.fit(train_features, Y_train)

    previsoes = random_forest.predict(test_features)

    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurácia da random forest: '+str(accuracy))
    ac_RF_TFIDF = str(accuracy)

    list_mc_RF_TFIDF = get_list_MT(mc)

    indices = calculaIndices(list_mc_RF_TFIDF)
    precisao_RF_TFIDF = indices['precisao']
    recall_RF_TFIDF = indices['recall']
    f_score_RF_TFIDF = indices['f_score']

  
    #SVM #########################################################
    svm = SVC(kernel='linear')
    svm.fit(X_train, Y_train)

    previsoes = svm.predict(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurárica SVM: '+str(accuracy))
    ac_SVM = str(accuracy)

    list_mc_SVM = get_list_MT(mc)

    indices = calculaIndices(list_mc_SVM)
    precisao_SVM = indices['precisao']
    recall_SVM = indices['recall']
    f_score_SVM = indices['f_score']

    
    #SVM com TF-IDF ###############################################
    svm = SVC(kernel='linear')
    svm.fit(train_features, Y_train)

    previsoes = svm.predict(test_features)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(Y_test, previsoes)

    mc = confusion_matrix(Y_test, previsoes)

    print('Acurárica SVM: '+str(accuracy))
    ac_SVM_TFIDF = str(accuracy)

    list_mc_SVM_TFIDF = get_list_MT(mc)

    indices = calculaIndices(list_mc_SVM_TFIDF)
    precisao_SVM_TFIDF = indices['precisao']
    recall_SVM_TFIDF = indices['recall']
    f_score_SVM_TFIDF = indices['f_score']

    
    #BERT ###########################################################
    #################################################################
    data = base
    dataTeste = baseTeste

    data['publicacao'] = data['publicacao'].apply(lambda x: ' '.join(x))
    data['publicacao'] = data['publicacao'].apply(lambda x:clean_bert(x))

    ######Aplicando o Label Enconder para deixar o target com valores númericos
    encoder = LabelEncoder()
    encoder.fit(data['classe'])
    data['encoded'] = encoder.transform(data['classe'])

    bert_model =  BertModel.from_pretrained('BERTikal/').to(device)
    bert_tokenizer = BertTokenizer.from_pretrained('BERTikal/vocab.txt', do_lower_case=False)
    bert_tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    bert = data['publicacao'].apply(lambda x: bert_tokenizer.encode(x, add_special_tokens=True,max_length=512, truncation = True))

    wrapper = textwrap.TextWrapper()
    data_text = list(data['publicacao'])

    encoded_inputs = bert_tokenizer(data_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded_inputs.keys()
    input_ids = encoded_inputs['input_ids'].to(device)

    # Criando o nosso vetor de features 
    features = []

    # Aplicando o modelo pré-treinado em cada frase e adicionando-o ao nosso vetor
    for i in tqdm(range(len(data_text))):
        with torch.no_grad():        
            last_hidden_states = bert_model(input_ids[i:(i+1)])[1].cpu().numpy().reshape(-1).tolist()
        features.append(last_hidden_states)

    features = np.array(features)

    df_features = pd.DataFrame(features)
    features_label = pd.concat([df_features, data['encoded']], axis = 1)

    y_train = Y_train
    y_test = Y_test
    X_val = X_test
    y_val = Y_test

    random_seed=42

    tunned_model = CatBoostClassifier(
        loss_function = 'MultiClass',
        random_seed=random_seed,
    )

    tunned_model.fit(
        X_train, y_train,
        verbose=500,
        eval_set = (X_val, y_val),
        early_stopping_rounds = 100
    )

    y_cat_pred = tunned_model.predict(X_test)

    ac_BERT = accuracy_score(y_test, y_cat_pred)

    tunned_model.predict_proba(X_test)[:5]

    cm = confusion_matrix(y_test, y_cat_pred)
    list_mc_BERT = get_list_MT(cm)

    indices = calculaIndices(list_mc_BERT)
    precisao_BERT = indices['precisao']
    recall_BERT = indices['recall']
    f_score_BERT = indices['f_score']


    #######################################GPT-3.5
    #print(baseTeste.head(5))
    mc_GPT = [0,0,0,0,0,0,0,0,0]
    #print(mc_GPT)
    for i in range(len(baseTeste)):
        if (baseTeste['classe'][i] == 'DECISÃO INTERLOCUTÓRIA'):
            if (baseTeste['chatgpt'][i] == 'DECISÃO INTERLOCUTÓRIA'):
                mc_GPT[0] = mc_GPT[0] + 1
            if (baseTeste['chatgpt'][i] == 'DESPACHO'):
                mc_GPT[1] = mc_GPT[1] + 1
            if (baseTeste['chatgpt'][i] == 'SENTENÇA'):
                mc_GPT[2] = mc_GPT[2] + 1
        if (baseTeste['classe'][i] == 'DESPACHO'):
            if (baseTeste['chatgpt'][i] == 'DECISÃO INTERLOCUTÓRIA'):
                mc_GPT[3] = mc_GPT[3] + 1
            if (baseTeste['chatgpt'][i] == 'DESPACHO'):
                mc_GPT[4] = mc_GPT[4] + 1
            if (baseTeste['chatgpt'][i] == 'SENTENÇA'):
                mc_GPT[5] = mc_GPT[5] + 1
        if (baseTeste['classe'][i] == 'SENTENÇA'):
            if (baseTeste['chatgpt'][i] == 'DECISÃO INTERLOCUTÓRIA'):
                mc_GPT[6] = mc_GPT[6] + 1
            if (baseTeste['chatgpt'][i] == 'DESPACHO'):
                mc_GPT[7] = mc_GPT[7] + 1
            if (baseTeste['chatgpt'][i] == 'SENTENÇA'):
                mc_GPT[8] = mc_GPT[8] + 1

    soma = 0
    for i in range(9):
        soma = soma + mc_GPT[i]

    ac_GPT = (mc_GPT[0] + mc_GPT[4] + mc_GPT[8])/soma
    print(ac_GPT)

    indices = calculaIndices(mc_GPT)
    precisao_GPT = indices['precisao']
    recall_GPT = indices['recall']
    f_score_GPT = indices['f_score']

    ##################################################################
    ##################################################################

    desvio_padrao = np.std(freq['classe'])

    list_data = [freq['classe']['SENTENÇA'],
                freq['classe']['DECISÃO INTERLOCUTÓRIA'],
                freq['classe']['DESPACHO'],
                str(round(desvio_padrao,2)),
                ac_NB,
                precisao_NB,
                recall_NB,
                f_score_NB,
                ac_NB_TFIDF,
                precisao_NB_TFIDF,
                recall_NB_TFIDF,
                f_score_NB_TFIDF,
                ac_DT,
                precisao_DT,
                recall_DT,
                f_score_DT,
                ac_DT_TFIDF,
                precisao_DT_TFIDF,
                recall_DT_TFIDF,
                f_score_DT_TFIDF,
                ac_RF,
                precisao_RF,
                recall_RF,
                f_score_RF,
                ac_RF_TFIDF,
                precisao_RF_TFIDF,
                recall_RF_TFIDF,
                f_score_RF_TFIDF,
                ac_SVM,
                precisao_SVM,
                recall_SVM,
                f_score_SVM,
                ac_SVM_TFIDF,
                precisao_SVM_TFIDF,
                recall_SVM_TFIDF,
                f_score_SVM_TFIDF,
                ac_BERT,
                precisao_BERT,
                recall_BERT,
                f_score_BERT,
                ac_GPT,
                precisao_GPT,
                recall_GPT,
                f_score_GPT,
                
                ]

    grava_MC('list_mc_NB', list_mc_NB)
    grava_MC('list_mc_NB_TFIDF', list_mc_NB_TFIDF)
    grava_MC('list_mc_DT', list_mc_DT)
    grava_MC('list_mc_DT_TFIDF', list_mc_DT_TFIDF)
    grava_MC('list_mc_RF', list_mc_RF)
    grava_MC('list_mc_RF_TFIDF', list_mc_RF_TFIDF)
    grava_MC('list_mc_SVM', list_mc_SVM)
    grava_MC('list_mc_SVM_TFIDF', list_mc_SVM_TFIDF)
    grava_MC('list_mc_BERT', list_mc_BERT)
    grava_MC('list_mc_GPT', mc_GPT)                
        
    with open('resultadosTOTAL.csv', 'a', newline='') as csv:  
        writer_object = writer(csv)    
        writer_object.writerow(list_data)  
        csv.close()

exit()