# Análise de Sentimento e Temas em Reclamações de Consumidores

## Visão Geral
Este projeto foca na análise de sentimento e na identificação de temas em um dataset de reclamações de consumidores. O objetivo é automatizar a classificação de sentimentos (positivo/negativo) nas reclamações e extrair os principais temas de insatisfação para cada categoria de produto, auxiliando na compreensão das dores dos clientes e na melhoria de serviços.

## Objetivo
- Classificar automaticamente reclamações como 'positivas' ou 'negativas' usando pseudo-rotulagem e modelos de Machine Learning/Deep Learning.
- Identificar e quantificar os temas mais recorrentes de insatisfação por produto.
- Gerar insights acionáveis sobre as áreas que demandam atenção por parte das empresas.

## Metodologia

1.  **Carga e Pré-processamento de Dados**: Carregamento de um grande volume de reclamações, filtragem por tamanho e relevância de produtos, e limpeza textual (lematização/stemização, remoção de stopwords).
2.  **Pseudo-Rotulagem de Sentimento**: Utilização de um modelo pré-treinado (`distilbert-base-uncased-finetuned-sst-2-english`) para atribuir rótulos de sentimento (positivo/negativo) às reclamações, com um limiar de confiança para garantir a qualidade dos rótulos.
3.  **Divisão dos Dados**: Separação estratificada dos dados em conjuntos de treino, validação e teste para garantir a representatividade dos rótulos.
4.  **Treinamento do Modelo**: Priorização de fine-tuning com **DistilBERT** para classificação binária. Em caso de falha ou limitação de recursos, um modelo de fallback **MLPClassifier com TF-IDF** é utilizado.
5.  **Avaliação do Modelo**: Cálculo de métricas como Acurácia, Precisão, Recall, F1-Score e ROC-AUC no conjunto de teste.
6.  **Análise de Temas para Reclamações Negativas**: Identificação de reclamações classificadas como negativas e aplicação de um dicionário de regras para categorizar temas de insatisfação (e.g., fraude, cobrança indevida, falha no atendimento).
7.  **Extração de Termos Chave**: Geração de unigramas e bigramas mais frequentes por categoria de produto a partir das reclamações negativas.
8.  **Visualização**: Geração de nuvens de palavras e gráficos de frequência de termos por produto para facilitar a interpretação dos resultados.

## Resultados Chave

-   **Modelo Utilizado**: `distilbert`
-   **Acurácia no Teste**: 0.9946
-   **Precisão no Teste**: 0.9352
-   **Recall no Teste**: 0.7537
-   **F1-Score no Teste**: 0.8347
-   **ROC-AUC no Teste**: 0.9839
-   **Reclamações Negativas Previstas**: 7257

### Temas mais recorrentes por produto (Top 5)

-   **Checking or savings account**: fraude, outros, taxas_juros, falha_no_atendimento, estorno_reembolso
-   **Credit card**: taxas_juros, falha_no_atendimento, fraude, cobranca_indevida, erro_info_credito
-   **Credit card or prepaid card**: falha_no_atendimento, taxas_juros, outros, fraude, cobranca_indevida
-   **Credit reporting or other personal consumer reports**: erro_info_credito, fraude, roubo_identidade, outros, problemas_pagamento
-   **Credit reporting, credit repair services, or other personal consumer reports**: erro_info_credito, outros, fraude, roubo_identidade, taxas_juros
-   **Debt collection**: erro_info_credito, cobranca_divida, outros, fraude, taxas_juros
-   **Money transfer, virtual currency, or money service**: fraude, falha_no_atendimento, estorno_reembolso, outros, taxas_juros
-   **Mortgage**: outros, taxas_juros, falha_no_atendimento, estorno_reembolso, cobranca_indevida
-   **Student loan**: outros, falha_no_atendimento, erro_info_credito, taxas_juros, cobranca_indevida
-   **Vehicle loan or lease**: outros, taxas_juros, falha_no_atendimento, erro_info_credito, fraude

## Dependências
As principais bibliotecas e frameworks utilizados neste projeto incluem:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `wordcloud`
-   `scikit-learn`
-   `nltk`
-   `spacy`
-   `transformers`
-   `torch`
-   `datasets`
-   `accelerate`
-   `sentencepiece`

As dependências podem ser instaladas executando:
`!pip -q install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk spacy transformers datasets accelerate sentencepiece`
E para spaCy:
`!python -m spacy download en_core_web_sm`
