# Predição de Desistência em Turmas de Programação Utilizando Sentimentos
Este repositório contém o código utilizado no artigo **"Predição de Desistência em Turmas de Programação com Utilização de Sentimentos de Estudantes"**. O trabalho explora o uso de dados de desempenho e sentimentos coletados de alunos para prever a desistência em disciplinas de programação, utilizando técnicas de aprendizado de máquina.

## Descrição do Código

1. **Coleta e Pré-processamento de Dados:**
   - Processamento de dados de presença, notas e sentimentos (dados coletados via questionários abertos).
   - Anonimização e limpeza dos dados.
   - Conversão de dados qualitativos (sentimentos) para atributos numéricos, utilizando diferentes abordagens descritas no artigo.

2. **Modelos de Predição:**
   - Modelos explícitos baseados em pontuações manuais.
   - Modelos implícitos utilizando algoritmos de aprendizado de máquina:
     - Naive Bayes
     - Random Forest
     - Multilayer Perceptron
     - Regressão Logística
     - Suporte a Vetor de Máquina (SVM)

3. **Avaliação e Resultados:**
   - Divisão dos dados em conjuntos de treino e teste (amostragem aleatória, holdout, validação cruzada 10-fold).
   - Comparação de diferentes proporções de dados (25%, 50%, 75%, 100%) para simular intervenções em diferentes estágios do semestre.
