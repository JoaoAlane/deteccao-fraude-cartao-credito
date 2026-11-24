# Detecção de fraude em cartão de crédito

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoaoAlane/deteccao-fraude-cartao-credito/blob/main/deteccao_fraude.ipynb)

Classificação de transações de cartão de crédito em normais ou fraudulentas, com foco em **recall, precisão e F1 da classe de fraude**. Todo o trabalho está no notebook [`deteccao_fraude.ipynb`](deteccao_fraude.ipynb), salvo com todas as saídas (tabelas e gráficos), então dá para ler os resultados direto no GitHub sem executar nada.

**Resultado principal:** XGBoost com `scale_pos_weight`, variáveis de tempo e limiar de decisão 0,15 escolhido na validação cruzada. No conjunto de teste: **recall 0,831, precisão 0,831 e F1 0,831**, ou seja, 123 das 148 fraudes detectadas com 25 alarmes falsos em 85.443 transações.

---

## 1. O problema

A base tem **284.807 transações** feitas por cartões europeus em dois dias de setembro de 2013. Apenas **492 são fraudes (0,17%)**.

| Coluna | Descrição |
|---|---|
| `Time` | Segundos desde a primeira transação da base |
| `Amount` | Valor da transação |
| `V1` a `V28` | Componentes de PCA (as variáveis originais foram anonimizadas) |
| `Class` | Resposta: 1 = fraude, 0 = normal |

### Por que 99% de acurácia não quer dizer nada aqui

Um "modelo" que responde **normal para todas as transações** acerta 284.315 de 284.807 casos: **99,83% de acurácia sem detectar uma única fraude**. Neste projeto, os três modelos iniciais têm 99,9% de acurácia, inclusive a Regressão Logística, que deixa passar 38% das fraudes do teste. Com uma classe tão rara, a acurácia é dominada pelas transações normais e não mede o que interessa.

Por isso a avaliação olha só para a classe de fraude:

| Métrica | Pergunta que responde | Por que importa |
|---|---|---|
| **Recall** | De todas as fraudes, quantas o modelo pegou? | Fraude perdida é prejuízo direto. É a métrica principal. |
| **Precisão** | De todos os alertas, quantos eram fraude de verdade? | Alarme falso bloqueia cliente e ocupa a equipe de análise. |
| **F1** | Equilíbrio entre recall e precisão | Resume os dois num número para comparar modelos. |
| **Average precision (AP)** | Área sob a curva de precisão e recall | Avalia o modelo em todos os limiares, sem depender de um corte. |

A curva ROC também é mostrada, mas com ressalva: como os falsos positivos são divididos por mais de 85 mil transações normais, a ROC fica alta para todos os modelos. A Regressão Logística, por exemplo, tem ROC AUC maior que a do Random Forest (0,957 contra 0,948), mas average precision bem menor (0,710 contra 0,809).

## 2. Preparação dos dados

- **Leitura direta da URL** com `pd.read_csv`. O CSV tem cerca de 150 MB e não fica no repositório (`*.csv` está no `.gitignore`).
- **`Amount_log = np.log1p(Amount)`**: o valor é muito assimétrico (máximo acima de 25 mil, mediana de 22). A transformação reduz a assimetria de 16,98 para 0,16.
- **Divisão treino e teste estratificada** (70/30, `stratify=y`, `random_state=42`), mantendo a proporção de fraudes:

  | Conjunto | Transações | Fraudes | Fraudes (%) |
  |---|---:|---:|---:|
  | Base completa | 284.807 | 492 | 0,1727 |
  | Treino | 199.364 | 344 | 0,1725 |
  | Teste | 85.443 | 148 | 0,1732 |

- **Padronização com `StandardScaler`** ajustada **só no treino** e aplicada ao teste, para que média e desvio do teste não vazem para o treino. Depois a padronização passa para dentro de um `Pipeline`, que garante o mesmo cuidado em cada fold da validação cruzada.
- Variáveis usadas nos modelos: `Time`, `V1` a `V28` e `Amount_log`. As variáveis de tempo da seção 5 entram como evolução.

## 3. Comparação entre os modelos

Os três modelos pedidos, avaliados no teste com o limiar padrão de 0,5:

| Modelo | Recall | Precisão | F1 | ROC AUC | AP | Fraudes detectadas | Alarmes falsos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística (baseline) | 0,622 | 0,860 | 0,722 | 0,957 | 0,710 | 92 de 148 | 15 |
| Random Forest (`class_weight='balanced'`) | 0,750 | **0,910** | 0,822 | 0,948 | 0,809 | 111 de 148 | **11** |
| XGBoost (`scale_pos_weight` = 578,5) | **0,791** | 0,886 | **0,836** | **0,973** | **0,838** | **117 de 148** | 15 |

- O **XGBoost** tem o maior recall, o maior F1 e a maior average precision: é o modelo que melhor ordena as transações por risco.
- O **Random Forest** é o mais conservador: menos alarmes falsos, mas perde mais fraudes.
- A **Regressão Logística** perde 56 das 148 fraudes. Ela serve de referência para medir o ganho dos modelos de árvore.

### Tabela final: todos os modelos e variações

Todos avaliados no mesmo conjunto de teste e ordenados pelo F1 da fraude. Em negrito, o modelo recomendado.

| # | Modelo | Variação | Limiar | Recall | Precisão | F1 | AP | Detectadas | Perdidas | Alarmes falsos |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | XGBoost | Sem reamostragem e sem `scale_pos_weight` | 0,50 | 0,784 | 0,959 | 0,862 | 0,834 | 116 | 32 | 5 |
| 2 | XGBoost | Originais + variáveis de tempo | 0,50 | 0,791 | 0,893 | 0,839 | 0,842 | 117 | 31 | 14 |
| 3 | XGBoost | `scale_pos_weight` | 0,50 | 0,791 | 0,886 | 0,836 | 0,838 | 117 | 31 | 15 |
| **4** | **XGBoost** | **Modelo final: variáveis de tempo + limiar escolhido na validação cruzada** | **0,15** | **0,831** | **0,831** | **0,831** | **0,842** | **123** | **25** | **25** |
| 5 | Random Forest | `class_weight='balanced'` | 0,50 | 0,750 | 0,910 | 0,822 | 0,809 | 111 | 37 | 11 |
| 6 | Random Forest | Pipeline + limiar escolhido na validação cruzada | 0,25 | 0,831 | 0,778 | 0,804 | 0,809 | 123 | 25 | 35 |
| 7 | XGBoost | Oversampling (SMOTE) | 0,50 | 0,811 | 0,745 | 0,777 | 0,825 | 120 | 28 | 41 |
| 8 | Regressão Logística | Baseline | 0,50 | 0,622 | 0,860 | 0,722 | 0,710 | 92 | 56 | 15 |
| 9 | Regressão Logística | Pipeline, limiar 0,3 | 0,30 | 0,655 | 0,795 | 0,719 | 0,710 | 97 | 51 | 25 |
| 10 | Regressão Logística | GridSearchCV (`scoring='recall'`): C=10, `class_weight='balanced'` | 0,50 | 0,872 | 0,065 | 0,121 | 0,702 | 129 | 19 | 1.859 |
| 11 | Regressão Logística | Oversampling (SMOTE) | 0,50 | 0,872 | 0,062 | 0,116 | 0,706 | 129 | 19 | 1.950 |
| 12 | Regressão Logística | Undersampling (RandomUnderSampler) | 0,50 | 0,878 | 0,058 | 0,110 | 0,547 | 130 | 18 | 2.096 |
| 13 | XGBoost | Undersampling (RandomUnderSampler) | 0,50 | 0,892 | 0,050 | 0,094 | 0,719 | 132 | 16 | 2.515 |

Como ler a tabela:

- **O maior F1 (linha 1) não é o modelo recomendado.** O XGBoost sem peso é o mais conservador (precisão 0,959), mas detecta 7 fraudes a menos que o modelo final. Como o foco é recall, o modelo final troca um pouco de precisão por mais fraudes detectadas. A average precision dos dois é parecida (0,834 e 0,842): a diferença está principalmente no ponto de corte.
- **O maior recall (linhas 10 a 13) também não serve.** São modelos que marcam fraude em excesso: de 1.859 a 2.515 alarmes falsos para menos de 135 fraudes detectadas.
- Contra o Random Forest com limiar ajustado (linha 6), o modelo final detecta as mesmas 123 fraudes com 10 alarmes falsos a menos.

## 4. Limiar de decisão e o que o SHAP mostrou

### Limiar de 0,5 contra 0,3

Na Regressão Logística em `Pipeline`, baixar o limiar de 0,5 para 0,3 pega 5 fraudes a mais (97 contra 92) ao custo de 10 alarmes falsos a mais (25 contra 15). O F1 praticamente não muda (0,722 e 0,719). O notebook traz a tabela completa de recall, precisão e F1 para limiares de 0,05 a 0,9 e o gráfico das três métricas em função do limiar.

### Limiar escolhido: 0,15 (XGBoost final)

**Regra:** o maior recall possível mantendo a precisão em pelo menos 80%, ou seja, no máximo 1 alarme falso para cada 4 fraudes detectadas. Entre limiares com o mesmo recall, fica o de maior precisão.

**Como foi escolhido:** com `cross_val_predict` (5 folds estratificados), cada transação do treino recebeu uma probabilidade prevista por um modelo que não a viu. O limiar foi escolhido nessas previsões e aplicado **uma única vez** no teste. Escolher o limiar olhando o teste seria vazamento e deixaria o resultado otimista.

| Limiar no teste | Recall | Precisão | F1 | Fraudes detectadas | Alarmes falsos |
|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,791 | 0,893 | 0,839 | 117 | 14 |
| 0,30 | 0,804 | 0,875 | 0,838 | 119 | 17 |
| **0,15 (escolhido)** | **0,831** | **0,831** | **0,831** | **123** | **25** |

**Justificativa:** em fraude de cartão os erros não custam o mesmo. Uma fraude perdida é prejuízo direto; um alarme falso custa uma verificação. Com o limiar 0,15 o modelo pega 6 fraudes a mais que com 0,5, em troca de 11 alarmes falsos a mais em mais de 85 mil transações. O piso de 80% de precisão (`PRECISAO_MINIMA` no notebook) é um parâmetro de negócio: se a equipe de análise tiver mais capacidade, ele pode descer e o recall sobe.

### O que o SHAP mostrou

O SHAP foi calculado com `TreeExplainer` no XGBoost final, sobre 700 transações normais sorteadas do teste mais todas as 148 fraudes do teste, usando 100 transações do treino como referência.

- **V4, V14, V12, V10 e V3 concentram a maior parte do impacto.** Valores altos de V4 e V11 empurram para fraude; valores muito negativos de V14, V12, V10 e V3 também. V14 dá os maiores empurrões individuais, passando de 6 em log-odds em algumas fraudes.
- **A importância por ganho do XGBoost engana.** Ela atribui metade da importância a V14 (0,505), que é usada nas primeiras divisões. O SHAP distribui o crédito de forma mais equilibrada e mostra variáveis que o ganho esconde: V3 é a 5ª no SHAP e só a 29ª no ganho.
- **Valor baixo aumenta o risco**, coerente com o pico de fraudes em valores muito pequenos visto na exploração.
- **Das variáveis de tempo, só `Segundos_desde_mesmo_valor` aparece entre as 15 de maior impacto** (11ª posição).
- **Explicação de uma fraude detectada (waterfall):** a transação 45732, de 1,00, parte do risco de uma transação típica (−11,83 em log-odds, 0,0007%) e chega a 99,99%. V14 = −10,9 soma +5,50 sozinha (75% das transações têm V14 acima de −0,43), seguida de V4 (+3,98), V10 (+2,85), V12 (+2,81) e V3 (+1,91). É a combinação de vários componentes fora do padrão que leva o risco de quase zero a quase certo.

Como `V1` a `V28` são componentes de PCA anonimizados, o SHAP diz quais componentes pesam, mas não permite traduzir isso em regras de negócio.

## 5. O que mudou em relação ao que a Expert fez na aula

A base do projeto segue o roteiro da aula: exploração, preparação com `Amount_log`, `StandardScaler` e divisão estratificada, os três modelos, avaliação com `classification_report`, matriz de confusão e curvas ROC e de precisão e recall, `Pipeline` com comparação dos limiares 0,5 e 0,3, `GridSearchCV` com `scoring='recall'` e `StratifiedKFold`, importância das variáveis e SHAP. As evoluções abaixo foram além disso:

| Evolução | O que foi feito | Resultado |
|---|---|---|
| **Undersampling contra oversampling** | `RandomUnderSampler` e `SMOTE` comparados pelo recall dentro de `imblearn.pipeline`, para que a reamostragem aconteça só na parte de treino de cada fold, sem vazamento para a validação e o teste. Testado na Regressão Logística e no XGBoost. | As duas técnicas sobem o recall (até 0,93 na validação), mas derrubam a precisão: o undersampling gera mais de 2 mil alarmes falsos no teste. O SMOTE é o mais equilibrado (XGBoost: recall 0,846 e precisão 0,739 na validação). Nenhuma melhora a average precision. |
| **Random Forest no Pipeline com ajuste de limiar** | Random Forest dentro de `Pipeline`, com limiar escolhido nas previsões fora da amostra do treino (não no teste). | Limiar 0,25: recall no teste de 0,750 para 0,831 (12 fraudes a mais), com 35 alarmes falsos. |
| **Variáveis de comportamento ao longo do tempo** | Seis variáveis criadas a partir de `Time`, usando só transações anteriores: hora do dia em forma cíclica (`Hora_sin`, `Hora_cos`), ritmo do sistema (`Segundos_desde_anterior`, `Transacoes_ultima_hora`) e repetição de valor (`Mesmo_valor_ultima_hora`, `Segundos_desde_mesmo_valor`). | Ganho pequeno: recall médio na validação cruzada de 0,831 para 0,840 (melhora em 3 de 5 folds, nenhuma piora), menor que o desvio entre folds. No teste, com limiar 0,5, o recall fica igual (0,791). Sem identificador de cartão, só é possível medir o comportamento do sistema inteiro, um sinal fraco. |
| **Tabela final** | Todos os modelos e variações comparados pelo recall, precisão e F1 da fraude no mesmo teste. | Ver a seção 3. |

Outros cuidados adicionados:

- **Limiar escolhido fora do teste**, com `cross_val_predict` e uma regra de negócio explícita.
- **Análise da armadilha do `scoring='recall'`:** o GridSearchCV fez exatamente o que foi pedido e escolheu `class_weight='balanced'`, com recall de 0,872 e precisão de 0,065 no teste (1.859 alarmes falsos). Recall precisa vir acompanhado de uma restrição de precisão.
- **Curva ROC também em escala log**, para mostrar a diferença entre os modelos nas taxas de falso positivo que importam na prática.
- **SHAP com referência explícita** (`feature_perturbation='interventional'` e 100 transações do treino como fundo), para que o valor base do waterfall represente uma transação típica. O modo padrão usa a cobertura das árvores, distorcida pelo `scale_pos_weight` de 578.

## 6. Como rodar

O notebook lê os dados direto da URL, então não é preciso baixar nada manualmente.

### Google Colab

1. Clique no botão **Abrir no Colab** no topo deste README (ou acesse `https://colab.research.google.com/github/JoaoAlane/deteccao-fraude-cartao-credito/blob/main/deteccao_fraude.ipynb`).
2. No menu, use **Ambiente de execução > Executar tudo**.

A primeira célula de código instala `xgboost`, `imbalanced-learn` e `shap` caso algum deles não esteja disponível no ambiente do Colab. Numa máquina com 12 núcleos o notebook inteiro roda em 5 a 6 minutos; no Colab gratuito, que tem menos núcleos, deve demorar mais.

### Localmente

```bash
git clone https://github.com/JoaoAlane/deteccao-fraude-cartao-credito.git
cd deteccao-fraude-cartao-credito
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook deteccao_fraude.ipynb
```

No Windows, ative o ambiente com `.venv\Scripts\activate` em vez de `source .venv/bin/activate`.

Para executar o notebook do início ao fim pela linha de comando e salvar todas as saídas no próprio arquivo:

```bash
jupyter nbconvert --to notebook --execute --inplace deteccao_fraude.ipynb
```

As saídas salvas foram geradas com Python 3.14, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1, imbalanced-learn 0.14.2 e shap 0.54.1.

## Estrutura do repositório

| Arquivo | Conteúdo |
|---|---|
| `deteccao_fraude.ipynb` | Notebook completo, com todas as saídas salvas |
| `README.md` | Este documento |
| `requirements.txt` | Bibliotecas necessárias |
| `.gitignore` | Ignora arquivos `.csv`, ambientes virtuais e checkpoints do Jupyter |

## Dados

Base de transações com cartão de crédito publicada pelo Machine Learning Group da Université Libre de Bruxelles (ULB) e disponibilizada no Kaggle. O notebook lê a cópia hospedada em `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`.
